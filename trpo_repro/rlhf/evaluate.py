from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from trpo_repro.config import load_config, save_config

from .data import build_prompt_records, load_helpsteer3_preference, save_jsonl
from .lm_policy import FrozenCausalLM, TokenPolicyWithValue
from .metrics import write_csv
from .reward_model import RewardModel
from .rollout import GenerationConfig, collect_lm_rollouts


def _device_from_cfg(cfg: dict[str, Any]) -> torch.device:
    name = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if name == "cuda" and not torch.cuda.is_available():
        name = "cpu"
    return torch.device(name)


def _load_policy_or_base(cfg, checkpoint_dir: str | None, device: torch.device):
    model_name = str(cfg.model.get("name", "Qwen/Qwen2.5-0.5B-Instruct"))
    if checkpoint_dir:
        policy = TokenPolicyWithValue.load_rlhf_pretrained(
            checkpoint_dir,
            base_model_name=model_name,
            torch_dtype=str(cfg.model.get("torch_dtype", "auto")),
            device_map=cfg.model.get("policy_device_map"),
            load_in_4bit=bool(cfg.model.get("policy_load_in_4bit", cfg.model.get("load_in_4bit", False))),
            load_in_8bit=bool(cfg.model.get("policy_load_in_8bit", False)),
            trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        )
    else:
        policy = TokenPolicyWithValue.from_model_name(
            model_name,
            torch_dtype=str(cfg.model.get("torch_dtype", "auto")),
            device_map=cfg.model.get("policy_device_map"),
            load_in_4bit=bool(cfg.model.get("policy_load_in_4bit", cfg.model.get("load_in_4bit", False))),
            load_in_8bit=bool(cfg.model.get("policy_load_in_8bit", False)),
            lora=None,
            gradient_checkpointing=False,
            trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        )
    if cfg.model.get("policy_device_map") is None:
        policy.to(device)
    policy.eval()
    return policy


def run_before_after_eval(config_path: str | Path, *, output_dir: str | Path | None = None) -> Path:
    cfg = load_config(config_path)
    output_dir = Path(output_dir or cfg.eval.get("output_dir", "outputs/rlhf/qwen25_05b_helpsteer3_eval"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config_resolved.yaml")

    from transformers import AutoTokenizer

    model_name = str(cfg.model.get("name", "Qwen/Qwen2.5-0.5B-Instruct"))
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=bool(cfg.model.get("trust_remote_code", False)))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = _device_from_cfg(cfg.eval)

    raw = load_helpsteer3_preference(str(cfg.data.get("eval_split", "validation")))
    records = build_prompt_records(raw, tokenizer, max_samples=int(cfg.eval.get("num_prompts", 100)), shuffle=False)
    prompts = [r["prompt"] for r in records]

    base_policy = _load_policy_or_base(cfg, None, device)
    ppo_policy = _load_policy_or_base(cfg, str(cfg.eval.get("policy_checkpoint_dir")), device)
    reference = FrozenCausalLM.from_model_name(
        model_name,
        torch_dtype=str(cfg.model.get("ref_torch_dtype", cfg.model.get("torch_dtype", "auto"))),
        device_map=cfg.model.get("ref_device_map"),
        load_in_4bit=bool(cfg.model.get("ref_load_in_4bit", True)),
        load_in_8bit=bool(cfg.model.get("ref_load_in_8bit", False)),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    )
    if cfg.model.get("ref_device_map") is None:
        reference.to(device)

    reward_model = RewardModel.load_rlhf_pretrained(
        str(cfg.reward_model.get("checkpoint_dir")),
        base_model_name=model_name,
        torch_dtype=str(cfg.reward_model.get("torch_dtype", cfg.model.get("torch_dtype", "auto"))),
        device_map=cfg.reward_model.get("device_map"),
        load_in_4bit=bool(cfg.reward_model.get("load_in_4bit", True)),
        load_in_8bit=bool(cfg.reward_model.get("load_in_8bit", False)),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    )
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad_(False)
    if cfg.reward_model.get("device_map") is None:
        reward_model.to(device)

    generation = GenerationConfig(**dict(cfg.get("generation", {})))
    batch_size = int(cfg.eval.get("batch_size", 4))
    rows: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="eval batches"):
        batch_prompts = prompts[start : start + batch_size]
        meta = records[start : start + batch_size]
        base = collect_lm_rollouts(
            base_policy,
            reference,
            reward_model,
            tokenizer,
            batch_prompts,
            generation=generation,
            kl_coef=0.0,
            device=device,
            metadata=meta,
        )
        tuned = collect_lm_rollouts(
            ppo_policy,
            reference,
            reward_model,
            tokenizer,
            batch_prompts,
            generation=generation,
            kl_coef=0.0,
            device=device,
            metadata=meta,
        )
        for i, prompt in enumerate(batch_prompts):
            base_score = float(base.scores[i].item())
            ppo_score = float(tuned.scores[i].item())
            rows.append(
                {
                    "idx": start + i,
                    "domain": meta[i].get("domain", "unknown"),
                    "language": meta[i].get("language", "unknown"),
                    "prompt": prompt,
                    "base_response": (base.responses or [""])[i],
                    "ppo_response": (tuned.responses or [""])[i],
                    "base_reward": base_score,
                    "ppo_reward": ppo_score,
                    "reward_delta": ppo_score - base_score,
                    "winner": "ppo" if ppo_score > base_score else "base",
                }
            )

    save_jsonl(rows, output_dir / "before_after_samples.jsonl")
    write_csv(rows, output_dir / "before_after_samples.csv")
    _write_markdown_table(rows[: int(cfg.eval.get("num_demo_rows", 12))], output_dir / "before_after_demo.md")
    return output_dir


def _write_markdown_table(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = ["# RLHF Before/After Demo\n"]
    for row in rows:
        parts.append(f"## Example {row['idx']} — {row.get('domain', 'unknown')}\n")
        parts.append("**Prompt**\n\n")
        parts.append(str(row["prompt"]).strip()[:2000] + "\n\n")
        parts.append(f"**Base reward:** {row['base_reward']:.3f}\n\n")
        parts.append(str(row["base_response"]).strip()[:2000] + "\n\n")
        parts.append(f"**PPO reward:** {row['ppo_reward']:.3f}\n\n")
        parts.append(str(row["ppo_response"]).strip()[:2000] + "\n\n")
        parts.append(f"**Reward delta:** {row['reward_delta']:.3f}; **winner:** {row['winner']}\n\n---\n")
    path.write_text("\n".join(parts), encoding="utf-8")
