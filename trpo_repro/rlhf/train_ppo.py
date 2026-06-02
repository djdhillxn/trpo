import random
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from trpo_repro.config import load_config, save_config

from .data import build_prompt_records, load_helpsteer3_preference, save_jsonl
from .lm_policy import FrozenCausalLM, TokenPolicyWithValue
from .metrics import (
    append_jsonl,
    collect_run_metadata,
    jsonl_to_csv,
    read_jsonl,
    save_metric_plots,
    write_json,
)
from .ppo_lm import AdaptiveKLController, LMPPOTrainer
from .reward_model import RewardModel
from .rollout import GenerationConfig, collect_lm_rollouts


def _device_from_cfg(cfg: dict[str, Any]) -> torch.device:
    name = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if name == "cuda" and not torch.cuda.is_available():
        name = "cpu"
    return torch.device(name)


def _batched_cycle(records: list[dict[str, str]], batch_size: int, seed: int):
    rng = random.Random(seed)
    while True:
        shuffled = list(records)
        rng.shuffle(shuffled)
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start : start + batch_size]
            if batch:
                yield batch


def _cuda_memory() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "cuda_memory_allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 4),
        "cuda_memory_reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 4),
        "cuda_max_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 4),
    }


def run_ppo_training(config_path: str | Path, *, output_dir: str | Path | None = None) -> Path:
    cfg = load_config(config_path)
    output_dir = Path(output_dir or cfg.train.get("output_dir", "outputs/rlhf/qwen25_05b_helpsteer3_ppo"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    save_config(cfg, output_dir / "config_resolved.yaml")

    from transformers import AutoTokenizer

    model_name = str(cfg.model.get("name", "Qwen/Qwen2.5-0.5B-Instruct"))
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=bool(cfg.model.get("trust_remote_code", False)))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    seed = int(cfg.train.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device_from_cfg(cfg.train)

    raw_prompts = load_helpsteer3_preference(str(cfg.data.get("prompt_split", "train")))
    prompt_records = build_prompt_records(
        raw_prompts,
        tokenizer,
        max_samples=cfg.data.get("max_prompt_samples"),
        seed=seed,
        shuffle=True,
    )
    if not prompt_records:
        raise RuntimeError("No prompts were loaded from HelpSteer3.")
    save_jsonl(prompt_records[: min(len(prompt_records), 1000)], output_dir / "prompt_preview.jsonl")
    write_json(
        collect_run_metadata(
            run_type="rlhf_ppo",
            config_path=config_path,
            extra={"model_name": model_name, "num_prompt_records": len(prompt_records)},
        ),
        output_dir / "run_metadata.json",
    )

    policy_init_checkpoint = cfg.model.get("policy_init_checkpoint_dir")
    if policy_init_checkpoint:
        policy = TokenPolicyWithValue.load_rlhf_pretrained(
            str(policy_init_checkpoint),
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
            lora=dict(cfg.get("lora", {})),
            gradient_checkpointing=bool(cfg.model.get("gradient_checkpointing", True)),
            trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        )
    if cfg.model.get("policy_device_map") is None:
        policy.to(device)

    ref_checkpoint = cfg.model.get("ref_checkpoint_dir") or policy_init_checkpoint
    if ref_checkpoint:
        reference = FrozenCausalLM.load_rlhf_pretrained(
            str(ref_checkpoint),
            base_model_name=model_name,
            torch_dtype=str(cfg.model.get("ref_torch_dtype", cfg.model.get("torch_dtype", "auto"))),
            device_map=cfg.model.get("ref_device_map"),
            load_in_4bit=bool(cfg.model.get("ref_load_in_4bit", True)),
            load_in_8bit=bool(cfg.model.get("ref_load_in_8bit", False)),
            trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        )
    else:
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

    reward_checkpoint = Path(str(cfg.reward_model.get("checkpoint_dir")))
    if not reward_checkpoint.exists():
        raise FileNotFoundError(
            f"Reward checkpoint not found: {reward_checkpoint}. Run scripts/rlhf_train_reward_model.py first."
        )
    reward_model = RewardModel.load_rlhf_pretrained(
        reward_checkpoint,
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
    ppo_trainer = LMPPOTrainer(policy, dict(cfg.get("ppo", {})))
    kl_ctl = AdaptiveKLController(
        init_kl_coef=float(cfg.kl.get("init_kl_coef", 0.05)),
        target_kl=float(cfg.kl.get("target_ref_kl", 0.05)),
        horizon=int(cfg.kl.get("horizon", 10000)),
        min_kl_coef=float(cfg.kl.get("min_kl_coef", 0.02)),
        max_kl_coef=float(cfg.kl.get("max_kl_coef", 1.0)),
        adaptive=bool(cfg.kl.get("adaptive", True)),
    )

    batch_size = int(cfg.train.get("rollout_batch_size", 8))
    total_updates = int(cfg.train.get("total_updates", 1000))
    save_every = int(cfg.train.get("save_every", 100))
    sample_every = int(cfg.train.get("sample_every", 25))
    prompt_iter = _batched_cycle(prompt_records, batch_size, seed=seed)
    start_time = time.time()

    progress = tqdm(range(1, total_updates + 1), desc="PPO updates")
    for update_idx in progress:
        records = next(prompt_iter)
        group_size = int(cfg.train.get("num_generations_per_prompt", 1))
        if group_size > 1:
            prompts = []
            expanded_records = []
            for r in records:
                for sample_idx in range(group_size):
                    prompts.append(r["prompt"])
                    rr = dict(r)
                    rr["group_sample_idx"] = sample_idx
                    rr["group_size"] = group_size
                    expanded_records.append(rr)
        else:
            prompts = [r["prompt"] for r in records]
            expanded_records = records
        shaping_cfg = dict(cfg.get("reward_shaping", {}))
        rollout = collect_lm_rollouts(
            policy,
            reference,
            reward_model,
            tokenizer,
            prompts,
            generation=generation,
            kl_coef=kl_ctl.value,
            device=device,
            metadata=expanded_records,
            reward_clip_min=shaping_cfg.get("reward_clip_min"),
            reward_clip_max=shaping_cfg.get("reward_clip_max"),
            length_penalty_coef=float(shaping_cfg.get("length_penalty_coef", 0.0)),
            missing_eos_penalty=float(shaping_cfg.get("missing_eos_penalty", 0.0)),
            group_size=group_size,
            group_normalize=bool(shaping_cfg.get("group_normalize", False)),
            group_advantage_eps=float(shaping_cfg.get("group_advantage_eps", 1e-6)),
        )
        rollout = ppo_trainer.prepare_batch(rollout)
        stats = ppo_trainer.update(rollout, kl_coef=kl_ctl.value)
        kl_ctl.update(stats.objective_kl, stats.num_response_tokens)

        record = stats.__dict__.copy()
        record.update(
            {
                "update": update_idx,
                "elapsed_sec": time.time() - start_time,
                "mean_response_tokens": float(rollout.response_mask.sum(dim=1).float().mean().item()),
                "mean_response_chars": float(sum(len(x) for x in (rollout.responses or [])) / max(len(rollout.responses or []), 1)),
            }
        )
        record.update(_cuda_memory())
        append_jsonl(record, output_dir / "ppo_metrics.jsonl")
        if int(cfg.train.get("artifact_every", 25)) > 0 and update_idx % int(cfg.train.get("artifact_every", 25)) == 0:
            jsonl_to_csv(output_dir / "ppo_metrics.jsonl", output_dir / "ppo_metrics.csv")
        progress.set_postfix(
            reward=f"{record['reward_model_score']:.3f}",
            kl=f"{record['objective_kl']:.4f}",
            loss=f"{record['loss']:.3f}",
        )

        if update_idx == 1 or (sample_every > 0 and update_idx % sample_every == 0):
            sample_rows = []
            for prompt, response, meta, score in zip(rollout.prompts or [], rollout.responses or [], rollout.metadata or [], rollout.scores):
                sample_rows.append(
                    {
                        "update": update_idx,
                        "domain": meta.get("domain", "unknown") if isinstance(meta, dict) else "unknown",
                        "language": meta.get("language", "unknown") if isinstance(meta, dict) else "unknown",
                        "prompt": prompt,
                        "response": response,
                        "reward_score": float(score.item()),
                    }
                )
            save_jsonl(sample_rows, output_dir / f"samples/update_{update_idx:05d}.jsonl")

        if save_every > 0 and update_idx % save_every == 0:
            policy.save_rlhf_pretrained(output_dir / f"checkpoint_{update_idx:05d}", tokenizer=tokenizer)
            jsonl_to_csv(output_dir / "ppo_metrics.jsonl", output_dir / "ppo_metrics.csv")

    policy.save_rlhf_pretrained(output_dir / "checkpoint_final", tokenizer=tokenizer)
    jsonl_to_csv(output_dir / "ppo_metrics.jsonl", output_dir / "ppo_metrics.csv")
    rows = read_jsonl(output_dir / "ppo_metrics.jsonl")
    plot_paths = save_metric_plots(
        rows,
        output_dir / "plots",
        x_key="update",
        y_keys=[
            "reward_model_score",
            "total_reward",
            "objective_kl",
            "kl_coef",
            "approx_kl",
            "abs_ref_logratio",
            "clip_fraction",
            "loss",
            "policy_loss",
            "value_loss",
            "mean_response_tokens",
        ],
        prefix="ppo",
    )
    write_json(
        {"total_updates": total_updates, "final_kl_coef": kl_ctl.value, "plot_paths": plot_paths},
        output_dir / "run_summary.json",
    )
    return output_dir
