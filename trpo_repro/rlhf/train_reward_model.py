from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from trpo_repro.config import load_config, save_config

from .data import build_preference_pairs, load_helpsteer3_preference, preference_pairs_to_dicts, save_jsonl
from .metrics import append_jsonl, write_json
from .reward_model import RewardModel


class PreferencePairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx]


class PreferenceCollator:
    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, pairs):
        chosen_texts = [p.chosen_text for p in pairs]
        rejected_texts = [p.rejected_text for p in pairs]
        chosen = self.tokenizer(
            chosen_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        rejected = self.tokenizer(
            rejected_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "weights": torch.tensor([max(1.0, float(p.margin)) for p in pairs], dtype=torch.float32),
            "domains": [p.domain for p in pairs],
        }


def _device_from_cfg(cfg: dict[str, Any]) -> torch.device:
    name = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if name == "cuda" and not torch.cuda.is_available():
        name = "cpu"
    return torch.device(name)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def evaluate_reward_model(model: RewardModel, loader: DataLoader, device: torch.device, *, max_batches: int | None = None) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    total_margin = 0.0
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            batch = _move_batch(batch, device)
            chosen = model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rejected = model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
            diff = chosen - rejected
            weights = batch["weights"].to(diff.device)
            loss = -(weights * F.logsigmoid(diff)).mean()
            total_loss += float(loss.item()) * diff.numel()
            total_correct += int((diff > 0).sum().item())
            total_margin += float(diff.sum().item())
            total += int(diff.numel())
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": total_correct / max(total, 1),
        "avg_margin": total_margin / max(total, 1),
        "num_pairs": total,
    }


def run_reward_training(config_path: str | Path, *, output_dir: str | Path | None = None) -> Path:
    cfg = load_config(config_path)
    output_dir = Path(output_dir or cfg.train.get("output_dir", "outputs/rlhf/qwen25_05b_helpsteer3_reward"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config_resolved.yaml")

    from transformers import AutoTokenizer

    model_name = str(cfg.model.get("name", "Qwen/Qwen2.5-0.5B-Instruct"))
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=bool(cfg.model.get("trust_remote_code", False)))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_raw = load_helpsteer3_preference(str(cfg.data.get("train_split", "train")))
    val_raw = load_helpsteer3_preference(str(cfg.data.get("eval_split", "validation")))
    train_pairs = build_preference_pairs(
        train_raw,
        tokenizer,
        max_samples=cfg.data.get("max_train_samples"),
        shuffle=True,
        seed=int(cfg.train.get("seed", 0)),
    )
    val_pairs = build_preference_pairs(
        val_raw,
        tokenizer,
        max_samples=cfg.data.get("max_eval_samples", 1000),
        shuffle=False,
        seed=int(cfg.train.get("seed", 0)),
    )
    save_jsonl(preference_pairs_to_dicts(train_pairs[: min(len(train_pairs), 1000)]), output_dir / "train_pairs_preview.jsonl")
    save_jsonl(preference_pairs_to_dicts(val_pairs[: min(len(val_pairs), 1000)]), output_dir / "eval_pairs_preview.jsonl")

    device = _device_from_cfg(cfg.train)
    model = RewardModel.from_model_name(
        model_name,
        torch_dtype=str(cfg.model.get("torch_dtype", "auto")),
        device_map=cfg.model.get("device_map"),
        load_in_4bit=bool(cfg.model.get("load_in_4bit", False)),
        load_in_8bit=bool(cfg.model.get("load_in_8bit", False)),
        lora=dict(cfg.get("lora", {})),
        gradient_checkpointing=bool(cfg.model.get("gradient_checkpointing", True)),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    )
    if cfg.model.get("device_map") is None:
        model.to(device)

    collator = PreferenceCollator(tokenizer, max_length=int(cfg.data.get("max_length", 1024)))
    train_loader = DataLoader(
        PreferencePairDataset(train_pairs),
        batch_size=int(cfg.train.get("batch_size", 2)),
        shuffle=True,
        collate_fn=collator,
        num_workers=int(cfg.train.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        PreferencePairDataset(val_pairs),
        batch_size=int(cfg.train.get("eval_batch_size", cfg.train.get("batch_size", 2))),
        shuffle=False,
        collate_fn=collator,
        num_workers=int(cfg.train.get("num_workers", 0)),
    )

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(cfg.train.get("learning_rate", 2e-5)),
        weight_decay=float(cfg.train.get("weight_decay", 0.0)),
    )
    grad_accum = int(cfg.train.get("gradient_accumulation_steps", 8))
    max_grad_norm = float(cfg.train.get("max_grad_norm", 1.0))
    num_epochs = int(cfg.train.get("epochs", 1))
    log_every = int(cfg.train.get("log_every", 10))
    eval_every = int(cfg.train.get("eval_every", 200))
    global_step = 0
    running_loss = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"reward epoch {epoch + 1}/{num_epochs}")
        optimizer.zero_grad(set_to_none=True)
        for local_step, batch in enumerate(pbar, start=1):
            batch = _move_batch(batch, device)
            chosen = model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rejected = model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
            diff = chosen - rejected
            weights = batch["weights"].to(diff.device)
            loss = -(weights * F.logsigmoid(diff)).mean()
            (loss / grad_accum).backward()
            running_loss += float(loss.item())

            if local_step % grad_accum == 0:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % log_every == 0:
                    record = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": running_loss / max(log_every, 1),
                        "chosen_reward": float(chosen.mean().item()),
                        "rejected_reward": float(rejected.mean().item()),
                        "accuracy_batch": float((diff > 0).float().mean().item()),
                        "elapsed_sec": time.time() - start_time,
                    }
                    running_loss = 0.0
                    append_jsonl(record, output_dir / "train_metrics.jsonl")
                    pbar.set_postfix(loss=record["loss"], acc=record["accuracy_batch"])

                if eval_every > 0 and global_step % eval_every == 0:
                    eval_metrics = evaluate_reward_model(
                        model,
                        val_loader,
                        device,
                        max_batches=cfg.train.get("eval_max_batches", 50),
                    )
                    eval_metrics["step"] = global_step
                    append_jsonl(eval_metrics, output_dir / "eval_metrics.jsonl")
                    model.train()

    final_metrics = evaluate_reward_model(model, val_loader, device, max_batches=cfg.train.get("final_eval_max_batches"))
    write_json(final_metrics, output_dir / "final_eval_metrics.json")
    model.save_rlhf_pretrained(output_dir / "checkpoint_final", tokenizer=tokenizer)
    return output_dir
