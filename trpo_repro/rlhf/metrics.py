from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask_f = mask.to(dtype=x.dtype)
    return (x * mask_f).sum() / mask_f.sum().clamp_min(eps)


def masked_var(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = masked_mean(x, mask, eps=eps)
    return masked_mean((x - mean) ** 2, mask, eps=eps)


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    with torch.no_grad():
        if mask is not None:
            valid = mask.bool()
            y_pred = y_pred[valid]
            y_true = y_true[valid]
        if y_true.numel() < 2:
            return float("nan")
        var_y = torch.var(y_true.float(), unbiased=False)
        if var_y.item() == 0:
            return float("nan")
        return float(1.0 - torch.var((y_true - y_pred).float(), unbiased=False).item() / var_y.item())


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(record: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def write_csv(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    records = list(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in records for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
