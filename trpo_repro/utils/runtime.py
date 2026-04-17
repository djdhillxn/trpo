from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch


RUN_ARTIFACT_NAMES = {
    "metrics.csv",
    "metrics.jsonl",
    "run_metadata.json",
    "config_resolved.yaml",
    "checkpoints",
    "aggregate_train_return_mean_by_iteration.csv",
    "aggregate_train_return_mean_by_iteration.png",
}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_env_spaces(env, seed: int) -> None:
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


def imported_package_path(package_name: str) -> str:
    module = __import__(package_name)
    return str(Path(module.__file__).resolve())


def prepare_run_dir(path: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists():
        has_known_artifacts = any((path / name).exists() for name in RUN_ARTIFACT_NAMES)
        has_any_contents = any(path.iterdir()) if path.is_dir() else True
        if (has_known_artifacts or has_any_contents) and not overwrite:
            raise FileExistsError(
                f"Run directory {path} already exists and is not empty. "
                "Pass --overwrite to replace it."
            )
        if overwrite:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class JsonlLogger:
    def __init__(self, output_dir: str | Path, mode: str = "w") -> None:
        if mode not in {"w", "a"}:
            raise ValueError(f"Unsupported logger mode: {mode}")
        self.output_dir = ensure_dir(output_dir)
        self.jsonl_path = self.output_dir / "metrics.jsonl"
        self.csv_path = self.output_dir / "metrics.csv"
        self.mode = mode
        self._fieldnames: list[str] | None = None
        self._csv_file = None
        self._csv_writer: csv.DictWriter[str] | None = None
        self._jsonl_file = self.jsonl_path.open(mode, encoding="utf-8")
        if mode == "a" and self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            with self.csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                self._fieldnames = next(reader, None)
        csv_needs_header = mode == "w" or not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        self._csv_file = self.csv_path.open(mode, newline="", encoding="utf-8")
        if self._fieldnames is not None:
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
        self._csv_needs_header = csv_needs_header

    def log(self, record: dict[str, Any]) -> None:
        if self._fieldnames is None:
            self._fieldnames = list(record.keys())
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            if self._csv_needs_header:
                self._csv_writer.writeheader()
        else:
            missing = [k for k in record.keys() if k not in self._fieldnames]
            if missing:
                raise ValueError(
                    f"Logger received new keys after initialization: {missing}. "
                    "Keep a stable logging schema within each run."
                )
        self._jsonl_file.write(json.dumps(record) + "\n")
        self._jsonl_file.flush()
        assert self._csv_writer is not None
        self._csv_writer.writerow({k: record.get(k) for k in self._fieldnames})
        self._csv_file.flush()

    def close(self) -> None:
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
