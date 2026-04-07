from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from trpo_repro.utils.io import ensure_dir

class JsonlLogger:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.jsonl_path = self.output_dir / "metrics.jsonl"
        self.csv_path = self.output_dir / "metrics.csv"
        self._csv_writer: csv.DictWriter[str] | None = None
        self._csv_file = None

    def log(self, record: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        
        if self._csv_writer is None:
            self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(record.keys()))
            self._csv_writer.writeheader()
        self._csv_writer.writerow(record)
        self._csv_file.flush()

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
