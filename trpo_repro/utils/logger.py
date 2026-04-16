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
        self._fieldnames: list[str] | None = None

    def log(self, record: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if self._csv_writer is None:
            self._fieldnames = list(record.keys())
            self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            self._csv_writer.writeheader()
        else:
            missing = [k for k in record.keys() if k not in self._fieldnames]
            if missing:
                raise ValueError(
                    f"Logger received new keys after initialization: {missing}. "
                    "Keep a stable logging schema within each run."
                )

        row = {k: record.get(k) for k in self._fieldnames}
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
