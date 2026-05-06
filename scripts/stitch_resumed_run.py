import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from _bootstrap import ensure_repo_root_on_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create one stitched run directory from a pre-checkpoint run and a resumed run."
    )
    parser.add_argument("--source-run-dir", type=str, required=True, help="Original seed dir with metrics.csv.")
    parser.add_argument("--resume-run-dir", type=str, required=True, help="Resumed seed dir with metrics.csv.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output seed dir for the stitched run.")
    parser.add_argument(
        "--resume-epoch",
        type=int,
        default=None,
        help="Checkpoint epoch. Source rows <= this epoch are kept; resumed rows > this epoch are kept.",
    )
    parser.add_argument("--metric", type=str, default="train_return_mean", help="Metric used to select best.pt.")
    parser.add_argument("--minimize", action="store_true", help="Select best checkpoint by minimizing --metric.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output dir if it already exists.")
    parser.add_argument(
        "--no-copy-checkpoints",
        dest="copy_checkpoints",
        action="store_false",
        help="Write stitched metrics/metadata only, without copying checkpoint files.",
    )
    parser.set_defaults(copy_checkpoints=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise ValueError(f"Metrics file has no header: {path}")
    return fieldnames, rows


def _row_epoch(row: dict[str, Any]) -> int:
    raw = row.get("epoch") or row.get("iteration")
    if raw in (None, ""):
        raise ValueError(f"Metric row has no epoch/iteration value: {row}")
    return int(float(raw))


def _float_or_none(value: Any) -> float | None:
    if value in (None, "", "nan", "NaN"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _union_fieldnames(*fieldname_sets: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for fieldnames in fieldname_sets:
        for fieldname in fieldnames:
            if fieldname in seen:
                continue
            seen.add(fieldname)
            out.append(fieldname)
    return out


def _infer_resume_epoch(resume_meta: dict[str, Any] | None, resume_summary: dict[str, Any] | None) -> int | None:
    for data in (resume_meta, resume_summary):
        if not isinstance(data, dict):
            continue
        resume = data.get("resume")
        if isinstance(resume, dict) and resume.get("checkpoint_epoch") is not None:
            return int(resume["checkpoint_epoch"])
    return None


def _metadata_value(source_meta: dict[str, Any] | None, resume_meta: dict[str, Any] | None, key: str, default=None):
    if isinstance(resume_meta, dict) and resume_meta.get(key) is not None:
        return resume_meta.get(key)
    if isinstance(source_meta, dict) and source_meta.get(key) is not None:
        return source_meta.get(key)
    return default


def _latest_record(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return dict(rows[-1])


def _best_record(rows: list[dict[str, str]], metric: str, maximize: bool = True) -> dict[str, Any] | None:
    best = None
    best_score = None
    for row in rows:
        score = _float_or_none(row.get(metric))
        if score is None:
            continue
        if best_score is None or ((score > best_score) if maximize else (score < best_score)):
            best_score = score
            best = row
    if best is None:
        return None
    epoch = _row_epoch(best)
    return {
        "epoch": epoch,
        "env_steps": int(float(best.get("env_steps", 0) or 0)),
        metric: best_score,
        "metric": metric,
        "metric_value": best_score,
        "row": dict(best),
    }


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _copy_run_context(source_dir: Path, resume_dir: Path, output_dir: Path) -> None:
    for name in ("config_runtime.yaml", "config_resolved.yaml", "environment.json"):
        src = resume_dir / name if (resume_dir / name).exists() else source_dir / name
        _copy_if_exists(src, output_dir / name)
    for name in ("run_metadata.json", "run_summary.json", "launch_metadata.json", "run_manager_summary.json"):
        _copy_if_exists(source_dir / name, output_dir / f"source_{name}")
        _copy_if_exists(resume_dir / name, output_dir / f"resume_{name}")


def _checkpoint_for_epoch(run_dir: Path, epoch: int) -> Path:
    return run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"


def _copy_checkpoints(
    source_dir: Path,
    resume_dir: Path,
    output_dir: Path,
    source_epochs: set[int],
    resume_epochs: set[int],
) -> list[Path]:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for run_dir, epochs in ((source_dir, source_epochs), (resume_dir, resume_epochs)):
        for epoch in sorted(epochs):
            src = _checkpoint_for_epoch(run_dir, epoch)
            if not src.exists():
                continue
            dst = checkpoint_dir / src.name
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def _write_checkpoint_aliases(output_dir: Path, rows: list[dict[str, str]], metric: str, maximize: bool) -> dict[str, str | None]:
    checkpoint_dir = output_dir / "checkpoints"
    ckpts = sorted(checkpoint_dir.glob("epoch_*.pt")) if checkpoint_dir.exists() else []
    aliases: dict[str, str | None] = {"last_checkpoint": None, "best_checkpoint": None}
    if not ckpts:
        return aliases

    last = ckpts[-1]
    last_dst = checkpoint_dir / "last.pt"
    shutil.copy2(last, last_dst)
    aliases["last_checkpoint"] = str(last_dst.resolve())

    by_epoch = {path.stem.removeprefix("epoch_"): path for path in ckpts}
    best_ckpt = None
    best_row = None
    best_score = None
    for row in rows:
        score = _float_or_none(row.get(metric))
        if score is None:
            continue
        epoch_key = f"{_row_epoch(row):04d}"
        ckpt = by_epoch.get(epoch_key)
        if ckpt is None:
            continue
        if best_score is None or ((score > best_score) if maximize else (score < best_score)):
            best_score = score
            best_row = row
            best_ckpt = ckpt

    if best_ckpt is not None:
        best_dst = checkpoint_dir / "best.pt"
        shutil.copy2(best_ckpt, best_dst)
        aliases["best_checkpoint"] = str(best_dst.resolve())
        _write_json(
            checkpoint_dir / "best_checkpoint_info.json",
            {
                "source_checkpoint": str(best_ckpt.resolve()),
                "metric": metric,
                "maximize": maximize,
                "metric_row": best_row,
            },
        )
    return aliases


def _write_metrics(output_dir: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({fieldname: row.get(fieldname, "") for fieldname in fieldnames})
    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    ensure_repo_root_on_path()

    source_dir = Path(args.source_run_dir).expanduser().resolve()
    resume_dir = Path(args.resume_run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_meta = _read_json(source_dir / "run_metadata.json")
    resume_meta = _read_json(resume_dir / "run_metadata.json")
    source_summary = _read_json(source_dir / "run_summary.json")
    resume_summary = _read_json(resume_dir / "run_summary.json")

    resume_epoch = args.resume_epoch
    if resume_epoch is None:
        resume_epoch = _infer_resume_epoch(resume_meta, resume_summary)
    if resume_epoch is None:
        raise ValueError("Could not infer resume epoch. Pass --resume-epoch explicitly.")

    source_fields, source_rows_all = _read_csv_rows(source_dir / "metrics.csv")
    resume_fields, resume_rows_all = _read_csv_rows(resume_dir / "metrics.csv")
    source_rows = [row for row in source_rows_all if _row_epoch(row) <= resume_epoch]
    resume_rows = [row for row in resume_rows_all if _row_epoch(row) > resume_epoch]
    stitched_rows = sorted([*source_rows, *resume_rows], key=_row_epoch)

    if not stitched_rows:
        raise ValueError("Stitching produced no metric rows.")

    epochs = [_row_epoch(row) for row in stitched_rows]
    duplicates = sorted({epoch for epoch in epochs if epochs.count(epoch) > 1})
    if duplicates:
        raise ValueError(f"Stitched metrics contain duplicate epochs: {duplicates[:10]}")

    fieldnames = _union_fieldnames(source_fields, resume_fields)
    _write_metrics(output_dir, fieldnames, stitched_rows)
    _copy_run_context(source_dir, resume_dir, output_dir)

    source_epochs = {_row_epoch(row) for row in source_rows}
    resume_epochs = {_row_epoch(row) for row in resume_rows}
    copied_checkpoints: list[Path] = []
    aliases = {"last_checkpoint": None, "best_checkpoint": None}
    maximize = not args.minimize
    if args.copy_checkpoints:
        copied_checkpoints = _copy_checkpoints(source_dir, resume_dir, output_dir, source_epochs, resume_epochs)
        aliases = _write_checkpoint_aliases(output_dir, stitched_rows, metric=args.metric, maximize=maximize)

    latest = _latest_record(stitched_rows)
    best = _best_record(stitched_rows, args.metric, maximize=maximize)
    last_epoch = max(epochs)
    last_row = stitched_rows[-1]
    total_env_steps = int(float(last_row.get("env_steps", 0) or 0))
    seed = _metadata_value(source_meta, resume_meta, "seed", output_dir.name)
    run_name = output_dir.parent.name

    stitch_info = {
        "source_run_dir": str(source_dir),
        "resume_run_dir": str(resume_dir),
        "output_dir": str(output_dir),
        "resume_epoch": resume_epoch,
        "source_rows_kept": len(source_rows),
        "resume_rows_kept": len(resume_rows),
        "discarded_source_rows_after_resume_epoch": len(source_rows_all) - len(source_rows),
        "discarded_resume_rows_at_or_before_resume_epoch": len(resume_rows_all) - len(resume_rows),
        "first_epoch": min(epochs),
        "last_epoch": last_epoch,
        "metric": args.metric,
        "maximize_metric": maximize,
        "copied_checkpoints": [str(path.resolve()) for path in copied_checkpoints],
        "aliases": aliases,
    }

    run_metadata = {
        "schema_version": 1,
        "status": "stitched",
        "stitched": True,
        "method": _metadata_value(source_meta, resume_meta, "method", "unknown"),
        "method_variant": _metadata_value(source_meta, resume_meta, "method_variant", "unknown"),
        "estimator": _metadata_value(source_meta, resume_meta, "estimator", None),
        "env_id": _metadata_value(source_meta, resume_meta, "env_id", "unknown"),
        "suite": _metadata_value(source_meta, resume_meta, "suite", "unknown"),
        "seed": seed,
        "run_name": run_name,
        "trainable": _metadata_value(source_meta, resume_meta, "trainable", True),
        "source_run_metadata_path": str((output_dir / "source_run_metadata.json").resolve()),
        "resume_run_metadata_path": str((output_dir / "resume_run_metadata.json").resolve()),
        "stitch": stitch_info,
    }
    run_summary = {
        "schema_version": 1,
        "status": "stitched",
        "completed_epochs": last_epoch,
        "target_epochs": int(float(last_row.get("epoch", last_epoch) or last_epoch)),
        "total_env_steps": total_env_steps,
        "latest_epoch_record": latest,
        "best_train_return": best,
        "checkpoints": [str(path.resolve()) for path in copied_checkpoints],
        "aliases": aliases,
        "stitch": stitch_info,
    }

    _write_json(output_dir / "stitch_metadata.json", stitch_info)
    _write_json(output_dir / "run_metadata.json", run_metadata)
    _write_json(output_dir / "run_summary.json", run_summary)
    print(json.dumps({"output_dir": str(output_dir), **stitch_info}, indent=2))


if __name__ == "__main__":
    main()
