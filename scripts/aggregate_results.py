from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=str,
        action="append",
        required=True,
        help="Method root like outputs/cartpole_linear, or a specific seed dir. Repeat to compare methods.",
    )
    parser.add_argument("--metric", type=str, default="train_return_mean")
    parser.add_argument("--x-axis", type=str, default="iteration", choices=["iteration", "epoch", "env_steps"])
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--labels", type=str, nargs="*", default=None)
    return parser.parse_args()


def _iter_seed_dirs(root: Path) -> Iterable[Path]:
    if (root / "metrics.csv").exists():
        yield root
        return
    for seed_dir in sorted([p for p in root.glob("seed_*") if p.is_dir()]):
        if (seed_dir / "metrics.csv").exists():
            yield seed_dir


def _read_metadata(seed_dir: Path) -> dict:
    meta_path = seed_dir / "run_metadata.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback for older runs.
    return {
        "method": seed_dir.parent.name,
        "method_variant": "unknown",
        "env_id": "unknown",
        "suite": "unknown",
        "seed": seed_dir.name,
        "run_name": seed_dir.parent.name,
    }


def _method_label(meta: dict) -> str:
    method = str(meta.get("method", "unknown"))
    variant = meta.get("method_variant")
    if variant in (None, "", "default"):
        return method
    return f"{method}:{variant}"


def _aggregate_seed_frames(seed_dirs: list[Path], metric: str, x_axis: str) -> tuple[pd.DataFrame, dict]:
    frames = []
    first_meta = None
    for seed_dir in seed_dirs:
        csv_path = seed_dir / "metrics.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if metric not in df.columns:
            raise KeyError(f"Metric '{metric}' not found in {csv_path}")
        if x_axis not in df.columns:
            raise KeyError(f"x-axis '{x_axis}' not found in {csv_path}")
        meta = _read_metadata(seed_dir)
        first_meta = first_meta or meta
        df = df[[x_axis, metric]].copy()
        df["seed"] = str(meta.get("seed", seed_dir.name))
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No metrics.csv files found for the requested runs.")

    full = pd.concat(frames, ignore_index=True)
    summary = (
        full.groupby(x_axis, as_index=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std", "count": "num_seeds"})
    )
    return summary, first_meta or {}


def main():
    args = parse_args()
    run_roots = [Path(p) for p in args.runs_root]
    if args.labels is not None and len(args.labels) not in {0, len(run_roots)}:
        raise ValueError("--labels must be omitted or match the number of --runs-root entries.")

    compare = bool(args.compare or len(run_roots) > 1)
    aggregate_frames: list[pd.DataFrame] = []
    plot_items: list[tuple[str, pd.DataFrame, Path]] = []

    for idx, run_root in enumerate(run_roots):
        seed_dirs = list(_iter_seed_dirs(run_root))
        if not seed_dirs:
            raise FileNotFoundError(f"No seed directories with metrics found under {run_root}")
        summary, meta = _aggregate_seed_frames(seed_dirs, metric=args.metric, x_axis=args.x_axis)
        label = args.labels[idx] if args.labels else _method_label(meta)
        summary["label"] = label
        summary["env_id"] = str(meta.get("env_id", "unknown"))
        aggregate_frames.append(summary)
        plot_items.append((label, summary, run_root))

    plt.figure(figsize=(8, 5))
    for label, summary, _run_root in plot_items:
        x = summary[args.x_axis]
        y = summary[f"{args.metric}_mean"]
        ystd = summary[f"{args.metric}_std"].fillna(0.0)
        plt.plot(x, y, label=label)
        plt.fill_between(x, y - ystd, y + ystd, alpha=0.2)

    plt.xlabel(args.x_axis)
    plt.ylabel(args.metric)
    if compare:
        plt.legend()
    default_title = "Method comparison" if compare else run_roots[0].name
    plt.title(args.title or default_title)
    plt.tight_layout()

    if compare:
        out_dir = Path("outputs") / "comparisons"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = args.save if args.save else f"compare_{args.metric}_by_{args.x_axis}"
        out_png = Path(stem) if stem.endswith(".png") else out_dir / f"{Path(stem).name}.png"
        out_csv = out_dir / f"{Path(stem).stem}.csv"
        combined = pd.concat(aggregate_frames, ignore_index=True)
        combined.to_csv(out_csv, index=False)
    else:
        run_root = run_roots[0]
        out_png = Path(args.save) if args.save else run_root / f"aggregate_{args.metric}_by_{args.x_axis}.png"
        out_csv = run_root / f"aggregate_{args.metric}_by_{args.x_axis}.csv"
        aggregate_frames[0].to_csv(out_csv, index=False)

    plt.savefig(out_png, dpi=150)

    print(f"Saved aggregate CSV to: {out_csv}")
    print(f"Saved plot to: {out_png}")


if __name__ == "__main__":
    main()
