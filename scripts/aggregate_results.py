from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=str, required=True, help="Root like outputs/swimmer_single_path")
    parser.add_argument("--metric", type=str, default="ep_return_mean")
    parser.add_argument("--save", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)
    seed_dirs = sorted([p for p in runs_root.glob("seed_*") if p.is_dir()])
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found under {runs_root}")

    frames = []
    for seed_dir in seed_dirs:
        csv_path = seed_dir / "metrics.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if args.metric not in df.columns:
            raise KeyError(f"Metric '{args.metric}' not found in {csv_path}")
        df = df[["epoch", args.metric]].copy()
        df["seed"] = seed_dir.name
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No metrics.csv files found")

    full = pd.concat(frames, ignore_index=True)
    summary = (
        full.groupby("epoch", as_index=False)[args.metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": f"{args.metric}_mean", "std": f"{args.metric}_std", "count": "num_seeds"})
    )

    out_csv = runs_root / f"aggregate_{args.metric}.csv"
    summary.to_csv(out_csv, index=False)

    plt.figure(figsize=(8, 5))
    x = summary["epoch"]
    y = summary[f"{args.metric}_mean"]
    ystd = summary[f"{args.metric}_std"].fillna(0.0)
    plt.plot(x, y)
    plt.fill_between(x, y - ystd, y + ystd, alpha=0.2)
    plt.xlabel("epoch")
    plt.ylabel(args.metric)
    plt.title(runs_root.name)
    plt.tight_layout()

    out_png = Path(args.save) if args.save else runs_root / f"aggregate_{args.metric}.png"
    plt.savefig(out_png, dpi=150)

    print(f"Saved aggregate CSV to: {out_csv}")
    print(f"Saved plot to: {out_png}")


if __name__ == "__main__":
    main()
