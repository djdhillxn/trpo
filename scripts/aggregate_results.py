import argparse
import json
from pathlib import Path
from typing import Iterable

import yaml

from _bootstrap import ensure_repo_root_on_path


def _load_plot_dependencies() -> None:
    global pd, plt
    try:
        import pandas as pd
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "aggregate_results.py requires working pandas, matplotlib, and numpy installations. "
            "Install the project dependencies with `python3 -m pip install -r requirements.txt`."
        ) from exc


_METRIC_ALIASES: dict[str, list[str]] = {
    "train_return_mean": ["ep_return_mean"],
    "train_return_std": ["ep_return_std"],
    "train_len_mean": ["ep_len_mean"],
    "ep_return_mean": ["train_return_mean"],
    "ep_return_std": ["train_return_std"],
    "ep_len_mean": ["train_len_mean"],
    "episodes_in_batch": ["episodes_in_epoch"],
    "episodes_in_epoch": ["episodes_in_batch"],
}

_X_AXIS_ALIASES: dict[str, list[str]] = {
    "iteration": ["epoch"],
    "epoch": ["iteration"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=str,
        action="append",
        required=True,
        help="Method root like outputs/cartpole_linear, or a specific seed dir. Repeat to compare methods.",
    )
    parser.add_argument("--metric", type=str, default="ep_return_mean")
    parser.add_argument("--x-axis", type=str, default="epoch", choices=["iteration", "epoch", "env_steps"])
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--labels", type=str, nargs="*", default=None)
    parser.add_argument("--allow-legacy-runs", action="store_true")
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--summary", action="store_true", help="Also save a final/best summary CSV for each plotted method.")
    return parser.parse_args()


def _iter_seed_dirs(root: Path) -> Iterable[Path]:
    if (root / "metrics.csv").exists():
        yield root
        return
    for seed_dir in sorted([p for p in root.glob("seed_*") if p.is_dir()]):
        if (seed_dir / "metrics.csv").exists():
            yield seed_dir


def _read_metadata(seed_dir: Path, *, allow_legacy_runs: bool) -> tuple[dict, bool]:
    meta_path = seed_dir / "run_metadata.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f), False
    if not allow_legacy_runs:
        raise FileNotFoundError(
            f"Legacy run directory detected at {seed_dir} (missing run_metadata.json). "
            "Re-run the experiment or pass --allow-legacy-runs."
        )
    print(f"Warning: legacy run directory without run_metadata.json: {seed_dir}")
    return {
        "method": seed_dir.parent.name,
        "method_variant": "unknown",
        "env_id": "unknown",
        "suite": "unknown",
        "seed": seed_dir.name,
        "run_name": seed_dir.parent.name,
    }, True


def _method_label(meta: dict) -> str:
    method = str(meta.get("method", "unknown"))
    variant = meta.get("method_variant")
    if variant in (None, "", "default"):
        return method
    return f"{method}:{variant}"


def _alias_candidates(name: str, aliases: dict[str, list[str]]) -> list[str]:
    candidates = [name, *aliases.get(name, [])]
    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def _resolve_column_name(columns: Iterable[str], requested: str, aliases: dict[str, list[str]]) -> str | None:
    available = set(columns)
    for candidate in _alias_candidates(requested, aliases):
        if candidate in available:
            return candidate
    return None


def _missing_column_error(kind: str, requested: str, csv_path: Path, columns: Iterable[str], aliases: dict[str, list[str]]) -> str:
    alias_candidates = [candidate for candidate in _alias_candidates(requested, aliases) if candidate != requested]
    alias_note = f" Checked aliases: {', '.join(alias_candidates)}." if alias_candidates else ""
    available = ", ".join(str(column) for column in columns)
    return f"{kind} '{requested}' not found in {csv_path}.{alias_note} Available columns: {available}"


def _load_steps_per_epoch(seed_dir: Path) -> int | None:
    config_path = seed_dir / "config_resolved.yaml"
    if not config_path.exists():
        return None
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        train_cfg = config.get("train") or {}
        steps_per_epoch = train_cfg.get("steps_per_epoch")
        return None if steps_per_epoch is None else int(steps_per_epoch)
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        return None


def _prepare_seed_frame(seed_dir: Path, metric: str, x_axis: str) -> "pd.DataFrame":
    csv_path = seed_dir / "metrics.csv"
    df = pd.read_csv(csv_path)
    metric_col = _resolve_column_name(df.columns, metric, _METRIC_ALIASES)
    if metric_col is None:
        raise KeyError(_missing_column_error("Metric", metric, csv_path, df.columns, _METRIC_ALIASES))
    x_col = _resolve_column_name(df.columns, x_axis, _X_AXIS_ALIASES)
    if x_col is None and x_axis == "env_steps":
        if "batch_env_steps" in df.columns:
            df = df.copy()
            df["env_steps"] = pd.to_numeric(df["batch_env_steps"], errors="coerce").cumsum()
            x_col = "env_steps"
        else:
            epoch_col = _resolve_column_name(df.columns, "epoch", _X_AXIS_ALIASES)
            steps_per_epoch = _load_steps_per_epoch(seed_dir)
            if epoch_col is not None and steps_per_epoch is not None:
                df = df.copy()
                df["env_steps"] = pd.to_numeric(df[epoch_col], errors="coerce") * steps_per_epoch
                x_col = "env_steps"
    if x_col is None:
        raise KeyError(_missing_column_error("x-axis", x_axis, csv_path, df.columns, _X_AXIS_ALIASES))
    out = df[[x_col, metric_col]].copy().rename(columns={x_col: x_axis, metric_col: metric})
    out = out.sort_values(x_axis).reset_index(drop=True)
    return out


def _aggregate_seed_frames(
    seed_dirs: list[Path],
    metric: str,
    x_axis: str,
    *,
    allow_legacy_runs: bool,
) -> tuple["pd.DataFrame", dict]:
    frames = []
    first_meta = None
    for seed_dir in seed_dirs:
        csv_path = seed_dir / "metrics.csv"
        if not csv_path.exists():
            continue
        meta, _legacy = _read_metadata(seed_dir, allow_legacy_runs=allow_legacy_runs)
        first_meta = first_meta or meta
        df = _prepare_seed_frame(seed_dir, metric=metric, x_axis=x_axis)
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
        .sort_values(x_axis)
        .reset_index(drop=True)
    )
    return summary, first_meta or {}


def _maybe_smooth(summary: "pd.DataFrame", metric: str, window: int) -> "pd.DataFrame":
    if window <= 1:
        return summary
    out = summary.copy()
    out[f"{metric}_mean"] = out[f"{metric}_mean"].rolling(window=window, min_periods=1).mean()
    out[f"{metric}_std"] = out[f"{metric}_std"].fillna(0.0).rolling(window=window, min_periods=1).mean()
    return out


def _summary_row(label: str, env_id: str, metric: str, x_axis: str, summary: "pd.DataFrame") -> dict[str, object]:
    ordered = summary.sort_values(x_axis).reset_index(drop=True)
    final_row = ordered.iloc[-1]
    best_idx = ordered[f"{metric}_mean"].idxmax()
    best_row = ordered.loc[best_idx]
    return {
        "label": label,
        "env_id": env_id,
        "metric": metric,
        "x_axis": x_axis,
        "final_x": float(final_row[x_axis]),
        "final_mean": float(final_row[f"{metric}_mean"]),
        "final_std": float(final_row[f"{metric}_std"]) if pd.notna(final_row[f"{metric}_std"]) else float("nan"),
        "best_x": float(best_row[x_axis]),
        "best_mean": float(best_row[f"{metric}_mean"]),
        "best_std": float(best_row[f"{metric}_std"]) if pd.notna(best_row[f"{metric}_std"]) else float("nan"),
        "num_seeds": int(ordered["num_seeds"].max()) if "num_seeds" in ordered.columns else 1,
    }


def main():
    args = parse_args()
    ensure_repo_root_on_path()
    _load_plot_dependencies()

    from trpo_repro.utils.utils import ensure_dir, slugify

    run_roots = [Path(p) for p in args.runs_root]
    if args.labels is not None and len(args.labels) not in {0, len(run_roots)}:
        raise ValueError("--labels must be omitted or match the number of --runs-root entries.")
    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be >= 1")

    compare = bool(args.compare or len(run_roots) > 1)
    aggregate_frames: list[pd.DataFrame] = []
    plot_items: list[tuple[str, pd.DataFrame, Path, dict]] = []
    summary_rows: list[dict[str, object]] = []

    for idx, run_root in enumerate(run_roots):
        seed_dirs = list(_iter_seed_dirs(run_root))
        if not seed_dirs:
            raise FileNotFoundError(f"No seed directories with metrics found under {run_root}")
        summary, meta = _aggregate_seed_frames(
            seed_dirs,
            metric=args.metric,
            x_axis=args.x_axis,
            allow_legacy_runs=args.allow_legacy_runs,
        )
        label = args.labels[idx] if args.labels else _method_label(meta)
        summary["label"] = label
        summary["env_id"] = str(meta.get("env_id", "unknown"))
        aggregate_frames.append(summary)
        plot_items.append((label, summary, run_root, meta))
        summary_rows.append(_summary_row(label, str(meta.get("env_id", "unknown")), args.metric, args.x_axis, summary))

    common_env = None
    if compare:
        env_ids = {str(meta.get("env_id", "unknown")) for *_rest, meta in plot_items}
        if len(env_ids) != 1:
            raise ValueError(f"Comparison plots require all runs to be from the same environment. Found: {sorted(env_ids)}")
        common_env = next(iter(env_ids))

    plt.figure(figsize=(8, 5))
    for label, summary, _run_root, _meta in plot_items:
        plot_summary = _maybe_smooth(summary, args.metric, args.smooth_window)
        x = plot_summary[args.x_axis]
        y = plot_summary[f"{args.metric}_mean"]
        ystd = plot_summary[f"{args.metric}_std"].fillna(0.0)
        plt.plot(x, y, label=label)
        plt.fill_between(x, y - ystd, y + ystd, alpha=0.2)

    plt.xlabel(args.x_axis)
    plt.ylabel(args.metric)
    if compare:
        plt.legend()
    default_title = f"{common_env} comparison" if compare else run_roots[0].name
    if args.smooth_window > 1:
        default_title = f"{default_title} (smoothed, window={args.smooth_window})"
    plt.title(args.title or default_title)
    plt.tight_layout()

    if compare:
        env_slug = slugify(common_env or "unknown_env")
        labels_slug = "__".join(slugify(label) for label, *_ in plot_items)
        out_dir = ensure_dir(Path("outputs") / "comparisons" / env_slug)
        stem = (
            Path(args.save).stem
            if args.save
            else f"{env_slug}__{labels_slug}__{slugify(args.metric)}__by_{slugify(args.x_axis)}"
        )
        out_png = Path(args.save) if args.save else out_dir / f"{stem}.png"
        out_csv = out_dir / f"{stem}.csv"
        pd.concat(aggregate_frames, ignore_index=True).to_csv(out_csv, index=False)
        summary_csv = out_dir / f"{stem}__summary.csv"
    else:
        run_root = run_roots[0]
        env_slug = slugify(str(plot_items[0][3].get("env_id", run_root.name)))
        label_slug = slugify(plot_items[0][0])
        stem = (
            Path(args.save).stem
            if args.save
            else f"{env_slug}__{label_slug}__aggregate_{slugify(args.metric)}_by_{slugify(args.x_axis)}"
        )
        out_png = Path(args.save) if args.save else run_root / f"{stem}.png"
        out_csv = run_root / f"{stem}.csv"
        aggregate_frames[0].to_csv(out_csv, index=False)
        summary_csv = run_root / f"{stem}__summary.csv"

    plt.savefig(out_png, dpi=150)
    print(f"Saved aggregate CSV to: {out_csv}")
    print(f"Saved plot to: {out_png}")

    if args.summary:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_csv, index=False)
        print(f"Saved summary CSV to: {summary_csv}")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
