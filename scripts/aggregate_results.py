import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

try:
    from _bootstrap import ensure_repo_root_on_path
except ImportError:
    from ._bootstrap import ensure_repo_root_on_path


def _load_data_dependencies() -> None:
    global pd
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "aggregate_results.py requires working pandas and numpy installations. "
            "Install the project dependencies with `python3 -m pip install -r requirements.txt`."
        ) from exc


def _load_plot_dependencies(*, backend: str | None = "Agg") -> None:
    global plt
    _load_data_dependencies()
    try:
        import matplotlib

        if backend is not None:
            matplotlib.use(backend)
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "Plotting aggregate results requires a working matplotlib installation. "
            "Install the project dependencies with `python3 -m pip install -r requirements.txt`."
        ) from exc


@dataclass
class AggregatePlotResult:
    figure: Any
    axis: Any
    aggregate: Any
    summary: Any
    plot_path: Path | None = None
    plot_paths: list[Path] | None = None
    aggregate_csv_path: Path | None = None
    summary_csv_path: Path | None = None


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
    parser.add_argument(
        "--interval",
        type=str,
        default="std",
        choices=["std", "sem", "ci95", "none"],
        help="Uncertainty band around the seed mean: std is mean +/- one seed standard deviation.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=None,
        choices=["png", "pdf", "eps", "svg", "jpg", "jpeg"],
        help="Figure format(s) to save. Default: extension from --save if provided, otherwise png.",
    )
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


def _read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_yaml_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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


def _normalize_labels(labels: Sequence[str] | None, expected: int) -> list[str] | None:
    if labels is None:
        return None
    labels = list(labels)
    if len(labels) == 0:
        return None
    if len(labels) != expected:
        raise ValueError("labels must be omitted or match the number of run roots.")
    return labels


def _normalize_interval(interval: str | None) -> str:
    if interval is None:
        return "none"
    normalized = str(interval).strip().lower()
    aliases = {
        "sd": "std",
        "stdev": "std",
        "sigma": "std",
        "standard_deviation": "std",
        "standard-error": "sem",
        "standard_error": "sem",
        "se": "sem",
        "95ci": "ci95",
        "95_ci": "ci95",
        "confidence": "ci95",
        "off": "none",
        "false": "none",
        "no": "none",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"std", "sem", "ci95", "none"}:
        raise ValueError("interval must be one of: std, sem, ci95, none")
    return normalized


def _normalize_save_formats(save_formats: Sequence[str] | None, save_path: str | Path | None) -> list[str]:
    if save_formats is None:
        suffix = Path(save_path).suffix.lstrip(".") if save_path is not None else ""
        save_formats = [suffix or "png"]

    normalized: list[str] = []
    for fmt in save_formats:
        fmt = str(fmt).strip().lower().lstrip(".")
        if fmt == "jpg":
            fmt = "jpeg"
        if fmt not in {"png", "pdf", "eps", "svg", "jpeg"}:
            raise ValueError("save_formats must contain only: png, pdf, eps, svg, jpg, jpeg")
        if fmt not in normalized:
            normalized.append(fmt)
    return normalized


def _plot_paths_for_formats(
    *,
    save_path: str | Path | None,
    base_dir: Path,
    stem: str,
    save_formats: Sequence[str],
) -> list[Path]:
    if save_path is None:
        return [base_dir / f"{stem}.{fmt}" for fmt in save_formats]

    explicit_path = Path(save_path)
    if explicit_path.suffix and len(save_formats) == 1:
        return [explicit_path]
    return [explicit_path.with_suffix(f".{fmt}") for fmt in save_formats]


def _interval_half_width(summary: "pd.DataFrame", metric: str, interval: str) -> "pd.Series | None":
    interval = _normalize_interval(interval)
    if interval == "none":
        return None
    std = summary[f"{metric}_std"].fillna(0.0)
    if interval == "std":
        return std
    count = summary["num_seeds"].clip(lower=1)
    sem = std / (count**0.5)
    if interval == "sem":
        return sem
    return 1.96 * sem


def _pretty_axis_label(name: str) -> str:
    labels = {
        "epoch": "Epoch",
        "iteration": "Iteration",
        "env_steps": "Environment steps",
        "ep_return_mean": "Mean episode return",
        "train_return_mean": "Mean episode return",
        "ep_len_mean": "Mean episode length",
        "train_len_mean": "Mean episode length",
    }
    return labels.get(name, name.replace("_", " ").title())


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _nested_get(data: dict, path: str, default: Any = None) -> Any:
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _first_not_null(values: Iterable[Any], default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _space_shape(space: str | None) -> tuple[int, ...] | None:
    if not space:
        return None
    matches = re.findall(r"\((\d+(?:\s*,\s*\d+)*\s*,?)\)", str(space))
    if not matches:
        return None
    values = [int(part.strip()) for part in matches[-1].split(",") if part.strip()]
    return tuple(values) if values else None


def _action_info(action_space: str | None) -> tuple[str | None, int | None]:
    if not action_space:
        return None, None
    action_space = str(action_space)
    discrete_match = re.search(r"Discrete\((\d+)\)", action_space)
    if discrete_match:
        return "discrete", int(discrete_match.group(1))
    shape = _space_shape(action_space)
    if shape is None:
        return None, None
    dim = 1
    for size in shape:
        dim *= size
    return "continuous", dim


def _prod(values: Sequence[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def _linear_param_count(input_dim: int, output_dim: int) -> int:
    return int(input_dim) * int(output_dim) + int(output_dim)


def _mlp_param_count(input_dim: int, hidden_sizes: Sequence[int], output_dim: int) -> int:
    sizes = [int(input_dim), *[int(size) for size in hidden_sizes], int(output_dim)]
    return sum(_linear_param_count(sizes[idx], sizes[idx + 1]) for idx in range(len(sizes) - 1))


def _atari_body_dims(in_channels: int = 4, fc_dim: int = 20) -> tuple[int, int]:
    conv1_spatial = (84 - 8) // 4 + 1
    conv2_spatial = (conv1_spatial - 4) // 2 + 1
    flat_dim = 16 * conv2_spatial * conv2_spatial
    conv1_params = 16 * int(in_channels) * 8 * 8 + 16
    conv2_params = 16 * 16 * 4 * 4 + 16
    fc_params = _linear_param_count(flat_dim, int(fc_dim))
    return flat_dim, conv1_params + conv2_params + fc_params


def _model_param_summary(config: dict, meta: dict) -> dict[str, Any]:
    model_cfg = config.get("model") or {}
    algo_cfg = config.get("algo") or {}
    method_cfg = config.get("method") or {}
    method = str(meta.get("method") or method_cfg.get("name") or "unknown")
    estimator = str(meta.get("estimator") or algo_cfg.get("estimator") or "unknown")
    obs_space = str(
        _first_not_null(
            [
                meta.get("observation_space"),
                _nested_get(meta, "runtime.observation_space"),
            ],
            "",
        )
    )
    action_space = str(
        _first_not_null(
            [
                meta.get("action_space"),
                _nested_get(meta, "runtime.action_space"),
            ],
            "",
        )
    )
    obs_shape = _space_shape(obs_space)
    action_kind, action_dim = _action_info(action_space)
    policy_hidden = [int(size) for size in _as_list(model_cfg.get("policy_hidden_sizes"))]
    value_hidden = [int(size) for size in _as_list(model_cfg.get("value_hidden_sizes"))]
    activation = str(model_cfg.get("activation", "unknown"))
    cnn_fc_dim = int(model_cfg.get("cnn_fc_dim", 20))
    uses_value_function = method == "ppo" or estimator in {"gae", "value_baseline"}

    policy_params = None
    value_params = None
    policy_arch = "unknown"
    value_arch = "none"
    if obs_shape is not None and action_dim is not None:
        if len(obs_shape) == 3:
            in_channels = obs_shape[0]
            flat_dim, body_params = _atari_body_dims(in_channels=in_channels, fc_dim=cnn_fc_dim)
            policy_params = body_params + _linear_param_count(cnn_fc_dim, action_dim)
            if action_kind == "continuous":
                policy_params += action_dim
            head_name = "logits" if action_kind == "discrete" else "mean"
            policy_arch = (
                f"Atari CNN: conv {in_channels}->16 k8/s4, conv 16->16 k4/s2, "
                f"fc {flat_dim}->{cnn_fc_dim}, {head_name} {cnn_fc_dim}->{action_dim}"
            )
            if uses_value_function:
                value_params = body_params + _linear_param_count(cnn_fc_dim, 1)
                value_arch = f"Atari CNN critic: shared shape, separate params, fc {cnn_fc_dim}->1"
        else:
            input_dim = _prod(obs_shape)
            body_out_dim = policy_hidden[-1] if policy_hidden else input_dim
            body_params = _mlp_param_count(input_dim, policy_hidden[:-1], policy_hidden[-1]) if policy_hidden else 0
            policy_params = body_params + _linear_param_count(body_out_dim, action_dim)
            if action_kind == "continuous":
                policy_params += action_dim
            hidden_text = " -> ".join(str(size) for size in policy_hidden)
            policy_arch = (
                f"MLP policy: {input_dim}"
                + (f" -> {hidden_text}" if hidden_text else "")
                + f" -> {action_dim} ({action_kind or 'unknown'})"
            )
            if uses_value_function:
                value_params = _mlp_param_count(input_dim, value_hidden, 1)
                hidden_text = " -> ".join(str(size) for size in value_hidden)
                value_arch = f"MLP value: {input_dim}" + (f" -> {hidden_text}" if hidden_text else "") + " -> 1"

    return {
        "obs_space": obs_space or None,
        "action_space": action_space or None,
        "obs_shape": obs_shape,
        "action_kind": action_kind,
        "action_dim": action_dim,
        "activation": activation,
        "policy_hidden_sizes": policy_hidden,
        "value_hidden_sizes": value_hidden,
        "cnn_fc_dim": cnn_fc_dim,
        "uses_value_function": uses_value_function,
        "policy_architecture": policy_arch,
        "value_architecture": value_arch,
        "policy_param_count": policy_params,
        "value_param_count": value_params,
        "total_param_count": None
        if policy_params is None
        else int(policy_params) + (0 if value_params is None else int(value_params)),
    }


def _seed_summary_row(
    seed_dir: Path,
    *,
    label: str,
    metric: str,
    x_axis: str,
    allow_legacy_runs: bool,
    kl_threshold: float,
) -> dict[str, Any]:
    meta, _legacy = _read_metadata(seed_dir, allow_legacy_runs=allow_legacy_runs)
    config = _read_yaml_if_exists(seed_dir / "config_resolved.yaml")
    run_summary = _read_json_if_exists(seed_dir / "run_summary.json")
    environment = _read_json_if_exists(seed_dir / "environment.json")
    csv_path = seed_dir / "metrics.csv"
    df = pd.read_csv(csv_path)
    metric_col = _resolve_column_name(df.columns, metric, _METRIC_ALIASES)
    if metric_col is None:
        raise KeyError(_missing_column_error("Metric", metric, csv_path, df.columns, _METRIC_ALIASES))
    x_col = _resolve_column_name(df.columns, x_axis, _X_AXIS_ALIASES) or x_axis
    if x_col not in df.columns:
        prepared = _prepare_seed_frame(seed_dir, metric=metric, x_axis=x_axis)
        df = df.copy()
        df[x_axis] = prepared[x_axis]
        x_col = x_axis

    ordered = df.sort_values(x_col).reset_index(drop=True)
    values = pd.to_numeric(ordered[metric_col], errors="coerce")
    final_idx = values.last_valid_index()
    final_row = ordered.loc[final_idx] if final_idx is not None else ordered.iloc[-1]
    best_idx = values.idxmax()
    best_row = ordered.loc[best_idx]
    kl = pd.to_numeric(ordered["approx_kl"], errors="coerce") if "approx_kl" in ordered.columns else pd.Series(dtype=float)
    returns_delta = values.diff().abs().dropna()

    def metric_mean(column: str) -> float:
        if column not in ordered.columns:
            return float("nan")
        series = pd.to_numeric(ordered[column], errors="coerce")
        return float(series.mean()) if not series.dropna().empty else float("nan")

    def final_metric(column: str) -> float:
        if column not in final_row.index or pd.isna(final_row[column]):
            return float("nan")
        return float(final_row[column])

    train_cfg = config.get("train") or {}
    algo_cfg = config.get("algo") or {}
    method_cfg = config.get("method") or {}
    model_info = _model_param_summary(config, meta)
    package_versions = environment.get("package_versions") or {}
    cuda = environment.get("cuda") or {}

    row = {
        "run_root": seed_dir.parent,
        "run_name": seed_dir.parent.name,
        "seed_dir": seed_dir,
        "label": label,
        "seed": str(meta.get("seed", seed_dir.name)),
        "status": meta.get("status") or run_summary.get("status"),
        "env_id": str(meta.get("env_id", "unknown")),
        "suite": str(meta.get("suite", "unknown")),
        "method": str(meta.get("method") or method_cfg.get("name") or "unknown"),
        "method_variant": str(meta.get("method_variant") or method_cfg.get("variant") or "unknown"),
        "estimator": str(meta.get("estimator") or algo_cfg.get("estimator") or "unknown"),
        "target_epochs": _first_not_null([meta.get("target_epochs"), run_summary.get("target_epochs"), train_cfg.get("epochs")]),
        "completed_epochs": _first_not_null(
            [meta.get("completed_epochs"), run_summary.get("completed_epochs"), int(final_row[x_col]) if x_col in final_row else None]
        ),
        "final_x": float(final_row[x_col]) if x_col in final_row and pd.notna(final_row[x_col]) else float("nan"),
        "final_return": float(final_row[metric_col]) if pd.notna(final_row[metric_col]) else float("nan"),
        "final_return_std_in_batch": float(final_row.get("train_return_std", float("nan")))
        if pd.notna(final_row.get("train_return_std", float("nan")))
        else float("nan"),
        "best_x": float(best_row[x_col]) if x_col in best_row and pd.notna(best_row[x_col]) else float("nan"),
        "best_return": float(best_row[metric_col]) if pd.notna(best_row[metric_col]) else float("nan"),
        "mean_kl": float(kl.mean()) if not kl.dropna().empty else float("nan"),
        "min_kl": float(kl.min()) if not kl.dropna().empty else float("nan"),
        "max_kl": float(kl.max()) if not kl.dropna().empty else float("nan"),
        "kl_count": int(kl.count()),
        "kl_sum": float(kl.sum()) if not kl.dropna().empty else 0.0,
        "kl_gt_threshold_count": int((kl > kl_threshold).sum()) if not kl.dropna().empty else 0,
        "kl_gt_threshold_pct": float((kl > kl_threshold).mean() * 100.0) if not kl.dropna().empty else float("nan"),
        "final_approx_kl": final_metric("approx_kl"),
        "mean_entropy": metric_mean("entropy"),
        "final_entropy": final_metric("entropy"),
        "mean_clip_fraction": metric_mean("clip_fraction"),
        "final_clip_fraction": final_metric("clip_fraction"),
        "mean_line_search_success": metric_mean("line_search_success"),
        "final_line_search_success": final_metric("line_search_success"),
        "mean_cg_norm": metric_mean("cg_norm"),
        "final_cg_norm": final_metric("cg_norm"),
        "mean_value_loss_after": metric_mean("value_loss_after"),
        "final_value_loss_after": final_metric("value_loss_after"),
        "mean_abs_return_delta": float(returns_delta.mean()) if not returns_delta.empty else float("nan"),
        "max_abs_return_delta": float(returns_delta.max()) if not returns_delta.empty else float("nan"),
        "mean_collect_time_sec": metric_mean("collect_time_sec"),
        "mean_update_time_sec": metric_mean("update_time_sec"),
        "mean_wall_time_sec": metric_mean("wall_time_sec"),
        "total_env_steps": _first_not_null([meta.get("total_env_steps"), run_summary.get("total_env_steps")]),
        "steps_per_epoch": _first_not_null([meta.get("steps_per_epoch"), _nested_get(meta, "runtime.steps_per_epoch"), train_cfg.get("steps_per_epoch")]),
        "max_ep_len": _first_not_null([meta.get("max_ep_len"), _nested_get(meta, "runtime.max_ep_len"), train_cfg.get("max_ep_len")]),
        "num_workers": _first_not_null([meta.get("num_workers"), _nested_get(meta, "runtime.num_workers"), train_cfg.get("num_workers")]),
        "parallel_rollouts": _first_not_null([meta.get("parallel_rollouts"), _nested_get(meta, "runtime.parallel_rollouts")]),
        "device": _first_not_null([meta.get("device"), environment.get("requested_device")]),
        "cuda_device_name": cuda.get("device_name"),
        "torch_cuda_version": cuda.get("torch_cuda_version"),
        "python_version": _nested_get(environment, "python.version"),
        "torch_version": package_versions.get("torch"),
        "numpy_version": package_versions.get("numpy"),
        "gymnasium_version": package_versions.get("gymnasium"),
        "mujoco_version": package_versions.get("mujoco"),
        "ale_py_version": package_versions.get("ale-py"),
        "memory_mode": _first_not_null([meta.get("memory_mode"), _nested_get(meta, "runtime.memory_mode"), train_cfg.get("memory_mode")]),
        "obs_storage": _first_not_null([meta.get("obs_storage"), _nested_get(meta, "runtime.obs_storage"), train_cfg.get("obs_storage")]),
        "normalize_obs": _first_not_null([meta.get("normalize_obs"), _nested_get(meta, "runtime.normalize_obs"), train_cfg.get("normalize_obs")]),
        "gamma": algo_cfg.get("gamma"),
        "lam": algo_cfg.get("lam"),
        "normalize_weights": algo_cfg.get("normalize_weights"),
        "bootstrap_truncated_paths": algo_cfg.get("bootstrap_truncated_paths"),
        "max_kl_config": algo_cfg.get("max_kl"),
        "cg_iters": algo_cfg.get("cg_iters"),
        "cg_damping": algo_cfg.get("cg_damping"),
        "cg_residual_tol": algo_cfg.get("cg_residual_tol"),
        "backtrack_coeff": algo_cfg.get("backtrack_coeff"),
        "backtrack_iters": algo_cfg.get("backtrack_iters"),
        "fvp_kl_metric": algo_cfg.get("fvp_kl_metric"),
        "fvp_estimator": algo_cfg.get("fvp_estimator"),
        "fvp_subsample_fraction": algo_cfg.get("fvp_subsample_fraction"),
        "npg_stepsize": algo_cfg.get("npg_stepsize"),
        "vf_lr": algo_cfg.get("vf_lr"),
        "vf_iters": algo_cfg.get("vf_iters"),
        "vf_batch_size": algo_cfg.get("vf_batch_size"),
        "ppo_clip_ratio": algo_cfg.get("ppo_clip_ratio"),
        "ppo_update_epochs": algo_cfg.get("ppo_update_epochs"),
        "ppo_minibatch_size": algo_cfg.get("ppo_minibatch_size"),
        "ppo_target_kl": algo_cfg.get("ppo_target_kl"),
        "ppo_pi_lr": algo_cfg.get("ppo_pi_lr"),
        "ppo_vf_lr": algo_cfg.get("ppo_vf_lr"),
        "ppo_vf_epochs": algo_cfg.get("ppo_vf_epochs"),
        "ppo_entropy_coef": algo_cfg.get("ppo_entropy_coef"),
        "ppo_max_grad_norm": algo_cfg.get("ppo_max_grad_norm"),
        "ppo_anneal_lr": algo_cfg.get("ppo_anneal_lr"),
        "ppo_anneal_clip_ratio": algo_cfg.get("ppo_anneal_clip_ratio"),
    }
    row.update(model_info)
    return row


def aggregate_run_roots(
    run_roots: Sequence[str | Path],
    *,
    metric: str = "ep_return_mean",
    x_axis: str = "epoch",
    labels: Sequence[str] | None = None,
    allow_legacy_runs: bool = False,
    require_same_env: bool = False,
) -> tuple[list[tuple[str, "pd.DataFrame", Path, dict]], "pd.DataFrame", "pd.DataFrame"]:
    """Aggregate one or more method roots over seeds.

    Returns the per-method plot items, the long aggregate dataframe used for
    plotting, and a compact final/best summary table.
    """

    ensure_repo_root_on_path()
    _load_data_dependencies()

    run_roots = [Path(p) for p in run_roots]
    if not run_roots:
        raise ValueError("At least one run root is required.")
    labels = _normalize_labels(labels, len(run_roots))

    aggregate_frames: list[pd.DataFrame] = []
    plot_items: list[tuple[str, pd.DataFrame, Path, dict]] = []
    summary_rows: list[dict[str, object]] = []

    for idx, run_root in enumerate(run_roots):
        seed_dirs = list(_iter_seed_dirs(run_root))
        if not seed_dirs:
            raise FileNotFoundError(f"No seed directories with metrics found under {run_root}")
        summary, meta = _aggregate_seed_frames(
            seed_dirs,
            metric=metric,
            x_axis=x_axis,
            allow_legacy_runs=allow_legacy_runs,
        )
        label = labels[idx] if labels else _method_label(meta)
        summary["label"] = label
        summary["env_id"] = str(meta.get("env_id", "unknown"))
        aggregate_frames.append(summary)
        plot_items.append((label, summary, run_root, meta))
        summary_rows.append(_summary_row(label, str(meta.get("env_id", "unknown")), metric, x_axis, summary))

    if require_same_env:
        env_ids = {str(meta.get("env_id", "unknown")) for *_rest, meta in plot_items}
        if len(env_ids) != 1:
            raise ValueError(f"Comparison plots require all runs to be from the same environment. Found: {sorted(env_ids)}")

    return plot_items, pd.concat(aggregate_frames, ignore_index=True), pd.DataFrame(summary_rows)


def discover_run_groups(
    runs_root: str | Path = "outputs",
    *,
    allow_legacy_runs: bool = False,
) -> "pd.DataFrame":
    """Return one row per run group under an outputs directory."""

    ensure_repo_root_on_path()
    _load_data_dependencies()

    root = Path(runs_root)
    rows: list[dict[str, object]] = []
    if not root.exists():
        return pd.DataFrame(
            columns=[
                "run_root",
                "run_name",
                "env_id",
                "suite",
                "method",
                "method_variant",
                "label",
                "num_seeds",
                "seeds",
                "completed_epochs",
            ]
        )

    for run_root in sorted([p for p in root.iterdir() if p.is_dir()]):
        seed_dirs = list(_iter_seed_dirs(run_root))
        if not seed_dirs:
            continue
        metas = []
        seeds = []
        completed_epochs = []
        for seed_dir in seed_dirs:
            meta, _legacy = _read_metadata(seed_dir, allow_legacy_runs=allow_legacy_runs)
            metas.append(meta)
            seeds.append(str(meta.get("seed", seed_dir.name)))
            if meta.get("completed_epochs") is not None:
                completed_epochs.append(int(meta["completed_epochs"]))
        meta = metas[0]
        rows.append(
            {
                "run_root": run_root,
                "run_name": run_root.name,
                "env_id": str(meta.get("env_id", "unknown")),
                "suite": str(meta.get("suite", "unknown")),
                "method": str(meta.get("method", "unknown")),
                "method_variant": str(meta.get("method_variant", "unknown")),
                "label": _method_label(meta),
                "num_seeds": len(seed_dirs),
                "seeds": ", ".join(seeds),
                "completed_epochs": min(completed_epochs) if completed_epochs else None,
            }
        )
    return pd.DataFrame(rows)


def _run_roots_and_labels(
    run_groups: Sequence[str | Path | tuple[str | Path, str]] | dict[str, Sequence[tuple[str | Path, str]]],
    labels: Sequence[str] | None = None,
) -> tuple[list[str | Path], list[str] | None]:
    if isinstance(run_groups, dict):
        entries: list[tuple[str | Path, str]] = []
        for group_entries in run_groups.values():
            entries.extend(group_entries)
        run_groups = entries

    groups = list(run_groups)
    if groups and isinstance(groups[0], tuple):
        roots = [entry[0] for entry in groups]  # type: ignore[index]
        entry_labels = [str(entry[1]) for entry in groups]  # type: ignore[index]
        return roots, entry_labels
    return [entry for entry in groups], None if labels is None else list(labels)  # type: ignore[list-item]


def collect_seed_run_details(
    run_groups: Sequence[str | Path | tuple[str | Path, str]] | dict[str, Sequence[tuple[str | Path, str]]],
    *,
    labels: Sequence[str] | None = None,
    metric: str = "train_return_mean",
    x_axis: str = "epoch",
    allow_legacy_runs: bool = False,
    kl_threshold: float = 0.01,
) -> "pd.DataFrame":
    """Collect one rich diagnostics row per seed directory.

    This reads metrics, metadata, resolved config, run summary, and environment
    snapshots so report tables can be regenerated from the output folders.
    """

    ensure_repo_root_on_path()
    _load_data_dependencies()

    run_roots, labels = _run_roots_and_labels(run_groups, labels=labels)
    if not run_roots:
        return pd.DataFrame()
    labels = _normalize_labels(labels, len(run_roots)) if labels is not None else None

    rows: list[dict[str, Any]] = []
    for idx, run_root_like in enumerate(run_roots):
        run_root = Path(run_root_like)
        seed_dirs = list(_iter_seed_dirs(run_root))
        if not seed_dirs:
            raise FileNotFoundError(f"No seed directories with metrics found under {run_root}")
        first_meta, _legacy = _read_metadata(seed_dirs[0], allow_legacy_runs=allow_legacy_runs)
        label = labels[idx] if labels else _method_label(first_meta)
        for seed_dir in seed_dirs:
            rows.append(
                _seed_summary_row(
                    seed_dir,
                    label=label,
                    metric=metric,
                    x_axis=x_axis,
                    allow_legacy_runs=allow_legacy_runs,
                    kl_threshold=kl_threshold,
                )
            )
    return pd.DataFrame(rows)


def _format_mean_pm(mean: float, std: float, *, digits: int = 1, count: int = 1) -> str:
    if pd.isna(mean):
        return ""
    if count <= 1 or pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def summarize_run_groups(
    run_groups: Sequence[str | Path | tuple[str | Path, str]] | dict[str, Sequence[tuple[str | Path, str]]],
    *,
    labels: Sequence[str] | None = None,
    metric: str = "train_return_mean",
    x_axis: str = "epoch",
    allow_legacy_runs: bool = False,
    kl_threshold: float = 0.01,
    std_ddof: int = 0,
) -> "pd.DataFrame":
    """Summarize final/best return and KL diagnostics by run group."""

    seed_df = collect_seed_run_details(
        run_groups,
        labels=labels,
        metric=metric,
        x_axis=x_axis,
        allow_legacy_runs=allow_legacy_runs,
        kl_threshold=kl_threshold,
    )
    rows: list[dict[str, Any]] = []
    if seed_df.empty:
        return pd.DataFrame()

    for (_run_root, label), group in seed_df.groupby(["run_root", "label"], sort=False):
        final_returns = pd.to_numeric(group["final_return"], errors="coerce")
        kl_count = int(pd.to_numeric(group["kl_count"], errors="coerce").fillna(0).sum())
        kl_sum = float(pd.to_numeric(group["kl_sum"], errors="coerce").fillna(0.0).sum())
        kl_gt_count = int(pd.to_numeric(group["kl_gt_threshold_count"], errors="coerce").fillna(0).sum())
        count = int(final_returns.count())
        final_mean = float(final_returns.mean()) if count else float("nan")
        final_std = float(final_returns.std(ddof=std_ddof)) if count > 1 else 0.0
        row = {
            "run_root": _run_root,
            "run_name": group["run_name"].iloc[0],
            "label": label,
            "env_id": group["env_id"].iloc[0],
            "suite": group["suite"].iloc[0],
            "method": group["method"].iloc[0],
            "method_variant": group["method_variant"].iloc[0],
            "estimator": group["estimator"].iloc[0],
            "num_seeds": count,
            "seeds": ", ".join(str(seed) for seed in group["seed"].tolist()),
            "target_epochs": group["target_epochs"].iloc[0],
            "completed_epochs_min": pd.to_numeric(group["completed_epochs"], errors="coerce").min(),
            "completed_epochs_max": pd.to_numeric(group["completed_epochs"], errors="coerce").max(),
            "final_return_mean": final_mean,
            "final_return_std": final_std,
            "final_return": _format_mean_pm(final_mean, final_std, count=count),
            "best_return": float(pd.to_numeric(group["best_return"], errors="coerce").max()),
            "mean_kl": kl_sum / kl_count if kl_count else float("nan"),
            "min_kl": float(pd.to_numeric(group["min_kl"], errors="coerce").min()),
            "max_kl": float(pd.to_numeric(group["max_kl"], errors="coerce").max()),
            "kl_gt_threshold_pct": (100.0 * kl_gt_count / kl_count) if kl_count else float("nan"),
            "mean_entropy": float(pd.to_numeric(group["mean_entropy"], errors="coerce").mean()),
            "mean_clip_fraction": float(pd.to_numeric(group["mean_clip_fraction"], errors="coerce").mean()),
            "line_search_success_rate": float(pd.to_numeric(group["mean_line_search_success"], errors="coerce").mean()),
            "mean_cg_norm": float(pd.to_numeric(group["mean_cg_norm"], errors="coerce").mean()),
            "mean_value_loss_after": float(pd.to_numeric(group["mean_value_loss_after"], errors="coerce").mean()),
            "mean_abs_return_delta": float(pd.to_numeric(group["mean_abs_return_delta"], errors="coerce").mean()),
            "max_abs_return_delta": float(pd.to_numeric(group["max_abs_return_delta"], errors="coerce").max()),
            "mean_collect_time_sec": float(pd.to_numeric(group["mean_collect_time_sec"], errors="coerce").mean()),
            "mean_update_time_sec": float(pd.to_numeric(group["mean_update_time_sec"], errors="coerce").mean()),
            "mean_wall_time_sec": float(pd.to_numeric(group["mean_wall_time_sec"], errors="coerce").mean()),
            "total_env_steps_mean": float(pd.to_numeric(group["total_env_steps"], errors="coerce").mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def make_locomotion_report_table(
    run_groups: Sequence[str | Path | tuple[str | Path, str]] | dict[str, Sequence[tuple[str | Path, str]]],
    *,
    labels: Sequence[str] | None = None,
    metric: str = "train_return_mean",
    x_axis: str = "epoch",
    allow_legacy_runs: bool = False,
) -> "pd.DataFrame":
    """Return the long MuJoCo table used in the report."""

    summary = summarize_run_groups(
        run_groups,
        labels=labels,
        metric=metric,
        x_axis=x_axis,
        allow_legacy_runs=allow_legacy_runs,
    )
    if summary.empty:
        return summary
    table = summary[summary["suite"].eq("mujoco")].copy()
    return table[
        [
            "env_id",
            "label",
            "num_seeds",
            "final_return",
            "final_return_mean",
            "final_return_std",
            "best_return",
            "mean_kl",
            "min_kl",
            "max_kl",
            "kl_gt_threshold_pct",
            "mean_abs_return_delta",
        ]
    ].rename(columns={"env_id": "environment", "label": "method"})


def make_atari_report_table(
    run_groups: Sequence[str | Path | tuple[str | Path, str]] | dict[str, Sequence[tuple[str | Path, str]]],
    *,
    labels: Sequence[str] | None = None,
    metric: str = "train_return_mean",
    x_axis: str = "epoch",
    allow_legacy_runs: bool = False,
    methods: Sequence[str] = ("TRPO", "PPO"),
) -> "pd.DataFrame":
    """Return a wide Atari comparison table with one row per game."""

    summary = summarize_run_groups(
        run_groups,
        labels=labels,
        metric=metric,
        x_axis=x_axis,
        allow_legacy_runs=allow_legacy_runs,
    )
    summary = summary[summary["suite"].eq("atari")].copy()
    rows: list[dict[str, Any]] = []
    for env_id, group in summary.groupby("env_id", sort=False):
        row: dict[str, Any] = {
            "game": str(env_id).replace("ALE/", "").replace("-v5", ""),
            "env_id": env_id,
        }
        for method in methods:
            match = group[group["label"].eq(method)]
            if match.empty:
                continue
            item = match.iloc[0]
            row[f"{method} final"] = item["final_return_mean"]
            row[f"{method} best"] = item["best_return"]
            row[f"{method} mean_kl"] = item["mean_kl"]
            row[f"{method} epochs"] = item["completed_epochs_max"]
        rows.append(row)
    return pd.DataFrame(rows)


def make_npg_ablation_table(
    run_groups: Sequence[str | Path | tuple[str | Path, str]] | dict[str, Sequence[tuple[str | Path, str]]],
    *,
    labels: Sequence[str] | None = None,
    metric: str = "train_return_mean",
    x_axis: str = "epoch",
    allow_legacy_runs: bool = False,
    baseline_stepsize: float | None = None,
    kl_threshold: float = 0.01,
) -> "pd.DataFrame":
    """Return the Hopper-style NPG step-size ablation diagnostics."""

    seed_df = collect_seed_run_details(
        run_groups,
        labels=labels,
        metric=metric,
        x_axis=x_axis,
        allow_legacy_runs=allow_legacy_runs,
        kl_threshold=kl_threshold,
    )
    summary = summarize_run_groups(
        run_groups,
        labels=labels,
        metric=metric,
        x_axis=x_axis,
        allow_legacy_runs=allow_legacy_runs,
        kl_threshold=kl_threshold,
    )
    if summary.empty:
        return summary

    step_by_root = (
        seed_df.groupby(["run_root", "label"], sort=False)["npg_stepsize"]
        .first()
        .reset_index()
        .rename(columns={"npg_stepsize": "npg_stepsize"})
    )
    out = summary.merge(step_by_root, on=["run_root", "label"], how="left")
    steps = pd.to_numeric(out["npg_stepsize"], errors="coerce")
    if baseline_stepsize is None:
        positive = steps[steps > 0]
        baseline_stepsize = float(positive.min()) if not positive.empty else None

    def step_label(value: Any) -> str:
        value = float(value) if pd.notna(value) else float("nan")
        if baseline_stepsize is None or pd.isna(value):
            return ""
        ratio = value / baseline_stepsize
        if abs(ratio - round(ratio)) < 1e-6:
            return f"{int(round(ratio))}x"
        return f"{ratio:.2g}x"

    out["step"] = [step_label(value) for value in out["npg_stepsize"]]
    return out[
        [
            "step",
            "label",
            "npg_stepsize",
            "num_seeds",
            "final_return",
            "final_return_mean",
            "final_return_std",
            "best_return",
            "mean_kl",
            "max_kl",
            "kl_gt_threshold_pct",
            "mean_abs_return_delta",
            "max_abs_return_delta",
        ]
    ]


def _unique_join(values: Iterable[Any]) -> str:
    seen: list[str] = []
    for value in values:
        if isinstance(value, float) and pd.isna(value):
            continue
        if value is None:
            continue
        text = str(value)
        if text not in seen:
            seen.append(text)
    return ", ".join(seen)


def collect_run_hyperparameters(
    run_groups: Sequence[str | Path | tuple[str | Path, str]] | dict[str, Sequence[tuple[str | Path, str]]],
    *,
    labels: Sequence[str] | None = None,
    metric: str = "train_return_mean",
    x_axis: str = "epoch",
    allow_legacy_runs: bool = False,
) -> "pd.DataFrame":
    """Return one configuration/runtime/model row per run group."""

    seed_df = collect_seed_run_details(
        run_groups,
        labels=labels,
        metric=metric,
        x_axis=x_axis,
        allow_legacy_runs=allow_legacy_runs,
    )
    rows: list[dict[str, Any]] = []
    if seed_df.empty:
        return pd.DataFrame()

    first_fields = [
        "run_name",
        "label",
        "env_id",
        "suite",
        "method",
        "method_variant",
        "estimator",
        "target_epochs",
        "steps_per_epoch",
        "max_ep_len",
        "num_workers",
        "parallel_rollouts",
        "device",
        "memory_mode",
        "obs_storage",
        "normalize_obs",
        "gamma",
        "lam",
        "normalize_weights",
        "bootstrap_truncated_paths",
        "max_kl_config",
        "cg_iters",
        "cg_damping",
        "cg_residual_tol",
        "backtrack_coeff",
        "backtrack_iters",
        "fvp_kl_metric",
        "fvp_estimator",
        "fvp_subsample_fraction",
        "npg_stepsize",
        "vf_lr",
        "vf_iters",
        "vf_batch_size",
        "ppo_clip_ratio",
        "ppo_update_epochs",
        "ppo_minibatch_size",
        "ppo_target_kl",
        "ppo_pi_lr",
        "ppo_vf_lr",
        "ppo_vf_epochs",
        "ppo_entropy_coef",
        "ppo_max_grad_norm",
        "ppo_anneal_lr",
        "ppo_anneal_clip_ratio",
        "obs_space",
        "action_space",
        "obs_shape",
        "action_kind",
        "action_dim",
        "activation",
        "policy_hidden_sizes",
        "value_hidden_sizes",
        "cnn_fc_dim",
        "uses_value_function",
        "policy_architecture",
        "value_architecture",
        "policy_param_count",
        "value_param_count",
        "total_param_count",
    ]
    for (_run_root, label), group in seed_df.groupby(["run_root", "label"], sort=False):
        first = group.iloc[0]
        row = {field: first.get(field) for field in first_fields}
        completed = pd.to_numeric(group["completed_epochs"], errors="coerce")
        total_steps = pd.to_numeric(group["total_env_steps"], errors="coerce")
        row.update(
            {
                "run_root": _run_root,
                "num_seeds": int(group["seed"].count()),
                "seeds": _unique_join(group["seed"]),
                "completed_epochs_min": completed.min(),
                "completed_epochs_max": completed.max(),
                "total_env_steps_mean": total_steps.mean(),
                "total_env_steps_sum": total_steps.sum(),
                "cuda_device_name": _unique_join(group["cuda_device_name"]),
                "torch_cuda_version": _unique_join(group["torch_cuda_version"]),
                "torch_version": _unique_join(group["torch_version"]),
                "numpy_version": _unique_join(group["numpy_version"]),
                "gymnasium_version": _unique_join(group["gymnasium_version"]),
                "mujoco_version": _unique_join(group["mujoco_version"]),
                "ale_py_version": _unique_join(group["ale_py_version"]),
            }
        )
        rows.append(row)
    ordered = [
        "run_name",
        "label",
        "env_id",
        "suite",
        "method",
        "method_variant",
        "estimator",
        "num_seeds",
        "seeds",
        "target_epochs",
        "completed_epochs_min",
        "completed_epochs_max",
        "steps_per_epoch",
        "max_ep_len",
        "gamma",
        "lam",
        "max_kl_config",
        "npg_stepsize",
        "ppo_clip_ratio",
        "ppo_update_epochs",
        "ppo_minibatch_size",
        "ppo_target_kl",
        "ppo_pi_lr",
        "ppo_vf_lr",
        "ppo_entropy_coef",
        "policy_architecture",
        "value_architecture",
        "policy_param_count",
        "value_param_count",
        "total_param_count",
        "device",
        "cuda_device_name",
        "torch_cuda_version",
        "memory_mode",
        "obs_storage",
        "num_workers",
        "total_env_steps_mean",
        "total_env_steps_sum",
    ]
    df = pd.DataFrame(rows)
    return df[[column for column in ordered if column in df.columns] + [column for column in df.columns if column not in ordered]]


def plot_aggregate_results(
    run_roots: Sequence[str | Path],
    *,
    metric: str = "ep_return_mean",
    x_axis: str = "epoch",
    labels: Sequence[str] | None = None,
    allow_legacy_runs: bool = False,
    compare: bool | None = None,
    title: str | None = None,
    smooth_window: int = 1,
    interval: str | None = "std",
    interval_alpha: float = 0.18,
    figsize: tuple[float, float] = (8, 5),
    save: bool = False,
    save_path: str | Path | None = None,
    save_formats: Sequence[str] | None = None,
    output_dir: str | Path | None = None,
    save_csv: bool = True,
    save_summary: bool = False,
    dpi: int = 150,
    show: bool = False,
    close: bool = False,
    backend: str | None = None,
    ax: Any | None = None,
) -> AggregatePlotResult:
    """Plot seed-aggregated training curves for one method or a comparison.

    The shaded band is controlled by ``interval``:
    ``"std"`` plots mean +/- one seed standard deviation, ``"sem"`` plots the
    standard error of the mean, ``"ci95"`` uses a normal-approximation 95%
    confidence interval, and ``"none"`` disables the band.
    """

    ensure_repo_root_on_path()
    _load_plot_dependencies(backend=backend)

    from trpo_repro.utils.utils import ensure_dir, slugify

    if smooth_window < 1:
        raise ValueError("smooth_window must be >= 1")

    run_roots = [Path(p) for p in run_roots]
    compare = bool(len(run_roots) > 1 if compare is None else compare)
    interval = _normalize_interval(interval)
    plot_items, aggregate_df, summary_df = aggregate_run_roots(
        run_roots,
        metric=metric,
        x_axis=x_axis,
        labels=labels,
        allow_legacy_runs=allow_legacy_runs,
        require_same_env=compare,
    )

    common_env = None
    if compare:
        common_env = str(plot_items[0][3].get("env_id", "unknown"))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for label, summary, _run_root, _meta in plot_items:
        plot_summary = _maybe_smooth(summary, metric, smooth_window)
        x = pd.to_numeric(plot_summary[x_axis], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(plot_summary[f"{metric}_mean"], errors="coerce").to_numpy(dtype=float)
        (line,) = ax.plot(x, y, label=label, linewidth=2.0)
        half_width = _interval_half_width(plot_summary, metric, interval)
        if half_width is not None:
            band = pd.to_numeric(half_width, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            ax.fill_between(x, y - band, y + band, color=line.get_color(), alpha=interval_alpha, linewidth=0)

    ax.set_xlabel(_pretty_axis_label(x_axis))
    ax.set_ylabel(_pretty_axis_label(metric))
    if len(plot_items) > 1:
        ax.legend(frameon=False)
    default_title = f"{common_env} comparison" if compare else run_roots[0].name
    if smooth_window > 1:
        default_title = f"{default_title} (smoothed, window={smooth_window})"
    ax.set_title(title or default_title)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    plot_path = None
    plot_paths = None
    aggregate_csv_path = None
    summary_csv_path = None
    if save or save_path is not None:
        normalized_formats = _normalize_save_formats(save_formats, save_path)
        if compare:
            env_slug = slugify(common_env or "unknown_env")
            labels_slug = "__".join(slugify(label) for label, *_ in plot_items)
            base_dir = ensure_dir(Path(output_dir) if output_dir is not None else Path("outputs") / "comparisons" / env_slug)
            stem = (
                Path(save_path).stem
                if save_path is not None
                else f"{env_slug}__{labels_slug}__{slugify(metric)}__by_{slugify(x_axis)}"
            )
        else:
            run_root = run_roots[0]
            env_slug = slugify(str(plot_items[0][3].get("env_id", run_root.name)))
            label_slug = slugify(plot_items[0][0])
            base_dir = ensure_dir(Path(output_dir) if output_dir is not None else run_root)
            stem = (
                Path(save_path).stem
                if save_path is not None
                else f"{env_slug}__{label_slug}__aggregate_{slugify(metric)}_by_{slugify(x_axis)}"
            )

        plot_paths = _plot_paths_for_formats(
            save_path=save_path,
            base_dir=base_dir,
            stem=stem,
            save_formats=normalized_formats,
        )
        for path in plot_paths:
            ensure_dir(path.parent)
            fig.savefig(path, dpi=dpi, format=path.suffix.lstrip("."))
        plot_path = plot_paths[0]
        if save_csv:
            aggregate_csv_path = base_dir / f"{stem}.csv"
            aggregate_df.to_csv(aggregate_csv_path, index=False)
        if save_summary:
            summary_csv_path = base_dir / f"{stem}__summary.csv"
            summary_df.to_csv(summary_csv_path, index=False)

    if show:
        plt.show()
    if close:
        plt.close(fig)

    return AggregatePlotResult(
        figure=fig,
        axis=ax,
        aggregate=aggregate_df,
        summary=summary_df,
        plot_path=plot_path,
        plot_paths=plot_paths,
        aggregate_csv_path=aggregate_csv_path,
        summary_csv_path=summary_csv_path,
    )


def main():
    args = parse_args()
    result = plot_aggregate_results(
        args.runs_root,
        metric=args.metric,
        x_axis=args.x_axis,
        labels=args.labels,
        allow_legacy_runs=args.allow_legacy_runs,
        compare=bool(args.compare or len(args.runs_root) > 1),
        title=args.title,
        smooth_window=args.smooth_window,
        interval=args.interval,
        save=True,
        save_path=args.save,
        save_formats=args.formats,
        save_summary=args.summary,
        show=False,
        close=True,
        backend="Agg",
    )

    if result.aggregate_csv_path is not None:
        print(f"Saved aggregate CSV to: {result.aggregate_csv_path}")
    if result.plot_paths:
        for path in result.plot_paths:
            print(f"Saved plot to: {path}")
    elif result.plot_path is not None:
        print(f"Saved plot to: {result.plot_path}")
    if args.summary and result.summary_csv_path is not None:
        print(f"Saved summary CSV to: {result.summary_csv_path}")
        print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
