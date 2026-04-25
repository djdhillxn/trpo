from __future__ import annotations

import argparse
from pathlib import Path

import trpo_repro
from trpo_repro.config import apply_overrides, load_config, save_config
from trpo_repro.envs.factory import make_env
from trpo_repro.runner import Runner
from trpo_repro.utils.utils import imported_package_path, prepare_run_dir, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--memory-mode", choices=["standard", "safe"], default=None)
    parser.add_argument("--progress-mode", choices=["auto", "terminal", "notebook", "off"], default=None)
    parser.add_argument("--obs-storage", "--obs_storage", dest="obs_storage", choices=["auto", "ram", "memmap"], default=None)
    parser.add_argument("--full-batch-chunk-size", "--full_batch_chunk_size", dest="full_batch_chunk_size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    overrides = {}
    if args.seed is not None:
        overrides["train.seed"] = args.seed
    if args.memory_mode is not None:
        overrides["train.memory_mode"] = args.memory_mode
    if args.progress_mode is not None:
        overrides["train.progress_mode"] = args.progress_mode
    if args.obs_storage is not None:
        overrides["train.obs_storage"] = args.obs_storage
    if args.full_batch_chunk_size is not None:
        overrides["algo.full_batch_chunk_size"] = args.full_batch_chunk_size
    if overrides:
        cfg = apply_overrides(cfg, overrides)

    seed = int(cfg.train.seed)
    set_seed(seed)

    run_name = str(cfg.train.get("run_name", Path(args.config).stem))
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / run_name / f"seed_{seed}"
    output_dir = prepare_run_dir(output_dir, overwrite=args.overwrite)
    save_config(cfg, output_dir / "config_resolved.yaml")

    resolved_method = str(cfg.get("method", {}).get("name", "trpo"))
    resolved_estimator = str(cfg.algo.get("estimator", cfg.algo.get("advantage_mode", "mc")))
    print({
        "package_path": imported_package_path("trpo_repro"),
        "package_init": str(Path(trpo_repro.__file__).resolve()),
        "config": str(Path(args.config).resolve()),
        "output_dir": str(output_dir.resolve()),
        "method": resolved_method,
        "estimator": resolved_estimator,
        "seed": seed,
    })

    env = make_env(cfg, seed=seed)
    runner = Runner(env=env, cfg=cfg, output_dir=output_dir, device=args.device)
    runner.train()


if __name__ == "__main__":
    main()
