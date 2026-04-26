import argparse
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--method-variant", "--method_variant", dest="method_variant", type=str, default=None)
    parser.add_argument("--estimator", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", "--steps_per_epoch", dest="steps_per_epoch", type=int, default=None)
    parser.add_argument("--save-interval", "--save_interval", dest="save_interval", type=int, default=None)
    parser.add_argument("--num-workers", "--num_workers", "--num-cores", "--num_cores", dest="num_workers", type=int, default=None)
    parser.add_argument("--memory-mode", choices=["standard", "safe"], default=None)
    parser.add_argument("--progress-mode", choices=["auto", "terminal", "notebook", "off"], default=None)
    parser.add_argument("--obs-storage", "--obs_storage", dest="obs_storage", choices=["auto", "ram", "memmap"], default=None)
    parser.add_argument("--full-batch-chunk-size", "--full_batch_chunk_size", dest="full_batch_chunk_size", type=int, default=None)
    parser.add_argument("--fvp-subsample-fraction", "--fvp_subsample_fraction", dest="fvp_subsample_fraction", type=float, default=None)
    parser.add_argument("--fvp-estimator", "--fvp_estimator", dest="fvp_estimator", choices=["analytic", "empirical"], default=None)
    parser.add_argument("--npg-stepsize", "--npg_stepsize", dest="npg_stepsize", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    ensure_repo_root_on_path()
    import trpo_repro
    from trpo_repro.algos.advantages import canonicalize_estimator
    from trpo_repro.config import apply_overrides, load_config, save_config
    from trpo_repro.envs.factory import make_env
    from trpo_repro.runner import Runner
    from trpo_repro.utils.utils import imported_package_path, prepare_run_dir, set_seed

    cfg = load_config(args.config)
    overrides = {}
    if args.seed is not None:
        overrides["train.seed"] = args.seed
    if args.method is not None:
        overrides["method.name"] = args.method
    if args.method_variant is not None:
        overrides["method.variant"] = args.method_variant
    if args.estimator is not None:
        overrides["algo.estimator"] = args.estimator
    if args.epochs is not None:
        overrides["train.epochs"] = int(args.epochs)
    if args.steps_per_epoch is not None:
        overrides["train.steps_per_epoch"] = int(args.steps_per_epoch)
    if args.save_interval is not None:
        overrides["train.save_interval"] = int(args.save_interval)
    if args.memory_mode is not None:
        overrides["train.memory_mode"] = args.memory_mode
    if args.progress_mode is not None:
        overrides["train.progress_mode"] = args.progress_mode
    if args.num_workers is not None:
        overrides["train.num_workers"] = max(1, int(args.num_workers))
    if args.obs_storage is not None:
        overrides["train.obs_storage"] = args.obs_storage
    if args.full_batch_chunk_size is not None:
        overrides["algo.full_batch_chunk_size"] = args.full_batch_chunk_size
    if args.fvp_subsample_fraction is not None:
        overrides["algo.fvp_subsample_fraction"] = args.fvp_subsample_fraction
    if args.fvp_estimator is not None:
        overrides["algo.fvp_estimator"] = args.fvp_estimator
    if args.npg_stepsize is not None:
        overrides["algo.npg_stepsize"] = float(args.npg_stepsize)
    if overrides:
        cfg = apply_overrides(cfg, overrides)

    seed = int(cfg.train.seed)
    set_seed(seed)

    run_name = str(cfg.train.get("run_name", Path(args.config).stem))
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / run_name / f"seed_{seed}"
    output_dir = prepare_run_dir(output_dir, overwrite=args.overwrite)
    save_config(cfg, output_dir / "config_resolved.yaml")

    resolved_method = str(cfg.get("method", {}).get("name", "trpo"))
    estimator_name = cfg.algo.get("estimator", cfg.algo.get("advantage_mode", "mc"))
    resolved_estimator = None if estimator_name is None else canonicalize_estimator(estimator_name)
    print({
        "package_path": imported_package_path("trpo_repro"),
        "package_init": str(Path(trpo_repro.__file__).resolve()),
        "config": str(Path(args.config).resolve()),
        "output_dir": str(output_dir.resolve()),
        "method": resolved_method,
        "estimator": resolved_estimator,
        "seed": seed,
        "num_workers": int(cfg.train.get("num_workers", 1)),
        "fvp_estimator": str(cfg.algo.get("fvp_estimator", "analytic")),
        "npg_stepsize": float(cfg.algo.get("npg_stepsize", 0.05)),
    })

    env = make_env(cfg, seed=seed)
    runner = Runner(env=env, cfg=cfg, output_dir=output_dir, device=args.device)
    runner.train()


if __name__ == "__main__":
    main()
