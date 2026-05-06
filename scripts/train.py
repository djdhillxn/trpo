import argparse
import sys
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
    parser.add_argument("--resume-from", "--resume_from", dest="resume_from", type=str, default=None)
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


def _resume_epoch_hint(path: str | None) -> int | None:
    if path is None:
        return None
    stem = Path(path).stem
    if not stem.startswith("epoch_"):
        return None
    try:
        return int(stem.removeprefix("epoch_"))
    except ValueError:
        return None


def main():
    args = parse_args()

    ensure_repo_root_on_path()
    import trpo_repro
    from trpo_repro.algos.advantages import canonicalize_estimator
    from trpo_repro.config import apply_overrides, load_config, save_config
    from trpo_repro.envs.factory import make_env
    from trpo_repro.runner import Runner
    from trpo_repro.utils.run_metadata import (
        build_failure_record,
        build_launch_metadata,
        monotonic_time,
        utc_now_iso,
        write_failure,
        write_run_summary,
    )
    from trpo_repro.utils.utils import imported_package_path, prepare_run_dir, set_seed, write_json

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
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.resume_from:
        resume_epoch = _resume_epoch_hint(args.resume_from)
        resume_suffix = "resume" if resume_epoch is None else f"resume_{resume_epoch:04d}"
        output_dir = Path("outputs") / run_name / f"seed_{seed}_{resume_suffix}"
    else:
        output_dir = Path("outputs") / run_name / f"seed_{seed}"
    output_dir = prepare_run_dir(output_dir, overwrite=args.overwrite)
    save_config(cfg, output_dir / "config_resolved.yaml")
    package_path = imported_package_path("trpo_repro")
    launch_metadata = build_launch_metadata(
        argv=sys.argv,
        cli_args=vars(args),
        cli_overrides=overrides,
        config_path=args.config,
        output_dir=output_dir,
        package_path=package_path,
    )
    write_json(launch_metadata, output_dir / "launch_metadata.json")

    resolved_method = str(cfg.get("method", {}).get("name", "trpo"))
    estimator_name = cfg.algo.get("estimator", cfg.algo.get("advantage_mode", "mc"))
    resolved_estimator = None if estimator_name is None else canonicalize_estimator(estimator_name)
    print({
        "package_path": package_path,
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

    init_started_at = utc_now_iso()
    init_start_time = monotonic_time()
    env = None
    try:
        env = make_env(cfg, seed=seed)
        runner = Runner(env=env, cfg=cfg, output_dir=output_dir, device=args.device, launch_metadata=launch_metadata)
        if args.resume_from is not None:
            runner.resume_from_checkpoint(args.resume_from)
    except BaseException as exc:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        failure_record = build_failure_record(exc, epoch=None)
        write_failure(output_dir, failure_record)
        write_run_summary(
            output_dir,
            {
                "schema_version": 1,
                "run_id": None,
                "status": status,
                "phase": "initialization",
                "started_at": init_started_at,
                "ended_at": utc_now_iso(),
                "duration_sec": monotonic_time() - init_start_time,
                "completed_epochs": 0,
                "target_epochs": int(cfg.train.epochs),
                "total_env_steps": 0,
                "failure": {
                    "exception_type": failure_record["exception_type"],
                    "exception_message": failure_record["exception_message"],
                    "epoch": failure_record["epoch"],
                    "failure_path": str((output_dir / "failure.json").resolve()),
                },
            },
        )
        raise
    runner.train()


if __name__ == "__main__":
    main()
