from __future__ import annotations

import argparse
from pathlib import Path

from trpo_repro.config import apply_overrides, load_config, save_config
from trpo_repro.envs.factory import make_env
from trpo_repro.runner import Runner
from trpo_repro.utils.io import ensure_dir
from trpo_repro.utils.seeding import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    overrides = {}
    if args.seed is not None:
        overrides["train.seed"] = args.seed
    if overrides:
        cfg = apply_overrides(cfg, overrides)

    seed = int(cfg.train.seed)
    set_seed(seed)

    run_name = str(cfg.train.get("run_name", Path(args.config).stem))
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / run_name / f"seed_{seed}"
    ensure_dir(output_dir)
    save_config(cfg, output_dir / "config_resolved.yaml")

    env = make_env(cfg, seed=seed)
    runner = Runner(env=env, cfg=cfg, output_dir=output_dir, device=args.device)
    runner.train()

if __name__ == "__main__":
    main()