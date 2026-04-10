from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SUITES = {
    "mujoco": [
        "configs/mujoco/swimmer_single_path.yaml",
        "configs/mujoco/hopper_single_path.yaml",
        "configs/mujoco/walker2d_single_path.yaml",
    ],
    "atari": [
        "configs/atari/beamrider_single_path.yaml",
        "configs/atari/breakout_single_path.yaml",
        "configs/atari/enduro_single_path.yaml",
        "configs/atari/pong_single_path.yaml",
        "configs/atari/qbert_single_path.yaml",
        "configs/atari/seaquest_single_path.yaml",
        "configs/atari/spaceinvaders_single_path.yaml",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=sorted(SUITES.keys()), required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    for config in SUITES[args.suite]:
        for seed in args.seeds:
            cmd = [
                sys.executable,
                str(repo_root / "scripts" / "train.py"),
                "--config",
                str(repo_root / config),
                "--seed",
                str(seed),
                "--device",
                args.device,
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
