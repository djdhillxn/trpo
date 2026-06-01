import argparse
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path


def main():
    parser = argparse.ArgumentParser(description="Train Qwen reward model on HelpSteer3 preference pairs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    ensure_repo_root_on_path()
    from trpo_repro.rlhf.train_reward_model import run_reward_training

    out = run_reward_training(args.config, output_dir=args.output_dir)
    print(f"Reward model saved to: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
