#!/usr/bin/env python3
import argparse

from trpo_repro.rlhf.train_sft import run_sft_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LoRA SFT warm-start policy on HelpSteer3 chosen responses.")
    parser.add_argument("--config", default="configs/rlhf/qwen25_05b_helpsteer3_sft.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out = run_sft_training(args.config, output_dir=args.output_dir)
    print(f"SFT policy saved to {out}")


if __name__ == "__main__":
    main()
