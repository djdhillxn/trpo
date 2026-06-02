import argparse
from pprint import pprint

from _bootstrap import ensure_repo_root_on_path


def parse_args():
    parser = argparse.ArgumentParser(description="Render a trained TRPO/PPO Atari policy to an MP4 video.")
    parser.add_argument("--config", required=True, help="Training config, e.g. configs/atari/qbert_single_path.yaml")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path, e.g. outputs/qbert_single_path/seed_0/checkpoints/best.pt")
    parser.add_argument("--output-dir", default="outputs/videos/qbert_demo")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--policy-mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--record-selection", choices=["best", "first"], default="best")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_repo_root_on_path()

    from trpo_repro.demos.atari_video import render_policy_video

    summary = render_policy_video(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        episodes=args.episodes,
        policy_mode=args.policy_mode,
        record_selection=args.record_selection,
        fps=args.fps,
        scale=args.scale,
        max_steps=args.max_steps,
        title=args.title,
    )
    pprint(summary.__dict__)


if __name__ == "__main__":
    main()
