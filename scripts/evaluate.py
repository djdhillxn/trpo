from __future__ import annotations

import argparse
import statistics

import torch

from trpo_repro.config import load_config
from trpo_repro.envs.factory import make_env
from trpo_repro.models.policies import make_policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    env = make_env(cfg, seed=args.seed)
    device = torch.device(args.device)

    policy = make_policy(env.observation_space, env.action_space, cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    policy.load_state_dict(ckpt["policy"])

    returns = []
    lengths = []
    for _ in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0
        while not done:
            obs_t = torch.as_tensor(obs[None, ...], dtype=torch.float32, device=device)
            action, _, _ = policy.act(obs_t, deterministic=args.deterministic)
            act_np = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(act_np)
            ep_ret += float(reward)
            ep_len += 1
            done = terminated or truncated
        returns.append(ep_ret)
        lengths.append(ep_len)

    print({
        "episodes": args.episodes,
        "return_mean": statistics.mean(returns),
        "return_std": statistics.pstdev(returns) if len(returns) > 1 else 0.0,
        "len_mean": statistics.mean(lengths),
    })


if __name__ == "__main__":
    main()
