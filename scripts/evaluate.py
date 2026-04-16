from __future__ import annotations

import argparse
import statistics

import torch

from trpo_repro.config import load_config
from trpo_repro.envs.factory import make_env
from trpo_repro.methods import make_method, resolve_method_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
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

    method = make_method(env.observation_space, env.action_space, cfg, device)
    method_name = resolve_method_name(cfg)
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("state")
        if state is None and "policy" in ckpt:
            # Backward compatibility with older checkpoints.
            state = {"policy": ckpt.get("policy"), "value_fn": ckpt.get("value_fn")}
        method.load_state_dict(state or {})
    elif method_name != "random":
        raise ValueError("A checkpoint is required for non-random methods.")

    returns = []
    lengths = []
    for _ in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0
        while not done:
            obs_t = torch.as_tensor(obs[None, ...], dtype=torch.float32, device=device)
            action, _, _ = method.act(obs_t, deterministic=args.deterministic)
            act_np = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(act_np)
            ep_ret += float(reward)
            ep_len += 1
            done = terminated or truncated
        returns.append(ep_ret)
        lengths.append(ep_len)

    print({
        "method": method.name,
        "method_variant": method.variant,
        "episodes": args.episodes,
        "return_mean": statistics.mean(returns),
        "return_std": statistics.pstdev(returns) if len(returns) > 1 else 0.0,
        "len_mean": statistics.mean(lengths),
    })


if __name__ == "__main__":
    main()
