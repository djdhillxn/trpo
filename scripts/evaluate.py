from __future__ import annotations

import argparse
import statistics

from _bootstrap import ensure_repo_root_on_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _restore_obs_rms(ckpt: dict, cfg, env):
    normalize_obs = bool(cfg.train.get("normalize_obs", False)) and len(env.observation_space.shape) == 1
    mean = ckpt.get("obs_rms_mean")
    var = ckpt.get("obs_rms_var")
    count = ckpt.get("obs_rms_count")
    if not normalize_obs or mean is None or var is None or count is None:
        return None
    rms = RunningMeanStd(shape=tuple(env.observation_space.shape))
    rms.mean = np.asarray(mean, dtype=np.float64)
    rms.var = np.asarray(var, dtype=np.float64)
    rms.count = float(count)
    return rms


def _preprocess_obs(obs, obs_rms):
    obs = np.asarray(obs)
    if obs_rms is not None:
        obs = obs_rms.normalize(obs)
    return obs.astype(np.float32)


def main():
    args = parse_args()

    ensure_repo_root_on_path()
    global np, RunningMeanStd
    import numpy as np
    import torch

    from trpo_repro.config import load_config
    from trpo_repro.envs.factory import make_env
    from trpo_repro.methods import make_method, resolve_method_name
    from trpo_repro.utils.torch_utils import RunningMeanStd

    cfg = load_config(args.config)
    env = make_env(cfg, seed=args.seed)
    device = torch.device(args.device)

    method = make_method(env.observation_space, env.action_space, cfg, device)
    method_name = resolve_method_name(cfg)
    ckpt = None
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("state")
        if state is None and "policy" in ckpt:
            state = {"policy": ckpt.get("policy"), "value_fn": ckpt.get("value_fn")}
        method.load_state_dict(state or {})
    elif method_name != "random":
        raise ValueError("A checkpoint is required for non-random methods.")

    obs_rms = _restore_obs_rms(ckpt or {}, cfg, env)
    max_ep_len = int(cfg.train.get("max_ep_len", 1000))

    returns = []
    lengths = []
    for _ in range(args.episodes):
        obs, _ = env.reset()
        obs = _preprocess_obs(obs, obs_rms)
        done = False
        ep_ret = 0.0
        ep_len = 0
        while not done:
            obs_t = torch.as_tensor(obs[None, ...], dtype=torch.float32, device=device)
            action, _, _ = method.act(obs_t, deterministic=args.deterministic)
            act_np = action.squeeze(0).cpu().numpy()
            env_action = int(act_np) if hasattr(env.action_space, "n") else act_np
            obs, reward, terminated, truncated, _ = env.step(env_action)
            obs = _preprocess_obs(obs, obs_rms)
            ep_ret += float(reward)
            ep_len += 1
            timeout = ep_len >= max_ep_len
            done = terminated or truncated or timeout
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
