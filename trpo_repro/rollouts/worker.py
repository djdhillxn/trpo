from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from trpo_repro.algos.advantages import discounted_cumsum
from trpo_repro.config import DotDict
from trpo_repro.envs.factory import make_env
from trpo_repro.methods import make_method
from trpo_repro.utils.utils import set_seed


def _to_preprocessed_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32)


def _compute_path_targets(
    estimator: str,
    rewards: list[float],
    values: list[float],
    gamma: float,
    lam: float,
    last_val: float,
) -> tuple[np.ndarray, np.ndarray]:
    rews = np.asarray(rewards, dtype=np.float32)
    vals = np.asarray(values, dtype=np.float32)
    if estimator == "paper_mc":
        rets = discounted_cumsum(np.append(rews, np.float32(last_val)), gamma)[:-1]
        return rets, rets.copy()
    if estimator in {"mc", "mc_baseline"}:
        rets = discounted_cumsum(np.append(rews, np.float32(last_val)), gamma)[:-1]
        weights = rets - vals
        return rets, weights.astype(np.float32, copy=False)
    if estimator == "gae":
        vals_ext = np.append(vals, np.float32(last_val))
        rews_ext = np.append(rews, np.float32(last_val))
        deltas = rews_ext[:-1] + gamma * vals_ext[1:] - vals_ext[:-1]
        adv = discounted_cumsum(deltas.astype(np.float32, copy=False), gamma * lam)
        rets = discounted_cumsum(rews_ext, gamma)[:-1]
        return rets.astype(np.float32, copy=False), adv.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported estimator for parallel rollouts: {estimator}")


def rollout_worker_loop(
    worker_id: int,
    task_queue,
    result_queue,
    cfg_dict: dict[str, Any],
    shared: dict[str, Any],
) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    cfg = DotDict(cfg_dict)
    seed = int(cfg.train.seed) + 10007 * (worker_id + 1)
    set_seed(seed)

    if bool(cfg.train.get("normalize_obs", False)):
        raise RuntimeError("Parallel rollout collection does not support normalize_obs=true.")

    env = make_env(cfg, seed=seed)
    method = make_method(env.observation_space, env.action_space, cfg, device=torch.device("cpu"))
    segment_start = int(shared["starts"][worker_id])
    segment_capacity = int(shared["capacities"][worker_id])
    gamma = float(cfg.algo.gamma)
    lam = float(cfg.algo.get("lam", 1.0))
    max_ep_len = int(cfg.train.get("max_ep_len", 1000))
    estimator = str(getattr(method, "estimator", cfg.algo.get("estimator", "paper_mc"))).lower()
    bootstrap_truncated_paths = bool(cfg.algo.get("bootstrap_truncated_paths", estimator != "paper_mc"))

    obs_buf = shared["obs"]
    act_buf = shared["act"]
    ret_buf = shared["ret"]
    weight_buf = shared["weight"]
    logp_buf = shared["logp"]

    try:
        while True:
            task = task_queue.get()
            cmd = task.get("cmd")
            if cmd == "close":
                break
            if cmd != "collect":
                continue

            try:
                request_id = int(task["request_id"])
                local_target_steps = int(task["target_steps"])
                method_state = task["method_state"]
                method.load_state_dict(method_state)

                if local_target_steps <= 0:
                    result_queue.put({
                        "worker_id": worker_id,
                        "request_id": request_id,
                        "count": 0,
                        "steps_collected": 0,
                        "episode_returns": [],
                        "episode_lengths": [],
                    })
                    continue

                ptr = 0
                steps_collected = 0
                episode_returns: list[float] = []
                episode_lengths: list[int] = []

                obs, _ = env.reset()
                obs = _to_preprocessed_obs(obs)
                ep_rewards: list[float] = []
                ep_values: list[float] = []
                ep_indices: list[int] = []
                ep_ret = 0.0
                ep_len = 0

                while True:
                    obs_tensor = torch.as_tensor(obs[None, ...], dtype=torch.float32)
                    with torch.inference_mode():
                        action_t, value_t, logp_t = method.act(obs_tensor, deterministic=False)
                    action = action_t.squeeze(0).cpu().numpy()
                    value = float(value_t.squeeze(0).cpu().item()) if value_t.ndim > 0 else float(value_t.cpu().item())
                    logp = float(logp_t.squeeze(0).cpu().item()) if logp_t.ndim > 0 else float(logp_t.cpu().item())

                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    next_obs = _to_preprocessed_obs(next_obs)

                    if ptr >= segment_capacity:
                        raise RuntimeError(
                            f"Worker {worker_id} exceeded shared segment capacity {segment_capacity}. "
                            f"Increase max_ep_len or reduce num_workers."
                        )

                    global_idx = segment_start + ptr
                    obs_buf[global_idx] = obs
                    act_buf[global_idx] = action
                    logp_buf[global_idx] = logp
                    ep_rewards.append(float(reward))
                    ep_values.append(value)
                    ep_indices.append(global_idx)

                    ptr += 1
                    steps_collected += 1
                    ep_ret += float(reward)
                    ep_len += 1

                    timeout = ep_len >= max_ep_len
                    terminal = bool(terminated or truncated or timeout)
                    reached_target = steps_collected >= local_target_steps

                    if terminal:
                        rets, weights = _compute_path_targets(estimator, ep_rewards, ep_values, gamma, lam, last_val=0.0)
                        idx_slice = np.asarray(ep_indices, dtype=np.int64)
                        ret_buf[idx_slice] = rets
                        weight_buf[idx_slice] = weights
                        episode_returns.append(ep_ret)
                        episode_lengths.append(ep_len)
                        if reached_target:
                            break
                        obs, _ = env.reset()
                        obs = _to_preprocessed_obs(obs)
                        ep_rewards.clear()
                        ep_values.clear()
                        ep_indices.clear()
                        ep_ret = 0.0
                        ep_len = 0
                        continue

                    obs = next_obs
                    if reached_target:
                        if estimator == "paper_mc":
                            continue
                        last_val = 0.0
                        if bootstrap_truncated_paths:
                            next_obs_tensor = torch.as_tensor(next_obs[None, ...], dtype=torch.float32)
                            with torch.inference_mode():
                                last_val = float(method.value(next_obs_tensor).cpu().item())
                        rets, weights = _compute_path_targets(estimator, ep_rewards, ep_values, gamma, lam, last_val=last_val)
                        idx_slice = np.asarray(ep_indices, dtype=np.int64)
                        ret_buf[idx_slice] = rets
                        weight_buf[idx_slice] = weights
                        break

                result_queue.put(
                    {
                        "worker_id": worker_id,
                        "request_id": request_id,
                        "count": ptr,
                        "steps_collected": steps_collected,
                        "episode_returns": episode_returns,
                        "episode_lengths": episode_lengths,
                    }
                )
            except Exception as exc:
                result_queue.put(
                    {
                        "worker_id": worker_id,
                        "request_id": int(task.get("request_id", -1)),
                        "error": repr(exc),
                    }
                )
    finally:
        env.close()
