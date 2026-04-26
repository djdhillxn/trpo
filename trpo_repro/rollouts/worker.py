import os
from typing import Any

import numpy as np
import torch

from trpo_repro.algos.advantages import canonicalize_estimator, compute_path_targets
from trpo_repro.config import DotDict
from trpo_repro.envs.factory import make_env
from trpo_repro.methods import make_method
from trpo_repro.utils.utils import set_seed


def _to_preprocessed_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32)



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
    estimator_name = getattr(method, "estimator", None) or cfg.algo.get("estimator") or "trpo_paper"
    estimator = canonicalize_estimator(estimator_name)
    bootstrap_truncated_paths = bool(cfg.algo.get("bootstrap_truncated_paths", estimator != "trpo_paper"))

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
                    env_action = int(action) if hasattr(env.action_space, "n") else action
                    value = float(value_t.squeeze(0).cpu().item()) if value_t.ndim > 0 else float(value_t.cpu().item())
                    logp = float(logp_t.squeeze(0).cpu().item()) if logp_t.ndim > 0 else float(logp_t.cpu().item())

                    next_obs, reward, terminated, truncated, _ = env.step(env_action)
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
                        rets, weights = compute_path_targets(estimator, ep_rewards, ep_values, gamma, lam, last_val=0.0)
                        idx_slice = np.asarray(ep_indices, dtype=np.int64)
                        if len(idx_slice) != len(rets) or len(idx_slice) != len(weights):
                            raise RuntimeError("Worker path target lengths do not match collected path length.")
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
                        if estimator == "trpo_paper":
                            continue
                        last_val = 0.0
                        if bootstrap_truncated_paths:
                            next_obs_tensor = torch.as_tensor(next_obs[None, ...], dtype=torch.float32)
                            with torch.inference_mode():
                                last_val = float(method.value(next_obs_tensor).cpu().item())
                        rets, weights = compute_path_targets(estimator, ep_rewards, ep_values, gamma, lam, last_val=last_val)
                        idx_slice = np.asarray(ep_indices, dtype=np.int64)
                        if len(idx_slice) != len(rets) or len(idx_slice) != len(weights):
                            raise RuntimeError("Worker bootstrap target lengths do not match collected path length.")
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
