from __future__ import annotations

import numpy as np
import torch


EstimatorName = str


def discounted_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(x))):
        running = x[i] + discount * running
        out[i] = running
    return out


class TrajectoryBuffer:
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        act_shape: tuple[int, ...],
        size: int,
        gamma: float,
        lam: float,
        estimator: EstimatorName,
        act_dtype: np.dtype,
        normalize_weights: bool = True,
    ) -> None:
        self.obs_buf = np.zeros((size, *obs_shape), dtype=np.float32)
        self.act_buf = np.zeros((size, *act_shape), dtype=act_dtype)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.weight_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.estimator = estimator.lower()
        self.normalize_weights = normalize_weights
        self.max_size = size
        self.ptr = 0
        self.path_start_idx = 0

    def store(self, obs, act, rew: float, val: float = 0.0, logp: float = 0.0) -> None:
        if self.ptr >= self.max_size:
            raise RuntimeError("TrajectoryBuffer overflow.")
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val: float = 0.0) -> None:
        if self.ptr == self.path_start_idx:
            return

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        if self.estimator == "gae":
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            self.weight_buf[path_slice] = discounted_cumsum(deltas, self.gamma * self.lam)
            self.ret_buf[path_slice] = discounted_cumsum(rews, self.gamma)[:-1]
        elif self.estimator in {"mc", "mc_baseline"}:
            self.ret_buf[path_slice] = discounted_cumsum(rews, self.gamma)[:-1]
            self.weight_buf[path_slice] = self.ret_buf[path_slice] - self.val_buf[path_slice]
        elif self.estimator == "paper_mc":
            self.ret_buf[path_slice] = discounted_cumsum(rews, self.gamma)[:-1]
            self.weight_buf[path_slice] = self.ret_buf[path_slice]
        else:
            raise ValueError(f"Unknown estimator mode: {self.estimator}")

        self.path_start_idx = self.ptr

    def get(self, device: torch.device) -> dict[str, torch.Tensor]:
        size = self.ptr
        if size <= 0:
            raise RuntimeError("Buffer is empty.")

        weight_buf = self.weight_buf[:size].copy()
        if self.normalize_weights and size > 1:
            weight_mean = np.mean(weight_buf)
            weight_std = np.std(weight_buf) + 1e-8
            weight_buf = (weight_buf - weight_mean) / weight_std

        weights_t = torch.as_tensor(weight_buf, dtype=torch.float32, device=device)
        data = {
            "obs": torch.as_tensor(self.obs_buf[:size], dtype=torch.float32, device=device),
            "act": torch.as_tensor(self.act_buf[:size], device=device),
            "ret": torch.as_tensor(self.ret_buf[:size], dtype=torch.float32, device=device),
            "weight": weights_t,
            # Backward-compatible alias for older code paths.
            "adv": weights_t,
            "logp": torch.as_tensor(self.logp_buf[:size], dtype=torch.float32, device=device),
        }
        self.ptr = 0
        self.path_start_idx = 0
        return data
