from __future__ import annotations

import numpy as np
import torch

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
        advantage_mode: str,
        act_dtype: np.dtype,
    ) -> None:
        self.obs_buf = np.zeros((size, *obs_shape), dtype=np.float32)
        self.act_buf = np.zeros((size, *act_shape), dtype=act_dtype)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.advantage_mode = advantage_mode
        self.max_size = size
        self.ptr = 0
        self.path_start_idx = 0

    def store(self, obs, act, rew: float, val: float, logp: float) -> None:
        if self.ptr >= self.max_size:
            raise RuntimeError("trajectory buffer overflow...")
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val: float = 0.0) -> None:
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        if self.advantage_mode.lower() == "gae":
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            self.adv_buf[path_slice] = discounted_cumsum(deltas, self.gamma * self.lam)
            self.ret_buf[path_slice] = discounted_cumsum(rews, self.gamma)[:-1]
        elif self.advantage_mode.lower() == "mc":
            self.ret_buf[path_slice] = discounted_cumsum(rews, self.gamma)[:-1]
            self.adv_buf[path_slice] = self.ret_buf[path_slice] - self.val_buf[path_slice]
        else:
            raise ValueError(f"unknown advantage mode: {self.advantage_mode}")
        
        self.path_start_idx = self.ptr

    def get(self, device: torch.device) -> dict[str, torch.Tensor]:
        if self.ptr != self.max_size:
            raise RuntimeError("buffer must be full before calling get..")
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf) + 1e-8
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = {
            "obs": torch.as_tensor(self.obs_buf, dtype=torch.float32, device=device),
            "act": torch.as_tensor(self.act_buf, device=device),
            "ret": torch.as_tensor(self.ret_buf, dtype=torch.float32, device=device),
            "adv": torch.as_tensor(self.adv_buf, dtype=torch.float32, device=device),
            "logp": torch.as_tensor(self.logp_buf, dtype=torch.float32, device=device),
        }
        self.ptr = 0
        self.path_start_idx = 0
        return data