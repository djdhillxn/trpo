from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn


def to_tensor(array, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(array, device=device)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


class RunningMeanStd:
    """Simple running mean/std tracker for observation normalization."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m_2 / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip: float | None = 10.0) -> np.ndarray:
        out = (x - self.mean) / np.sqrt(self.var + 1e-8)
        if clip is not None:
            out = np.clip(out, -clip, clip)
        return out.astype(np.float32)


def flat_params(module: nn.Module) -> torch.Tensor:
    return torch.cat([param.data.view(-1) for param in module.parameters()])


@torch.no_grad()
def set_flat_params(module: nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    for param in module.parameters():
        numel = param.numel()
        param.copy_(flat[offset : offset + numel].view_as(param))
        offset += numel


def flat_grad(grads: Iterable[torch.Tensor | None]) -> torch.Tensor:
    flat = []
    for grad in grads:
        if grad is None:
            continue
        flat.append(grad.contiguous().view(-1))
    return torch.cat(flat) if flat else torch.tensor([])
