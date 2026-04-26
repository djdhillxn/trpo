from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass
class MethodUpdateStats:
    policy_loss_before: float = float('nan')
    policy_loss_after: float = float('nan')
    value_loss_before: float = float('nan')
    value_loss_after: float = float('nan')
    entropy: float = float('nan')
    approx_kl: float = float('nan')
    line_search_success: float = float('nan')
    cg_norm: float = float('nan')
    clip_fraction: float = float('nan')
    kl_coef: float = float('nan')
    value_explained_variance_before: float = float('nan')
    value_explained_variance_after: float = float('nan')
    did_update: float = 0.0

    def to_log_dict(self) -> dict[str, float]:
        return asdict(self)


class BaseMethod:
    trainable: bool = False
    supports_checkpoints: bool = False
    batch_obs_to_device: bool = True

    def __init__(self, cfg, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def variant(self) -> str:
        return "default"

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        raise NotImplementedError

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(obs.shape[0], dtype=torch.float32, device=obs.device)

    def update(self, batch: dict[str, torch.Tensor] | None) -> MethodUpdateStats:
        return MethodUpdateStats(did_update=0.0)

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        del state

    def set_training_progress(self, epoch: int, total_epochs: int) -> None:
        del epoch, total_epochs
