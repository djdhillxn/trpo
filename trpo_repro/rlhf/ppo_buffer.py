from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LMRolloutBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    old_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    scores: torch.Tensor
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None
    prompts: list[str] | None = None
    responses: list[str] | None = None
    metadata: list[dict[str, Any]] | None = None

    def to(self, device: torch.device | str) -> "LMRolloutBatch":
        kwargs = {}
        for field_name, value in self.__dict__.items():
            if torch.is_tensor(value):
                kwargs[field_name] = value.to(device)
            else:
                kwargs[field_name] = value
        return LMRolloutBatch(**kwargs)

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.size(0))

    @property
    def num_response_tokens(self) -> int:
        return int(self.response_mask.sum().item())
