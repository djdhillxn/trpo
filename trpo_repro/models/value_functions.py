from __future__ import annotations

import numpy as np
import torch
from torch import nn

from trpo_repro.models.cnn import AtariBody
from trpo_repro.models.mlp import build_mlp

class ValueFunction(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], cfg) -> None:
        super().__init__()
        activation = cfg.model.get("activation", "tanh")
        value_hidden_sizes = list(cfg.model.get("value_hidden_sizes", [64, 64]))

        if len(obs_shape) == 3:
            self.body = AtariBody(in_channels=obs_shape[0], hidden_dim=int(cfg.model.get("cnn_fc_dim", 20)))
            self.head = nn.Linear(self.body.output_dim, 1)
        else:
            input_dim = int(np.prod(obs_shape))
            self.body = build_mlp(input_dim, value_hidden_sizes, 1, activation=activation)
            self.head = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim > 2 and self.head is not None:
            return self.head(self.body(obs)).squeeze(-1)
        if obs.ndim > 2:
            obs = obs.view(obs.shape[0], -1)
        return self.body(obs).squeeze(-1)