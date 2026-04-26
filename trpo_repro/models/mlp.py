from typing import Iterable

import torch
from torch import nn


_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
}


def build_mlp(
    input_dim: int,
    hidden_sizes: Iterable[int],
    output_dim: int,
    activation: str = "tanh",
    output_activation: nn.Module | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    sizes = [input_dim, *hidden_sizes, output_dim]
    act_cls = _ACTIVATIONS[activation.lower()]
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act_cls())
        elif output_activation is not None:
            layers.append(output_activation)
    return nn.Sequential(*layers)
