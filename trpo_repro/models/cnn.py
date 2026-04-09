from __future__ import annotations

import torch
from torch import nn

class AtariBody(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 20) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            n_flat = self.conv(dummy).view(1, -1).shape[1]
        self.fc = nn.Sequential(nn.Linear(n_flat, hidden_dim), nn.ReLU())
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        if x.ndim != 4:
            raise ValueError(f"expected Atari input shape [B, C, H, W], got {tuple(x.shape)}")
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)
        