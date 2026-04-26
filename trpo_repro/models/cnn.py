import torch
from torch import nn


class AtariBody(nn.Module):
    """CNN close to the parameter count described in the TRPO paper.

    Uses 16-channel conv layers and a 20-unit fully-connected layer. The kernel / stride
    choices here are the common 8x8/4 and 4x4/2 setup, which matches the cited total
    parameter count much more closely than using 4x4 filters in both layers.
    """

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
            raise ValueError(f"Expected Atari input shape [B, C, H, W], got {tuple(x.shape)}")
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)
