import torch
from torch import nn


class TinyRefiner(nn.Module):
    """A tiny residual refiner operating on [img(3) + alpha(1)] -> delta_alpha(1)."""

    def __init__(self, in_ch: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


