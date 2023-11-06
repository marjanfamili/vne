import torch

import numpy as np

from typing import Tuple

from vne.encoders.base import BaseEncoder
from vne.vae import dims_after_pooling


class Encoder3D(BaseEncoder):
    """Really simple encoder."""

    def __init__(
        self,
        *,
        layer_channels: Tuple[int] = (8, 16, 32, 64),
        input_shape: Tuple[int] = (32, 32, 32),
    ):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Conv3d(1, layer_channels[0], 3, stride=2, padding=1),
            torch.nn.ReLU(),
        )

        for in_channels, out_channels in zip(
            layer_channels, layer_channels[1:]
        ):
            self.model.append(
                torch.nn.Conv3d(
                    in_channels, out_channels, 3, stride=2, padding=1
                )
            )
            self.model.append(torch.nn.ReLU())

        self.model.append(torch.nn.Flatten())
        xd, yd, zd = [
            dims_after_pooling(d, len(layer_channels)) for d in input_shape
        ]
        self.unflat_shape = (layer_channels[-1], xd, yd, zd)

    @property
    def flat_shape(self) -> tuple:
        return np.prod(self.unflat_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
