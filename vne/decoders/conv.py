import torch

import numpy as np

from typing import Tuple

from vne.decoders.base import BaseDecoder


class Decoder3D(BaseDecoder):
    """Simple convolutional decoder."""

    def __init__(
        self,
        latent_dims: int,
        pose_dims: int,
        *,
        layer_channels: Tuple[int] = (64, 32, 18, 8),
        output_shape: Tuple[int] = (32, 32, 32),
    ):
        unflat_shape = (64, 2, 2, 2)
        self.flat_shape = np.prod(unflat_shape)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dims + pose_dims, self.flat_shape),
            torch.nn.Unflatten(-1, unflat_shape),
            torch.nn.ConvTranspose3d(64, 32, 3, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(16, 8, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(8, 1, 2, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        z_pose = torch.cat([z, pose], dim=-1)
        return self.model(z_pose)
