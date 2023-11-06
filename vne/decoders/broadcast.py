import torch

from typing import Tuple
from vne.decoders.base import BaseDecoder
from vne.decoders.spatial import SpatialDims, RotatedCoordinates


class BroadcastDecoder(BaseDecoder):
    """Broadcast decoder with explicit rotation transformations.

    Parameters
    ----------
    shape : tuple
        A tuple describing the output shape of the image data. Can be 2- or 3-
        dimensional.
    latent_dims : int
        The dimensions of the latent representation.
    hidden_channels : int
        The number of hidden channels in the hidden convolutional layer.
    output_channels : int
        The number of output channels in the final image volume.

    Notes
    -----
    Implements a spatial broadcast decoder [1]_ with a learnable spatial
    transform, provided by manipulating the coordinate channels with an explicit
    rotation transform parameterised by the pose channel of Affinity-VAE.

    References
    ----------
    .. [1] 'Spatial broadcast decoder: A simple architecture for learning
      disentangled representations in VAEs' Nicholas Watters, Loic Matthey,
      Christopher P. Burgess, Alexander Lerchner. https://arxiv.org/abs/1901.07017
    """

    def __init__(
        self,
        shape: Tuple[int],
        *,
        latent_dims: int = 8,
        hidden_channels: int = 128,
        output_channels: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self._shape = shape
        self._ndim = len(self._shape)
        self._extra_z_dims = [1] * self._ndim

        # this generates a grid of rotated coordinates in the forward pass
        self.coordinates = RotatedCoordinates(shape=shape, device=device)

        # need to swap the 2d conv for 3d when using 3d data
        conv = (
            torch.nn.Conv2d
            if self._ndim == SpatialDims.TWO
            else torch.nn.Conv3d
        )

        # build a simple decoder that takes the broadcasted latents and
        # generates an output image/volume use 1x1(x1) convolutions
        # NOTE(arl): this could use some testing
        self.decoder = torch.nn.Sequential(
            conv(latent_dims + self._ndim, hidden_channels, 3, padding="same"),
            torch.nn.ReLU(),
            conv(hidden_channels, hidden_channels, 3, padding="same"),
            torch.nn.ReLU(),
            conv(hidden_channels, output_channels, 1),
        ).to(device)

        self.device = device

    def forward(self, z: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """Decode the latents to an image volume given an explicit transform.

        Parameters
        ----------
        z : tensor
            An (N, D) tensor specifying the D dimensional latent encodings for
            the minibatch of N images.
        pose : tensor
            An (N, 1 | 4) tensor specifying the pose in terms of a single
            rotation (assumed around the z-axis) or a full axis-angle rotation.

        Returns
        -------
        x : tensor
            The decoded image from the latents and pose.
        """
        # we need to expand the dimensions of the latents to be able to broadcast
        # to the size of the image domain, e.g. N, Z, H, W for 2D
        z_expand = z.view(*z.shape, *self._extra_z_dims)
        z_broadcast = torch.tile(z_expand, (1, 1, *self._shape))
        z_coords = self.coordinates(pose)
        z_concat = torch.concat([z_broadcast, z_coords], axis=1)
        return self.decoder(z_concat)
