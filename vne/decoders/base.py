import abc
import torch


class BaseDecoder(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, z: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Decoder `forward` method must be implemented with `z` and `pose` as"
            " arguments."
        )
