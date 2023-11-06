import abc
import torch


class BaseEncoder(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractproperty
    def flat_shape(self) -> tuple:
        raise NotImplementedError
