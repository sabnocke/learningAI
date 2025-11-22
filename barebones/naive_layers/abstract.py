from abc import abstractmethod, ABC
from typing import Any, List, Union, Tuple, Iterable

from torch import Tensor
import torch

type Number = int | float
type Device = torch.device

class AbstractLayer(ABC):
    def __init__(self):
        self.training = True
        self._weights = torch.zeros(1)

    def train(self, train: bool) -> None:
        self.training = train

    @property
    def layers(self) -> Any:
        return tuple()

    @property
    def weights(self) -> Any:
        return self._weights

    @weights.setter
    def weights(self, value: Tensor) -> None:
        self._weights = value

    @abstractmethod
    def __call__(self, inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def to(self, device: Device) -> "AbstractLayer":
        pass

    @property
    @abstractmethod
    def parameters(self) -> List[Tensor]: ...


class BaseLayer(AbstractLayer):
    def __call__(self, inputs: Tensor) -> Tensor:
        return inputs

    def to(self, device: Device) -> "BaseLayer":
        return self

    @property
    def parameters(self) -> List[Tensor]:
        return []
