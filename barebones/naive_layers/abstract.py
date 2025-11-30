from abc import abstractmethod, ABC
from typing import Any, List, Union, Tuple, Iterable
from pathlib import Path

from torch import Tensor
import torch as th

type Number = int | float
type Device = th.device

class AbstractLayer(ABC):
    """
    Basis for any layer
    """
    def __init__(self):
        super().__init__()
        self.training = True
        self._weights = th.zeros(1)

    def train(self, train: bool) -> None:
        """
        Whether the model is training or being evaluated

        Args:
            train: if it is being trained

        """
        self.training = train

    @property
    def layers(self) -> Any:
        """
        Method to obtain layers of model
        Returns:
            A tuple of layers
        """
        return tuple()

    @property
    def weights(self) -> Any:
        return self._weights

    @weights.setter
    def weights(self, value: Tensor) -> None:
        self._weights = value

    @abstractmethod
    def __call__(self, inputs: Tensor) -> Tensor: ...

    @abstractmethod
    def to(self, device: Device) -> "AbstractLayer": ...

    @property
    @abstractmethod
    def parameters(self) -> List[Tensor]: ...


class BaseLayer(AbstractLayer):
    """
    A minimal implementation of AbstractLayer,
    to be used for layers that don't need to implement entire AbstractLayer
    """
    def __call__(self, inputs: Tensor) -> Tensor:
        return inputs

    def to(self, device: Device) -> "BaseLayer":
        return self

    @property
    def parameters(self) -> List[Tensor]:
        return []


class SerializableModel(ABC):
    """
    Defines a model that can be serialized and deserialized
    """
    def __init__(self):
        super().__init__()

        self.save_path = Path(__file__).resolve().parent.parent / "weights"


    @property
    @abstractmethod
    def layers(self) -> Tuple[AbstractLayer, ...]:
        """
        Method to obtain layers of model

        Returns:
            Tuple of layers
        """


    def save_weights(self, name: str, path: None | str = None):
        """
        Save the weights to a file at <path>/<name>.pth

        If path is None (by default) the path is /barebones/weights/<name>.pth
        Args:
            name: name of resulting file (required)
            path: path to save weights (optional)
        """
        state = {}
        for i, layer in enumerate(self.layers):
            state[f"layer_{i}"] = layer.parameters

        destination = Path(path).resolve() if path is not None else self.save_path
        destination = destination / f"{name}.pth"
        th.save(state, destination)
        print(f"Saved weights to {destination}")

    def load_weights(self, name: str, path: None | str = None):
        """
        Loads weights from a file

        If path is None (by default) the path is /barebones/weights/<name>.pth

        Args:
            name: name of the file to load (required)
            path: it's path

        Returns:

        """
        source = Path(path).resolve() if path is not None else self.save_path
        source = source / name
        state = th.load(source)

        for i, layer in enumerate(self.layers):
            saved_params = state[f"layer_{i}"]

            for p_current, p_saved in zip(layer.parameters, saved_params):
                p_current.data.copy_(p_saved)

        print(f"Weights loaded from {source}")
