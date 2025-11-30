"""
Defines structure of sequential layer, a layer that goes through each of its sub-layers

Example:
    NaiveSequential(
        NaiveLinear(...),
        NaiveReLU(...),

        NaiveLinear(...),
        ...
    )
"""

from typing import List, Optional, Tuple, override
from pathlib import Path
import torch as th
from torch import Tensor
from barebones import TrainingConfig
from barebones.naive_layers import abstract

type Layer = abstract.AbstractLayer


class NaiveSequential(abstract.AbstractLayer, abstract.SerializableModel):
    r"""
    Represents a sequential layer, a layer that goes through each of its sub-layers
    """

    def __init__(self, *layers: Layer, config: TrainingConfig):
        """

        Args:
            *layers (Layer): Every layer to be used by model
            config: Configuration for training
        """
        super().__init__()

        self._layers = layers
        self.testing: Optional[Tensor] = None
        self.test_labels: Optional[Tensor] = None
        self.learning_rate = config.learning_rate
        self.config = config
        self.loss_collection: List[Tensor] = []
        self.acc_collection: List[float] = []
        self.lr_collection: List[float] = []
        self.verbose: bool = False

        # self.save_path = Path(__file__).resolve().parent.parent / "weights"

    @property
    @override
    def layers(self) -> Tuple[Layer, ...]:
        return self._layers

    def __call__(self, inputs: Tensor):
        for layer in self._layers:
            inputs = layer(inputs)
            # print(f"{layer.__class__.__name__}: {inputs.shape}")

        return inputs

    def to(self, device: th.device):
        """
        Helper method to send all layers to given device

        Args:
            device (th.device):
                device to send all layers to

        Returns: self
        """
        for layer in self._layers:
            layer.to(device=device)

        return self

    @property
    def parameters(self) -> List[Tensor]:
        all_params = []
        for layer in self._layers:
            if hasattr(layer, "parameters"):
                all_params.extend(layer.parameters)

        return all_params

    def train(self, train: bool):
        for layer in self._layers:
            layer.train(train)

    # def save_weights(self, name: str, path: None | str = None):
    #     state = {}
    #     for i, layer in enumerate(self._layers):
    #         state[f"layer_{i}"] = layer.parameters
    #
    #     destination = Path(path).resolve() if path is not None else self.save_path
    #     destination = destination / f"{name}.pth"
    #     th.save(state, destination)
    #     print(f"Saved weights to {destination}")
    #
    # def load_weights(self, path: None | str = None):
    #     source = path if path is not None else self.save_path
    #     state = th.load(source)
    #
    #     for i, layer in enumerate(self._layers):
    #         saved_params = state[f"layer_{i}"]
    #
    #         for p_current, p_saved in zip(layer.parameters, saved_params):
    #             p_current.data.copy_(p_saved)
    #
    #     print(f"Weights loaded from {source}")