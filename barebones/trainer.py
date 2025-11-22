from typing import Callable, Dict, List, Optional
import torch as th

from torch import Tensor
from barebones.naive_layers import AbstractLayer as Layer
from barebones import batch_generator, optim
from barebones.aux import TrainingConfig

class Trainer:
    def __init__(
            self,
            model: Layer, /,
            loss_fn: Callable,
            optimizer: optim.NaiveOptimizer, *,
            config: TrainingConfig
    ):
        self.model = model
        self.config = config

        self.history: Dict[str, List[float]] = {
            "loss": [],
            "accuracy": [],
        }

        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.test: Optional[Callable[[], float]] = None

    def fit(self, _input: Tensor, labels: Tensor):
        N = _input.shape[0]

        for epoch in range(self.config.epochs):
            self.model.train(True)
            gen = batch_generator.BatchGenerator(_input, labels, self.config.batch_size)

            epoch_loss = 0.
            for _in, _lab in gen:
                predictions = self.model(_in)

                loss = self.loss_fn(predictions, _lab)
                epoch_loss += loss.item()

                loss.backward()



                value = epoch if self.optimizer.config.lrs is not None else None
                self.optimizer.step(value)
                self.optimizer.zero_grad()

            avg_loss = epoch_loss / (N / self.config.batch_size)
            self.history["loss"].append(avg_loss)
            if (epoch + 1) % self.config.every_nth == 0:
                print(
                    f"Epoch {epoch + 1}, average loss: {avg_loss:.4f}, "
                    f"test accuracy: {self.test()}" if self.test is not None else ""
                )

    def run_test(self, test_data: Tensor, test_labels: Tensor):
        def test():
            with th.no_grad():
                self.model.train(False)
                predictions = self.model(test_data)
                predicted = th.argmax(predictions, 1)
                correct = th.eq(predicted, test_labels).sum().item()
                accuracy = correct / test_labels.size(0)
                self.model.train(True)
                return accuracy

        self.test = test
