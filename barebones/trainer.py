from typing import Callable, Dict, List, Optional, Tuple
import torch as th

from torch import Tensor
from barebones.naive_layers import NaiveSequential as SequentialLayer
from barebones import batch_generator, optim
from barebones.config import TrainingConfig, History
from barebones.helper import measure

class Trainer:
    def __init__(
            self,
            model: SequentialLayer, /,
            loss_fn: Callable,
            optimizer: optim.NaiveOptimizer, *,
            config: TrainingConfig
    ):
        self.model = model
        self.config = config

        self.history: History = History()

        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.test: Callable[[], None] = lambda: None
        self.current_epoch = 0

        self.improv: Callable[[], None] = lambda: None

    @measure
    def fit(self, _input: Tensor, labels: Tensor):
        N = _input.shape[0]
        best_acc: float = 0.

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            self.model.train(True)
            gen = batch_generator.BatchGenerator(
                _input,
                labels,
                self.config.batch_size,
                shuffle=self.config.batch_shuffle,
                augment=self.config.batch_augment,
            )
            # print(f"Batch size: {len(gen)}")

            epoch_loss = 0.
            for _in, _lab in gen:
                # print(f"{gen.num_iterations} - {_in.shape}")
                predictions = self.model(_in)

                loss = self.loss_fn(predictions, _lab)
                epoch_loss += loss.item()

                loss.backward()

                value = epoch if self.optimizer.config.lrs is not None else None

                self.optimizer.step(value)
                self.optimizer.zero_grad()

            avg_loss = epoch_loss / (N / self.config.batch_size)
            self.history.central_loss.append(avg_loss)

            if (epoch + 1) % self.config.every_nth == 0 or (epoch + 1) <= self.config.first_n:
                self.test()
                self.improv()

            if (epoch + 1) % 10 == 0:
                self.history.analyze(10)

            if self.history.test_acc and best_acc < self.history.test_acc[-1]:
                best_acc = self.history.test_acc[-1]
                print(f"Epoch {epoch + 1}, best test acc: {best_acc:.4f}")
                self.model.save_weights("best_model")


            if self.history.test_loss and self.history.test_loss[-1] < self.config.patience:
                print(f"Epoch {epoch + 1}, early stopping")
                break

    def run_test(self, test_data: Tensor, test_labels: Tensor):
        def test():
            self.model.train(False)
            test_gen = batch_generator.BatchGenerator(test_data, test_labels, self.config.batch_size)
            total_correct = 0
            total_loss = 0.0
            num_test_samples = 0

            with th.no_grad():
                for t_imgs, t_lab in test_gen:
                    # print(f"Iteration #{idx}")
                    predictions = self.model(t_imgs)

                    batch_loss = th.nn.functional.cross_entropy(predictions, t_lab).item()
                    total_loss += batch_loss * t_imgs.shape[0]

                    predicted = th.argmax(predictions, 1)
                    total_correct += th.eq(predicted, t_lab).sum().item()

                    num_test_samples += t_imgs.shape[0]

                avg_test_loss = total_loss / num_test_samples
                test_acc = total_correct / num_test_samples

                gap = avg_test_loss - self.history.central_loss[-1]
                self.history.entry(self.history.central_loss[-1], avg_test_loss, test_acc, gap)

                print(
                    f"Epoch {self.current_epoch + 1}, "
                    f"average train loss: {self.history.loss[-1]:.4f}, "
                    f"average test loss: {avg_test_loss:.4f}, "
                    f"generalization gap: {gap:.4f}, "
                    f"test accuracy: {test_acc:.4f}"
                )

        self.test = test

        def measure_improvement():
            if len(self.history.test_acc) < 2 or len(self.history.test_loss) < 2:
                return

            acc = self.history.test_acc[-2]
            acc_new = self.history.test_acc[-1]
            length = len(test_data)

            data_improvement = length * (1 - acc)
            data_improvement_new = length * (1 - acc_new)
            improvement = abs(data_improvement - data_improvement_new) / data_improvement

            delta = self.history.test_loss[-2] - self.history.test_loss[-1]

            print(
                f"Improvement measure: {acc:.2%} -> {acc_new:.2%} => {improvement:.2%}",
                f"Loss improvement measure: {self.history.test_loss[-2]:.4f} -> {self.history.test_loss[-1]:.4f} => {delta:.4f}",
                sep="\n"
            )

        self.improv = measure_improvement
