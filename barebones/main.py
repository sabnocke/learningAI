from pathlib import Path
from typing import Union

import torch as th
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

# from barebones.naive_layers import linear, sequential, activation
from barebones.aux import TrainingConfig
from barebones.trainer import Trainer
from barebones.optim import NaiveOptimizer
from barebones.naive_lrs.cosine_annealing import CosineAnnealingLR


# from barebones.naive_layers.dropout import NaiveDropout

from barebones.naive_layers import NaiveLinear, NaiveSequential, NaiveDropout, NaiveReLU

type Datasets = Union[tv.datasets.MNIST, tv.datasets.FashionMNIST]
type Tensor = th.Tensor

# NaiveSequential = sequential.NaiveSequential
# NaiveLinear = linear.NaiveLinear

def transform(_in: Datasets, device: th.device):
    one: Tensor = _in.data.reshape(-1, 28 * 28).to(th.float32) / 255
    # The division converts int B&W to float B&W
    one = one.to(device)

    labels = _in.targets
    labels = labels.to(device)

    return one, labels

def provide_data(device: th.device):
    data_train = tv.datasets.MNIST(
        root=Path.cwd(),train=True,download=True,transform=tv.transforms.ToTensor()
    )
    data_test = tv.datasets.MNIST(
        root=Path.cwd(),train=False,download=True,transform=tv.transforms.ToTensor()
    )

    train, train_labels = transform(data_train, device)
    test, test_labels = transform(data_test, device)

    return train, train_labels, test, test_labels

def provide_data2(device: th.device):
    train_data = tv.datasets.FashionMNIST(
        root=Path.cwd(), train=True, download=True, transform=tv.transforms.ToTensor()
    )
    test_data = tv.datasets.FashionMNIST(
        root=Path.cwd(), train=False, download=True, transform=tv.transforms.ToTensor()
    )

    train, train_labels = transform(train_data, device)
    test, test_labels = transform(test_data, device)

    return train, train_labels, test, test_labels

def display_graph(model, epochs):
    x = np.arange(0, epochs)
    y1 = np.array(model.loss_collection)
    y2 = np.array(model.acc_collection)
    y3 = np.array(model.lr_collection)

    y3_norm = (y3 - min(y3)) / (max(y3) - min(y3) + 1e-10)

    fig, ax = plt.subplots()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss", color="r")
    ax.plot(x, y1, label="Training Loss", color="r")
    ax.tick_params(axis="y", labelcolor="r")

    ax2 = ax.twinx()
    ax2.set_ylabel("Accuracy", color="b")
    ax2.plot(x, y2, label="Training Accuracy", color="b")
    ax2.tick_params(axis="y", labelcolor="b")
    ax2.plot(x, y3_norm, label="Learning rate", color="g")

    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.85))

    plt.show()

def main() -> None:
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    # device = th.device("cpu")

    cfg = TrainingConfig(
        learning_rate=1e-4,
        epochs=100,
        momentum=0.9,
        batch_size=128,
        every_nth=10,
        lrs=CosineAnnealingLR(1e-4, 1e-2, 20, True),
        weight_decay=1e-4,
    )

    use_bn = True

    model = NaiveSequential(
        NaiveLinear(28 * 28, 512, batch_normalization=use_bn),
        NaiveReLU(),
        NaiveDropout(0.2),

        NaiveLinear(512, 256, batch_normalization=use_bn),
        NaiveReLU(),
        NaiveDropout(0.2),

        NaiveLinear(256, 128, batch_normalization=use_bn),
        NaiveReLU(),
        NaiveDropout(0.2),

        NaiveLinear(128, 10, batch_normalization=False),
        config=cfg,
    ).to(device)

    trainer = Trainer(
        model,
        loss_fn=nn.functional.cross_entropy,
        optimizer=NaiveOptimizer(model.parameters, cfg),
        config=cfg,
    )

    train, train_labels, test, test_labels = provide_data2(device)

    # model.testing = test
    # model.test_labels = test_labels
    # model.to(device)
    # model.verbose = True

    # epochs = 100

    print(f"Model device: {model.layers[0].weights.device}")
    print(f"Optimizer state device: {trainer.optimizer.velocities[0].device}")

    assert model.layers[0].weights.device == trainer.optimizer.velocities[0].device
    print("System is ready for GPU training!")

    trainer.run_test(test, test_labels)
    trainer.fit(train, train_labels)

if __name__ == '__main__':
    main()
