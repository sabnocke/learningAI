from pathlib import Path
from typing import Union
import torch as th
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch import Tensor
from barebones.config import TrainingConfig
from barebones.naive_layers.naive_conv import NaiveConv2d
from barebones.trainer import Trainer
from barebones.optim import NaiveOptimizer
from barebones.naive_lrs.cosine_annealing import CosineAnnealingLR
from barebones.naive_layers import NaiveLinear, NaiveSequential, NaiveDropout, NaiveReLU, NaiveMaxPool2d, \
    NaiveBatchNorm1d, NaiveBatchNorm2d
from naive_layers import NaiveFlatten

type Datasets = Union[tv.datasets.MNIST, tv.datasets.FashionMNIST]


def transform(_in: Datasets, device: th.device):
    one: Tensor = _in.data.reshape(-1, 28 * 28).to(th.float32) / 255
    # The division converts int B&W to float B&W
    one = one.to(device)

    labels = _in.targets
    labels = labels.to(device)

    return one, labels


def transform_no_reshape(_in: Datasets, device: th.device):
    one: Tensor = _in.data.unsqueeze(1).float() / 255
    one = one.to(device)

    labels: Tensor = _in.targets.to(device)

    return one, labels


def provide_data(device: th.device, reshape: bool = True):
    data_train = tv.datasets.MNIST(
        root=Path.cwd(), train=True, download=True, transform=tv.transforms.ToTensor()
    )
    data_test = tv.datasets.MNIST(
        root=Path.cwd(), train=False, download=True, transform=tv.transforms.ToTensor()
    )

    if reshape:
        train, train_labels = transform(data_train, device)
        test, test_labels = transform(data_test, device)
    else:
        train, train_labels = transform_no_reshape(data_train, device)
        test, test_labels = transform_no_reshape(data_test, device)

    return train, train_labels, test, test_labels


def provide_data2(device: th.device, reshape: bool = True):
    train_data = tv.datasets.FashionMNIST(
        root=Path.cwd(), train=True, download=True, transform=tv.transforms.ToTensor()
    )
    test_data = tv.datasets.FashionMNIST(
        root=Path.cwd(), train=False, download=True, transform=tv.transforms.ToTensor()
    )

    if reshape:
        train, train_labels = transform(train_data, device)
        test, test_labels = transform(test_data, device)
    else:
        train, train_labels = transform_no_reshape(train_data, device)
        test, test_labels = transform_no_reshape(test_data, device)

    print(train.shape)

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
        epochs=50,
        momentum=0.9,
        batch_size=128,
        every_nth=1,
        first_n=0,
        lrs=CosineAnnealingLR(1e-4, 1e-2, 20, True),
        weight_decay=1e-4,
        batch_augment=True,
        batch_shuffle=True,
    )

    # MLP (Multi-Layer Perceptron) - old model
    # model = NaiveSequential(
    #     NaiveLinear(28 * 28, 512),
    #     NaiveBatchNorm1d(512),
    #     NaiveReLU(),
    #     NaiveDropout(0.2),
    #
    #     NaiveLinear(512, 256),
    #     NaiveBatchNorm1d(256),
    #     NaiveReLU(),
    #     NaiveDropout(0.2),
    #
    #     NaiveLinear(256, 128),
    #     NaiveBatchNorm1d(128),
    #     NaiveReLU(),
    #     NaiveDropout(0.2),
    #
    #     NaiveLinear(128, 10),
    #     config=cfg,
    # ).to(device)

    model = NaiveSequential(
        # (1, 28, 28) -> (32, 26, 26)
        # pixels are lost due to kernel size
        NaiveConv2d(1, 32, (3, 3)),
        NaiveBatchNorm2d(32),
        NaiveReLU(),
        # (32, 26, 26) -> (32, 13, 13)
        NaiveMaxPool2d(),

        # (32, 13, 13) -> (64, 11, 11)
        NaiveConv2d(32, 64, (3, 3)),
        NaiveBatchNorm2d(64),
        NaiveReLU(),
        # (64, 11, 11) -> (64, 5, 5)
        NaiveMaxPool2d(),

        # Input: (64, 5, 5) -> Output: (64 * 5 * 5) = 1600
        NaiveFlatten(),

        NaiveLinear(64 * 5 * 5, 128),
        NaiveBatchNorm1d(128),
        NaiveReLU(),
        NaiveDropout(0.2),

        NaiveLinear(128, 10),
        config=cfg
    ).to(device)

    trainer = Trainer(
        model,
        loss_fn=nn.functional.cross_entropy,
        optimizer=NaiveOptimizer(model.parameters, cfg),
        config=cfg,
    )

    train, train_labels, test, test_labels = provide_data(device, False)
    print(train.shape)

    assert model.layers[0].weights.device == trainer.optimizer.velocities[0].device
    print("System is ready for GPU training!")

    print(cfg)
    trainer.run_test(test, test_labels)
    trainer.fit(train, train_labels)
    trainer.history.plot()


if __name__ == '__main__':
    main()
