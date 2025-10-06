import torch as th
import torchvision as tv
from pathlib import Path
from icecream import ic
from NaiveLinear import NaiveLinear
from NaiveSequential import NaiveSequential
import numpy as np

import matplotlib.pyplot as plt

def naive_relu(_in: th.Tensor) -> th.Tensor:
    return th.max(_in, th.zeros_like(_in))

def naive_softmax(_in: th.Tensor) -> th.Tensor:
    return th.exp(_in) / th.sum(th.exp(_in), -1, keepdim=True)

def identity(_in: th.Tensor) -> th.Tensor:
    return _in

def main() -> None:
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    ic(device)

    model = NaiveSequential(
        NaiveLinear(28 * 28, 512, naive_relu, batch_normalization=True),
        NaiveLinear(512, 256, naive_relu, batch_normalization=True),
        NaiveLinear(256, 128, naive_relu, batch_normalization=True),
        NaiveLinear(128, 10, identity, batch_normalization=True),
        learning_rate=1e-4,
    )

    data_train = tv.datasets.MNIST(root=Path.cwd(), train=True, download=True, transform=tv.transforms.ToTensor())
    data_test = tv.datasets.MNIST(root=Path.cwd(), train=False, download=True, transform=tv.transforms.ToTensor())

    train = data_train.data.reshape(-1, 28 * 28).to(th.float32) / 255
    train = train.to(device)

    train_labels = data_train.targets
    train_labels = train_labels.to(device)

    test = data_test.data.reshape(-1, 28 * 28).to(th.float32) / 255
    test = test.to(device)

    test_labels = data_test.targets
    test_labels = test_labels.to(device)

    model.testing = test
    model.test_labels = test_labels
    model.to(device)

    epochs = 100

    model.fit(train, train_labels, epochs=epochs, verbose=True)

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

    # plt.show()

    plt.savefig("training.png")

if __name__ == '__main__':
    main()