from math import ceil
from torch import Tensor


class BatchGenerator:
    def __init__(self, images, labels, batch_size: int = 128):
        assert len(images) == len(labels)
        self.index = 0
        self.images: Tensor = images
        self.labels: Tensor = labels
        self.batch_size = batch_size
        self.num_batches = ceil(len(self.images) / self.batch_size)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.images):
            raise StopIteration

        images = self.images[self.index: self.index + self.batch_size].clone()
        labels = self.labels[self.index: self.index + self.batch_size].clone()
        self.index += self.batch_size
        return images, labels