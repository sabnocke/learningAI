from math import ceil
from torch import Tensor
import torch as th


class BatchGenerator:
    def __init__(self, images, labels, batch_size: int = 128, shuffle: bool = False, augment: bool = False):
        assert len(images) == len(labels)
        self.index = 0
        self.images: Tensor = images
        self.labels: Tensor = labels
        self.batch_size = batch_size
        self.num_batches = ceil(len(self.images) / self.batch_size)
        self.num_iterations = 0

        self.current_idx = 0
        self.indices = th.arange(len(self.images))
        self.n_samples = len(self.images)

        self.shuffle = shuffle
        self.augment = augment

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        self.index = 0

        if self.shuffle:
            self.indices = th.randperm(len(self.images))
        else:
            self.indices = th.arange(0, len(self.images))

        return self

    def __next__(self):
        if self.current_idx >= len(self.images):
            raise StopIteration

        end_idx = min(self.current_idx + self.batch_size, len(self.images))
        batch_indices = self.indices[self.current_idx:end_idx]
        images = self.images[batch_indices]
        labels = self.labels[batch_indices]

        if self.augment:
            flip_mask = th.rand(images.shape[0], 1, 1, 1) > 0.5
            flip_mask = flip_mask.to(images.device)

            flipped_images = th.flip(images, dims=[3])

            assert flip_mask.device == flipped_images.device == images.device, \
                f"{flip_mask.device} != {flipped_images.device} != {images.device}"
            images = th.where(flip_mask, images, flipped_images)

        self.current_idx += self.batch_size
        self.num_iterations += 1



        return images, labels
