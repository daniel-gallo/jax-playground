import numpy as np
from torch.utils import data
import torchvision
from jax.tree_util import tree_map


def numpy_collate(batch):
    return tree_map(np.asarray, data.default_collate(batch))


def get_mnist_dataloaders(batch_size=64):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5),
    ])
    train_dataset = torchvision.datasets.MNIST("datasets", train=True, download=True, transform=transforms)
    test_dataset = torchvision.datasets.MNIST("datasets", train=False, download=True, transform=transforms)

    loader_kwargs = dict(collate_fn=numpy_collate, batch_size=batch_size, shuffle=True, drop_last=True)
    train_dataloader = data.DataLoader(train_dataset, **loader_kwargs)
    test_dataloader = data.DataLoader(test_dataset, **loader_kwargs)

    return train_dataloader, test_dataloader

