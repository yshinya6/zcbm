import os
from typing import Any, Tuple

import torchvision.datasets


class GenericDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform, test=False) -> None:
        assert (root is not None) and (transform is not None)
        train = not test
        if train:
            split_data_dir = os.path.join(root, "train")
        else:
            split_data_dir = os.path.join(root, "test")
        super(GenericDataset, self).__init__(root=split_data_dir, transform=transform)


class Aircraft(GenericDataset):
    def __init__(self, transform, test=True):
        super(Aircraft, self).__init__(
            root="/dataset/Aircraft",
            transform=transform,
            test=test,
        )


class Bird(GenericDataset):
    def __init__(self, transform, test=True):
        super(Bird, self).__init__(
            root="~/dataset/CUB_200_2011",
            transform=transform,
            test=test,
        )


class Car(GenericDataset):
    def __init__(self, transform, test=True):
        super(Car, self).__init__(
            root="/dataset/StanfordCars",
            transform=transform,
            test=test,
        )


class Caltech101(GenericDataset):
    def __init__(self, transform, test=True):
        super(Caltech101, self).__init__(
            root="/dataset/Caltech101",
            transform=transform,
            test=test,
        )


class DTD(GenericDataset):
    def __init__(self, transform, test=True):
        super(DTD, self).__init__(
            root="/dataset/DTD",
            transform=transform,
            test=test,
        )


class EuroSAT(GenericDataset):
    def __init__(self, transform, test=True):
        super(EuroSAT, self).__init__(
            root="/dataset/EuroSAT",
            transform=transform,
            test=test,
        )


class Food(GenericDataset):
    def __init__(self, transform, test=True):
        super(Food, self).__init__(
            root="/dataset/Food101",
            transform=transform,
            test=test,
        )


class Flower(GenericDataset):
    def __init__(self, transform, test=True):
        super(Flower, self).__init__(
            root="/dataset/OxfordFlower102",
            transform=transform,
            test=test,
        )


class ImageNet(torchvision.datasets.ImageFolder):
    def __init__(self, transform, test=False):
        root = "/dataset/imagenet"
        train = not test
        if train:
            split_data_dir = os.path.join(root, "train")
        else:
            split_data_dir = os.path.join(root, "val")
        super(ImageNet, self).__init__(root=split_data_dir, transform=transform)


class Pet(GenericDataset):
    def __init__(self, transform, test=True):
        super(Pet, self).__init__(
            root="/dataset/OxfordPets",
            transform=transform,
            test=test,
        )


class SUN397(GenericDataset):
    def __init__(self, transform, test=True):
        super(SUN397, self).__init__(
            root="/dataset/SUN397",
            transform=transform,
            test=test,
        )


class UCF101(GenericDataset):
    def __init__(self, transform, test=True):
        super(UCF101, self).__init__(
            root="/dataset/UCF101",
            transform=transform,
            test=test,
        )
