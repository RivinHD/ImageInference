# SPDX-FileCopyrightText: © 2024 Vincent Gerlach
#
# SPDX-License-Identifier: MIT

import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
import torch.utils.data as data
import torch

MAX_DATA_SIZE = 1500


def getImageNetDataset() -> datasets.ImageNet:
    if "IMAGENET_DATASET_2012" not in os.environ:
        raise RuntimeError("Environment variable IMAGENET_DATASET_2012 must be set")
    print(f"IMAGENET_DATASET_2012={os.getenv('IMAGENET_DATASET_2012')}")

    dataset_path = os.getenv('IMAGENET_DATASET_2012')
    # The IMAGENET1K_V1 transforms is chosen on purpose as it matches with the actions during inference.
    # For more information see: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html.
    transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return datasets.ImageNet(
        root=dataset_path,
        split='val',
        transform=transformer
    )


def getImageNetSampler() -> data.RandomSampler:
    torch.manual_seed(123)
    dataset = getImageNetDataset()
    return data.RandomSampler(dataset, num_samples=MAX_DATA_SIZE)


def getImageNet() -> data.RandomSampler:
    dataset = getImageNetDataset()
    ran_sampler = getImageNetSampler()
    return (dataset[i][0].reshape(1, 3, 224, 224) for i in ran_sampler)
