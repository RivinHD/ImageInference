# SPDX-FileCopyrightText: © 2024 Vincent Gerlach
#
# SPDX-License-Identifier: MIT

import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
import torch.utils.data as data
import torch
import shutil

IMAGES_TO_COPY = 5000


if __name__ == "__main__":
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

    dataset = datasets.ImageNet(
        root=dataset_path,
        split='val',
        transform=transformer
    )

    torch.manual_seed(123)
    ran_sampler = data.RandomSampler(dataset, num_samples=IMAGES_TO_COPY)

    working_directory = os.path.dirname(os.path.dirname(__file__))
    destinationDirectory = os.path.join(
        working_directory,
        "android", "app", "src", "main", "assets",
        "labeled_collections", "imagenet_2012"
    )

    if os.path.exists(destinationDirectory):
        shutil.rmtree(destinationDirectory)
    os.makedirs(destinationDirectory, exist_ok=True)

    print("Copying ImageNet 2012 as benchmark collection to android app.")

    for i in ran_sampler:
        sample, index = dataset.samples[i]
        head, tail = os.path.split(sample)
        destinationPath = os.path.join(destinationDirectory, f"{index}_{tail}")
        shutil.copy(sample, destinationPath)
