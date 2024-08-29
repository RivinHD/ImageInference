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
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.execu_python.datasets import getImageNetDataset, getImageNetSampler

SKIP_IMAGES = 2000  # The 2000 images can be skipped as they are might be used for quantization.
IMAGES_TO_COPY = 5000

if __name__ == "__main__":
    dataset = getImageNetDataset()
    quantization_sampler = getImageNetSampler()

    torch.manual_seed(123)
    ran_sampler = data.RandomSampler(dataset, num_samples=IMAGES_TO_COPY+SKIP_IMAGES)

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

    collectedImages = 0
    quantization_samples = set(quantization_sampler)
    for sampleIndex in ran_sampler:
        if sampleIndex in quantization_samples:
            continue
        sample, index = dataset.samples[sampleIndex]
        head, tail = os.path.split(sample)
        destinationPath = os.path.join(destinationDirectory, f"{index}_{tail}")
        shutil.copy(sample, destinationPath)
        collectedImages += 1
        if collectedImages >= IMAGES_TO_COPY:
            break
