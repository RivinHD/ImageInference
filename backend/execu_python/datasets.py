#  Copyright (c) 2024 by Vincent Gerlach. All rights reserved.
#
#  SPDX-License-Identifier: GPL-3.0-or-later
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  in the root folder of this project with the name LICENSE. If not, see <http://www.gnu.org/licenses/>.

import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
import torch.utils.data as data
import ultralytics
import ultralytics.data

MAX_DATA_SIZE = 1500


def get_imagenet(transform: transforms.Compose, image_sizes: tuple) -> data.RandomSampler:
    if "IMAGENET_DATASET_2012" not in os.environ:
        raise RuntimeError("Environment variable IMAGENET_DATASET_2012 must be set")
    print(f"IMAGENET_DATASET_2012={os.getenv('IMAGENET_DATASET_2012')}")

    dataset_path = os.getenv('IMAGENET_DATASET_2012')

    dataset = datasets.ImageNet(
        root=dataset_path,
        split='val',
        transform=transform
    )

    ran_sampler = data.RandomSampler(dataset, num_samples=MAX_DATA_SIZE)
    return (dataset[i][0].reshape(1, 3, image_sizes[0], image_sizes[1]) for i in ran_sampler)


def get_coco() -> data.RandomSampler:
    dataset = ultralytics.data.dataset.YOLODataset(
        data='coco8.yaml',
        split='val',
    )

    ran_sampler = data.RandomSampler(dataset, num_samples=MAX_DATA_SIZE)

    return (dataset[i][0].reshape(1, 3, dataset.imgsz, dataset.imgsz) for i in ran_sampler)
