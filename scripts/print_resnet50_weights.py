# SPDX-FileCopyrightText: © 2024 Vincent Gerlach
#
# SPDX-License-Identifier: MIT

import sys
import os
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# Path to parent directory of this script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.baremetal.export_utils import getResnet50Weights

if __name__ == "__main__":
    for i, (name, param) in enumerate(getResnet50Weights(ResNet50_Weights.IMAGENET1K_V2).items()):
        print(f"{i:<3} - {name}: {list(param.size())}")

    print("\n\nC++ Enum of the weights:\n\n")
    print("enum weightIndex")
    print("{")
    for i, (name, param) in enumerate(getResnet50Weights(ResNet50_Weights.IMAGENET1K_V2).items()):
        print(f"    {name.replace('.', '_')} = {i},  // {list(param.size())}")
    print("};")
