# Copyright (c) 2024 by Vincent Gerlach. All rights reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# in the root folder of this project with the name LICENSE. If not, see <http://www.gnu.org/licenses/>.

import argparse
import os
import sys

import torch
from torchvision import models
from torchvision.models import ResNet50_Weights
from torchvision.models._api import WeightsEnum
from executorch.examples.portable.utils import export_to_exec_prog
from executorch.exir import EdgeCompileConfig
from typing import Dict, Optional
from torch.nn.parameter import Parameter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.baremetal.resnet50v15_module import custom_resnet50
from backend.baremetal.export_utils import getResnet50Weights, compressParameters


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--so_library",
        required=True,
        help="Provide path to so library. E.g., cmake-out/portable/custom_ops/baremetal_ops_aot_lib.so",
    )
    args = parser.parse_args()

    print("Starting the testing process.")

    # See if we have custom op  baremetal_ops::resnet50.out registered
    has_out_ops = True
    try:
        op = torch.ops.baremetal_ops.resnet50.out
    except AttributeError:
        print("No registered custom op baremetal_ops::resnet50.out")
        has_out_ops = False
    if not has_out_ops:
        if args.so_library:
            print(f"Loading library at {args.so_library}")
            torch.ops.load_library(args.so_library)
        else:
            raise RuntimeError(
                "Need to specify shared library path to register custom op baremetal_ops::resnet50.out into"
                "EXIR. The required shared library is defined as `baremetal_ops_aot_lib` in "
                "backend/baremetal/CMakeLists.txt if you are using CMake build,"
                "libcustom_ops_aot_lib.[so|dylib]."
            )

    # Lowering the Model with Executorch
    parameters = getResnet50Weights(ResNet50_Weights.IMAGENET1K_V2)
    model = custom_resnet50(compressParameters(parameters)).eval()
    torch_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()

    torch.manual_seed(123)

    for i in range(100):
        with torch.no_grad():
            input = torch.randn(1, 3, 224, 224)
            output = model(input.clone().detach())

            expected_output = torch_model(input)
            torch.testing.assert_close(output, expected_output[0])

    print("Successfully finished testing.")
