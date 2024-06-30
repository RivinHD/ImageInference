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

import torch
from executorch.exir import EdgeCompileConfig
from torchvision import models
from torchvision.models import ResNet50_Weights
from torchvision.models._api import WeightsEnum


class custom_resnet50(torch.nn.Module):
    def __init__(self, weights: WeightsEnum):
        super(custom_resnet50, self).__init__()

        model = models.resnet50(weights=weights)
        self.weights = [weight.data for weight in model.parameters()]

    def forward(self, a):
        return torch.ops.baremetal_ops.resnet50.default(a, self.weights)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--so_library",
        required=True,
        help="Provide path to so library. E.g., cmake-android-out/portable/custom_ops/libcustom_ops_aot_lib.so",
    )
    args = parser.parse_args()

    # See if we have custom op my_ops::mul4.out registered
    has_out_ops = True
    try:
        op = torch.ops.baremetal_ops.resnet50.out
    except AttributeError:
        print("No registered custom op baremetal_ops::resnet50.out")
        has_out_ops = False
    if not has_out_ops:
        if args.so_library:
            torch.ops.load_library(args.so_library)
        else:
            raise RuntimeError(
                "Need to specify shared library path to register custom op my_ops::mul4.out into"
                "EXIR. The required shared library is defined as `custom_ops_aot_lib` in "
                "examples/portable/custom_ops/CMakeLists.txt if you are using CMake build,"
                " or `custom_ops_aot_lib_2` in "
                "examples/portable/custom_ops/targets.bzl for buck2."
                "One example path would be cmake-out/examples/portable/custom_ops/"
                "libcustom_ops_aot_lib.[so|dylib]."
            )
    print(args.so_library)

    # Lowering the Model with Executorch
    model = custom_resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
