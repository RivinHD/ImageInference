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

        self._torch_model = models.resnet50(weights=weights)
        self.register_parameter("resnet50_weights", [weight.data for weight in self._torch_model.parameters()])

    def forward(self, a):
        return torch.ops.baremetal_ops.resnet50.default(a, self.get_parameter("resnet50_weights"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--so_library",
        required=True,
        help="Provide path to so library. E.g., cmake-out/portable/custom_ops/baremetal_ops_aot_lib.so",
    )
    args = parser.parse_args()

    # See if we have custom op  baremetal_ops::resnet50.out registered
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
                "Need to specify shared library path to register custom op baremetal_ops::resnet50.out into"
                "EXIR. The required shared library is defined as `baremetal_ops_aot_lib` in "
                "backend/baremetal/CMakeLists.txt if you are using CMake build,"
                "libcustom_ops_aot_lib.[so|dylib]."
            )
    print(args.so_library)

    # Lowering the Model with Executorch
    model = custom_resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
