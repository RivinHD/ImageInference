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

import torch
from torchvision import models
from torchvision.models import ResNet50_Weights
from torchvision.models._api import WeightsEnum
from executorch.examples.portable.utils import export_to_exec_prog
from executorch.exir import EdgeCompileConfig


class custom_resnet50(torch.nn.Module):
    def __init__(self, weights: WeightsEnum):
        super(custom_resnet50, self).__init__()

        self._torch_model = models.resnet50(weights=weights)
        self._parameters = self._torch_model._parameters
        self.weight_list = [weight.data for weight in self.parameters()]

    def forward(self, a):
        return torch.ops.baremetal_ops.resnet50.default(a, self.weight_list)


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

    op = torch.ops.baremetal_ops.test_resnet50_conv3x3_channels16x16.default

    # Lowering the Model with Executorch
    model = custom_resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    torch_model = model._torch_model.eval()

    torch.manual_seed(123)

    with torch.no_grad():
        input = torch.randn(1, 3, 244, 244)
        output = model(input.clone().detach())

        out = model._torch_model.conv1(input)
        # out = model._torch_model.bn1(out)
        # out = model._torch_model.relu(out)

        torch.testing.assert_close(output, out[0])

        # expected_output = torch_model(input)
        # torch.testing.assert_close(output, expected_output[0])

    print("Successfully finished testing.")
