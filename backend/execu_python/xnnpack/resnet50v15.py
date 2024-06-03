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

import argparse
import os

import torch
import torchvision.models as models
from executorch.backends.xnnpack.partition.xnnpack_partitioner import \
    XnnpackPartitioner
from executorch.exir import EdgeCompileConfig, EdgeProgramManager, to_edge
from torch.export import ExportedProgram, export
from torchvision.models import ResNet50_Weights

from .builder import quantize
from ..datasets import getImageNet
from torch._export import capture_pre_autograd_graph

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q",
        "--quantize",
        required=False,
        default=False,
        help="Flag for producing quantized or floating-point model",
        action='store_true',
    )
    args = parser.parse_args()

    print("Processing ResNet50v15 model with XNNPACK partitioner.")

    # Lowering the model to XNNPACK
    resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    sample_input = (torch.randn(1, 3, 224, 224),)

    resnet50 = capture_pre_autograd_graph(resnet50, sample_input)

    compile_config = EdgeCompileConfig()

    if args.quantize:
        print("Starting quantization")
        imagenet_dataset = getImageNet()
        resnet50 = quantize(resnet50, imagenet_dataset)
        compile_config = EdgeCompileConfig(_check_ir_validity=False)

    exported_program: ExportedProgram = export(resnet50, sample_input)
    edge: EdgeProgramManager = to_edge(exported_program, compile_config=compile_config)

    edge = edge.to_backend(XnnpackPartitioner())

    exec_program = edge.to_executorch()

    quantize_tag = "q8" if args.quantize else "fp32"
    os.makedirs("models-out", exist_ok=True)
    with open(f"models-out/resnet50v15_xnnpack_{quantize_tag}.pte", "wb") as file:
        exec_program.write_to_file(file)

    print("Finished processing ResNet50v15 model with XNNPACK partitioner.")
