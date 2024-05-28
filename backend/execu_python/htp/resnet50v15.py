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
from executorch.exir import EdgeProgramManager, to_edge
from torch.export import ExportedProgram
from torchvision.models import ResNet50_Weights
from torch._export import capture_pre_autograd_graph
from .builder import quantize
from ..datasets import getImageNet
from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.qualcomm.utils.utils import (
    capture_program,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
)
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q",
        "--quantize",
        required=False,
        default=False,
        help="Flag for producing quantized or floating-point model",
        action='store_true'
    )
    parser.add_argument(
        "-m",
        "--model",
        help="SoC model of current device." +
        "Available 'SM8650', 'SM8550', 'SM8475', 'SM8450' e.g. 'SM8550' for Snapdragon 8 Gen 2.",
        type=str,
        required=True,
        choices=["SM8650", "SM8550", "SM8475", "SM8450"],
    )
    args = parser.parse_args()

    # Check for need environment variables.
    if "QNN_SDK_ROOT" not in os.environ:
        raise RuntimeError("Environment variable QNN_SDK_ROOT must be set")
    print(f"QNN_SDK_ROOT={os.getenv('QNN_SDK_ROOT')}")

    if "LD_LIBRARY_PATH" not in os.environ:
        raise RuntimeError("Environment variable LD_LIBRARY_PATH must be set")
    print(f"LD_LIBRARY_PATH={os.getenv('LD_LIBRARY_PATH')}")

    resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    sample_input = (torch.randn(1, 3, 224, 224),)

    resnet50 = capture_pre_autograd_graph(resnet50, sample_input)

    if args.quantize:
        imagenet_dataset = getImageNet()
        resnet50 = quantize(resnet50, imagenet_dataset)

    exported_program: ExportedProgram = capture_program(resnet50, sample_input).exported_program
    edge: EdgeProgramManager = to_edge(exported_program)

    backend_options = generate_htp_compiler_spec(
        use_fp16=False
    )
    qnn_partitioner = QnnPartitioner(
        generate_qnn_executorch_compiler_spec(
            soc_model=args.model,
            backend_options=backend_options,
            debug=False,
            saver=False,
            shared_buffer=False,
        )
    )

    edge = edge.to_backend(qnn_partitioner)

    print(edge.exported_program().graph_module)

    exec_program = edge.to_executorch(config=ExecutorchBackendConfig(
        extract_constant_segment=False,
        # For shared buffer, user must pass the memory address
        # which is allocated by RPC memory to executor runner.
        # Therefore, won't want to pre-allocate
        # by memory manager in runtime.
        memory_planning_pass=MemoryPlanningPass(
            memory_planning_algo="greedy",
        ),
        extract_delegate_segments=True,
    ))
    quantize_tag = "q8" if args.quantize else "fp32"
    os.makedirs("models-out", exist_ok=True)
    with open(f"models-out/resnet50v15_htp_{quantize_tag}.pte", "wb") as file:
        exec_program.write_to_file(file)
