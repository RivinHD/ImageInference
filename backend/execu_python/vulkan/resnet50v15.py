# SPDX-FileCopyrightText: © 2024 Vincent Gerlach
#
# SPDX-License-Identifier: MIT

import argparse
import os
import torch
import torchvision.models as models

from torch.export import export, ExportedProgram
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from torchvision.models import ResNet50_Weights
from executorch.exir import EdgeProgramManager, to_edge
from torch._export import capture_pre_autograd_graph
from ..datasets import getImageNet

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

    print("Processing ResNet50v15 model with Vulkan partitioner.")

    # Lowering the model to Vulkan
    resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    sample_input = (torch.randn(1, 3, 224, 224),)

    resnet50 = capture_pre_autograd_graph(resnet50, sample_input)

    if args.quantize:
        print("Starting quantization")
        imagenet_dataset = getImageNet()
        # resnet50 = quantize(resnet50, imagenet_dataset)
        raise RuntimeError("Vulkan currently does not support int8 quantization")

    exported_program: ExportedProgram = export(resnet50, sample_input)
    edge: EdgeProgramManager = to_edge(exported_program)

    edge = edge.to_backend(VulkanPartitioner())

    exec_program = edge.to_executorch()

    quantize_tag = "q8" if args.quantize else "fp32"
    os.makedirs("models-out", exist_ok=True)
    with open(f"models-out/resnet50v15_vulkan_{quantize_tag}.pte", "wb") as file:
        exec_program.write_to_file(file)

    print("Finished processing ResNet50v15 model with Vulkan partitioner.")
