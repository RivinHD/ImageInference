# SPDX-FileCopyrightText: © 2024 Vincent Gerlach
#
# SPDX-License-Identifier: MIT

import argparse
import os

import logging
import torch
from torchvision import models
from torchvision.models import ResNet50_Weights
from torchvision.models._api import WeightsEnum
from executorch.examples.portable.utils import export_to_exec_prog, _core_aten_to_edge, _to_core_aten
from executorch.exir import EdgeCompileConfig
from .resnet50v15_module import custom_resnet50
from . import export_utils
from torch._export import capture_pre_autograd_graph
from torch.export import export, ExportedProgram


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--so_library",
        required=True,
        help="Provide path to so library. E.g., cmake-out/portable/custom_ops/baremetal_ops_aot_lib.so",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        required=False,
        default=False,
        help="Flag for producing quantized or floating-point model",
        choices=["false"],
    )
    args = parser.parse_args()

    print("Processing ResNet50v15 model with Custom implementation.")

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

    # Registering a opaque operator for the custom implementation to not trace into it
    @torch.library.register_fake("baremetal_ops::resnet50")
    def _(input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.zeros([1, 1000])

    # Lowering the Model with Executorch
    parameters = export_utils.getResnet50Weights(ResNet50_Weights.IMAGENET1K_V2)
    model = custom_resnet50(export_utils.compressParameters(parameters))
    sample_input = (torch.randn(1, 3, 224, 224),)

    exec_program = export_to_exec_prog(
        model,
        sample_input,
        edge_compile_config=EdgeCompileConfig(_check_ir_validity=False)
    )
    exec_program.dump_executorch_program()

    quantize_tag = args.quantize if args.quantize != "false" else "fp32"
    os.makedirs("models-out", exist_ok=True)
    with open(f"models-out/resnet50v15_custom_{quantize_tag}.pte", "wb") as file:
        exec_program.write_to_file(file)

    print("Finished processing ResNet50v15 model with Custom implementation.")
