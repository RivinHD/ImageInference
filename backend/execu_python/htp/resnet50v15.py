# SPDX-FileCopyrightText: © 2024 Vincent Gerlach
#
# SPDX-License-Identifier: MIT

import argparse
import os

import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torch._export import capture_pre_autograd_graph
from .builder import quantize, ExtendedQuantDtype
from ..datasets import getImageNet
from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.qualcomm.utils.utils import (
    capture_program,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
)
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import (
    QcomChipset
)
from executorch.exir import ExirExportedProgram
from executorch.exir.backend.backend_api import to_backend

chipset_parse = {
    "SM8650": QcomChipset.SM8650,
    "SM8550": QcomChipset.SM8550,
    "SM8475": QcomChipset.SM8475,
    "SM8450": QcomChipset.SM8450,
}

quantize_parser = {
    "int8": ExtendedQuantDtype.use_8a8w,
    "int16": ExtendedQuantDtype.use_16a16w,
    "int16int4": ExtendedQuantDtype.use_16a4w,
    "int8int4": ExtendedQuantDtype.use_8a4w
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q",
        "--quantize",
        required=False,
        default=False,
        help="Flag for producing quantized or floating-point model. If false is chosen nothing is done.\n"
        + "if multiple types are listed the first one is for the activation functions and the second one is"
        + "for the weights.\n"
        + "E.g. int16int4 uses int16 for the activation functions and int4 for the weights.",
        choices=["false", "int8", "int16", "int16int4", "int8int4"],
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

    print("Processing ResNet50v15 model with HTP partitioner.")

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

    if args.quantize != "false":
        print("Starting quantization")
        imagenet_dataset = getImageNet()
        resnet50 = quantize(resnet50, imagenet_dataset, quantize_parser[args.quantize])
    else:
        print("WARNING: The non-quantized model does not work correctly, "
              + "therefore it is disabled in the app and will not be generated.")
        exit(0)

    exec_program: ExirExportedProgram = capture_program(resnet50, sample_input)

    backend_options = generate_htp_compiler_spec(
        use_fp16=False
    )
    qnn_partitioner = QnnPartitioner(
        generate_qnn_executorch_compiler_spec(
            soc_model=chipset_parse[args.model],
            backend_options=backend_options,
            debug=False,
            saver=False,
            shared_buffer=False,
        ),
        skip_node_id_set=set(
        ),
        skip_node_op_set=set(
        ),
    )

    exec_program.exported_program = to_backend(exec_program.exported_program, qnn_partitioner)

    exec_program = exec_program.to_executorch(config=ExecutorchBackendConfig(
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
    quantize_tag = args.quantize if args.quantize != "False" else "fp32"
    os.makedirs("models-out", exist_ok=True)
    with open(f"models-out/resnet50v15_htp_{quantize_tag}_{args.model}.pte", "wb") as file:
        exec_program.write_to_file(file)

    print("Finished processing ResNet50v15 model with XNNPACK partitioner.")
