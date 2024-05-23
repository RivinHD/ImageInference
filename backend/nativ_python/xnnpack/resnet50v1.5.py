import argparse

import torch
import torchvision.models as models
from executorch.backends.xnnpack.partition.xnnpack_partitioner import \
    XnnpackPartitioner
from executorch.exir import EdgeCompileConfig, EdgeProgramManager, to_edge
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config)
from torch.export import ExportedProgram, export
from torchvision.models import ResNet50_Weights


def quantize(model, example_inputs):
    """This is the official recommended flow for quantization in pytorch 2.0 export"""
    print(f"Original model: {model}")
    quantizer = XNNPACKQuantizer()
    # if we set is_per_channel to True, we also need to add out_variant of quantize_per_channel/dequantize_per_channel
    operator_config = get_symmetric_quantization_config(is_per_channel=False)
    quantizer.set_global(operator_config)
    m = prepare_pt2e(model, quantizer)
    # calibration
    m(*example_inputs)
    m = convert_pt2e(m)
    print(f"Quantized model: {m}")
    # make sure we can export to flat buffer
    return m


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q",
        "--quantize",
        required=False,
        default=False,
        help="Flag for producing quantized or floating-point model",
    )
    args = parser.parse_args()

    # Lowering the model to XNNPACK
    resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    sample_input = (torch.randn(1, 3, 224, 224),)

    compile_config = EdgeCompileConfig()

    if args.quantize:
        resnet50 = quantize(resnet50, sample_input)
        compile_config = EdgeCompileConfig(_check_ir_validity=False)

    exported_program: ExportedProgram = export(resnet50, sample_input)
    edge: EdgeProgramManager = to_edge(exported_program, compile_config=compile_config)

    edge = edge.to_backend(XnnpackPartitioner())

    print(edge.exported_program().graph_module)

    exec_program = edge.to_executorch()

    quantize_tag = "q8" if args.quantize else "fp32"
    with open(f"resnet50v1.5_xnnpack_{quantize_tag}.pte", "wb") as file:
        exec_program.write_to_file(file)
