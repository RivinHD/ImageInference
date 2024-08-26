# SPDX-FileCopyrightText: © 2024 Vincent Gerlach
#
# SPDX-License-Identifier: MIT

import sys
import os
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import json
import time
from collections import defaultdict
import aimet_common.defs
import aimet_torch.quantsim
from aimet_torch import model_preparer
from aimet_torch.quantsim import QuantizationDataType

EVALUATE_WARMUP = 100
EVALUATE_REPETITIONS = 5000

# Ensures that the backend is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.execu_python.datasets import getImageNet

# See https://quic.github.io/aimet-pages/AimetDocs/api_docs/torch_quantsim.html for example on how to use the AIMET


def calibrate(model, forward_pass_args=None):

    imagenet = getImageNet()

    with torch.no_grad():
        for data in imagenet:
            model(data)


def quantize(model, param_bw, output_bw, quantType: QuantizationDataType) -> torch.nn.Module:
    # Quantize the resnet50 model with Aimet
    dummy_input = torch.randn(1, 3, 224, 224)
    quant_scheme = aimet_common.defs.QuantScheme.post_training_tf_enhanced

    prepared_model = model_preparer.prepare_model(resnet)

    config_file = os.path.join(os.path.dirname(__file__), "config.json")

    sim = aimet_torch.quantsim.QuantizationSimModel(prepared_model,
                                                    dummy_input=dummy_input,
                                                    quant_scheme=quant_scheme,
                                                    default_param_bw=param_bw,
                                                    default_output_bw=output_bw,
                                                    default_data_type=quantType,
                                                    config_file=config_file)

    print(f"Quantizing the model with {param_bw} bits for parameters" +
          f"and {output_bw} bits for output of type {quantType}")
    sys.stdout.flush()

    sim.compute_encodings(forward_pass_callback=calibrate,
                          forward_pass_callback_args=None)

    print("Finished quantizing the model")
    sys.stdout.flush()

    return sim.model


if __name__ == "__main__":
    resnet = models.resnet50(ResNet50_Weights.IMAGENET1K_V2).eval()
    image_sizes = [
        224,  # Default resnet50 input size
        256,  # typical  low image resolution
        448,  # double resnet50 input size
        512,  # typical medium image resolution
        672,  # triple resnet50 input size
        720,  # typical high image resolution
    ]

    data = defaultdict(lambda: defaultdict(dict))

    models = {
        "fp32": resnet,
        "fp16": quantize(resnet, 16, 16, QuantizationDataType.float),
        "int8": quantize(resnet, 8, 8, QuantizationDataType.int),
        "int16": quantize(resnet, 16, 16, QuantizationDataType.int),
    }

    with torch.no_grad():
        for quantType, model in models.items():
            print(f"Processing model {quantType}")
            for size in image_sizes:
                print(f"Processing image size {size} x {size}")
                sys.stdout.flush()
                input_tensor = torch.rand(1, 3, size, size)
                data[quantType][size]["average"] = 0
                data[quantType][size]["min"] = float("inf")
                data[quantType][size]["max"] = float("-inf")
                for _ in range(EVALUATE_WARMUP):
                    output = model(input_tensor)
                for i in range(EVALUATE_REPETITIONS):
                    timer = time.time()
                    output = model(input_tensor)
                    delta_time = time.time() - timer
                    data[quantType][size]["average"] += delta_time
                    data[quantType][size]["min"] = min(data[quantType][size]["min"], delta_time)
                    data[quantType][size]["max"] = max(data[quantType][size]["max"], delta_time)
                data[quantType][size]["average"] /= EVALUATE_REPETITIONS

                with open("results.json", "w") as file:
                    json.dump(data, file, indent=4)
