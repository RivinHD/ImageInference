# SPDX-FileCopyrightText: © 2024 Vincent Gerlach
#
# SPDX-License-Identifier: MIT

import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import os
import numpy as np
import io
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.baremetal.export_utils import getResnet50Weights


def writeTensor(file: io.BufferedWriter, tensor: torch.Tensor):
    """
    Writes a tensor in binary format into a tensor.

    The create binary has the structure:
    Tensor<countSizes><sizes><data>
    Tensor is a raw ascii text, which indicates that a new Tensor starts
    <countSizes> is in binary int64 and indicates the number of elements in the <sizes>
    <sizes> is in binary int64 and indicates the size of the tensor
    <data> is in binary float32 and contains the data of the tensor

    Args:
        file (io.BufferedWriter): The file to write the tensor.
        tensor (torch.Tensor): The tensor to write.
    """
    # Writing in 'C' style/order mean little endian
    file.write("Tensor".encode(encoding="ascii"))
    file.write(len(tensor.size()).to_bytes(8, byteorder="little"))
    file.write(np.array(tensor.size()).tobytes('C'))
    file.write(tensor.detach().numpy().tobytes('C'))


if __name__ == "__main__":

    torch.manual_seed(0)
    resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()

    base_directory = os.path.dirname(os.path.dirname(__file__))
    directory = "test_data"
    filePath = os.path.join(base_directory, directory, "resnet50_weights_v2.bin")
    os.makedirs(os.path.dirname(filePath), exist_ok=True)
    with open(filePath, "wb") as f:
        for name, param in getResnet50Weights(ResNet50_Weights.IMAGENET1K_V2).items():
            writeTensor(f, param)

    for i in range(10):
        testImage = torch.rand(1, 3, 224, 224)
        output: torch.Tensor = resnet50(testImage)
        filePath = os.path.join(base_directory, directory, f"resnet50_test{i}.bin")
        with open(filePath, "wb") as f:
            writeTensor(f, testImage)
            writeTensor(f, output)

        testBlock0 = torch.rand(1, 64, 56, 56)
        output: torch.Tensor = resnet50.layer1(testBlock0)
        filePath = os.path.join(base_directory, directory, f"resnet50_block0_test{i}.bin")
        with open(filePath, "wb") as f:
            writeTensor(f, testBlock0)
            writeTensor(f, output)

        testBlock1 = torch.rand(1, 256, 56, 56)
        output: torch.Tensor = resnet50.layer2(testBlock1)
        filePath = os.path.join(base_directory, directory, f"resnet50_block1_test{i}.bin")
        with open(filePath, "wb") as f:
            writeTensor(f, testBlock1)
            writeTensor(f, output)

        testBlock2 = torch.rand(1, 512, 28, 28)
        output: torch.Tensor = resnet50.layer3(testBlock2)
        filePath = os.path.join(base_directory, directory, f"resnet50_block2_test{i}.bin")
        with open(filePath, "wb") as f:
            writeTensor(f, testBlock2)
            writeTensor(f, output)

        testBlock3 = torch.rand(1, 1024, 14, 14)
        output: torch.Tensor = resnet50.layer4(testBlock3)
        filePath = os.path.join(base_directory, directory, f"resnet50_block3_test{i}.bin")
        with open(filePath, "wb") as f:
            writeTensor(f, testBlock3)
            writeTensor(f, output)

        testBatchNorm = torch.rand(1, 64, 1, 1)
        output = resnet50.bn1(testBatchNorm)
        filePath = os.path.join(base_directory, directory, f"resnet50_batchNorm1_test{i}.bin")
        with open(filePath, "wb") as f:
            writeTensor(f, testBatchNorm)
            writeTensor(f, output)

    testImage = torch.ones(1, 3, 224, 224)
    output: torch.Tensor = resnet50(testImage)
    filePath = os.path.join(base_directory, directory, "resnet50_test_ones.bin")
    with open(filePath, "wb") as f:
        writeTensor(f, testImage)
        writeTensor(f, output)

    testBlock0 = torch.ones(1, 64, 56, 56)
    output: torch.Tensor = resnet50.layer1(testBlock0)
    filePath = os.path.join(base_directory, directory, "resnet50_block0_test_ones.bin")
    with open(filePath, "wb") as f:
        writeTensor(f, testBlock0)
        writeTensor(f, output)

    testBlock1 = torch.ones(1, 256, 56, 56)
    output: torch.Tensor = resnet50.layer2(testBlock1)
    filePath = os.path.join(base_directory, directory, "resnet50_block1_test_ones.bin")
    with open(filePath, "wb") as f:
        writeTensor(f, testBlock1)
        writeTensor(f, output)

    testBlock2 = torch.ones(1, 512, 28, 28)
    output: torch.Tensor = resnet50.layer3(testBlock2)
    filePath = os.path.join(base_directory, directory, "resnet50_block2_test_ones.bin")
    with open(filePath, "wb") as f:
        writeTensor(f, testBlock2)
        writeTensor(f, output)

    testBlock3 = torch.ones(1, 1024, 14, 14)
    output: torch.Tensor = resnet50.layer4(testBlock3)
    filePath = os.path.join(base_directory, directory, "resnet50_block3_test_ones.bin")
    with open(filePath, "wb") as f:
        writeTensor(f, testBlock3)
        writeTensor(f, output)

    testconv7x7 = torch.ones(1, 3, 224, 224)
    output = resnet50.conv1(testconv7x7)
    output = resnet50.bn1(output)
    output = resnet50.relu(output)
    filePath = os.path.join(base_directory, directory, "resnet50_conv7x7_ones.bin")
    with open(filePath, "wb") as f:
        writeTensor(f, testconv7x7)
        writeTensor(f, output)

    testBatchNorm = torch.ones(1, 64, 1, 1)
    output = resnet50.bn1(testBatchNorm)
    filePath = os.path.join(base_directory, directory, "resnet50_batchNorm1_ones.bin")
    with open(filePath, "wb") as f:
        writeTensor(f, testBatchNorm)
        writeTensor(f, output)
