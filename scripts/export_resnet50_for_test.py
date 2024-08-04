
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import os
import numpy as np
import io


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

    torch.manual_seed(123)
    resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()

    base_directory = os.path.dirname(os.path.dirname(__file__))
    directory = "test_data"
    filePath = os.path.join(base_directory, directory, "resnet50_weights_v2.bin")
    os.makedirs(os.path.dirname(filePath), exist_ok=True)
    with open(filePath, "wb") as f:
        for name, param in resnet50.named_parameters():
            writeTensor(f, param)

    for i in range(10):
        testImage = torch.rand(1, 3, 224, 224)
        output: torch.Tensor = resnet50(testImage)
        filePath = os.path.join(base_directory, directory, f"resnet50_test{i}.bin")
        with open(filePath, "wb") as f:
            writeTensor(f, testImage)
            writeTensor(f, output)
