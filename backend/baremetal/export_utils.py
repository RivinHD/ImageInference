import torch
from torchvision import models
from torchvision.models._api import WeightsEnum
from typing import Dict, Optional
from torch.nn.parameter import Parameter


def getResnet50Weights(weights: WeightsEnum) -> Dict[str, Optional[Parameter]]:
    def getMeanAndVar(bn, name) -> dict[str, Optional[Parameter]]:
        return {f"{name}.running_mean": Parameter(bn.running_mean), f"{name}.running_var": Parameter(bn.running_var)}

    def getBottleneckMeanAndVar(layer, name) -> dict[str, Optional[Parameter]]:
        data = {}
        data.update(getMeanAndVar(layer.bn1, f"{name}.bn1"))
        data.update(getMeanAndVar(layer.bn2, f"{name}.bn2"))
        data.update(getMeanAndVar(layer.bn3, f"{name}.bn3"))
        if layer.downsample is not None:
            data.update(getMeanAndVar(layer.downsample[1], f"{name}.downsample.1"))
        return data

    resnet50 = models.resnet50(weights=weights).eval()
    parameters = {name: param for name, param in resnet50.named_parameters()}

    # Adding missing running_mean and running_var parameters
    parameters.update(getMeanAndVar(resnet50.bn1, "bn1"))

    parameters.update(getBottleneckMeanAndVar(resnet50.layer1[0], "layer1.0"))
    parameters.update(getBottleneckMeanAndVar(resnet50.layer1[1], "layer1.1"))
    parameters.update(getBottleneckMeanAndVar(resnet50.layer1[2], "layer1.2"))

    parameters.update(getBottleneckMeanAndVar(resnet50.layer2[0], "layer2.0"))
    parameters.update(getBottleneckMeanAndVar(resnet50.layer2[1], "layer2.1"))
    parameters.update(getBottleneckMeanAndVar(resnet50.layer2[2], "layer2.2"))
    parameters.update(getBottleneckMeanAndVar(resnet50.layer2[3], "layer2.3"))

    parameters.update(getBottleneckMeanAndVar(resnet50.layer3[0], "layer3.0"))
    parameters.update(getBottleneckMeanAndVar(resnet50.layer3[1], "layer3.1"))
    parameters.update(getBottleneckMeanAndVar(resnet50.layer3[2], "layer3.2"))
    parameters.update(getBottleneckMeanAndVar(resnet50.layer3[3], "layer3.3"))
    parameters.update(getBottleneckMeanAndVar(resnet50.layer3[4], "layer3.4"))
    parameters.update(getBottleneckMeanAndVar(resnet50.layer3[5], "layer3.5"))

    parameters.update(getBottleneckMeanAndVar(resnet50.layer4[0], "layer4.0"))
    parameters.update(getBottleneckMeanAndVar(resnet50.layer4[1], "layer4.1"))
    parameters.update(getBottleneckMeanAndVar(resnet50.layer4[2], "layer4.2"))

    return parameters


def compressParameters(parameters: Dict[str, Optional[Parameter]]) -> Dict[str, Optional[Parameter]]:
    """
    Compress all parameters into a single parameter with the key 'weight'

    Args:
        parameters (Dict[str, Optional[Parameter]]): The parameters to compress.

    Returns:
        Dict[str, Optional[Parameter]]: The compressed parameters with key 'weight'.
    """

    total_length = sum(param.numel() for param in parameters.values())
    weightCompressed = torch.empty(total_length, dtype=torch.float32)
    offset = 0
    for param in parameters.values():
        numel = param.numel()
        weightCompressed[offset: offset + numel] = param.data.view([param.numel()])
        offset += numel

    return {"weight": Parameter(weightCompressed)}
