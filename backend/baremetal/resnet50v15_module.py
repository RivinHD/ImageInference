# SPDX-FileCopyrightText: © 2024 Vincent Gerlach
#
# SPDX-License-Identifier: MIT

import torch
from torch import Tensor
from typing import Dict, Optional
from torch.nn.parameter import Parameter


class custom_resnet50(torch.nn.Module):

    weight: Tensor

    def __init__(self, compressedParameters: Dict[str, Optional[Parameter]]):
        super(custom_resnet50, self).__init__()

        self._parameters = compressedParameters
        self.weight = self.get_parameter("weight")

    def forward(self, a):
        return torch.ops.baremetal_ops.resnet50.default(a, self.weight)
