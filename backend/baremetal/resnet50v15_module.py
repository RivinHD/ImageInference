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

    def _forward_impl(self, x: Tensor) -> Tensor:
        return torch.ops.baremetal_ops.resnet50.default(x, self.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
