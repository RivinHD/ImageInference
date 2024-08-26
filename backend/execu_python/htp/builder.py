# SPDX-FileCopyrightText: © 2024 Vincent Gerlach
#
# SPDX-License-Identifier: MIT

from enum import IntEnum, unique
from typing import Any, Dict
from executorch.backends.qualcomm.quantizer.quantizer import (
    get_16a4w_qnn_ptq_config,
    get_default_16bit_qnn_ptq_config,
    QnnQuantizer
)
import torch
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.fx.graph_module import GraphModule
from torch.ao.quantization.observer import (
    MinMaxObserver,
    MovingAverageMinMaxObserver
)
from torch.ao.quantization.quantizer import (
    QuantizationSpec,
)

from submodules.executorch.backends.arm.quantizer.quantization_config import QuantizationConfig


@unique
class ExtendedQuantDtype(IntEnum):
    """
    bits of activation and bits of weight
    """

    use_16a16w = 0
    use_16a4w = 1
    use_8a8w = 2
    use_8a4w = 3


def quantize(model, dataset, type: ExtendedQuantDtype) -> GraphModule:
    quantizer = QnnQuantizer()
    if type == ExtendedQuantDtype.use_16a4w:
        quantizer.add_16bit_quant_ops(quantizer.SUPPORTED_OPS)
        quantizer.set_bit16_op_quant_config(get_16a4w_qnn_ptq_config())
        quantizer.set_per_channel_weight_dtype(weight_dtype_for_16bit_act="int4")
    elif type == ExtendedQuantDtype.use_16a16w:
        quantizer.add_16bit_quant_ops(quantizer.SUPPORTED_OPS)
        quantizer.set_bit16_op_quant_config(get_default_16bit_qnn_ptq_config())
    elif type == ExtendedQuantDtype.use_8a4w:
        quantizer.set_bit8_op_quant_config(get_8a4w_qnn_ptq_config())
        quantizer.set_per_channel_weight_dtype(weight_dtype_for_8bit_act="int4")

    m = prepare_pt2e(model, quantizer)
    # calibration
    for data in dataset:
        m(data)
    m = convert_pt2e(m)
    return m


def get_8a4w_qnn_ptq_config() -> QuantizationConfig:
    # 4 bits quantization only supports specific ops.
    extra_args: Dict[str, Any] = {"eps": 2**-20}
    act_quantization_spec = QuantizationSpec(
        dtype=torch.uint8,
        quant_min=0,
        quant_max=torch.iinfo(torch.uint8).max,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=MovingAverageMinMaxObserver.with_args(**extra_args),
    )

    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-7,
        quant_max=7,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    bias_quantization_spec = QuantizationSpec(
        dtype=torch.int32,
        quant_min=torch.iinfo(torch.int32).min,
        quant_max=torch.iinfo(torch.int32).max,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=MinMaxObserver.with_args(**extra_args),
    )

    quantization_config = QuantizationConfig(
        input_activation=act_quantization_spec,
        output_activation=act_quantization_spec,
        weight=weight_quantization_spec,
        bias=bias_quantization_spec,
    )

    return quantization_config
