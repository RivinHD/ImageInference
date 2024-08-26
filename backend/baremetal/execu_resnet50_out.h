// SPDX-FileCopyrightText: © 2024 Vincent Gerlach
//
// SPDX-License-Identifier: MIT

#ifndef IMAGEINFERENCE_EXECU_RESNET50_OUT_H
#define IMAGEINFERENCE_EXECU_RESNET50_OUT_H

#include <executorch/runtime/kernel/kernel_includes.h>

#ifdef USE_ATEN_LIB
#define MAKE_ARRAY_REF(...) __VA_ARGS__
#define ARRAY_REF IntArrayRef
#endif // !USE_ATEN_LIB
#ifndef USE_ATEN_LIB
#define MAKE_ARRAY_REF(...) torch::executor::makeArrayRef(__VA_ARGS__)
#define ARRAY_REF torch::executor::ArrayRef<torch::executor::TensorImpl::SizesType>
#endif // USE_ATEN_LIB

#include "model/ResNet50.h"
#include <string>

namespace custom
{
    namespace native
    {
        using exec_aten::IntArrayRef;
        using exec_aten::optional;
        using exec_aten::Tensor;
        using exec_aten::TensorList;
        using ImageInference::model::ResNet50;
        using torch::executor::RuntimeContext;

        Tensor &resnet50_out_impl(const Tensor &in, const Tensor &weights, Tensor &out);

        Tensor &resnet50_out_impl(RuntimeContext &ctx, const Tensor &in, const Tensor &weights, Tensor &out);
    } // namespace native
} // namespace custom

#endif // IMAGEINFERENCE_EXECU_RESNET50_OUT_H