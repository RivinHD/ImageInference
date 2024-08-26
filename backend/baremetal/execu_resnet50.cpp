// SPDX-FileCopyrightText: © 2024 Vincent Gerlach
//
// SPDX-License-Identifier: MIT

#ifndef USE_ATEN_LIB
#define USE_ATEN_LIB
#endif // !USE_ATEN_LIB
#include <ATen/ATen.h>
#include <torch/library.h>
#include "execu_resnet50_out.h"

namespace custom
{
    namespace native
    {
        Tensor resnet50_impl(const Tensor &in, const Tensor &weights)
        {
            Tensor out = at::zeros({1, 1000});
            resnet50_out_impl(in, weights, out);
            return out;
        }

        // standard API to register ops into PyTorch
        TORCH_LIBRARY_FRAGMENT(baremetal_ops, m)
        {
            m.def("baremetal_ops::resnet50(Tensor input, Tensor weights) -> Tensor");
        }

        TORCH_LIBRARY_IMPL(baremetal_ops, CompositeExplicitAutograd, m)
        {
            m.impl("resnet50", TORCH_FN(resnet50_impl));
        }
    } // namespace native
} // namespace custom
