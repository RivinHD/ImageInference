// Copyright (c) 2024 by Vincent Gerlach. All rights reserved.
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// in the root folder of this project with the name LICENSE. If not, see <http://www.gnu.org/licenses/>.

#include <torch/library.h>
#include "execu_resnet50_out.cpp"
#include <make_aten_functor_from_et_functor.h>

namespace custom
{
    namespace native
    {
        using at::Tensor;
        using c10::ScalarType;

        Tensor resnet50_impl(const Tensor &in, const Tensor &weights)
        {
            at::Tensor out = at::zeros({1000});

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
