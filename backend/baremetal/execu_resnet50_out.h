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

#define USE_ATEN_LIB
#include <executorch/runtime/kernel/kernel_includes.h>
#include <model/ResNet50.h>
#include <string>

namespace custom
{
    namespace native
    {
        using exec_aten::optional;
        using exec_aten::ScalarType;
        using exec_aten::IntArrayRef;
        using exec_aten::Tensor;
        using exec_aten::TensorList;
        using ImageInference::model::ResNet50;
        using torch::executor::RuntimeContext;

        namespace
        {
            std::string to_string(IntArrayRef shape);
            void check_preconditions(const Tensor &in, const TensorList &weights, Tensor &out);
            void check_weights(const TensorList &weights);
        } // namespace

        Tensor &resnet50_out_impl(const Tensor &in, const TensorList &weights, Tensor &out);

        Tensor &resnet50_out_impl(RuntimeContext &ctx, const Tensor &in, const TensorList &weights, Tensor &out);
    } // namespace native
} // namespace custom