// SPDX-FileCopyrightText: © 2024 Vincent Gerlach
//
// SPDX-License-Identifier: MIT
#include <executorch/runtime/kernel/kernel_includes.h>

#include "execu_resnet50_out.h"
#include <sstream>
#include <iostream>

namespace custom
{
    namespace native
    {
        constexpr const size_t weightsCount = 267;
        constexpr const size_t sizes[weightsCount] = {
            (64 * 3 * 7 * 7),
            (64),
            (64),
            (64 * 64 * 1 * 1),
            (64),
            (64),
            (64 * 64 * 3 * 3),
            (64),
            (64),
            (256 * 64 * 1 * 1),
            (256),
            (256),
            (256 * 64 * 1 * 1),
            (256),
            (256),
            (64 * 256 * 1 * 1),
            (64),
            (64),
            (64 * 64 * 3 * 3),
            (64),
            (64),
            (256 * 64 * 1 * 1),
            (256),
            (256),
            (64 * 256 * 1 * 1),
            (64),
            (64),
            (64 * 64 * 3 * 3),
            (64),
            (64),
            (256 * 64 * 1 * 1),
            (256),
            (256),
            (128 * 256 * 1 * 1),
            (128),
            (128),
            (128 * 128 * 3 * 3),
            (128),
            (128),
            (512 * 128 * 1 * 1),
            (512),
            (512),
            (512 * 256 * 1 * 1),
            (512),
            (512),
            (128 * 512 * 1 * 1),
            (128),
            (128),
            (128 * 128 * 3 * 3),
            (128),
            (128),
            (512 * 128 * 1 * 1),
            (512),
            (512),
            (128 * 512 * 1 * 1),
            (128),
            (128),
            (128 * 128 * 3 * 3),
            (128),
            (128),
            (512 * 128 * 1 * 1),
            (512),
            (512),
            (128 * 512 * 1 * 1),
            (128),
            (128),
            (128 * 128 * 3 * 3),
            (128),
            (128),
            (512 * 128 * 1 * 1),
            (512),
            (512),
            (256 * 512 * 1 * 1),
            (256),
            (256),
            (256 * 256 * 3 * 3),
            (256),
            (256),
            (1024 * 256 * 1 * 1),
            (1024),
            (1024),
            (1024 * 512 * 1 * 1),
            (1024),
            (1024),
            (256 * 1024 * 1 * 1),
            (256),
            (256),
            (256 * 256 * 3 * 3),
            (256),
            (256),
            (1024 * 256 * 1 * 1),
            (1024),
            (1024),
            (256 * 1024 * 1 * 1),
            (256),
            (256),
            (256 * 256 * 3 * 3),
            (256),
            (256),
            (1024 * 256 * 1 * 1),
            (1024),
            (1024),
            (256 * 1024 * 1 * 1),
            (256),
            (256),
            (256 * 256 * 3 * 3),
            (256),
            (256),
            (1024 * 256 * 1 * 1),
            (1024),
            (1024),
            (256 * 1024 * 1 * 1),
            (256),
            (256),
            (256 * 256 * 3 * 3),
            (256),
            (256),
            (1024 * 256 * 1 * 1),
            (1024),
            (1024),
            (256 * 1024 * 1 * 1),
            (256),
            (256),
            (256 * 256 * 3 * 3),
            (256),
            (256),
            (1024 * 256 * 1 * 1),
            (1024),
            (1024),
            (512 * 1024 * 1 * 1),
            (512),
            (512),
            (512 * 512 * 3 * 3),
            (512),
            (512),
            (2048 * 512 * 1 * 1),
            (2048),
            (2048),
            (2048 * 1024 * 1 * 1),
            (2048),
            (2048),
            (512 * 2048 * 1 * 1),
            (512),
            (512),
            (512 * 512 * 3 * 3),
            (512),
            (512),
            (2048 * 512 * 1 * 1),
            (2048),
            (2048),
            (512 * 2048 * 1 * 1),
            (512),
            (512),
            (512 * 512 * 3 * 3),
            (512),
            (512),
            (2048 * 512 * 1 * 1),
            (2048),
            (2048),
            (1000 * 2048),
            (1000),
            (64),
            (64),
            (64),
            (64),
            (64),
            (64),
            (256),
            (256),
            (256),
            (256),
            (64),
            (64),
            (64),
            (64),
            (256),
            (256),
            (64),
            (64),
            (64),
            (64),
            (256),
            (256),
            (128),
            (128),
            (128),
            (128),
            (512),
            (512),
            (512),
            (512),
            (128),
            (128),
            (128),
            (128),
            (512),
            (512),
            (128),
            (128),
            (128),
            (128),
            (512),
            (512),
            (128),
            (128),
            (128),
            (128),
            (512),
            (512),
            (256),
            (256),
            (256),
            (256),
            (1024),
            (1024),
            (1024),
            (1024),
            (256),
            (256),
            (256),
            (256),
            (1024),
            (1024),
            (256),
            (256),
            (256),
            (256),
            (1024),
            (1024),
            (256),
            (256),
            (256),
            (256),
            (1024),
            (1024),
            (256),
            (256),
            (256),
            (256),
            (1024),
            (1024),
            (256),
            (256),
            (256),
            (256),
            (1024),
            (1024),
            (512),
            (512),
            (512),
            (512),
            (2048),
            (2048),
            (2048),
            (2048),
            (512),
            (512),
            (512),
            (512),
            (2048),
            (2048),
            (512),
            (512),
            (512),
            (512),
            (2048),
            (2048),
        };

        namespace
        {
            void check_preconditions(const Tensor &in, const Tensor &weights, Tensor &out)
            {
                // Type checks
                ET_CHECK_MSG(
                    out.scalar_type() == exec_aten::ScalarType::Float,
                    "Expected out tensor to have dtype Float, but got %hhd instead",
                    out.scalar_type());
                ET_CHECK_MSG(
                    weights.scalar_type() == exec_aten::ScalarType::Float, // Float
                    "Expected weights tensor to have dtype Float, but got %hhd instead",
                    weights.scalar_type());
                ET_CHECK_MSG(
                    in.scalar_type() == exec_aten::ScalarType::Float,
                    "Expected in tensor to have dtype Float, but got %hhd instead",
                    in.scalar_type());

                // Check Input Shape
                ET_CHECK_MSG(
                    in.dim() == 4,
                    "Exepcted input tensor to have 4 dimensions (Batch, Channels, Height, Width), but got %d instead",
                    in.dim());
                ET_CHECK_MSG(
                    in.size(1) == 3,
                    "Expected input tensor to have 3 channels for Red, Green, Blue, but got %d instead",
                    in.size(1));
                ET_CHECK_MSG(
                    in.size(2) == 224,
                    "Expected input tensor to have 224 height, but got %d instead",
                    in.size(2));
                ET_CHECK_MSG(
                    in.size(3) == 224,
                    "Expected input tensor to have 224 width, but got %d instead",
                    in.size(3));

                ET_CHECK_MSG(
                    weights.dim() == 1,
                    "Expected weights tensor to have 1 dimension (CompressedWeights), but got %d instead",
                    weights.dim());
                ET_CHECK_MSG(
                    weights.size(0) == 25610152,
                    "Expected weights tensor to have 25610152 elements, but got %d instead",
                    weights.size(0));


                // Check Output Shape
                ET_CHECK_MSG(
                    out.dim() == 2,
                    "Expected out tensor to have 1 dimension (Batch, Classes), but got %d instead",
                    out.dim());
                ET_CHECK_MSG(
                    out.size(1) == 1000,
                    "Expected out tensor to have 1000 classes, but got %d instead",
                    out.size(1));
            }

            template <typename T>
            static void expandToTensorList(const Tensor &tensor, std::vector<void *> &out)
            {
                auto ptr = tensor.mutable_data_ptr<T>();
                for (size_t i = 0; i < weightsCount; i++)
                {
                    out[i] = ptr;
                    ptr += sizes[i];
                }
            }
        } // namespace

        Tensor &resnet50_out_impl(const Tensor &in, const Tensor &weights, Tensor &out)
        {
            std::vector<void *> raw_weights = std::vector<void *>(weightsCount);

            ImageInference::types::ScalarType type = ImageInference::types::ScalarType::Undefined;
            switch (weights.scalar_type())
            {
            case exec_aten::ScalarType::Float:
                type = ImageInference::types::ScalarType::Float;
                expandToTensorList<float>(weights, raw_weights);
                break;
            default:
                ET_CHECK_MSG(false, "Unsupported scalar type");
                break;
            }

            check_preconditions(in, weights, out);

            ResNet50 resnet50 = ResNet50(raw_weights, type);

            if (resnet50.getType() == ImageInference::types::ScalarType::Float)
            {
                // Float
                float *out_data = out.mutable_data_ptr<float>();
                const float *in_data = in.const_data_ptr<float>();

                resnet50.inference(in_data, out_data);
            }

            return out;
        }

        Tensor &resnet50_out_impl(RuntimeContext &ctx, const Tensor &in, const Tensor &weights, Tensor &out)
        {
            (void)ctx;
            resnet50_out_impl(in, weights, out);
            return out;
        }
    } // namespace native
} // namespace custom