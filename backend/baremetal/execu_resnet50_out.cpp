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

#include "execu_resnet50_out.h"
#include <sstream>

namespace custom
{
    namespace native
    {
        namespace
        {
            std::string to_string(IntArrayRef shape)
            {
                std::stringstream ss;
                ss << shape;
                return ss.str();
            }

            void check_preconditions(const Tensor &in, const TensorList &weights, Tensor &out)
            {
                // Type checks
                ET_CHECK_MSG(
                    out.scalar_type() == ScalarType::Float,
                    "Expected out tensor to have dtype Float, but got %hhd instead",
                    out.scalar_type());
                ET_CHECK_MSG(
                    weights[0].scalar_type() == ScalarType::Float || weights[0].scalar_type() == ScalarType::Char, // Float or Int8
                    "Expected weights tensor to have dtype Float or Byte (Int8), but got %hhd instead",
                    weights[0].scalar_type());
                ET_CHECK_MSG(
                    in.scalar_type() == ScalarType::Float,
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

                // Check Output Shape
                ET_CHECK_MSG(
                    out.dim() == 1,
                    "Expected out tensor to have 1 dimension (Classes), but got %d instead",
                    out.dim());
                ET_CHECK_MSG(
                    out.size(0) == 1000,
                    "Expected out tensor to have 1000 classes, but got %d instead",
                    out.size(0));

                check_weights(weights);
            }

            void check_weights(const TensorList &weights)
            {
                ET_CHECK_MSG(
                    weights[ResNet50::conv1_weight].sizes().equals({64, 3, 7, 7}),
                    "Expected weight at index %d to be of shape [64, 3, 7, 7], but got %s",
                    ResNet50::conv1_weight,
                    to_string(weights[ResNet50::conv1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::bn1_weight].sizes().equals({64}),
                    "Expected weight at index %d to be of shape [64], but got %s",
                    ResNet50::bn1_weight,
                    to_string(weights[ResNet50::bn1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::bn1_bias].sizes().equals({64}),
                    "Expected weight at index %d to be of shape [64], but got %s",
                    ResNet50::bn1_bias,
                    to_string(weights[ResNet50::bn1_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_0_conv1_weight].sizes().equals({64, 64, 1, 1}),
                    "Expected weight at index %d to be of shape [64, 64, 1, 1], but got %s",
                    ResNet50::layer1_0_conv1_weight,
                    to_string(weights[ResNet50::layer1_0_conv1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_0_bn1_weight].sizes().equals({64}),
                    "Expected weight at index %d to be of shape [64], but got %s",
                    ResNet50::layer1_0_bn1_weight,
                    to_string(weights[ResNet50::layer1_0_bn1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_0_bn1_bias].sizes().equals({64}),
                    "Expected weight at index %d to be of shape [64], but got %s",
                    ResNet50::layer1_0_bn1_bias,
                    to_string(weights[ResNet50::layer1_0_bn1_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_0_conv2_weight].sizes().equals({64, 64, 3, 3}),
                    "Expected weight at index %d to be of shape [64, 64, 3, 3], but got %s",
                    ResNet50::layer1_0_conv2_weight,
                    to_string(weights[ResNet50::layer1_0_conv2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_0_bn2_weight].sizes().equals({64}),
                    "Expected weight at index %d to be of shape [64], but got %s",
                    ResNet50::layer1_0_bn2_weight,
                    to_string(weights[ResNet50::layer1_0_bn2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_0_bn2_bias].sizes().equals({64}),
                    "Expected weight at index %d to be of shape [64], but got %s",
                    ResNet50::layer1_0_bn2_bias,
                    to_string(weights[ResNet50::layer1_0_bn2_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_0_conv3_weight].sizes().equals({256, 64, 1, 1}),
                    "Expected weight at index %d to be of shape [256, 64, 1, 1], but got %s",
                    ResNet50::layer1_0_conv3_weight,
                    to_string(weights[ResNet50::layer1_0_conv3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_0_bn3_weight].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer1_0_bn3_weight,
                    to_string(weights[ResNet50::layer1_0_bn3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_0_bn3_bias].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer1_0_bn3_bias,
                    to_string(weights[ResNet50::layer1_0_bn3_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_0_downsample_0_weight].sizes().equals({256, 64, 1, 1}),
                    "Expected weight at index %d to be of shape [256, 64, 1, 1], but got %s",
                    ResNet50::layer1_0_downsample_0_weight,
                    to_string(weights[ResNet50::layer1_0_downsample_0_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_0_downsample_1_weight].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer1_0_downsample_1_weight,
                    to_string(weights[ResNet50::layer1_0_downsample_1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_0_downsample_1_bias].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer1_0_downsample_1_bias,
                    to_string(weights[ResNet50::layer1_0_downsample_1_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_1_conv1_weight].sizes().equals({64, 256, 1, 1}),
                    "Expected weight at index %d to be of shape [64, 256, 1, 1], but got %s",
                    ResNet50::layer1_1_conv1_weight,
                    to_string(weights[ResNet50::layer1_1_conv1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_1_bn1_weight].sizes().equals({64}),
                    "Expected weight at index %d to be of shape [64], but got %s",
                    ResNet50::layer1_1_bn1_weight,
                    to_string(weights[ResNet50::layer1_1_bn1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_1_bn1_bias].sizes().equals({64}),
                    "Expected weight at index %d to be of shape [64], but got %s",
                    ResNet50::layer1_1_bn1_bias,
                    to_string(weights[ResNet50::layer1_1_bn1_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_1_conv2_weight].sizes().equals({64, 64, 3, 3}),
                    "Expected weight at index %d to be of shape [64, 64, 3, 3], but got %s",
                    ResNet50::layer1_1_conv2_weight,
                    to_string(weights[ResNet50::layer1_1_conv2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_1_bn2_weight].sizes().equals({64}),
                    "Expected weight at index %d to be of shape [64], but got %s",
                    ResNet50::layer1_1_bn2_weight,
                    to_string(weights[ResNet50::layer1_1_bn2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_1_bn2_bias].sizes().equals({64}),
                    "Expected weight at index %d to be of shape [64], but got %s",
                    ResNet50::layer1_1_bn2_bias,
                    to_string(weights[ResNet50::layer1_1_bn2_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_1_conv3_weight].sizes().equals({256, 64, 1, 1}),
                    "Expected weight at index %d to be of shape [256, 64, 1, 1], but got %s",
                    ResNet50::layer1_1_conv3_weight,
                    to_string(weights[ResNet50::layer1_1_conv3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_1_bn3_weight].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer1_1_bn3_weight,
                    to_string(weights[ResNet50::layer1_1_bn3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_1_bn3_bias].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer1_1_bn3_bias,
                    to_string(weights[ResNet50::layer1_1_bn3_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_2_conv1_weight].sizes().equals({64, 256, 1, 1}),
                    "Expected weight at index %d to be of shape [64, 256, 1, 1], but got %s",
                    ResNet50::layer1_2_conv1_weight,
                    to_string(weights[ResNet50::layer1_2_conv1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_2_bn1_weight].sizes().equals({64}),
                    "Expected weight at index %d to be of shape [64], but got %s",
                    ResNet50::layer1_2_bn1_weight,
                    to_string(weights[ResNet50::layer1_2_bn1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_2_bn1_bias].sizes().equals({64}),
                    "Expected weight at index %d to be of shape [64], but got %s",
                    ResNet50::layer1_2_bn1_bias,
                    to_string(weights[ResNet50::layer1_2_bn1_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_2_conv2_weight].sizes().equals({64, 64, 3, 3}),
                    "Expected weight at index %d to be of shape [64, 64, 3, 3], but got %s",
                    ResNet50::layer1_2_conv2_weight,
                    to_string(weights[ResNet50::layer1_2_conv2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_2_bn2_weight].sizes().equals({64}),
                    "Expected weight at index %d to be of shape [64], but got %s",
                    ResNet50::layer1_2_bn2_weight,
                    to_string(weights[ResNet50::layer1_2_bn2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_2_bn2_bias].sizes().equals({64}),
                    "Expected weight at index %d to be of shape [64], but got %s",
                    ResNet50::layer1_2_bn2_bias,
                    to_string(weights[ResNet50::layer1_2_bn2_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_2_conv3_weight].sizes().equals({256, 64, 1, 1}),
                    "Expected weight at index %d to be of shape [256, 64, 1, 1], but got %s",
                    ResNet50::layer1_2_conv3_weight,
                    to_string(weights[ResNet50::layer1_2_conv3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_2_bn3_weight].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer1_2_bn3_weight,
                    to_string(weights[ResNet50::layer1_2_bn3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer1_2_bn3_bias].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer1_2_bn3_bias,
                    to_string(weights[ResNet50::layer1_2_bn3_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_0_conv1_weight].sizes().equals({128, 256, 1, 1}),
                    "Expected weight at index %d to be of shape [128, 256, 1, 1], but got %s",
                    ResNet50::layer2_0_conv1_weight,
                    to_string(weights[ResNet50::layer2_0_conv1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_0_bn1_weight].sizes().equals({128}),
                    "Expected weight at index %d to be of shape [128], but got %s",
                    ResNet50::layer2_0_bn1_weight,
                    to_string(weights[ResNet50::layer2_0_bn1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_0_bn1_bias].sizes().equals({128}),
                    "Expected weight at index %d to be of shape [128], but got %s",
                    ResNet50::layer2_0_bn1_bias,
                    to_string(weights[ResNet50::layer2_0_bn1_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_0_conv2_weight].sizes().equals({128, 128, 3, 3}),
                    "Expected weight at index %d to be of shape [128, 128, 3, 3], but got %s",
                    ResNet50::layer2_0_conv2_weight,
                    to_string(weights[ResNet50::layer2_0_conv2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_0_bn2_weight].sizes().equals({128}),
                    "Expected weight at index %d to be of shape [128], but got %s",
                    ResNet50::layer2_0_bn2_weight,
                    to_string(weights[ResNet50::layer2_0_bn2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_0_bn2_bias].sizes().equals({128}),
                    "Expected weight at index %d to be of shape [128], but got %s",
                    ResNet50::layer2_0_bn2_bias,
                    to_string(weights[ResNet50::layer2_0_bn2_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_0_conv3_weight].sizes().equals({512, 128, 1, 1}),
                    "Expected weight at index %d to be of shape [512, 128, 1, 1], but got %s",
                    ResNet50::layer2_0_conv3_weight,
                    to_string(weights[ResNet50::layer2_0_conv3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_0_bn3_weight].sizes().equals({512}),
                    "Expected weight at index %d to be of shape [512], but got %s",
                    ResNet50::layer2_0_bn3_weight,
                    to_string(weights[ResNet50::layer2_0_bn3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_0_bn3_bias].sizes().equals({512}),
                    "Expected weight at index %d to be of shape [512], but got %s",
                    ResNet50::layer2_0_bn3_bias,
                    to_string(weights[ResNet50::layer2_0_bn3_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_0_downsample_0_weight].sizes().equals({512, 256, 1, 1}),
                    "Expected weight at index %d to be of shape [512, 256, 1, 1], but got %s",
                    ResNet50::layer2_0_downsample_0_weight,
                    to_string(weights[ResNet50::layer2_0_downsample_0_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_0_downsample_1_weight].sizes().equals({512}),
                    "Expected weight at index %d to be of shape [512], but got %s",
                    ResNet50::layer2_0_downsample_1_weight,
                    to_string(weights[ResNet50::layer2_0_downsample_1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_0_downsample_1_bias].sizes().equals({512}),
                    "Expected weight at index %d to be of shape [512], but got %s",
                    ResNet50::layer2_0_downsample_1_bias,
                    to_string(weights[ResNet50::layer2_0_downsample_1_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_1_conv1_weight].sizes().equals({128, 512, 1, 1}),
                    "Expected weight at index %d to be of shape [128, 512, 1, 1], but got %s",
                    ResNet50::layer2_1_conv1_weight,
                    to_string(weights[ResNet50::layer2_1_conv1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_1_bn1_weight].sizes().equals({128}),
                    "Expected weight at index %d to be of shape [128], but got %s",
                    ResNet50::layer2_1_bn1_weight,
                    to_string(weights[ResNet50::layer2_1_bn1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_1_bn1_bias].sizes().equals({128}),
                    "Expected weight at index %d to be of shape [128], but got %s",
                    ResNet50::layer2_1_bn1_bias,
                    to_string(weights[ResNet50::layer2_1_bn1_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_1_conv2_weight].sizes().equals({128, 128, 3, 3}),
                    "Expected weight at index %d to be of shape [128, 128, 3, 3], but got %s",
                    ResNet50::layer2_1_conv2_weight,
                    to_string(weights[ResNet50::layer2_1_conv2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_1_bn2_weight].sizes().equals({128}),
                    "Expected weight at index %d to be of shape [128], but got %s",
                    ResNet50::layer2_1_bn2_weight,
                    to_string(weights[ResNet50::layer2_1_bn2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_1_bn2_bias].sizes().equals({128}),
                    "Expected weight at index %d to be of shape [128], but got %s",
                    ResNet50::layer2_1_bn2_bias,
                    to_string(weights[ResNet50::layer2_1_bn2_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_1_conv3_weight].sizes().equals({512, 128, 1, 1}),
                    "Expected weight at index %d to be of shape [512, 128, 1, 1], but got %s",
                    ResNet50::layer2_1_conv3_weight,
                    to_string(weights[ResNet50::layer2_1_conv3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_1_bn3_weight].sizes().equals({512}),
                    "Expected weight at index %d to be of shape [512], but got %s",
                    ResNet50::layer2_1_bn3_weight,
                    to_string(weights[ResNet50::layer2_1_bn3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_1_bn3_bias].sizes().equals({512}),
                    "Expected weight at index %d to be of shape [512], but got %s",
                    ResNet50::layer2_1_bn3_bias,
                    to_string(weights[ResNet50::layer2_1_bn3_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_2_conv1_weight].sizes().equals({128, 512, 1, 1}),
                    "Expected weight at index %d to be of shape [128, 512, 1, 1], but got %s",
                    ResNet50::layer2_2_conv1_weight,
                    to_string(weights[ResNet50::layer2_2_conv1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_2_bn1_weight].sizes().equals({128}),
                    "Expected weight at index %d to be of shape [128], but got %s",
                    ResNet50::layer2_2_bn1_weight,
                    to_string(weights[ResNet50::layer2_2_bn1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_2_bn1_bias].sizes().equals({128}),
                    "Expected weight at index %d to be of shape [128], but got %s",
                    ResNet50::layer2_2_bn1_bias,
                    to_string(weights[ResNet50::layer2_2_bn1_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_2_conv2_weight].sizes().equals({128, 128, 3, 3}),
                    "Expected weight at index %d to be of shape [128, 128, 3, 3], but got %s",
                    ResNet50::layer2_2_conv2_weight,
                    to_string(weights[ResNet50::layer2_2_conv2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_2_bn2_weight].sizes().equals({128}),
                    "Expected weight at index %d to be of shape [128], but got %s",
                    ResNet50::layer2_2_bn2_weight,
                    to_string(weights[ResNet50::layer2_2_bn2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_2_bn2_bias].sizes().equals({128}),
                    "Expected weight at index %d to be of shape [128], but got %s",
                    ResNet50::layer2_2_bn2_bias,
                    to_string(weights[ResNet50::layer2_2_bn2_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_2_conv3_weight].sizes().equals({512, 128, 1, 1}),
                    "Expected weight at index %d to be of shape [512, 128, 1, 1], but got %s",
                    ResNet50::layer2_2_conv3_weight,
                    to_string(weights[ResNet50::layer2_2_conv3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_2_bn3_weight].sizes().equals({512}),
                    "Expected weight at index %d to be of shape [512], but got %s",
                    ResNet50::layer2_2_bn3_weight,
                    to_string(weights[ResNet50::layer2_2_bn3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_2_bn3_bias].sizes().equals({512}),
                    "Expected weight at index %d to be of shape [512], but got %s",
                    ResNet50::layer2_2_bn3_bias,
                    to_string(weights[ResNet50::layer2_2_bn3_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_3_conv1_weight].sizes().equals({128, 512, 1, 1}),
                    "Expected weight at index %d to be of shape [128, 512, 1, 1], but got %s",
                    ResNet50::layer2_3_conv1_weight,
                    to_string(weights[ResNet50::layer2_3_conv1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_3_bn1_weight].sizes().equals({128}),
                    "Expected weight at index %d to be of shape [128], but got %s",
                    ResNet50::layer2_3_bn1_weight,
                    to_string(weights[ResNet50::layer2_3_bn1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_3_bn1_bias].sizes().equals({128}),
                    "Expected weight at index %d to be of shape [128], but got %s",
                    ResNet50::layer2_3_bn1_bias,
                    to_string(weights[ResNet50::layer2_3_bn1_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_3_conv2_weight].sizes().equals({128, 128, 3, 3}),
                    "Expected weight at index %d to be of shape [128, 128, 3, 3], but got %s",
                    ResNet50::layer2_3_conv2_weight,
                    to_string(weights[ResNet50::layer2_3_conv2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_3_bn2_weight].sizes().equals({128}),
                    "Expected weight at index %d to be of shape [128], but got %s",
                    ResNet50::layer2_3_bn2_weight,
                    to_string(weights[ResNet50::layer2_3_bn2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_3_bn2_bias].sizes().equals({128}),
                    "Expected weight at index %d to be of shape [128], but got %s",
                    ResNet50::layer2_3_bn2_bias,
                    to_string(weights[ResNet50::layer2_3_bn2_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_3_conv3_weight].sizes().equals({512, 128, 1, 1}),
                    "Expected weight at index %d to be of shape [512, 128, 1, 1], but got %s",
                    ResNet50::layer2_3_conv3_weight,
                    to_string(weights[ResNet50::layer2_3_conv3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_3_bn3_weight].sizes().equals({512}),
                    "Expected weight at index %d to be of shape [512], but got %s",
                    ResNet50::layer2_3_bn3_weight,
                    to_string(weights[ResNet50::layer2_3_bn3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer2_3_bn3_bias].sizes().equals({512}),
                    "Expected weight at index %d to be of shape [512], but got %s",
                    ResNet50::layer2_3_bn3_bias,
                    to_string(weights[ResNet50::layer2_3_bn3_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_0_conv1_weight].sizes().equals({256, 512, 1, 1}),
                    "Expected weight at index %d to be of shape [256, 512, 1, 1], but got %s",
                    ResNet50::layer3_0_conv1_weight,
                    to_string(weights[ResNet50::layer3_0_conv1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_0_bn1_weight].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_0_bn1_weight,
                    to_string(weights[ResNet50::layer3_0_bn1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_0_bn1_bias].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_0_bn1_bias,
                    to_string(weights[ResNet50::layer3_0_bn1_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_0_conv2_weight].sizes().equals({256, 256, 3, 3}),
                    "Expected weight at index %d to be of shape [256, 256, 3, 3], but got %s",
                    ResNet50::layer3_0_conv2_weight,
                    to_string(weights[ResNet50::layer3_0_conv2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_0_bn2_weight].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_0_bn2_weight,
                    to_string(weights[ResNet50::layer3_0_bn2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_0_bn2_bias].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_0_bn2_bias,
                    to_string(weights[ResNet50::layer3_0_bn2_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_0_conv3_weight].sizes().equals({1024, 256, 1, 1}),
                    "Expected weight at index %d to be of shape [1024, 256, 1, 1], but got %s",
                    ResNet50::layer3_0_conv3_weight,
                    to_string(weights[ResNet50::layer3_0_conv3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_0_bn3_weight].sizes().equals({1024}),
                    "Expected weight at index %d to be of shape [1024], but got %s",
                    ResNet50::layer3_0_bn3_weight,
                    to_string(weights[ResNet50::layer3_0_bn3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_0_bn3_bias].sizes().equals({1024}),
                    "Expected weight at index %d to be of shape [1024], but got %s",
                    ResNet50::layer3_0_bn3_bias,
                    to_string(weights[ResNet50::layer3_0_bn3_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_0_downsample_0_weight].sizes().equals({1024, 512, 1, 1}),
                    "Expected weight at index %d to be of shape [1024, 512, 1, 1], but got %s",
                    ResNet50::layer3_0_downsample_0_weight,
                    to_string(weights[ResNet50::layer3_0_downsample_0_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_0_downsample_1_weight].sizes().equals({1024}),
                    "Expected weight at index %d to be of shape [1024], but got %s",
                    ResNet50::layer3_0_downsample_1_weight,
                    to_string(weights[ResNet50::layer3_0_downsample_1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_0_downsample_1_bias].sizes().equals({1024}),
                    "Expected weight at index %d to be of shape [1024], but got %s",
                    ResNet50::layer3_0_downsample_1_bias,
                    to_string(weights[ResNet50::layer3_0_downsample_1_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_1_conv1_weight].sizes().equals({256, 1024, 1, 1}),
                    "Expected weight at index %d to be of shape [256, 1024, 1, 1], but got %s",
                    ResNet50::layer3_1_conv1_weight,
                    to_string(weights[ResNet50::layer3_1_conv1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_1_bn1_weight].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_1_bn1_weight,
                    to_string(weights[ResNet50::layer3_1_bn1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_1_bn1_bias].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_1_bn1_bias,
                    to_string(weights[ResNet50::layer3_1_bn1_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_1_conv2_weight].sizes().equals({256, 256, 3, 3}),
                    "Expected weight at index %d to be of shape [256, 256, 3, 3], but got %s",
                    ResNet50::layer3_1_conv2_weight,
                    to_string(weights[ResNet50::layer3_1_conv2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_1_bn2_weight].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_1_bn2_weight,
                    to_string(weights[ResNet50::layer3_1_bn2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_1_bn2_bias].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_1_bn2_bias,
                    to_string(weights[ResNet50::layer3_1_bn2_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_1_conv3_weight].sizes().equals({1024, 256, 1, 1}),
                    "Expected weight at index %d to be of shape [1024, 256, 1, 1], but got %s",
                    ResNet50::layer3_1_conv3_weight,
                    to_string(weights[ResNet50::layer3_1_conv3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_1_bn3_weight].sizes().equals({1024}),
                    "Expected weight at index %d to be of shape [1024], but got %s",
                    ResNet50::layer3_1_bn3_weight,
                    to_string(weights[ResNet50::layer3_1_bn3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_1_bn3_bias].sizes().equals({1024}),
                    "Expected weight at index %d to be of shape [1024], but got %s",
                    ResNet50::layer3_1_bn3_bias,
                    to_string(weights[ResNet50::layer3_1_bn3_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_2_conv1_weight].sizes().equals({256, 1024, 1, 1}),
                    "Expected weight at index %d to be of shape [256, 1024, 1, 1], but got %s",
                    ResNet50::layer3_2_conv1_weight,
                    to_string(weights[ResNet50::layer3_2_conv1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_2_bn1_weight].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_2_bn1_weight,
                    to_string(weights[ResNet50::layer3_2_bn1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_2_bn1_bias].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_2_bn1_bias,
                    to_string(weights[ResNet50::layer3_2_bn1_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_2_conv2_weight].sizes().equals({256, 256, 3, 3}),
                    "Expected weight at index %d to be of shape [256, 256, 3, 3], but got %s",
                    ResNet50::layer3_2_conv2_weight,
                    to_string(weights[ResNet50::layer3_2_conv2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_2_bn2_weight].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_2_bn2_weight,
                    to_string(weights[ResNet50::layer3_2_bn2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_2_bn2_bias].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_2_bn2_bias,
                    to_string(weights[ResNet50::layer3_2_bn2_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_2_conv3_weight].sizes().equals({1024, 256, 1, 1}),
                    "Expected weight at index %d to be of shape [1024, 256, 1, 1], but got %s",
                    ResNet50::layer3_2_conv3_weight,
                    to_string(weights[ResNet50::layer3_2_conv3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_2_bn3_weight].sizes().equals({1024}),
                    "Expected weight at index %d to be of shape [1024], but got %s",
                    ResNet50::layer3_2_bn3_weight,
                    to_string(weights[ResNet50::layer3_2_bn3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_2_bn3_bias].sizes().equals({1024}),
                    "Expected weight at index %d to be of shape [1024], but got %s",
                    ResNet50::layer3_2_bn3_bias,
                    to_string(weights[ResNet50::layer3_2_bn3_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_3_conv1_weight].sizes().equals({256, 1024, 1, 1}),
                    "Expected weight at index %d to be of shape [256, 1024, 1, 1], but got %s",
                    ResNet50::layer3_3_conv1_weight,
                    to_string(weights[ResNet50::layer3_3_conv1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_3_bn1_weight].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_3_bn1_weight,
                    to_string(weights[ResNet50::layer3_3_bn1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_3_bn1_bias].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_3_bn1_bias,
                    to_string(weights[ResNet50::layer3_3_bn1_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_3_conv2_weight].sizes().equals({256, 256, 3, 3}),
                    "Expected weight at index %d to be of shape [256, 256, 3, 3], but got %s",
                    ResNet50::layer3_3_conv2_weight,
                    to_string(weights[ResNet50::layer3_3_conv2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_3_bn2_weight].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_3_bn2_weight,
                    to_string(weights[ResNet50::layer3_3_bn2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_3_bn2_bias].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_3_bn2_bias,
                    to_string(weights[ResNet50::layer3_3_bn2_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_3_conv3_weight].sizes().equals({1024, 256, 1, 1}),
                    "Expected weight at index %d to be of shape [1024, 256, 1, 1], but got %s",
                    ResNet50::layer3_3_conv3_weight,
                    to_string(weights[ResNet50::layer3_3_conv3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_3_bn3_weight].sizes().equals({1024}),
                    "Expected weight at index %d to be of shape [1024], but got %s",
                    ResNet50::layer3_3_bn3_weight,
                    to_string(weights[ResNet50::layer3_3_bn3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_3_bn3_bias].sizes().equals({1024}),
                    "Expected weight at index %d to be of shape [1024], but got %s",
                    ResNet50::layer3_3_bn3_bias,
                    to_string(weights[ResNet50::layer3_3_bn3_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_4_conv1_weight].sizes().equals({256, 1024, 1, 1}),
                    "Expected weight at index %d to be of shape [256, 1024, 1, 1], but got %s",
                    ResNet50::layer3_4_conv1_weight,
                    to_string(weights[ResNet50::layer3_4_conv1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_4_bn1_weight].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_4_bn1_weight,
                    to_string(weights[ResNet50::layer3_4_bn1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_4_bn1_bias].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_4_bn1_bias,
                    to_string(weights[ResNet50::layer3_4_bn1_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_4_conv2_weight].sizes().equals({256, 256, 3, 3}),
                    "Expected weight at index %d to be of shape [256, 256, 3, 3], but got %s",
                    ResNet50::layer3_4_conv2_weight,
                    to_string(weights[ResNet50::layer3_4_conv2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_4_bn2_weight].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_4_bn2_weight,
                    to_string(weights[ResNet50::layer3_4_bn2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_4_bn2_bias].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_4_bn2_bias,
                    to_string(weights[ResNet50::layer3_4_bn2_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_4_conv3_weight].sizes().equals({1024, 256, 1, 1}),
                    "Expected weight at index %d to be of shape [1024, 256, 1, 1], but got %s",
                    ResNet50::layer3_4_conv3_weight,
                    to_string(weights[ResNet50::layer3_4_conv3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_4_bn3_weight].sizes().equals({1024}),
                    "Expected weight at index %d to be of shape [1024], but got %s",
                    ResNet50::layer3_4_bn3_weight,
                    to_string(weights[ResNet50::layer3_4_bn3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_4_bn3_bias].sizes().equals({1024}),
                    "Expected weight at index %d to be of shape [1024], but got %s",
                    ResNet50::layer3_4_bn3_bias,
                    to_string(weights[ResNet50::layer3_4_bn3_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_5_conv1_weight].sizes().equals({256, 1024, 1, 1}),
                    "Expected weight at index %d to be of shape [256, 1024, 1, 1], but got %s",
                    ResNet50::layer3_5_conv1_weight,
                    to_string(weights[ResNet50::layer3_5_conv1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_5_bn1_weight].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_5_bn1_weight,
                    to_string(weights[ResNet50::layer3_5_bn1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_5_bn1_bias].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_5_bn1_bias,
                    to_string(weights[ResNet50::layer3_5_bn1_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_5_conv2_weight].sizes().equals({256, 256, 3, 3}),
                    "Expected weight at index %d to be of shape [256, 256, 3, 3], but got %s",
                    ResNet50::layer3_5_conv2_weight,
                    to_string(weights[ResNet50::layer3_5_conv2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_5_bn2_weight].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_5_bn2_weight,
                    to_string(weights[ResNet50::layer3_5_bn2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_5_bn2_bias].sizes().equals({256}),
                    "Expected weight at index %d to be of shape [256], but got %s",
                    ResNet50::layer3_5_bn2_bias,
                    to_string(weights[ResNet50::layer3_5_bn2_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_5_conv3_weight].sizes().equals({1024, 256, 1, 1}),
                    "Expected weight at index %d to be of shape [1024, 256, 1, 1], but got %s",
                    ResNet50::layer3_5_conv3_weight,
                    to_string(weights[ResNet50::layer3_5_conv3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_5_bn3_weight].sizes().equals({1024}),
                    "Expected weight at index %d to be of shape [1024], but got %s",
                    ResNet50::layer3_5_bn3_weight,
                    to_string(weights[ResNet50::layer3_5_bn3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer3_5_bn3_bias].sizes().equals({1024}),
                    "Expected weight at index %d to be of shape [1024], but got %s",
                    ResNet50::layer3_5_bn3_bias,
                    to_string(weights[ResNet50::layer3_5_bn3_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_0_conv1_weight].sizes().equals({512, 1024, 1, 1}),
                    "Expected weight at index %d to be of shape [512, 1024, 1, 1], but got %s",
                    ResNet50::layer4_0_conv1_weight,
                    to_string(weights[ResNet50::layer4_0_conv1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_0_bn1_weight].sizes().equals({512}),
                    "Expected weight at index %d to be of shape [512], but got %s",
                    ResNet50::layer4_0_bn1_weight,
                    to_string(weights[ResNet50::layer4_0_bn1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_0_bn1_bias].sizes().equals({512}),
                    "Expected weight at index %d to be of shape [512], but got %s",
                    ResNet50::layer4_0_bn1_bias,
                    to_string(weights[ResNet50::layer4_0_bn1_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_0_conv2_weight].sizes().equals({512, 512, 3, 3}),
                    "Expected weight at index %d to be of shape [512, 512, 3, 3], but got %s",
                    ResNet50::layer4_0_conv2_weight,
                    to_string(weights[ResNet50::layer4_0_conv2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_0_bn2_weight].sizes().equals({512}),
                    "Expected weight at index %d to be of shape [512], but got %s",
                    ResNet50::layer4_0_bn2_weight,
                    to_string(weights[ResNet50::layer4_0_bn2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_0_bn2_bias].sizes().equals({512}),
                    "Expected weight at index %d to be of shape [512], but got %s",
                    ResNet50::layer4_0_bn2_bias,
                    to_string(weights[ResNet50::layer4_0_bn2_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_0_conv3_weight].sizes().equals({2048, 512, 1, 1}),
                    "Expected weight at index %d to be of shape [2048, 512, 1, 1], but got %s",
                    ResNet50::layer4_0_conv3_weight,
                    to_string(weights[ResNet50::layer4_0_conv3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_0_bn3_weight].sizes().equals({2048}),
                    "Expected weight at index %d to be of shape [2048], but got %s",
                    ResNet50::layer4_0_bn3_weight,
                    to_string(weights[ResNet50::layer4_0_bn3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_0_bn3_bias].sizes().equals({2048}),
                    "Expected weight at index %d to be of shape [2048], but got %s",
                    ResNet50::layer4_0_bn3_bias,
                    to_string(weights[ResNet50::layer4_0_bn3_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_0_downsample_0_weight].sizes().equals({2048, 1024, 1, 1}),
                    "Expected weight at index %d to be of shape [2048, 1024, 1, 1], but got %s",
                    ResNet50::layer4_0_downsample_0_weight,
                    to_string(weights[ResNet50::layer4_0_downsample_0_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_0_downsample_1_weight].sizes().equals({2048}),
                    "Expected weight at index %d to be of shape [2048], but got %s",
                    ResNet50::layer4_0_downsample_1_weight,
                    to_string(weights[ResNet50::layer4_0_downsample_1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_0_downsample_1_bias].sizes().equals({2048}),
                    "Expected weight at index %d to be of shape [2048], but got %s",
                    ResNet50::layer4_0_downsample_1_bias,
                    to_string(weights[ResNet50::layer4_0_downsample_1_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_1_conv1_weight].sizes().equals({512, 2048, 1, 1}),
                    "Expected weight at index %d to be of shape [512, 2048, 1, 1], but got %s",
                    ResNet50::layer4_1_conv1_weight,
                    to_string(weights[ResNet50::layer4_1_conv1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_1_bn1_weight].sizes().equals({512}),
                    "Expected weight at index %d to be of shape [512], but got %s",
                    ResNet50::layer4_1_bn1_weight,
                    to_string(weights[ResNet50::layer4_1_bn1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_1_bn1_bias].sizes().equals({512}),
                    "Expected weight at index %d to be of shape [512], but got %s",
                    ResNet50::layer4_1_bn1_bias,
                    to_string(weights[ResNet50::layer4_1_bn1_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_1_conv2_weight].sizes().equals({512, 512, 3, 3}),
                    "Expected weight at index %d to be of shape [512, 512, 3, 3], but got %s",
                    ResNet50::layer4_1_conv2_weight,
                    to_string(weights[ResNet50::layer4_1_conv2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_1_bn2_weight].sizes().equals({512}),
                    "Expected weight at index %d to be of shape [512], but got %s",
                    ResNet50::layer4_1_bn2_weight,
                    to_string(weights[ResNet50::layer4_1_bn2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_1_bn2_bias].sizes().equals({512}),
                    "Expected weight at index %d to be of shape [512], but got %s",
                    ResNet50::layer4_1_bn2_bias,
                    to_string(weights[ResNet50::layer4_1_bn2_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_1_conv3_weight].sizes().equals({2048, 512, 1, 1}),
                    "Expected weight at index %d to be of shape [2048, 512, 1, 1], but got %s",
                    ResNet50::layer4_1_conv3_weight,
                    to_string(weights[ResNet50::layer4_1_conv3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_1_bn3_weight].sizes().equals({2048}),
                    "Expected weight at index %d to be of shape [2048], but got %s",
                    ResNet50::layer4_1_bn3_weight,
                    to_string(weights[ResNet50::layer4_1_bn3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_1_bn3_bias].sizes().equals({2048}),
                    "Expected weight at index %d to be of shape [2048], but got %s",
                    ResNet50::layer4_1_bn3_bias,
                    to_string(weights[ResNet50::layer4_1_bn3_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_2_conv1_weight].sizes().equals({512, 2048, 1, 1}),
                    "Expected weight at index %d to be of shape [512, 2048, 1, 1], but got %s",
                    ResNet50::layer4_2_conv1_weight,
                    to_string(weights[ResNet50::layer4_2_conv1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_2_bn1_weight].sizes().equals({512}),
                    "Expected weight at index %d to be of shape [512], but got %s",
                    ResNet50::layer4_2_bn1_weight,
                    to_string(weights[ResNet50::layer4_2_bn1_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_2_bn1_bias].sizes().equals({512}),
                    "Expected weight at index %d to be of shape [512], but got %s",
                    ResNet50::layer4_2_bn1_bias,
                    to_string(weights[ResNet50::layer4_2_bn1_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_2_conv2_weight].sizes().equals({512, 512, 3, 3}),
                    "Expected weight at index %d to be of shape [512, 512, 3, 3], but got %s",
                    ResNet50::layer4_2_conv2_weight,
                    to_string(weights[ResNet50::layer4_2_conv2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_2_bn2_weight].sizes().equals({512}),
                    "Expected weight at index %d to be of shape [512], but got %s",
                    ResNet50::layer4_2_bn2_weight,
                    to_string(weights[ResNet50::layer4_2_bn2_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_2_bn2_bias].sizes().equals({512}),
                    "Expected weight at index %d to be of shape [512], but got %s",
                    ResNet50::layer4_2_bn2_bias,
                    to_string(weights[ResNet50::layer4_2_bn2_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_2_conv3_weight].sizes().equals({2048, 512, 1, 1}),
                    "Expected weight at index %d to be of shape [2048, 512, 1, 1], but got %s",
                    ResNet50::layer4_2_conv3_weight,
                    to_string(weights[ResNet50::layer4_2_conv3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_2_bn3_weight].sizes().equals({2048}),
                    "Expected weight at index %d to be of shape [2048], but got %s",
                    ResNet50::layer4_2_bn3_weight,
                    to_string(weights[ResNet50::layer4_2_bn3_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::layer4_2_bn3_bias].sizes().equals({2048}),
                    "Expected weight at index %d to be of shape [2048], but got %s",
                    ResNet50::layer4_2_bn3_bias,
                    to_string(weights[ResNet50::layer4_2_bn3_bias].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::fc_weight].sizes().equals({1000, 2048}),
                    "Expected weight at index %d to be of shape [1000, 2048], but got %s",
                    ResNet50::fc_weight,
                    to_string(weights[ResNet50::fc_weight].sizes()).c_str());
                ET_CHECK_MSG(
                    weights[ResNet50::fc_bias].sizes().equals({1000}),
                    "Expected weight at index %d to be of shape [1000], but got %s",
                    ResNet50::fc_bias,
                    to_string(weights[ResNet50::fc_bias].sizes()).c_str());
            }
        } // namespace

        Tensor &resnet50_out_impl(const Tensor &in, const TensorList &weights, Tensor &out)
        {
            check_preconditions(in, weights, out);
            ResNet50 resnet50 = ResNet50(weights);

            if (resnet50.getType() == ScalarType::Float)
            {
                // Float
                float *out_data = out.mutable_data_ptr<float>();
                const float *in_data = in.const_data_ptr<float>();

                const float *output = resnet50.inference(in_data);
                std::memcpy(out_data, output, out.numel() * sizeof(float));
            }
            else if (resnet50.getType() == ScalarType::Char)
            {
                // Int8
                // Convert Input to Int8
                Tensor in_int8 = in.to(ScalarType::Char);

                const int8_t *in_data = in_int8.const_data_ptr<int8_t>();
                float *out_data = out.mutable_data_ptr<float>();

                const int8_t *output = resnet50.inference(in_data);
                Tensor out_int8 = at::from_blob((void *)output, {1000}, at::TensorOptions().dtype(at::kChar));

                // Convert output back to Float
                std::memcpy(out_data, out_int8.to(ScalarType::Float).mutable_data_ptr<float>(), out.numel() * sizeof(float));
            }

            return out;
        }

        Tensor &resnet50_out_impl(RuntimeContext &ctx, const Tensor &in, const TensorList &weights, Tensor &out)
        {
            (void)ctx;
            resnet50_out_impl(in, weights, out);
            return out;
        }
    } // namespace native
} // namespace custom