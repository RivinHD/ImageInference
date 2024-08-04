//  Copyright (c) 2024 by Vincent Gerlach. All rights reserved.
//
//  SPDX-License-Identifier: GPL-3.0-or-later
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  in the root folder of this project with the name LICENSE. If not, see <http://www.gnu.org/licenses/>.

#ifndef IMAGEINFERENCE_RESNET50_H
#define IMAGEINFERENCE_RESNET50_H

#include "IModel.h"
#include "../types/Image.h"
#include "../types/Kernel.h"
#include "../types/Array.h"
#include "../types/BatchNorm.h"
#include "../types/Matrix.h"
#include "../types/ScalarTypes.h"
#include <vector>
#include <stdint.h>
#include <omp.h>
#include <iostream>
#ifndef FASTOR_USE_LIBXSMM
#define FASTOR_USE_LIBXSMM
#endif // !FASTOR_USE_LIBXSMM
#include <Fastor/Fastor.h>
#include <libxsmm.h>

#define MAX_RESNET50_SIZE 122 * 122 * 64 * 2 * 2 // 967936 additional 2x for zero padding
#define RESNET50_BLOCK_SIZE 16

#ifdef IMAGEINFERENCE_TESTING
namespace ImageInference::model::test
{
    class ResNet50Test;
}
#endif // IMAGEINFERENCE_TESTING

namespace ImageInference
{
    namespace model
    {
        /// @brief The resnet50 v1.5 model from https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch
        class ResNet50 : public IModel<float>
        {
        private:
            std::vector<void *> modelWeights;
            ImageInference::types::ScalarType type;

            // All the blocks start with a 1x1 kernel. Therefore no padding is required.

            template <typename T, size_t BlockSize>
            ImageInference::types::Image<T, 0, BlockSize, 256, 61, 61> block0(ImageInference::types::Image<T, 0, BlockSize, 64, 61, 61> &input);

            template <typename T, size_t BlockSize>
            ImageInference::types::Image<T, 0, BlockSize, 512, 30, 30> block1(ImageInference::types::Image<T, 0, BlockSize, 256, 61, 61> &input);

            template <typename T, size_t BlockSize>
            ImageInference::types::Image<T, 0, BlockSize, 1024, 15, 15> block2(ImageInference::types::Image<T, 0, BlockSize, 512, 30, 30> &input);

            template <typename T, size_t BlockSize>
            ImageInference::types::Image<T, 0, BlockSize, 2048, 7, 7> block3(ImageInference::types::Image<T, 0, BlockSize, 1024, 15, 15> &input);

            template <size_t Stride, size_t OutPadding, size_t InPadding,
                      typename T, size_t BlockSizeCount, size_t BlockSizeChannel,
                      size_t ImageChannels, size_t ImageHeight, size_t ImageWidth,
                      size_t KernelCount, size_t KernelHeight, size_t KernelWidth>
            static ImageInference::types::Image<T, OutPadding, BlockSizeCount, KernelCount, ImageHeight / Stride, ImageWidth / Stride> convBlock(
                ImageInference::types::Image<T, InPadding, BlockSizeChannel, ImageChannels, ImageHeight, ImageWidth> &image,
                ImageInference::types::Kernel<T, BlockSizeCount, BlockSizeChannel, KernelCount, ImageChannels, KernelHeight, KernelWidth> &kernel,
                ImageInference::types::BatchNorm<T, KernelCount> &batchNorm);

            template <size_t OutPadding, size_t InPadding, size_t ShortcutPadding,
                      typename T, size_t BlockSizeCount, size_t BlockSizeChannel,
                      size_t ImageChannels, size_t ImageHeight, size_t ImageWidth,
                      size_t KernelCount, size_t KernelHeight, size_t KernelWidth>
            static ImageInference::types::Image<T, OutPadding, BlockSizeCount, KernelCount, ImageHeight, ImageWidth> convBlockAddIdentity(
                ImageInference::types::Image<T, InPadding, BlockSizeChannel, ImageChannels, ImageHeight, ImageWidth> &image,
                ImageInference::types::Kernel<T, BlockSizeCount, BlockSizeChannel, KernelCount, ImageChannels, KernelHeight, KernelWidth> &kernel,
                ImageInference::types::BatchNorm<T, KernelCount> &batchNorm,
                ImageInference::types::Image<T, ShortcutPadding, BlockSizeCount, KernelCount, ImageHeight, ImageWidth> &shortcut);

            template <size_t Stride, size_t ShortcutDimExpand,
                      size_t OutPadding, size_t InPadding, size_t ShortcutPadding,
                      typename T, size_t BlockSizeCount, size_t BlockSizeChannel,
                      size_t ImageChannels, size_t ImageHeight, size_t ImageWidth,
                      size_t KernelCount, size_t KernelHeight, size_t KernelWidth>
            static ImageInference::types::Image<T, OutPadding, BlockSizeCount, KernelCount, ImageHeight / Stride, ImageWidth / Stride> convBlockAddProjection(
                ImageInference::types::Image<T, InPadding, BlockSizeChannel, ImageChannels, ImageHeight / Stride, ImageWidth / Stride> &image,
                ImageInference::types::Kernel<T, BlockSizeCount, BlockSizeChannel, KernelCount, ImageChannels, KernelHeight, KernelWidth> &kernel,
                ImageInference::types::BatchNorm<T, KernelCount> &batchNorm,
                ImageInference::types::Image<T, ShortcutPadding, BlockSizeCount, KernelCount / ShortcutDimExpand, ImageHeight, ImageWidth> &shortcut,
                ImageInference::types::Kernel<T, BlockSizeCount, BlockSizeCount, KernelCount, KernelCount / ShortcutDimExpand, 1, 1> &projectionKernel,
                ImageInference::types::BatchNorm<T, KernelCount> &projectionBatchNorm);

            template <size_t Stride, size_t OutPadding, size_t InPadding,
                      typename T, size_t BlockSize,
                      size_t ImageChannels, size_t ImageHeight, size_t ImageWidth>
            static ImageInference::types::Image<T, OutPadding, BlockSize, ImageChannels, ImageHeight / Stride, ImageWidth / Stride> maxPool(
                ImageInference::types::Image<T, InPadding, BlockSize, ImageChannels, ImageHeight, ImageWidth> &image);

            template <size_t OutPadding, size_t InPadding, typename T, size_t BlockSize,
                      size_t ImageChannels, size_t ImageHeight, size_t ImageWidth>
            static ImageInference::types::Image<T, OutPadding, BlockSize, ImageChannels, 1, 1> globalAveragePool(
                ImageInference::types::Image<T, InPadding, BlockSize, ImageChannels, ImageHeight, ImageWidth> &image);

            template <typename T, size_t Columns, size_t Rows>
            static void fullyConnectedLayer(
                ImageInference::types::Array<T, Rows> &input,
                ImageInference::types::Matrix<T, Columns, Rows> &weight,
                ImageInference::types::Array<T, Columns> &biasAccumulator);

            template <typename T>
            T *getWeight(size_t index);

            template <typename T>
            static T relu(T value);

            template <typename T>
            static T batchNorm(T value, T gamma, T beta, T mean, T batchVariance);

        public:
            /// @brief Initialize the model with the weights
            /// @param weights The weights of the model with the following shape.
            /// @param type The scalar type of the weights.
            ///
            /// @code
            ///     conv1.weight: [64, 3, 7, 7]
            ///     bn1.weight: [64]
            ///     bn1.bias: [64]
            ///     layer1.0.conv1.weight: [64, 64, 1, 1]
            ///     layer1.0.bn1.weight: [64]
            ///     layer1.0.bn1.bias: [64]
            ///     layer1.0.conv2.weight: [64, 64, 3, 3]
            ///     layer1.0.bn2.weight: [64]
            ///     layer1.0.bn2.bias: [64]
            ///     layer1.0.conv3.weight: [256, 64, 1, 1]
            ///     layer1.0.bn3.weight: [256]
            ///     layer1.0.bn3.bias: [256]
            ///     layer1.0.downsample.0.weight: [256, 64, 1, 1]
            ///     layer1.0.downsample.1.weight: [256]
            ///     layer1.0.downsample.1.bias: [256]
            ///     layer1.1.conv1.weight: [64, 256, 1, 1]
            ///     layer1.1.bn1.weight: [64]
            ///     layer1.1.bn1.bias: [64]
            ///     layer1.1.conv2.weight: [64, 64, 3, 3]
            ///     layer1.1.bn2.weight: [64]
            ///     layer1.1.bn2.bias: [64]
            ///     layer1.1.conv3.weight: [256, 64, 1, 1]
            ///     layer1.1.bn3.weight: [256]
            ///     layer1.1.bn3.bias: [256]
            ///     layer1.2.conv1.weight: [64, 256, 1, 1]
            ///     layer1.2.bn1.weight: [64]
            ///     layer1.2.bn1.bias: [64]
            ///     layer1.2.conv2.weight: [64, 64, 3, 3]
            ///     layer1.2.bn2.weight: [64]
            ///     layer1.2.bn2.bias: [64]
            ///     layer1.2.conv3.weight: [256, 64, 1, 1]
            ///     layer1.2.bn3.weight: [256]
            ///     layer1.2.bn3.bias: [256]
            ///     layer2.0.conv1.weight: [128, 256, 1, 1]
            ///     layer2.0.bn1.weight: [128]
            ///     layer2.0.bn1.bias: [128]
            ///     layer2.0.conv2.weight: [128, 128, 3, 3]
            ///     layer2.0.bn2.weight: [128]
            ///     layer2.0.bn2.bias: [128]
            ///     layer2.0.conv3.weight: [512, 128, 1, 1]
            ///     layer2.0.bn3.weight: [512]
            ///     layer2.0.bn3.bias: [512]
            ///     layer2.0.downsample.0.weight: [512, 256, 1, 1]
            ///     layer2.0.downsample.1.weight: [512]
            ///     layer2.0.downsample.1.bias: [512]
            ///     layer2.1.conv1.weight: [128, 512, 1, 1]
            ///     layer2.1.bn1.weight: [128]
            ///     layer2.1.bn1.bias: [128]
            ///     layer2.1.conv2.weight: [128, 128, 3, 3]
            ///     layer2.1.bn2.weight: [128]
            ///     layer2.1.bn2.bias: [128]
            ///     layer2.1.conv3.weight: [512, 128, 1, 1]
            ///     layer2.1.bn3.weight: [512]
            ///     layer2.1.bn3.bias: [512]
            ///     layer2.2.conv1.weight: [128, 512, 1, 1]
            ///     layer2.2.bn1.weight: [128]
            ///     layer2.2.bn1.bias: [128]
            ///     layer2.2.conv2.weight: [128, 128, 3, 3]
            ///     layer2.2.bn2.weight: [128]
            ///     layer2.2.bn2.bias: [128]
            ///     layer2.2.conv3.weight: [512, 128, 1, 1]
            ///     layer2.2.bn3.weight: [512]
            ///     layer2.2.bn3.bias: [512]
            ///     layer2.3.conv1.weight: [128, 512, 1, 1]
            ///     layer2.3.bn1.weight: [128]
            ///     layer2.3.bn1.bias: [128]
            ///     layer2.3.conv2.weight: [128, 128, 3, 3]
            ///     layer2.3.bn2.weight: [128]
            ///     layer2.3.bn2.bias: [128]
            ///     layer2.3.conv3.weight: [512, 128, 1, 1]
            ///     layer2.3.bn3.weight: [512]
            ///     layer2.3.bn3.bias: [512]
            ///     layer3.0.conv1.weight: [256, 512, 1, 1]
            ///     layer3.0.bn1.weight: [256]
            ///     layer3.0.bn1.bias: [256]
            ///     layer3.0.conv2.weight: [256, 256, 3, 3]
            ///     layer3.0.bn2.weight: [256]
            ///     layer3.0.bn2.bias: [256]
            ///     layer3.0.conv3.weight: [1024, 256, 1, 1]
            ///     layer3.0.bn3.weight: [1024]
            ///     layer3.0.bn3.bias: [1024]
            ///     layer3.0.downsample.0.weight: [1024, 512, 1, 1]
            ///     layer3.0.downsample.1.weight: [1024]
            ///     layer3.0.downsample.1.bias: [1024]
            ///     layer3.1.conv1.weight: [256, 1024, 1, 1]
            ///     layer3.1.bn1.weight: [256]
            ///     layer3.1.bn1.bias: [256]
            ///     layer3.1.conv2.weight: [256, 256, 3, 3]
            ///     layer3.1.bn2.weight: [256]
            ///     layer3.1.bn2.bias: [256]
            ///     layer3.1.conv3.weight: [1024, 256, 1, 1]
            ///     layer3.1.bn3.weight: [1024]
            ///     layer3.1.bn3.bias: [1024]
            ///     layer3.2.conv1.weight: [256, 1024, 1, 1]
            ///     layer3.2.bn1.weight: [256]
            ///     layer3.2.bn1.bias: [256]
            ///     layer3.2.conv2.weight: [256, 256, 3, 3]
            ///     layer3.2.bn2.weight: [256]
            ///     layer3.2.bn2.bias: [256]
            ///     layer3.2.conv3.weight: [1024, 256, 1, 1]
            ///     layer3.2.bn3.weight: [1024]
            ///     layer3.2.bn3.bias: [1024]
            ///     layer3.3.conv1.weight: [256, 1024, 1, 1]
            ///     layer3.3.bn1.weight: [256]
            ///     layer3.3.bn1.bias: [256]
            ///     layer3.3.conv2.weight: [256, 256, 3, 3]
            ///     layer3.3.bn2.weight: [256]
            ///     layer3.3.bn2.bias: [256]
            ///     layer3.3.conv3.weight: [1024, 256, 1, 1]
            ///     layer3.3.bn3.weight: [1024]
            ///     layer3.3.bn3.bias: [1024]
            ///     layer3.4.conv1.weight: [256, 1024, 1, 1]
            ///     layer3.4.bn1.weight: [256]
            ///     layer3.4.bn1.bias: [256]
            ///     layer3.4.conv2.weight: [256, 256, 3, 3]
            ///     layer3.4.bn2.weight: [256]
            ///     layer3.4.bn2.bias: [256]
            ///     layer3.4.conv3.weight: [1024, 256, 1, 1]
            ///     layer3.4.bn3.weight: [1024]
            ///     layer3.4.bn3.bias: [1024]
            ///     layer3.5.conv1.weight: [256, 1024, 1, 1]
            ///     layer3.5.bn1.weight: [256]
            ///     layer3.5.bn1.bias: [256]
            ///     layer3.5.conv2.weight: [256, 256, 3, 3]
            ///     layer3.5.bn2.weight: [256]
            ///     layer3.5.bn2.bias: [256]
            ///     layer3.5.conv3.weight: [1024, 256, 1, 1]
            ///     layer3.5.bn3.weight: [1024]
            ///     layer3.5.bn3.bias: [1024]
            ///     layer4.0.conv1.weight: [512, 1024, 1, 1]
            ///     layer4.0.bn1.weight: [512]
            ///     layer4.0.bn1.bias: [512]
            ///     layer4.0.conv2.weight: [512, 512, 3, 3]
            ///     layer4.0.bn2.weight: [512]
            ///     layer4.0.bn2.bias: [512]
            ///     layer4.0.conv3.weight: [2048, 512, 1, 1]
            ///     layer4.0.bn3.weight: [2048]
            ///     layer4.0.bn3.bias: [2048]
            ///     layer4.0.downsample.0.weight: [2048, 1024, 1, 1]
            ///     layer4.0.downsample.1.weight: [2048]
            ///     layer4.0.downsample.1.bias: [2048]
            ///     layer4.1.conv1.weight: [512, 2048, 1, 1]
            ///     layer4.1.bn1.weight: [512]
            ///     layer4.1.bn1.bias: [512]
            ///     layer4.1.conv2.weight: [512, 512, 3, 3]
            ///     layer4.1.bn2.weight: [512]
            ///     layer4.1.bn2.bias: [512]
            ///     layer4.1.conv3.weight: [2048, 512, 1, 1]
            ///     layer4.1.bn3.weight: [2048]
            ///     layer4.1.bn3.bias: [2048]
            ///     layer4.2.conv1.weight: [512, 2048, 1, 1]
            ///     layer4.2.bn1.weight: [512]
            ///     layer4.2.bn1.bias: [512]
            ///     layer4.2.conv2.weight: [512, 512, 3, 3]
            ///     layer4.2.bn2.weight: [512]
            ///     layer4.2.bn2.bias: [512]
            ///     layer4.2.conv3.weight: [2048, 512, 1, 1]
            ///     layer4.2.bn3.weight: [2048]
            ///     layer4.2.bn3.bias: [2048]
            ///     fc.weight: [1000, 2048]
            ///     fc.bias: [1000]
            /// @endcode
            ResNet50(const std::vector<void *> &modelWeights, ImageInference::types::ScalarType type);
            ~ResNet50();

            enum weightIndex
            {
                conv1_weight = 0,
                bn1_weight = 1,
                bn1_bias = 2,
                layer1_0_conv1_weight = 3,
                layer1_0_bn1_weight = 4,
                layer1_0_bn1_bias = 5,
                layer1_0_conv2_weight = 6,
                layer1_0_bn2_weight = 7,
                layer1_0_bn2_bias = 8,
                layer1_0_conv3_weight = 9,
                layer1_0_bn3_weight = 10,
                layer1_0_bn3_bias = 11,
                layer1_0_downsample_0_weight = 12,
                layer1_0_downsample_1_weight = 13,
                layer1_0_downsample_1_bias = 14,
                layer1_1_conv1_weight = 15,
                layer1_1_bn1_weight = 16,
                layer1_1_bn1_bias = 17,
                layer1_1_conv2_weight = 18,
                layer1_1_bn2_weight = 19,
                layer1_1_bn2_bias = 20,
                layer1_1_conv3_weight = 21,
                layer1_1_bn3_weight = 22,
                layer1_1_bn3_bias = 23,
                layer1_2_conv1_weight = 24,
                layer1_2_bn1_weight = 25,
                layer1_2_bn1_bias = 26,
                layer1_2_conv2_weight = 27,
                layer1_2_bn2_weight = 28,
                layer1_2_bn2_bias = 29,
                layer1_2_conv3_weight = 30,
                layer1_2_bn3_weight = 31,
                layer1_2_bn3_bias = 32,
                layer2_0_conv1_weight = 33,
                layer2_0_bn1_weight = 34,
                layer2_0_bn1_bias = 35,
                layer2_0_conv2_weight = 36,
                layer2_0_bn2_weight = 37,
                layer2_0_bn2_bias = 38,
                layer2_0_conv3_weight = 39,
                layer2_0_bn3_weight = 40,
                layer2_0_bn3_bias = 41,
                layer2_0_downsample_0_weight = 42,
                layer2_0_downsample_1_weight = 43,
                layer2_0_downsample_1_bias = 44,
                layer2_1_conv1_weight = 45,
                layer2_1_bn1_weight = 46,
                layer2_1_bn1_bias = 47,
                layer2_1_conv2_weight = 48,
                layer2_1_bn2_weight = 49,
                layer2_1_bn2_bias = 50,
                layer2_1_conv3_weight = 51,
                layer2_1_bn3_weight = 52,
                layer2_1_bn3_bias = 53,
                layer2_2_conv1_weight = 54,
                layer2_2_bn1_weight = 55,
                layer2_2_bn1_bias = 56,
                layer2_2_conv2_weight = 57,
                layer2_2_bn2_weight = 58,
                layer2_2_bn2_bias = 59,
                layer2_2_conv3_weight = 60,
                layer2_2_bn3_weight = 61,
                layer2_2_bn3_bias = 62,
                layer2_3_conv1_weight = 63,
                layer2_3_bn1_weight = 64,
                layer2_3_bn1_bias = 65,
                layer2_3_conv2_weight = 66,
                layer2_3_bn2_weight = 67,
                layer2_3_bn2_bias = 68,
                layer2_3_conv3_weight = 69,
                layer2_3_bn3_weight = 70,
                layer2_3_bn3_bias = 71,
                layer3_0_conv1_weight = 72,
                layer3_0_bn1_weight = 73,
                layer3_0_bn1_bias = 74,
                layer3_0_conv2_weight = 75,
                layer3_0_bn2_weight = 76,
                layer3_0_bn2_bias = 77,
                layer3_0_conv3_weight = 78,
                layer3_0_bn3_weight = 79,
                layer3_0_bn3_bias = 80,
                layer3_0_downsample_0_weight = 81,
                layer3_0_downsample_1_weight = 82,
                layer3_0_downsample_1_bias = 83,
                layer3_1_conv1_weight = 84,
                layer3_1_bn1_weight = 85,
                layer3_1_bn1_bias = 86,
                layer3_1_conv2_weight = 87,
                layer3_1_bn2_weight = 88,
                layer3_1_bn2_bias = 89,
                layer3_1_conv3_weight = 90,
                layer3_1_bn3_weight = 91,
                layer3_1_bn3_bias = 92,
                layer3_2_conv1_weight = 93,
                layer3_2_bn1_weight = 94,
                layer3_2_bn1_bias = 95,
                layer3_2_conv2_weight = 96,
                layer3_2_bn2_weight = 97,
                layer3_2_bn2_bias = 98,
                layer3_2_conv3_weight = 99,
                layer3_2_bn3_weight = 100,
                layer3_2_bn3_bias = 101,
                layer3_3_conv1_weight = 102,
                layer3_3_bn1_weight = 103,
                layer3_3_bn1_bias = 104,
                layer3_3_conv2_weight = 105,
                layer3_3_bn2_weight = 106,
                layer3_3_bn2_bias = 107,
                layer3_3_conv3_weight = 108,
                layer3_3_bn3_weight = 109,
                layer3_3_bn3_bias = 110,
                layer3_4_conv1_weight = 111,
                layer3_4_bn1_weight = 112,
                layer3_4_bn1_bias = 113,
                layer3_4_conv2_weight = 114,
                layer3_4_bn2_weight = 115,
                layer3_4_bn2_bias = 116,
                layer3_4_conv3_weight = 117,
                layer3_4_bn3_weight = 118,
                layer3_4_bn3_bias = 119,
                layer3_5_conv1_weight = 120,
                layer3_5_bn1_weight = 121,
                layer3_5_bn1_bias = 122,
                layer3_5_conv2_weight = 123,
                layer3_5_bn2_weight = 124,
                layer3_5_bn2_bias = 125,
                layer3_5_conv3_weight = 126,
                layer3_5_bn3_weight = 127,
                layer3_5_bn3_bias = 128,
                layer4_0_conv1_weight = 129,
                layer4_0_bn1_weight = 130,
                layer4_0_bn1_bias = 131,
                layer4_0_conv2_weight = 132,
                layer4_0_bn2_weight = 133,
                layer4_0_bn2_bias = 134,
                layer4_0_conv3_weight = 135,
                layer4_0_bn3_weight = 136,
                layer4_0_bn3_bias = 137,
                layer4_0_downsample_0_weight = 138,
                layer4_0_downsample_1_weight = 139,
                layer4_0_downsample_1_bias = 140,
                layer4_1_conv1_weight = 141,
                layer4_1_bn1_weight = 142,
                layer4_1_bn1_bias = 143,
                layer4_1_conv2_weight = 144,
                layer4_1_bn2_weight = 145,
                layer4_1_bn2_bias = 146,
                layer4_1_conv3_weight = 147,
                layer4_1_bn3_weight = 148,
                layer4_1_bn3_bias = 149,
                layer4_2_conv1_weight = 150,
                layer4_2_bn1_weight = 151,
                layer4_2_bn1_bias = 152,
                layer4_2_conv2_weight = 153,
                layer4_2_bn2_weight = 154,
                layer4_2_bn2_bias = 155,
                layer4_2_conv3_weight = 156,
                layer4_2_bn3_weight = 157,
                layer4_2_bn3_bias = 158,
                fc_weight = 159,
                fc_bias = 160
            };

            void inference(const float *input, float *output) override;
            ImageInference::types::ScalarType getType();

#ifdef IMAGEINFERENCE_TESTING
            friend class ImageInference::model::test::ResNet50Test;
#endif // IMAGEINFERENCE_TESTING
        };

        template <typename T, size_t BlockSize>
        inline ImageInference::types::Image<T, 0, BlockSize, 256, 61, 61> ResNet50::block0(ImageInference::types::Image<T, 0, BlockSize, 64, 61, 61> &input)
        {
            auto kernel_0_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 64, 64, 1, 1>(getWeight<T>(weightIndex::layer1_0_conv1_weight));
            auto batchNorm_0_0 = ImageInference::types::BatchNorm<T, 64>(getWeight<T>(weightIndex::layer1_0_bn1_weight), getWeight<T>(weightIndex::layer1_0_bn1_bias));
            auto image_0_0 = convBlock<1, 1>(input, kernel_0_0, batchNorm_0_0); // OutPadding of 1 is because a 3x3 kernel is coming next
            auto kernel_0_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 64, 64, 3, 3>(getWeight<T>(weightIndex::layer1_0_conv2_weight));
            auto batchNorm_0_1 = ImageInference::types::BatchNorm<T, 64>(getWeight<T>(weightIndex::layer1_0_bn2_weight), getWeight<T>(weightIndex::layer1_0_bn2_bias));
            auto image_0_1 = convBlock<1, 0>(image_0_0, kernel_0_1, batchNorm_0_1); // OutPadding of 0 is because a 1x1 kernel is coming next
            auto kernel_0_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 64, 1, 1>(getWeight<T>(weightIndex::layer1_0_conv3_weight));
            auto batchNorm_0_2 = ImageInference::types::BatchNorm<T, 256>(getWeight<T>(weightIndex::layer1_0_bn3_weight), getWeight<T>(weightIndex::layer1_0_bn3_bias));
            auto projectionKernel = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 64, 1, 1>(getWeight<T>(weightIndex::layer1_0_downsample_0_weight));
            auto projectionBatchNorm = ImageInference::types::BatchNorm<T, 256>(getWeight<T>(weightIndex::layer1_0_downsample_1_weight), getWeight<T>(weightIndex::layer1_0_downsample_1_bias));
            // OutPadding of 0 is because a 1x1 kernel is coming next
            auto image_0_2 = convBlockAddProjection<1, 4, 0>(image_0_1, kernel_0_2, batchNorm_0_2, input, projectionKernel, projectionBatchNorm);

            auto kernel_1_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 64, 256, 1, 1>(getWeight<T>(weightIndex::layer1_1_conv1_weight));
            auto batchNorm_1_0 = ImageInference::types::BatchNorm<T, 64>(getWeight<T>(weightIndex::layer1_1_bn1_weight), getWeight<T>(weightIndex::layer1_1_bn1_bias));
            auto image_1_0 = convBlock<1, 1>(image_0_2, kernel_1_0, batchNorm_1_0); // OutPadding of 1 is because a 3x3 kernel is coming next
            auto kernel_1_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 64, 64, 3, 3>(getWeight<T>(weightIndex::layer1_1_conv2_weight));
            auto batchNorm_1_1 = ImageInference::types::BatchNorm<T, 64>(getWeight<T>(weightIndex::layer1_1_bn2_weight), getWeight<T>(weightIndex::layer1_1_bn2_bias));
            auto image_1_1 = convBlock<1, 0>(image_1_0, kernel_1_1, batchNorm_1_1); // OutPadding of 0 is because a 1x1 kernel is coming next
            auto kernel_1_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 64, 1, 1>(getWeight<T>(weightIndex::layer1_1_conv3_weight));
            auto batchNorm_1_2 = ImageInference::types::BatchNorm<T, 256>(getWeight<T>(weightIndex::layer1_1_bn3_weight), getWeight<T>(weightIndex::layer1_1_bn3_bias));
            // OutPadding of 0 is because a 1x1 kernel is coming next
            auto image_1_2 = convBlockAddIdentity<0>(image_1_1, kernel_1_2, batchNorm_1_2, image_0_2);

            auto kernel_2_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 64, 256, 1, 1>(getWeight<T>(weightIndex::layer1_2_conv1_weight));
            auto batchNorm_2_0 = ImageInference::types::BatchNorm<T, 64>(getWeight<T>(weightIndex::layer1_2_bn1_weight), getWeight<T>(weightIndex::layer1_2_bn1_bias));
            auto image_2_0 = convBlock<1, 1>(image_1_2, kernel_2_0, batchNorm_2_0);
            auto kernel_2_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 64, 64, 3, 3>(getWeight<T>(weightIndex::layer1_2_conv2_weight));
            auto batchNorm_2_1 = ImageInference::types::BatchNorm<T, 64>(getWeight<T>(weightIndex::layer1_2_bn2_weight), getWeight<T>(weightIndex::layer1_2_bn2_bias));
            auto image_2_1 = convBlock<1, 0>(image_2_0, kernel_2_1, batchNorm_2_1);
            auto kernel_2_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 64, 1, 1>(getWeight<T>(weightIndex::layer1_2_conv3_weight));
            auto batchNorm_2_2 = ImageInference::types::BatchNorm<T, 256>(getWeight<T>(weightIndex::layer1_2_bn3_weight), getWeight<T>(weightIndex::layer1_2_bn3_bias));
            // OutPadding of 0 is the next block starts with a 1x1 kernel
            auto image_2_2 = convBlockAddIdentity<0>(image_2_1, kernel_2_2, batchNorm_2_2, image_1_2);

            return image_2_2;
        }

        template <typename T, size_t BlockSize>
        ImageInference::types::Image<T, 0, BlockSize, 512, 30, 30> ResNet50::block1(ImageInference::types::Image<T, 0, BlockSize, 256, 61, 61> &input)
        {
            auto kernel_0_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 128, 256, 1, 1>(getWeight<T>(weightIndex::layer2_0_conv1_weight));
            auto batchNorm_0_0 = ImageInference::types::BatchNorm<T, 128>(getWeight<T>(weightIndex::layer2_0_bn1_weight), getWeight<T>(weightIndex::layer2_0_bn1_bias));
            auto image_0_0 = convBlock<1, 1>(input, kernel_0_0, batchNorm_0_0);
            auto kernel_0_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 128, 128, 3, 3>(getWeight<T>(weightIndex::layer2_0_conv2_weight));
            auto batchNorm_0_1 = ImageInference::types::BatchNorm<T, 128>(getWeight<T>(weightIndex::layer2_0_bn2_weight), getWeight<T>(weightIndex::layer2_0_bn2_bias));
            auto image_0_1 = convBlock<2, 0>(image_0_0, kernel_0_1, batchNorm_0_1);
            auto kernel_0_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 512, 128, 1, 1>(getWeight<T>(weightIndex::layer2_0_conv3_weight));
            auto batchNorm_0_2 = ImageInference::types::BatchNorm<T, 512>(getWeight<T>(weightIndex::layer2_0_bn3_weight), getWeight<T>(weightIndex::layer2_0_bn3_bias));
            auto projectionKernel = ImageInference::types::Kernel<T, BlockSize, BlockSize, 512, 256, 1, 1>(getWeight<T>(weightIndex::layer2_0_downsample_0_weight));
            auto projectionBatchNorm = ImageInference::types::BatchNorm<T, 512>(getWeight<T>(weightIndex::layer2_0_downsample_1_weight), getWeight<T>(weightIndex::layer2_0_downsample_1_bias));
            auto image_0_2 = convBlockAddProjection<2, 2, 0>(image_0_1, kernel_0_2, batchNorm_0_2, input, projectionKernel, projectionBatchNorm);

            auto kernel_1_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 128, 512, 1, 1>(getWeight<T>(weightIndex::layer2_1_conv1_weight));
            auto batchNorm_1_0 = ImageInference::types::BatchNorm<T, 128>(getWeight<T>(weightIndex::layer2_1_bn1_weight), getWeight<T>(weightIndex::layer2_1_bn1_bias));
            auto image_1_0 = convBlock<1, 1>(image_0_2, kernel_1_0, batchNorm_1_0);
            auto kernel_1_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 128, 128, 3, 3>(getWeight<T>(weightIndex::layer2_1_conv2_weight));
            auto batchNorm_1_1 = ImageInference::types::BatchNorm<T, 128>(getWeight<T>(weightIndex::layer2_1_bn2_weight), getWeight<T>(weightIndex::layer2_1_bn2_bias));
            auto image_1_1 = convBlock<1, 0>(image_1_0, kernel_1_1, batchNorm_1_1);
            auto kernel_1_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 512, 128, 1, 1>(getWeight<T>(weightIndex::layer2_1_conv3_weight));
            auto batchNorm_1_2 = ImageInference::types::BatchNorm<T, 512>(getWeight<T>(weightIndex::layer2_1_bn3_weight), getWeight<T>(weightIndex::layer2_1_bn3_bias));
            auto image_1_2 = convBlockAddIdentity<0>(image_1_1, kernel_1_2, batchNorm_1_2, image_0_2);

            auto kernel_2_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 128, 512, 1, 1>(getWeight<T>(weightIndex::layer2_2_conv1_weight));
            auto batchNorm_2_0 = ImageInference::types::BatchNorm<T, 128>(getWeight<T>(weightIndex::layer2_2_bn1_weight), getWeight<T>(weightIndex::layer2_2_bn1_bias));
            auto image_2_0 = convBlock<1, 1>(image_1_2, kernel_2_0, batchNorm_2_0);
            auto kernel_2_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 128, 128, 3, 3>(getWeight<T>(weightIndex::layer2_2_conv2_weight));
            auto batchNorm_2_1 = ImageInference::types::BatchNorm<T, 128>(getWeight<T>(weightIndex::layer2_2_bn2_weight), getWeight<T>(weightIndex::layer2_2_bn2_bias));
            auto image_2_1 = convBlock<1, 0>(image_2_0, kernel_2_1, batchNorm_2_1);
            auto kernel_2_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 512, 128, 1, 1>(getWeight<T>(weightIndex::layer2_2_conv3_weight));
            auto batchNorm_2_2 = ImageInference::types::BatchNorm<T, 512>(getWeight<T>(weightIndex::layer2_2_bn3_weight), getWeight<T>(weightIndex::layer2_2_bn3_bias));
            auto image_2_2 = convBlockAddIdentity<0>(image_2_1, kernel_2_2, batchNorm_2_2, image_1_2);

            auto kernel_3_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 128, 512, 1, 1>(getWeight<T>(weightIndex::layer2_3_conv1_weight));
            auto batchNorm_3_0 = ImageInference::types::BatchNorm<T, 128>(getWeight<T>(weightIndex::layer2_3_bn1_weight), getWeight<T>(weightIndex::layer2_3_bn1_bias));
            auto image_3_0 = convBlock<1, 1>(image_2_2, kernel_3_0, batchNorm_3_0);
            auto kernel_3_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 128, 128, 3, 3>(getWeight<T>(weightIndex::layer2_3_conv2_weight));
            auto batchNorm_3_1 = ImageInference::types::BatchNorm<T, 128>(getWeight<T>(weightIndex::layer2_3_bn2_weight), getWeight<T>(weightIndex::layer2_3_bn2_bias));
            auto image_3_1 = convBlock<1, 0>(image_3_0, kernel_3_1, batchNorm_3_1);
            auto kernel_3_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 512, 128, 1, 1>(getWeight<T>(weightIndex::layer2_3_conv3_weight));
            auto batchNorm_3_2 = ImageInference::types::BatchNorm<T, 512>(getWeight<T>(weightIndex::layer2_3_bn3_weight), getWeight<T>(weightIndex::layer2_3_bn3_bias));
            auto image_3_2 = convBlockAddIdentity<0>(image_3_1, kernel_3_2, batchNorm_3_2, image_2_2);

            return image_3_2;
        }

        template <typename T, size_t BlockSize>
        ImageInference::types::Image<T, 0, BlockSize, 1024, 15, 15> ResNet50::block2(ImageInference::types::Image<T, 0, BlockSize, 512, 30, 30> &input)
        {
            auto kernel_0_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 512, 1, 1>(getWeight<T>(weightIndex::layer3_0_conv1_weight));
            auto batchNorm_0_0 = ImageInference::types::BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_0_bn1_weight), getWeight<T>(weightIndex::layer3_0_bn1_bias));
            auto image_0_0 = convBlock<1, 1>(input, kernel_0_0, batchNorm_0_0);
            auto kernel_0_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 256, 3, 3>(getWeight<T>(weightIndex::layer3_0_conv2_weight));
            auto batchNorm_0_1 = ImageInference::types::BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_0_bn2_weight), getWeight<T>(weightIndex::layer3_0_bn2_bias));
            auto image_0_1 = convBlock<2, 0>(image_0_0, kernel_0_1, batchNorm_0_1);
            auto kernel_0_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 1024, 256, 1, 1>(getWeight<T>(weightIndex::layer3_0_conv3_weight));
            auto batchNorm_0_2 = ImageInference::types::BatchNorm<T, 1024>(getWeight<T>(weightIndex::layer3_0_bn3_weight), getWeight<T>(weightIndex::layer3_0_bn3_bias));
            auto projectionKernel = ImageInference::types::Kernel<T, BlockSize, BlockSize, 1024, 512, 1, 1>(getWeight<T>(weightIndex::layer3_0_downsample_0_weight));
            auto projectionBatchNorm = ImageInference::types::BatchNorm<T, 1024>(getWeight<T>(weightIndex::layer3_0_downsample_1_weight), getWeight<T>(weightIndex::layer3_0_downsample_1_bias));
            auto image_0_2 = convBlockAddProjection<2, 2, 0>(image_0_1, kernel_0_2, batchNorm_0_2, input, projectionKernel, projectionBatchNorm);

            auto kernel_1_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 1024, 1, 1>(getWeight<T>(weightIndex::layer3_1_conv1_weight));
            auto batchNorm_1_0 = ImageInference::types::BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_1_bn1_weight), getWeight<T>(weightIndex::layer3_1_bn1_bias));
            auto image_1_0 = convBlock<1, 1>(image_0_2, kernel_1_0, batchNorm_1_0);
            auto kernel_1_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 256, 3, 3>(getWeight<T>(weightIndex::layer3_1_conv2_weight));
            auto batchNorm_1_1 = ImageInference::types::BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_1_bn2_weight), getWeight<T>(weightIndex::layer3_1_bn2_bias));
            auto image_1_1 = convBlock<1, 0>(image_1_0, kernel_1_1, batchNorm_1_1);
            auto kernel_1_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 1024, 256, 1, 1>(getWeight<T>(weightIndex::layer3_1_conv3_weight));
            auto batchNorm_1_2 = ImageInference::types::BatchNorm<T, 1024>(getWeight<T>(weightIndex::layer3_1_bn3_weight), getWeight<T>(weightIndex::layer3_1_bn3_bias));
            auto image_1_2 = convBlockAddIdentity<0>(image_1_1, kernel_1_2, batchNorm_1_2, image_0_2);

            auto kernel_2_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 1024, 1, 1>(getWeight<T>(weightIndex::layer3_2_conv1_weight));
            auto batchNorm_2_0 = ImageInference::types::BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_2_bn1_weight), getWeight<T>(weightIndex::layer3_2_bn1_bias));
            auto image_2_0 = convBlock<1, 1>(image_1_2, kernel_2_0, batchNorm_2_0);
            auto kernel_2_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 256, 3, 3>(getWeight<T>(weightIndex::layer3_2_conv2_weight));
            auto batchNorm_2_1 = ImageInference::types::BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_2_bn2_weight), getWeight<T>(weightIndex::layer3_2_bn2_bias));
            auto image_2_1 = convBlock<1, 0>(image_2_0, kernel_2_1, batchNorm_2_1);
            auto kernel_2_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 1024, 256, 1, 1>(getWeight<T>(weightIndex::layer3_2_conv3_weight));
            auto batchNorm_2_2 = ImageInference::types::BatchNorm<T, 1024>(getWeight<T>(weightIndex::layer3_2_bn3_weight), getWeight<T>(weightIndex::layer3_2_bn3_bias));
            auto image_2_2 = convBlockAddIdentity<0>(image_2_1, kernel_2_2, batchNorm_2_2, image_1_2);

            auto kernel_3_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 1024, 1, 1>(getWeight<T>(weightIndex::layer3_3_conv1_weight));
            auto batchNorm_3_0 = ImageInference::types::BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_3_bn1_weight), getWeight<T>(weightIndex::layer3_3_bn1_bias));
            auto image_3_0 = convBlock<1, 1>(image_2_2, kernel_3_0, batchNorm_3_0);
            auto kernel_3_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 256, 3, 3>(getWeight<T>(weightIndex::layer3_3_conv2_weight));
            auto batchNorm_3_1 = ImageInference::types::BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_3_bn2_weight), getWeight<T>(weightIndex::layer3_3_bn2_bias));
            auto image_3_1 = convBlock<1, 0>(image_3_0, kernel_3_1, batchNorm_3_1);
            auto kernel_3_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 1024, 256, 1, 1>(getWeight<T>(weightIndex::layer3_3_conv3_weight));
            auto batchNorm_3_2 = ImageInference::types::BatchNorm<T, 1024>(getWeight<T>(weightIndex::layer3_3_bn3_weight), getWeight<T>(weightIndex::layer3_3_bn3_bias));
            auto image_3_2 = convBlockAddIdentity<0>(image_3_1, kernel_3_2, batchNorm_3_2, image_2_2);

            auto kernel_4_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 1024, 1, 1>(getWeight<T>(weightIndex::layer3_4_conv1_weight));
            auto batchNorm_4_0 = ImageInference::types::BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_4_bn1_weight), getWeight<T>(weightIndex::layer3_4_bn1_bias));
            auto image_4_0 = convBlock<1, 1>(image_3_2, kernel_4_0, batchNorm_4_0);
            auto kernel_4_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 256, 3, 3>(getWeight<T>(weightIndex::layer3_4_conv2_weight));
            auto batchNorm_4_1 = ImageInference::types::BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_4_bn2_weight), getWeight<T>(weightIndex::layer3_4_bn2_bias));
            auto image_4_1 = convBlock<1, 0>(image_4_0, kernel_4_1, batchNorm_4_1);
            auto kernel_4_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 1024, 256, 1, 1>(getWeight<T>(weightIndex::layer3_4_conv3_weight));
            auto batchNorm_4_2 = ImageInference::types::BatchNorm<T, 1024>(getWeight<T>(weightIndex::layer3_4_bn3_weight), getWeight<T>(weightIndex::layer3_4_bn3_bias));
            auto image_4_2 = convBlockAddIdentity<0>(image_4_1, kernel_4_2, batchNorm_4_2, image_3_2);

            auto kernel_5_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 1024, 1, 1>(getWeight<T>(weightIndex::layer3_5_conv1_weight));
            auto batchNorm_5_0 = ImageInference::types::BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_5_bn1_weight), getWeight<T>(weightIndex::layer3_5_bn1_bias));
            auto image_5_0 = convBlock<1, 1>(image_4_2, kernel_5_0, batchNorm_5_0);
            auto kernel_5_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 256, 3, 3>(getWeight<T>(weightIndex::layer3_5_conv2_weight));
            auto batchNorm_5_1 = ImageInference::types::BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_5_bn2_weight), getWeight<T>(weightIndex::layer3_5_bn2_bias));
            auto image_5_1 = convBlock<1, 0>(image_5_0, kernel_5_1, batchNorm_5_1);
            auto kernel_5_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 1024, 256, 1, 1>(getWeight<T>(weightIndex::layer3_5_conv3_weight));
            auto batchNorm_5_2 = ImageInference::types::BatchNorm<T, 1024>(getWeight<T>(weightIndex::layer3_5_bn3_weight), getWeight<T>(weightIndex::layer3_5_bn3_bias));
            auto image_5_2 = convBlockAddIdentity<0>(image_5_1, kernel_5_2, batchNorm_5_2, image_4_2);

            return image_5_2;
        }

        template <typename T, size_t BlockSize>
        ImageInference::types::Image<T, 0, BlockSize, 2048, 7, 7> ResNet50::block3(ImageInference::types::Image<T, 0, BlockSize, 1024, 15, 15> &input)
        {
            auto kernel_0_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 512, 1024, 1, 1>(getWeight<T>(weightIndex::layer4_0_conv1_weight));
            auto batchNorm_0_0 = ImageInference::types::BatchNorm<T, 512>(getWeight<T>(weightIndex::layer4_0_bn1_weight), getWeight<T>(weightIndex::layer4_0_bn1_bias));
            auto image_0_0 = convBlock<1, 1>(input, kernel_0_0, batchNorm_0_0);
            auto kernel_0_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 512, 512, 3, 3>(getWeight<T>(weightIndex::layer4_0_conv2_weight));
            auto batchNorm_0_1 = ImageInference::types::BatchNorm<T, 512>(getWeight<T>(weightIndex::layer4_0_bn2_weight), getWeight<T>(weightIndex::layer4_0_bn2_bias));
            auto image_0_1 = convBlock<2, 0>(image_0_0, kernel_0_1, batchNorm_0_1);
            auto kernel_0_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 2048, 512, 1, 1>(getWeight<T>(weightIndex::layer4_0_conv3_weight));
            auto batchNorm_0_2 = ImageInference::types::BatchNorm<T, 2048>(getWeight<T>(weightIndex::layer4_0_bn3_weight), getWeight<T>(weightIndex::layer4_0_bn3_bias));
            auto projectionKernel = ImageInference::types::Kernel<T, BlockSize, BlockSize, 2048, 1024, 1, 1>(getWeight<T>(weightIndex::layer4_0_downsample_0_weight));
            auto projectionBatchNorm = ImageInference::types::BatchNorm<T, 2048>(getWeight<T>(weightIndex::layer4_0_downsample_1_weight), getWeight<T>(weightIndex::layer4_0_downsample_1_bias));
            auto image_0_2 = convBlockAddProjection<2, 2, 0>(image_0_1, kernel_0_2, batchNorm_0_2, input, projectionKernel, projectionBatchNorm);

            auto kernel_1_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 512, 2048, 1, 1>(getWeight<T>(weightIndex::layer4_1_conv1_weight));
            auto batchNorm_1_0 = ImageInference::types::BatchNorm<T, 512>(getWeight<T>(weightIndex::layer4_1_bn1_weight), getWeight<T>(weightIndex::layer4_1_bn1_bias));
            auto image_1_0 = convBlock<1, 1>(image_0_2, kernel_1_0, batchNorm_1_0);
            auto kernel_1_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 512, 512, 3, 3>(getWeight<T>(weightIndex::layer4_1_conv2_weight));
            auto batchNorm_1_1 = ImageInference::types::BatchNorm<T, 512>(getWeight<T>(weightIndex::layer4_1_bn2_weight), getWeight<T>(weightIndex::layer4_1_bn2_bias));
            auto image_1_1 = convBlock<1, 0>(image_1_0, kernel_1_1, batchNorm_1_1);
            auto kernel_1_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 2048, 512, 1, 1>(getWeight<T>(weightIndex::layer4_1_conv3_weight));
            auto batchNorm_1_2 = ImageInference::types::BatchNorm<T, 2048>(getWeight<T>(weightIndex::layer4_1_bn3_weight), getWeight<T>(weightIndex::layer4_1_bn3_bias));
            auto image_1_2 = convBlockAddIdentity<0>(image_1_1, kernel_1_2, batchNorm_1_2, image_0_2);

            auto kernel_2_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 512, 2048, 1, 1>(getWeight<T>(weightIndex::layer4_2_conv1_weight));
            auto batchNorm_2_0 = ImageInference::types::BatchNorm<T, 512>(getWeight<T>(weightIndex::layer4_2_bn1_weight), getWeight<T>(weightIndex::layer4_2_bn1_bias));
            auto image_2_0 = convBlock<1, 1>(image_1_2, kernel_2_0, batchNorm_2_0);
            auto kernel_2_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 512, 512, 3, 3>(getWeight<T>(weightIndex::layer4_2_conv2_weight));
            auto batchNorm_2_1 = ImageInference::types::BatchNorm<T, 512>(getWeight<T>(weightIndex::layer4_2_bn2_weight), getWeight<T>(weightIndex::layer4_2_bn2_bias));
            auto image_2_1 = convBlock<1, 0>(image_2_0, kernel_2_1, batchNorm_2_1);
            auto kernel_2_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 2048, 512, 1, 1>(getWeight<T>(weightIndex::layer4_2_conv3_weight));
            auto batchNorm_2_2 = ImageInference::types::BatchNorm<T, 2048>(getWeight<T>(weightIndex::layer4_2_bn3_weight), getWeight<T>(weightIndex::layer4_2_bn3_bias));
            auto image_2_2 = convBlockAddIdentity<0>(image_2_1, kernel_2_2, batchNorm_2_2, image_1_2);

            return image_2_2;
        }

        template <size_t Stride, size_t OutPadding, size_t InPadding,
                  typename T, size_t BlockSizeCount, size_t BlockSizeChannel,
                  size_t ImageChannels, size_t ImageHeight, size_t ImageWidth,
                  size_t KernelCount, size_t KernelHeight, size_t KernelWidth>
        inline ImageInference::types::Image<T, OutPadding, BlockSizeCount, KernelCount, ImageHeight / Stride, ImageWidth / Stride> ResNet50::convBlock(
            ImageInference::types::Image<T, InPadding, BlockSizeChannel, ImageChannels, ImageHeight, ImageWidth> &image,
            ImageInference::types::Kernel<T, BlockSizeCount, BlockSizeChannel, KernelCount, ImageChannels, KernelHeight, KernelWidth> &kernel,
            ImageInference::types::BatchNorm<T, KernelCount> &batchNorm)
        {
            if constexpr (InPadding != KernelHeight / 2 || InPadding != KernelWidth / 2)
            {
                std::cerr << "Padding is too small or to large for the kernel size. Padding: " << InPadding
                          << " KernelHeight: " << KernelHeight << " KernelWidth: " << KernelWidth << std::endl
                          << "Should be KernelHeight / 2 or KernelWidth / 2 = Padding." << std::endl;
                throw std::runtime_error("Padding is too small or to large for the kernel size!");
            }

            constexpr size_t countBlocks = KernelCount / BlockSizeCount;
            constexpr size_t channelBlocks = ImageChannels / BlockSizeChannel;
            constexpr size_t outputHeight = ImageHeight / Stride;
            constexpr size_t outputWidth = ImageWidth / Stride;

            ImageInference::types::Image<T, OutPadding, BlockSizeCount, KernelCount, ImageHeight / Stride, ImageWidth / Stride> output;
            auto outputPtr = output.getPointer() + output.paddingOffset; // We skip the padding as we want to start at the data section.
            auto meanPtr = output.getMeanPointer();                      // Count = CountBlocks x CountElements
            auto variancePtr = output.getBatchVariancePointer();         // Count = CountBlocks x CountElements

            auto imagePtr = image.getPointer();          // ChannelBlocks x Height x Width x ChannelElements
            auto kernelPtr = kernel.getPointer();        // CountBlocks x ChannelBlocks x Height x Width x ChannelElements x CountElements
            auto gammaPtr = batchNorm.getGammaPointer(); // Count = CountBlocks x CountElements
            auto betaPtr = batchNorm.getBetaPointer();   // Count = CountBlocks x CountElements

            size_t meanVarianceCount[KernelCount]{0};

            for (size_t iBCount = 0; iBCount < countBlocks; iBCount++)
            {
                for (size_t iHeight = 0; iHeight < outputHeight; iHeight++)
                {
                    // Do convolution calculation
                    for (size_t iBChannel = 0; iBChannel < channelBlocks; iBChannel++)
                    {
                        for (size_t kHeight = 0; kHeight < KernelHeight; kHeight++)
                        {
                            for (size_t kWidth = 0; kWidth < KernelWidth; kWidth++)
                            {
                                size_t inputOffset = image.getOffset(iBChannel, iHeight * Stride + kHeight, kWidth, 0);
                                size_t kernelOffset = kernel.getOffset(iBCount, iBChannel, kHeight, kWidth, 0, 0);
                                size_t outputOffset = output.getOffset(iBCount, iHeight, 0, 0);

                                // Kernel of shape BlockSizeChannel x BlockSizeCount
                                // Input of shape ImageWidth x BlockSizeChannel
                                // Output of shape outputWidth x BlockSizeCount === ImageWidth / Stride x BlockSizeCount

                                // If we use libxsmm directly we don't need to do add separately!
                                // If we use the leading dimension on the image we can use it as stride.
                                // With the leading dimension we skip the next blocks as they should be skipped by the stride.
                                constexpr int MM = outputWidth;
                                constexpr int KK = BlockSizeChannel;
                                constexpr int NN = BlockSizeCount;
                                constexpr int ldImage = NN * Stride;
                                constexpr float alpha = 1.0;
                                constexpr float beta = 1.0;

                                constexpr char transa = 'N';
                                constexpr char transb = 'N';

                                libxsmm_sgemm(
                                    &transa /*transa*/,
                                    &transb /*transb*/,
                                    &NN /*required*/,
                                    &MM /*required*/,
                                    &KK /*required*/,
                                    &alpha /*alpha*/,
                                    kernelPtr + kernelOffset /*required*/,
                                    &NN /*lda*/,
                                    imagePtr + inputOffset /*required*/,
                                    &ldImage /*ldb*/,
                                    &beta /*beta*/,
                                    outputPtr + outputOffset /*required*/,
                                    &NN /*ldc*/
                                );
                            }
                        }
                    }

                    // At this point we completed a complete row of the output.
                    // We will also update the mean and variance as we already loaded the data.
                    for(size_t iWidth = 0; iWidth < outputWidth; iWidth++)
                    {
                        for (size_t iCount = 0; iCount < BlockSizeCount; iCount++)
                        {
                            size_t offsetOutput = output.getOffset(iBCount, iHeight, iWidth, iCount);
                            size_t offsetCount = iBCount * BlockSizeCount + iCount;
                            output.updateMeanVariance(outputPtr[offsetOutput], offsetCount, ++meanVarianceCount[offsetCount]);
                        }
                    }
                }
            }

            for (size_t iBCount = 0; iBCount < countBlocks; iBCount++)
            {
                for (size_t iCount = 0; iCount < BlockSizeCount; iCount++)
                {
                    size_t offsetCount = iBCount * BlockSizeCount + iCount;
                    output.finalizeMeanVariance(offsetCount, meanVarianceCount[offsetCount]);
                }

                // Now we apply the batch norm and relu.
                for (size_t iHeight = 0; iHeight < outputHeight; iHeight++)
                {
                    for(size_t iWidth = 0; iWidth < outputWidth; iWidth++)
                    {
                        for (size_t iCount = 0; iCount < BlockSizeCount; iCount++)
                        {
                            size_t offsetOutput = output.getOffset(iBCount, iHeight, iWidth, iCount);
                            size_t offsetCount = iBCount * BlockSizeCount + iCount;
                            outputPtr[offsetOutput] = relu<T>(ResNet50::batchNorm<T>(
                                outputPtr[offsetOutput],
                                gammaPtr[offsetCount],
                                betaPtr[offsetCount],
                                meanPtr[offsetCount],
                                variancePtr[offsetCount]));
                        }
                    }
                }
            }

            return output;
        }

        template <size_t Stride, size_t OutPadding, size_t InPadding,
                  typename T, size_t BlockSize,
                  size_t ImageChannels, size_t ImageHeight, size_t ImageWidth>
        inline ImageInference::types::Image<T, OutPadding, BlockSize, ImageChannels, ImageHeight / Stride, ImageWidth / Stride> ResNet50::maxPool(
            ImageInference::types::Image<T, InPadding, BlockSize, ImageChannels, ImageHeight, ImageWidth> &image)
        {
            if constexpr (InPadding != 1)
            {
                std::cerr << "Padding is too small or to large for the kernel size. Padding is " << InPadding
                          << " but should be 1." << std::endl;
                throw std::runtime_error("Padding is too small or to large for 3x3 Max Pooling!");
            }

            constexpr size_t channelBlocks = ImageChannels / BlockSize;
            constexpr size_t outputHeight = ImageHeight / Stride;
            constexpr size_t outputWidth = ImageWidth / Stride;

            ImageInference::types::Image<T, OutPadding, BlockSize, ImageChannels, ImageHeight / Stride, ImageWidth / Stride> output;

            auto outputPtr = output.getPointer() + output.paddingOffset; // We skip the padding as we want to start at the data section.

            auto imagePtr = image.getPointer();

            // 3x3 Stencil that gets the max value
            for (size_t iBChannel = 0; iBChannel < channelBlocks; iBChannel++)
            {
                for (size_t iHeight = 0; iHeight < outputHeight; iHeight++)
                {
                    for (size_t iWidth = 0; iWidth < outputWidth; iWidth++)
                    {
                        size_t preOffsetOutput = output.getOffset(iBChannel, iHeight, iWidth, 0);

                        for (size_t iChannel = 0; iChannel < BlockSize; iChannel++)
                        {
                            auto offsetOutput = preOffsetOutput + iChannel * output.strideChannel;
                            outputPtr[offsetOutput] = std::numeric_limits<T>::lowest();
                        }

                        for (size_t kHeight = 0; kHeight < 3; kHeight++)
                        {
                            for (size_t kWidth = 0; kWidth < 3; kWidth++)
                            {
                                for (size_t iChannel = 0; iChannel < BlockSize; iChannel++)
                                {
                                    auto offsetOutput = preOffsetOutput + iChannel * output.strideChannel;
                                    auto offsetImage = image.getOffset(iBChannel, iHeight * Stride + kHeight, iWidth * Stride + kWidth, iChannel);
                                    outputPtr[offsetOutput] = std::max(outputPtr[offsetOutput], imagePtr[offsetImage]);
                                }
                            }
                        }
                    }
                }
            }

            return output;
        }

        template <size_t OutPadding, size_t InPadding, typename T, size_t BlockSize,
                  size_t ImageChannels, size_t ImageHeight, size_t ImageWidth>
        inline ImageInference::types::Image<T, OutPadding, BlockSize, ImageChannels, 1, 1> ResNet50::globalAveragePool(
            ImageInference::types::Image<T, InPadding, BlockSize, ImageChannels, ImageHeight, ImageWidth> &image)
        {
            constexpr size_t channelBlocks = ImageChannels / BlockSize;

            ImageInference::types::Image<T, OutPadding, BlockSize, ImageChannels, 1, 1> output;
            auto outputPtr = output.getPointer() + output.paddingOffset; // We skip the padding as we want to start at the data section.

            auto imagePtr = image.getPointer() + image.paddingOffset; // We skip the padding as padding should not be averaged.
            constexpr float scale = 1.0f / (ImageHeight * ImageWidth);
#ifdef USE_OMP
#pragma omp parallel for
#endif // USE_OMP
            for (size_t iBChannel = 0; iBChannel < channelBlocks; iBChannel++)
            {
                for (size_t iHeight = 0; iHeight < ImageHeight; iHeight++)
                {
                    for (size_t iWidth = 0; iWidth < ImageWidth; iWidth++)
                    {
                        for (size_t iChannel = 0; iChannel < BlockSize; iChannel++)
                        {
                            auto offsetOutput = output.getOffset(iBChannel, 0, 0, iChannel);
                            auto offsetImage = image.getOffset(iBChannel, iHeight, iWidth, iChannel);
                            outputPtr[offsetOutput] += imagePtr[offsetImage];
                        }
                    }
                }

                for (size_t iChannel = 0; iChannel < BlockSize; iChannel++)
                {
                    auto offsetOutput = output.getOffset(iBChannel, 0, 0, iChannel);
                    outputPtr[offsetOutput] = static_cast<T>(outputPtr[offsetOutput] * scale);
                }
            }

            return output;
        }

        template <typename T, size_t Columns, size_t Rows>
        void ResNet50::fullyConnectedLayer(
            ImageInference::types::Array<T, Rows> &input,
            ImageInference::types::Matrix<T, Columns, Rows> &weight,
            ImageInference::types::Array<T, Columns> &biasAccumulator)
        {
            auto inputPtr = input.getPointer();
            auto weightPtr = weight.getPointer();
            auto biasPtr = biasAccumulator.getPointer();

            Fastor::TensorMap<T, Rows> inputMap(inputPtr);
            Fastor::TensorMap<T, Columns, Rows> weightMap(weightPtr);
            Fastor::TensorMap<T, Columns> biasMap(biasPtr);
            biasMap += Fastor::matmul(weightMap, inputMap);
        }

        template <typename T>
        inline T *ResNet50::getWeight(size_t index)
        {
            return static_cast<T *>(modelWeights[index]);
        }

        template <typename T>
        inline T ResNet50::relu(T value)
        {
            return (value > 0) * value;
        }

        template <typename T>
        inline T ResNet50::batchNorm(T value, T gamma, T beta, T mean, T batchVariance)
        {
            return gamma * (value - mean) * batchVariance + beta;
        }
    } // namespace model
} // namespace ImageInference

#endif // IMAGEINFERENCE_RESNET50_H
