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

#define MAX_RESNET50_SIZE 122 * 122 * 64

namespace ImageInference
{
    namespace model
    {
        using types::Array;
        using types::BatchNorm;
        using types::Image;
        using types::Kernel;
        using types::Matrix;
        using types::ScalarType;

        /// @brief The resnet50 v1.5 model from https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch
        class ResNet50 : public IModel<float>, public IModel<int8_t>
        {
        private:
            ScalarType type;
            std::vector<void *> weights;
            void *inputBuffer;
            void *outputBuffer;
            void *shortcutBuffer;

            template <typename T>
            Image<T, 256, 61, 61> block0(Image<T, 64, 61, 61> &input);

            template <typename T>
            Image<T, 512, 30, 30> block1(Image<T, 256, 61, 61> &input);

            template <typename T>
            Image<T, 1024, 15, 15> block2(Image<T, 512, 30, 30> &input);

            template <typename T>
            Image<T, 2048, 7, 7> block3(Image<T, 1024, 15, 15> &input);

            template <size_t Stride, typename T,
                      size_t ImageChannels, size_t ImageHeight, size_t ImageWidth,
                      size_t KernelCount, size_t KernelHeight, size_t KernelWidth>
            Image<T, KernelCount, ImageHeight / Stride, ImageWidth / Stride> ConvBlock(
                Image<T, ImageChannels, ImageHeight, ImageWidth> image,
                Kernel<T, KernelCount, ImageChannels, KernelHeight, KernelWidth> kernel,
                BatchNorm<T, KernelCount> batchNorm);

            template <typename T,
                      size_t ImageChannels, size_t ImageHeight, size_t ImageWidth,
                      size_t KernelCount, size_t KernelHeight, size_t KernelWidth>
            Image<T, KernelCount, ImageHeight, ImageWidth> ConvBlockAddIdentity(
                Image<T, ImageChannels, ImageHeight, ImageWidth> image,
                Kernel<T, KernelCount, ImageChannels, KernelHeight, KernelWidth> kernel,
                BatchNorm<T, KernelCount> batchNorm,
                Image<T, KernelCount, ImageHeight, ImageWidth> shortcut);

            template <size_t Stride, size_t ShortcutDimExpand, typename T,
                      size_t ImageChannels, size_t ImageHeight, size_t ImageWidth,
                      size_t KernelCount, size_t KernelHeight, size_t KernelWidth>
            Image<T, KernelCount, ImageHeight / Stride, ImageWidth / Stride> ConvBlockAddProjection(
                Image<T, ImageChannels, ImageHeight / Stride, ImageWidth / Stride> image,
                Kernel<T, KernelCount, ImageChannels, KernelHeight, KernelWidth> kernel,
                BatchNorm<T, KernelCount> batchNorm,
                Image<T, KernelCount / ShortcutDimExpand, ImageHeight, ImageWidth> shortcut,
                Kernel<T, KernelCount, KernelCount / ShortcutDimExpand, 1, 1> projectionKernel,
                BatchNorm<T, KernelCount> projectionBatchNorm);

            template <size_t Stride, typename T,
                      size_t ImageChannels, size_t ImageHeight, size_t ImageWidth>
            Image<T, ImageChannels, ImageHeight / Stride, ImageWidth / Stride> MaxPool(
                Image<T, ImageChannels, ImageHeight, ImageWidth> image);

            template <typename T,
                      size_t ImageChannels, size_t ImageHeight, size_t ImageWidth>
            Image<T, ImageChannels, 1, 1> GlobalAveragePool(Image<T, ImageChannels, ImageHeight, ImageWidth> image);

            template <typename T>
            Array<T, 1000> fullyConnectedLayer(
                Array<T, 2048> input,
                Matrix<T, 1000, 2048> weights,
                Array<T, 1000> bias);

            template <typename T>
            T *getWeight(size_t index);

            template <typename T>
            T relu(T value);

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
            ResNet50(const std::vector<void *> &weights, ScalarType type);
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

            const float *inference(const float *input) override;
            const int8_t *inference(const int8_t *input) override;
            ScalarType getType();
        };

        template <typename T>
        inline Image<T, 256, 61, 61> ResNet50::block0(Image<T, 64, 61, 61> &input)
        {
            std::swap(inputBuffer, shortcutBuffer); // buffer swap so that the input is not overwritten by convolution block as is needed for the shortcut
            auto kernel_0_0 = Kernel<T, 64, 64, 1, 1>(getWeight<T>(weightIndex::layer1_0_conv1_weight));
            auto batchNorm_0_0 = BatchNorm<T, 64>(getWeight<T>(weightIndex::layer1_0_bn1_weight), getWeight<T>(weightIndex::layer1_0_bn1_bias));
            auto image_0_0 = ConvBlock<1>(input, kernel_0_0, batchNorm_0_0);
            auto kernel_0_1 = Kernel<T, 64, 64, 3, 3>(getWeight<T>(weightIndex::layer1_0_conv2_weight));
            auto batchNorm_0_1 = BatchNorm<T, 64>(getWeight<T>(weightIndex::layer1_0_bn2_weight), getWeight<T>(weightIndex::layer1_0_bn2_bias));
            auto image_0_1 = ConvBlock<1>(image_0_0, kernel_0_1, batchNorm_0_1);
            auto kernel_0_2 = Kernel<T, 256, 64, 1, 1>(getWeight<T>(weightIndex::layer1_0_conv3_weight));
            auto batchNorm_0_2 = BatchNorm<T, 256>(getWeight<T>(weightIndex::layer1_0_bn3_weight), getWeight<T>(weightIndex::layer1_0_bn3_bias));
            auto projectionKernel = Kernel<T, 256, 64, 1, 1>(getWeight<T>(weightIndex::layer1_0_downsample_0_weight));
            auto projectionBatchNorm = BatchNorm<T, 256>(getWeight<T>(weightIndex::layer1_0_downsample_1_weight), getWeight<T>(weightIndex::layer1_0_downsample_1_bias));
            auto image_0_2 = ConvBlockAddProjection<1, 4>(image_0_1, kernel_0_2, batchNorm_0_2, input, projectionKernel, projectionBatchNorm);

            std::swap(inputBuffer, shortcutBuffer); // buffer swap so that the image_0_2 is not overwritten by convolution block as is needed for the shortcut
            auto kernel_1_0 = Kernel<T, 64, 256, 1, 1>(getWeight<T>(weightIndex::layer1_1_conv1_weight));
            auto batchNorm_1_0 = BatchNorm<T, 64>(getWeight<T>(weightIndex::layer1_1_bn1_weight), getWeight<T>(weightIndex::layer1_1_bn1_bias));
            auto image_1_0 = ConvBlock<1>(image_0_2, kernel_1_0, batchNorm_1_0);
            auto kernel_1_1 = Kernel<T, 64, 64, 3, 3>(getWeight<T>(weightIndex::layer1_1_conv2_weight));
            auto batchNorm_1_1 = BatchNorm<T, 64>(getWeight<T>(weightIndex::layer1_1_bn2_weight), getWeight<T>(weightIndex::layer1_1_bn2_bias));
            auto image_1_1 = ConvBlock<1>(image_1_0, kernel_1_1, batchNorm_1_1);
            auto kernel_1_2 = Kernel<T, 256, 64, 1, 1>(getWeight<T>(weightIndex::layer1_1_conv3_weight));
            auto batchNorm_1_2 = BatchNorm<T, 256>(getWeight<T>(weightIndex::layer1_1_bn3_weight), getWeight<T>(weightIndex::layer1_1_bn3_bias));
            auto image_1_2 = ConvBlockAddIdentity(image_1_1, kernel_1_2, batchNorm_1_2, image_0_2);

            std::swap(inputBuffer, shortcutBuffer); // buffer swap so that the image_1_2 is not overwritten by convolution block as is needed for the shortcut
            auto kernel_2_0 = Kernel<T, 64, 256, 1, 1>(getWeight<T>(weightIndex::layer1_2_conv1_weight));
            auto batchNorm_2_0 = BatchNorm<T, 64>(getWeight<T>(weightIndex::layer1_2_bn1_weight), getWeight<T>(weightIndex::layer1_2_bn1_bias));
            auto image_2_0 = ConvBlock<1>(image_1_2, kernel_2_0, batchNorm_2_0);
            auto kernel_2_1 = Kernel<T, 64, 64, 3, 3>(getWeight<T>(weightIndex::layer1_2_conv2_weight));
            auto batchNorm_2_1 = BatchNorm<T, 64>(getWeight<T>(weightIndex::layer1_2_bn2_weight), getWeight<T>(weightIndex::layer1_2_bn2_bias));
            auto image_2_1 = ConvBlock<1>(image_2_0, kernel_2_1, batchNorm_2_1);
            auto kernel_2_2 = Kernel<T, 256, 64, 1, 1>(getWeight<T>(weightIndex::layer1_2_conv3_weight));
            auto batchNorm_2_2 = BatchNorm<T, 256>(getWeight<T>(weightIndex::layer1_2_bn3_weight), getWeight<T>(weightIndex::layer1_2_bn3_bias));
            auto image_2_2 = ConvBlockAddIdentity(image_2_1, kernel_2_2, batchNorm_2_2, image_1_2);

            return image_2_2;
        }

        template <typename T>
        Image<T, 512, 30, 30> ResNet50::block1(Image<T, 256, 61, 61> &input)
        {
            std::swap(inputBuffer, shortcutBuffer); // buffer swap so that the input is not overwritten by convolution block as is needed for the shortcut
            auto kernel_0_0 = Kernel<T, 128, 256, 1, 1>(getWeight<T>(weightIndex::layer2_0_conv1_weight));
            auto batchNorm_0_0 = BatchNorm<T, 128>(getWeight<T>(weightIndex::layer2_0_bn1_weight), getWeight<T>(weightIndex::layer2_0_bn1_bias));
            auto image_0_0 = ConvBlock<1>(input, kernel_0_0, batchNorm_0_0);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_0_1 = Kernel<T, 128, 128, 3, 3>(getWeight<T>(weightIndex::layer2_0_conv2_weight));
            auto batchNorm_0_1 = BatchNorm<T, 128>(getWeight<T>(weightIndex::layer2_0_bn2_weight), getWeight<T>(weightIndex::layer2_0_bn2_bias));
            auto image_0_1 = ConvBlock<2>(image_0_0, kernel_0_1, batchNorm_0_1);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_0_2 = Kernel<T, 512, 128, 1, 1>(getWeight<T>(weightIndex::layer2_0_conv3_weight));
            auto batchNorm_0_2 = BatchNorm<T, 512>(getWeight<T>(weightIndex::layer2_0_bn3_weight), getWeight<T>(weightIndex::layer2_0_bn3_bias));
            auto projectionKernel = Kernel<T, 512, 256, 1, 1>(getWeight<T>(weightIndex::layer2_0_downsample_0_weight));
            auto projectionBatchNorm = BatchNorm<T, 512>(getWeight<T>(weightIndex::layer2_0_downsample_1_weight), getWeight<T>(weightIndex::layer2_0_downsample_1_bias));
            auto image_0_2 = ConvBlockAddProjection<2, 2>(image_0_1, kernel_0_2, batchNorm_0_2, input, projectionKernel, projectionBatchNorm);
            std::swap(inputBuffer, outputBuffer);

            std::swap(inputBuffer, shortcutBuffer); // buffer swap so that the image_0_2 is not overwritten by convolution block as is needed for the shortcut
            auto kernel_1_0 = Kernel<T, 128, 512, 1, 1>(getWeight<T>(weightIndex::layer2_1_conv1_weight));
            auto batchNorm_1_0 = BatchNorm<T, 128>(getWeight<T>(weightIndex::layer2_1_bn1_weight), getWeight<T>(weightIndex::layer2_1_bn1_bias));
            auto image_1_0 = ConvBlock<1>(image_0_2, kernel_1_0, batchNorm_1_0);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_1_1 = Kernel<T, 128, 128, 3, 3>(getWeight<T>(weightIndex::layer2_1_conv2_weight));
            auto batchNorm_1_1 = BatchNorm<T, 128>(getWeight<T>(weightIndex::layer2_1_bn2_weight), getWeight<T>(weightIndex::layer2_1_bn2_bias));
            auto image_1_1 = ConvBlock<1>(image_1_0, kernel_1_1, batchNorm_1_1);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_1_2 = Kernel<T, 512, 128, 1, 1>(getWeight<T>(weightIndex::layer2_1_conv3_weight));
            auto batchNorm_1_2 = BatchNorm<T, 512>(getWeight<T>(weightIndex::layer2_1_bn3_weight), getWeight<T>(weightIndex::layer2_1_bn3_bias));
            auto image_1_2 = ConvBlockAddIdentity(image_1_1, kernel_1_2, batchNorm_1_2, image_0_2);
            std::swap(inputBuffer, outputBuffer);

            std::swap(inputBuffer, shortcutBuffer); // buffer swap so that the image_1_2 is not overwritten by convolution block as is needed for the shortcut
            auto kernel_2_0 = Kernel<T, 128, 512, 1, 1>(getWeight<T>(weightIndex::layer2_2_conv1_weight));
            auto batchNorm_2_0 = BatchNorm<T, 128>(getWeight<T>(weightIndex::layer2_2_bn1_weight), getWeight<T>(weightIndex::layer2_2_bn1_bias));
            auto image_2_0 = ConvBlock<1>(image_1_2, kernel_2_0, batchNorm_2_0);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_2_1 = Kernel<T, 128, 128, 3, 3>(getWeight<T>(weightIndex::layer2_2_conv2_weight));
            auto batchNorm_2_1 = BatchNorm<T, 128>(getWeight<T>(weightIndex::layer2_2_bn2_weight), getWeight<T>(weightIndex::layer2_2_bn2_bias));
            auto image_2_1 = ConvBlock<1>(image_2_0, kernel_2_1, batchNorm_2_1);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_2_2 = Kernel<T, 512, 128, 1, 1>(getWeight<T>(weightIndex::layer2_2_conv3_weight));
            auto batchNorm_2_2 = BatchNorm<T, 512>(getWeight<T>(weightIndex::layer2_2_bn3_weight), getWeight<T>(weightIndex::layer2_2_bn3_bias));
            auto image_2_2 = ConvBlockAddIdentity(image_2_1, kernel_2_2, batchNorm_2_2, image_1_2);
            std::swap(inputBuffer, outputBuffer);

            std::swap(inputBuffer, shortcutBuffer); // buffer swap so that the image_2_2 is not overwritten by convolution block as is needed for the shortcut
            auto kernel_3_0 = Kernel<T, 128, 512, 1, 1>(getWeight<T>(weightIndex::layer2_3_conv1_weight));
            auto batchNorm_3_0 = BatchNorm<T, 128>(getWeight<T>(weightIndex::layer2_3_bn1_weight), getWeight<T>(weightIndex::layer2_3_bn1_bias));
            auto image_3_0 = ConvBlock<1>(image_2_2, kernel_3_0, batchNorm_3_0);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_3_1 = Kernel<T, 128, 128, 3, 3>(getWeight<T>(weightIndex::layer2_3_conv2_weight));
            auto batchNorm_3_1 = BatchNorm<T, 128>(getWeight<T>(weightIndex::layer2_3_bn2_weight), getWeight<T>(weightIndex::layer2_3_bn2_bias));
            auto image_3_1 = ConvBlock<1>(image_3_0, kernel_3_1, batchNorm_3_1);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_3_2 = Kernel<T, 512, 128, 1, 1>(getWeight<T>(weightIndex::layer2_3_conv3_weight));
            auto batchNorm_3_2 = BatchNorm<T, 512>(getWeight<T>(weightIndex::layer2_3_bn3_weight), getWeight<T>(weightIndex::layer2_3_bn3_bias));
            auto image_3_2 = ConvBlockAddIdentity(image_3_1, kernel_3_2, batchNorm_3_2, image_2_2);
            std::swap(inputBuffer, outputBuffer);

            return image_3_2;
        }

        template <typename T>
        Image<T, 1024, 15, 15> ResNet50::block2(Image<T, 512, 30, 30> &input)
        {
            std::swap(inputBuffer, shortcutBuffer); // buffer swap so that the input is not overwritten by convolution block as is needed for the shortcut
            auto kernel_0_0 = Kernel<T, 256, 512, 1, 1>(getWeight<T>(weightIndex::layer3_0_conv1_weight));
            auto batchNorm_0_0 = BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_0_bn1_weight), getWeight<T>(weightIndex::layer3_0_bn1_bias));
            auto image_0_0 = ConvBlock<1>(input, kernel_0_0, batchNorm_0_0);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_0_1 = Kernel<T, 256, 256, 3, 3>(getWeight<T>(weightIndex::layer3_0_conv2_weight));
            auto batchNorm_0_1 = BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_0_bn2_weight), getWeight<T>(weightIndex::layer3_0_bn2_bias));
            auto image_0_1 = ConvBlock<2>(image_0_0, kernel_0_1, batchNorm_0_1);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_0_2 = Kernel<T, 1024, 256, 1, 1>(getWeight<T>(weightIndex::layer3_0_conv3_weight));
            auto batchNorm_0_2 = BatchNorm<T, 1024>(getWeight<T>(weightIndex::layer3_0_bn3_weight), getWeight<T>(weightIndex::layer3_0_bn3_bias));
            auto projectionKernel = Kernel<T, 1024, 512, 1, 1>(getWeight<T>(weightIndex::layer3_0_downsample_0_weight));
            auto projectionBatchNorm = BatchNorm<T, 1024>(getWeight<T>(weightIndex::layer3_0_downsample_1_weight), getWeight<T>(weightIndex::layer3_0_downsample_1_bias));
            auto image_0_2 = ConvBlockAddProjection<2, 2>(image_0_1, kernel_0_2, batchNorm_0_2, input, projectionKernel, projectionBatchNorm);
            std::swap(inputBuffer, outputBuffer);

            std::swap(inputBuffer, shortcutBuffer); // buffer swap so that the image_0_2 is not overwritten by convolution block as is needed for the shortcut
            auto kernel_1_0 = Kernel<T, 256, 1024, 1, 1>(getWeight<T>(weightIndex::layer3_1_conv1_weight));
            auto batchNorm_1_0 = BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_1_bn1_weight), getWeight<T>(weightIndex::layer3_1_bn1_bias));
            auto image_1_0 = ConvBlock<1>(image_0_2, kernel_1_0, batchNorm_1_0);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_1_1 = Kernel<T, 256, 256, 3, 3>(getWeight<T>(weightIndex::layer3_1_conv2_weight));
            auto batchNorm_1_1 = BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_1_bn2_weight), getWeight<T>(weightIndex::layer3_1_bn2_bias));
            auto image_1_1 = ConvBlock<1>(image_1_0, kernel_1_1, batchNorm_1_1);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_1_2 = Kernel<T, 1024, 256, 1, 1>(getWeight<T>(weightIndex::layer3_1_conv3_weight));
            auto batchNorm_1_2 = BatchNorm<T, 1024>(getWeight<T>(weightIndex::layer3_1_bn3_weight), getWeight<T>(weightIndex::layer3_1_bn3_bias));
            auto image_1_2 = ConvBlockAddIdentity(image_1_1, kernel_1_2, batchNorm_1_2, image_0_2);
            std::swap(inputBuffer, outputBuffer);

            std::swap(inputBuffer, shortcutBuffer); // buffer swap so that the image_1_2 is not overwritten by convolution block as is needed for the shortcut
            auto kernel_2_0 = Kernel<T, 256, 1024, 1, 1>(getWeight<T>(weightIndex::layer3_2_conv1_weight));
            auto batchNorm_2_0 = BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_2_bn1_weight), getWeight<T>(weightIndex::layer3_2_bn1_bias));
            auto image_2_0 = ConvBlock<1>(image_1_2, kernel_2_0, batchNorm_2_0);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_2_1 = Kernel<T, 256, 256, 3, 3>(getWeight<T>(weightIndex::layer3_2_conv2_weight));
            auto batchNorm_2_1 = BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_2_bn2_weight), getWeight<T>(weightIndex::layer3_2_bn2_bias));
            auto image_2_1 = ConvBlock<1>(image_2_0, kernel_2_1, batchNorm_2_1);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_2_2 = Kernel<T, 1024, 256, 1, 1>(getWeight<T>(weightIndex::layer3_2_conv3_weight));
            auto batchNorm_2_2 = BatchNorm<T, 1024>(getWeight<T>(weightIndex::layer3_2_bn3_weight), getWeight<T>(weightIndex::layer3_2_bn3_bias));
            auto image_2_2 = ConvBlockAddIdentity(image_2_1, kernel_2_2, batchNorm_2_2, image_1_2);
            std::swap(inputBuffer, outputBuffer);

            std::swap(inputBuffer, shortcutBuffer); // buffer swap so that the image_2_2 is not overwritten by convolution block as is needed for the shortcut
            auto kernel_3_0 = Kernel<T, 256, 1024, 1, 1>(getWeight<T>(weightIndex::layer3_3_conv1_weight));
            auto batchNorm_3_0 = BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_3_bn1_weight), getWeight<T>(weightIndex::layer3_3_bn1_bias));
            auto image_3_0 = ConvBlock<1>(image_2_2, kernel_3_0, batchNorm_3_0);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_3_1 = Kernel<T, 256, 256, 3, 3>(getWeight<T>(weightIndex::layer3_3_conv2_weight));
            auto batchNorm_3_1 = BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_3_bn2_weight), getWeight<T>(weightIndex::layer3_3_bn2_bias));
            auto image_3_1 = ConvBlock<1>(image_3_0, kernel_3_1, batchNorm_3_1);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_3_2 = Kernel<T, 1024, 256, 1, 1>(getWeight<T>(weightIndex::layer3_3_conv3_weight));
            auto batchNorm_3_2 = BatchNorm<T, 1024>(getWeight<T>(weightIndex::layer3_3_bn3_weight), getWeight<T>(weightIndex::layer3_3_bn3_bias));
            auto image_3_2 = ConvBlockAddIdentity(image_3_1, kernel_3_2, batchNorm_3_2, image_2_2);
            std::swap(inputBuffer, outputBuffer);

            std::swap(inputBuffer, shortcutBuffer); // buffer swap so that the image_3_2 is not overwritten by convolution block as is needed for the shortcut
            auto kernel_4_0 = Kernel<T, 256, 1024, 1, 1>(getWeight<T>(weightIndex::layer3_4_conv1_weight));
            auto batchNorm_4_0 = BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_4_bn1_weight), getWeight<T>(weightIndex::layer3_4_bn1_bias));
            auto image_4_0 = ConvBlock<1>(image_3_2, kernel_4_0, batchNorm_4_0);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_4_1 = Kernel<T, 256, 256, 3, 3>(getWeight<T>(weightIndex::layer3_4_conv2_weight));
            auto batchNorm_4_1 = BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_4_bn2_weight), getWeight<T>(weightIndex::layer3_4_bn2_bias));
            auto image_4_1 = ConvBlock<1>(image_4_0, kernel_4_1, batchNorm_4_1);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_4_2 = Kernel<T, 1024, 256, 1, 1>(getWeight<T>(weightIndex::layer3_4_conv3_weight));
            auto batchNorm_4_2 = BatchNorm<T, 1024>(getWeight<T>(weightIndex::layer3_4_bn3_weight), getWeight<T>(weightIndex::layer3_4_bn3_bias));
            auto image_4_2 = ConvBlockAddIdentity(image_4_1, kernel_4_2, batchNorm_4_2, image_3_2);
            std::swap(inputBuffer, outputBuffer);

            std::swap(inputBuffer, shortcutBuffer); // buffer swap so that the image_4_2 is not overwritten by convolution block as is needed for the shortcut
            auto kernel_5_0 = Kernel<T, 256, 1024, 1, 1>(getWeight<T>(weightIndex::layer3_5_conv1_weight));
            auto batchNorm_5_0 = BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_5_bn1_weight), getWeight<T>(weightIndex::layer3_5_bn1_bias));
            auto image_5_0 = ConvBlock<1>(image_4_2, kernel_5_0, batchNorm_5_0);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_5_1 = Kernel<T, 256, 256, 3, 3>(getWeight<T>(weightIndex::layer3_5_conv2_weight));
            auto batchNorm_5_1 = BatchNorm<T, 256>(getWeight<T>(weightIndex::layer3_5_bn2_weight), getWeight<T>(weightIndex::layer3_5_bn2_bias));
            auto image_5_1 = ConvBlock<1>(image_5_0, kernel_5_1, batchNorm_5_1);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_5_2 = Kernel<T, 1024, 256, 1, 1>(getWeight<T>(weightIndex::layer3_5_conv3_weight));
            auto batchNorm_5_2 = BatchNorm<T, 1024>(getWeight<T>(weightIndex::layer3_5_bn3_weight), getWeight<T>(weightIndex::layer3_5_bn3_bias));
            auto image_5_2 = ConvBlockAddIdentity(image_5_1, kernel_5_2, batchNorm_5_2, image_4_2);
            std::swap(inputBuffer, outputBuffer);

            return image_5_2;
        }

        template <typename T>
        Image<T, 2048, 7, 7> ResNet50::block3(Image<T, 1024, 15, 15> &input)
        {
            std::swap(inputBuffer, shortcutBuffer); // buffer swap so that the input is not overwritten by convolution block as is needed for the shortcut
            auto kernel_0_0 = Kernel<T, 512, 1024, 1, 1>(getWeight<T>(weightIndex::layer4_0_conv1_weight));
            auto batchNorm_0_0 = BatchNorm<T, 512>(getWeight<T>(weightIndex::layer4_0_bn1_weight), getWeight<T>(weightIndex::layer4_0_bn1_bias));
            auto image_0_0 = ConvBlock<1>(input, kernel_0_0, batchNorm_0_0);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_0_1 = Kernel<T, 512, 512, 3, 3>(getWeight<T>(weightIndex::layer4_0_conv2_weight));
            auto batchNorm_0_1 = BatchNorm<T, 512>(getWeight<T>(weightIndex::layer4_0_bn2_weight), getWeight<T>(weightIndex::layer4_0_bn2_bias));
            auto image_0_1 = ConvBlock<2>(image_0_0, kernel_0_1, batchNorm_0_1);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_0_2 = Kernel<T, 2048, 512, 1, 1>(getWeight<T>(weightIndex::layer4_0_conv3_weight));
            auto batchNorm_0_2 = BatchNorm<T, 2048>(getWeight<T>(weightIndex::layer4_0_bn3_weight), getWeight<T>(weightIndex::layer4_0_bn3_bias));
            auto projectionKernel = Kernel<T, 2048, 1024, 1, 1>(getWeight<T>(weightIndex::layer4_0_downsample_0_weight));
            auto projectionBatchNorm = BatchNorm<T, 2048>(getWeight<T>(weightIndex::layer4_0_downsample_1_weight), getWeight<T>(weightIndex::layer4_0_downsample_1_bias));
            auto image_0_2 = ConvBlockAddProjection<2, 2>(image_0_1, kernel_0_2, batchNorm_0_2, input, projectionKernel, projectionBatchNorm);
            std::swap(inputBuffer, outputBuffer);

            std::swap(inputBuffer, shortcutBuffer); // buffer swap so that the image_0_2 is not overwritten by convolution block as is needed for the shortcut
            auto kernel_1_0 = Kernel<T, 512, 2048, 1, 1>(getWeight<T>(weightIndex::layer4_1_conv1_weight));
            auto batchNorm_1_0 = BatchNorm<T, 512>(getWeight<T>(weightIndex::layer4_1_bn1_weight), getWeight<T>(weightIndex::layer4_1_bn1_bias));
            auto image_1_0 = ConvBlock<1>(image_0_2, kernel_1_0, batchNorm_1_0);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_1_1 = Kernel<T, 512, 512, 3, 3>(getWeight<T>(weightIndex::layer4_1_conv2_weight));
            auto batchNorm_1_1 = BatchNorm<T, 512>(getWeight<T>(weightIndex::layer4_1_bn2_weight), getWeight<T>(weightIndex::layer4_1_bn2_bias));
            auto image_1_1 = ConvBlock<1>(image_1_0, kernel_1_1, batchNorm_1_1);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_1_2 = Kernel<T, 2048, 512, 1, 1>(getWeight<T>(weightIndex::layer4_1_conv3_weight));
            auto batchNorm_1_2 = BatchNorm<T, 2048>(getWeight<T>(weightIndex::layer4_1_bn3_weight), getWeight<T>(weightIndex::layer4_1_bn3_bias));
            auto image_1_2 = ConvBlockAddIdentity(image_1_1, kernel_1_2, batchNorm_1_2, image_0_2);
            std::swap(inputBuffer, outputBuffer);

            std::swap(inputBuffer, shortcutBuffer); // buffer swap so that the image_1_2 is not overwritten by convolution block as is needed for the shortcut
            auto kernel_2_0 = Kernel<T, 512, 2048, 1, 1>(getWeight<T>(weightIndex::layer4_2_conv1_weight));
            auto batchNorm_2_0 = BatchNorm<T, 512>(getWeight<T>(weightIndex::layer4_2_bn1_weight), getWeight<T>(weightIndex::layer4_2_bn1_bias));
            auto image_2_0 = ConvBlock<1>(image_1_2, kernel_2_0, batchNorm_2_0);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_2_1 = Kernel<T, 512, 512, 3, 3>(getWeight<T>(weightIndex::layer4_2_conv2_weight));
            auto batchNorm_2_1 = BatchNorm<T, 512>(getWeight<T>(weightIndex::layer4_2_bn2_weight), getWeight<T>(weightIndex::layer4_2_bn2_bias));
            auto image_2_1 = ConvBlock<1>(image_2_0, kernel_2_1, batchNorm_2_1);
            std::swap(inputBuffer, outputBuffer);
            auto kernel_2_2 = Kernel<T, 2048, 512, 1, 1>(getWeight<T>(weightIndex::layer4_2_conv3_weight));
            auto batchNorm_2_2 = BatchNorm<T, 2048>(getWeight<T>(weightIndex::layer4_2_bn3_weight), getWeight<T>(weightIndex::layer4_2_bn3_bias));
            auto image_2_2 = ConvBlockAddIdentity(image_2_1, kernel_2_2, batchNorm_2_2, image_1_2);
            std::swap(inputBuffer, outputBuffer);

            return image_2_2;
        }

        template <size_t Stride, typename T, size_t ImageChannels, size_t ImageHeight, size_t ImageWidth, size_t KernelCount, size_t KernelHeight, size_t KernelWidth>
        inline Image<T, KernelCount, ImageHeight / Stride, ImageWidth / Stride> ResNet50::ConvBlock(
            Image<T, ImageChannels, ImageHeight, ImageWidth> image,
            Kernel<T, KernelCount, ImageChannels, KernelHeight, KernelWidth> kernel,
            BatchNorm<T, KernelCount> batchNorm)
        {
            constexpr size_t strideImageChannels = ImageHeight * ImageWidth;
            constexpr size_t strideImageHeight = ImageWidth;
            constexpr size_t strideImageWidth = Stride;

            constexpr size_t strideOutputChannels = ImageHeight * ImageWidth / (Stride * Stride);
            constexpr size_t strideOutputHeight = ImageWidth / Stride;
            constexpr size_t strideOutputWidth = 1;

            constexpr size_t strideKernelCount = ImageChannels * KernelHeight * KernelWidth;
            constexpr size_t strideKernelChannels = KernelHeight * KernelWidth;
            constexpr int strideKernelHeight = static_cast<int>(KernelWidth);
            constexpr int strideKernelWidth = 1;

            constexpr size_t lengthHeightCount = ImageHeight / Stride;
            constexpr size_t lengthWidthCount = ImageWidth / Stride;

            constexpr int lowerHalfKernelHeight = static_cast<int>(KernelHeight) / 2;
            constexpr int upperHalfKernelHeight = static_cast<int>(KernelHeight - 1) / 2;
            constexpr int lowerHalfKernelWidth = static_cast<int>(KernelWidth) / 2;
            constexpr int upperHalfKernelWidth = static_cast<int>(KernelWidth - 1) / 2;

            auto output = Image<T, KernelCount, ImageHeight / Stride, ImageWidth / Stride>(outputBuffer);
            auto outputPtr = output.getPointer();

            auto imagePtr = image.getPointer();
            auto kernelPtr = kernel.getPointer();
            auto betaPtr = batchNorm.getBetaPointer();
            auto gammaPtr = batchNorm.getGammaPointer();

#pragma omp parallel for collapse(2)
            for (size_t kernel = 0; kernel < KernelCount; kernel++)
            {
                auto offsetKernelCount = kernel * strideKernelCount;
                auto offsetOutputChannels = kernel * strideOutputChannels;

                for (size_t height = 0; height < lengthHeightCount; height++)
                {
                    int lowerHeight = static_cast<int>(height);
                    int upperHeight = static_cast<int>(lengthHeightCount - height);
                    int lowerKernelHeight = -1 * (lowerHalfKernelHeight & (lowerHeight || (lowerHeight > lowerHalfKernelHeight)));
                    int upperKernelHeight = upperHalfKernelHeight & (upperHeight || (upperHeight > upperHalfKernelHeight));

                    for (size_t width = 0; width < lengthWidthCount; width++)
                    {
                        int lowerWidth = static_cast<int>(width);
                        int upperWidth = static_cast<int>(lengthHeightCount - width);
                        int lowerKernelWidth = -1 * (lowerHalfKernelWidth & (lowerWidth || (lowerWidth > lowerHalfKernelWidth)));
                        int upperKernelWidth = upperHalfKernelWidth & (upperWidth || (upperWidth > upperHalfKernelWidth));

                        auto offsetOutput = offsetOutputChannels + height * strideOutputHeight + width * strideOutputWidth;
                        auto offsetImage = offsetImageChannels + height * strideImageHeight + width * strideImageWidth;
                        outputPtr[offsetOutput] = 0;

                        for (size_t channel = 0; channel < ImageChannels; channel++)
                        {
                            auto offsetKernel = offsetKernelCount + channel * strideKernelChannels - lowerKernelHeight * strideKernelHeight - lowerKernelWidth * strideKernelWidth;
                            auto offsetImageChannels = channel * strideImageChannels;

#pragma omp simd collapse(2) reduction(+ : outputPtr[offsetOutput])
                            for (int kernelHeight = lowerKernelHeight; kernelHeight < upperKernelHeight; kernelHeight += strideKernelHeight)
                            {
                                for (int kernelWidth = lowerKernelWidth; kernelWidth < upperKernelWidth; kernelWidth += strideKernelWidth)
                                {
                                    outputPtr[offsetOutput] +=
                                        imagePtr[offsetImage + kernelHeight * strideImageHeight + kernelWidth * strideImageWidth] *
                                        kernelPtr[offsetKernel + kernelHeight * strideKernelHeight + kernelWidth * strideKernelWidth];
                                }
                            }
                        }

                        outputPtr[offsetOutput] = relu<T>(gammaPtr[kernel] * outputPtr[offsetOutput] + betaPtr[kernel]);
                    }
                }
            }

            return output;
        }

        template <typename T, size_t ImageChannels, size_t ImageHeight, size_t ImageWidth, size_t KernelCount, size_t KernelHeight, size_t KernelWidth>
        inline Image<T, KernelCount, ImageHeight, ImageWidth> ResNet50::ConvBlockAddIdentity(
            Image<T, ImageChannels, ImageHeight, ImageWidth> image,
            Kernel<T, KernelCount, ImageChannels, KernelHeight, KernelWidth> kernel,
            BatchNorm<T, KernelCount> batchNorm,
            Image<T, KernelCount, ImageHeight, ImageWidth> shortcut)
        {
            constexpr size_t strideImageChannels = ImageHeight * ImageWidth;
            constexpr size_t strideImageHeight = ImageWidth;
            constexpr size_t strideImageWidth = Stride;

            constexpr size_t strideOutputChannels = ImageHeight * ImageWidth / (Stride * Stride);
            constexpr size_t strideOutputHeight = ImageWidth / Stride;
            constexpr size_t strideOutputWidth = 1;

            constexpr size_t strideKernelCount = ImageChannels * KernelHeight * KernelWidth;
            constexpr size_t strideKernelChannels = KernelHeight * KernelWidth;
            constexpr int strideKernelHeight = static_cast<int>(KernelWidth);
            constexpr int strideKernelWidth = 1;

            constexpr size_t lengthHeightCount = ImageHeight / Stride;
            constexpr size_t lengthWidthCount = ImageWidth / Stride;

            constexpr int lowerHalfKernelHeight = static_cast<int>(KernelHeight) / 2;
            constexpr int upperHalfKernelHeight = static_cast<int>(KernelHeight - 1) / 2;
            constexpr int lowerHalfKernelWidth = static_cast<int>(KernelWidth) / 2;
            constexpr int upperHalfKernelWidth = static_cast<int>(KernelWidth - 1) / 2;

            auto output = Image<T, KernelCount, ImageHeight / Stride, ImageWidth / Stride>(outputBuffer);
            auto outputPtr = output.getPointer();

            auto imagePtr = image.getPointer();
            auto kernelPtr = kernel.getPointer();
            auto betaPtr = batchNorm.getBetaPointer();
            auto gammaPtr = batchNorm.getGammaPointer();
            auto shortcutPtr = shortcut.getPointer();

#pragma omp parallel for collapse(2)
            for (size_t kernel = 0; kernel < KernelCount; kernel++)
            {
                auto offsetKernelCount = kernel * strideKernelCount;
                auto offsetOutputChannels = kernel * strideOutputChannels;

                for (size_t height = 0; height < lengthHeightCount; height++)
                {
                    int lowerHeight = static_cast<int>(height);
                    int upperHeight = static_cast<int>(lengthHeightCount - height);
                    int lowerKernelHeight = -1 * (lowerHalfKernelHeight & (lowerHeight || (lowerHeight > lowerHalfKernelHeight)));
                    int upperKernelHeight = upperHalfKernelHeight & (upperHeight || (upperHeight > upperHalfKernelHeight));

                    for (size_t width = 0; width < lengthWidthCount; width++)
                    {
                        int lowerWidth = static_cast<int>(width);
                        int upperWidth = static_cast<int>(lengthHeightCount - width);
                        int lowerKernelWidth = -1 * (lowerHalfKernelWidth & (lowerWidth || (lowerWidth > lowerHalfKernelWidth)));
                        int upperKernelWidth = upperHalfKernelWidth & (upperWidth || (upperWidth > upperHalfKernelWidth));

                        auto offsetOutput = offsetOutputChannels + height * strideOutputHeight + width * strideOutputWidth;
                        auto offsetImage = offsetImageChannels + height * strideImageHeight + width * strideImageWidth;
                        outputPtr[offsetOutput] = 0;

                        for (size_t channel = 0; channel < ImageChannels; channel++)
                        {
                            auto offsetKernel = offsetKernelCount + channel * strideKernelChannels - lowerKernelHeight * strideKernelHeight - lowerKernelWidth * strideKernelWidth;
                            auto offsetImageChannels = channel * strideImageChannels;

#pragma omp simd collapse(2) reduction(+ : outputPtr[offsetOutput])
                            for (int kernelHeight = lowerKernelHeight; kernelHeight < upperKernelHeight; kernelHeight += strideKernelHeight)
                            {
                                for (int kernelWidth = lowerKernelWidth; kernelWidth < upperKernelWidth; kernelWidth += strideKernelWidth)
                                {
                                    outputPtr[offsetOutput] +=
                                        imagePtr[offsetImage + kernelHeight * strideImageHeight + kernelWidth * strideImageWidth] *
                                        kernelPtr[offsetKernel + kernelHeight * strideKernelHeight + kernelWidth * strideKernelWidth];
                                }
                            }
                        }

                        outputPtr[offsetOutput] = relu<T>(gammaPtr[kernel] * outputPtr[offsetOutput] + betaPtr[kernel] + shortcutPtr[offsetOutput]);
                    }
                }
            }

            return output;
        }

        template <size_t Stride, size_t ShortcutDimExpand, typename T, size_t ImageChannels, size_t ImageHeight, size_t ImageWidth, size_t KernelCount, size_t KernelHeight, size_t KernelWidth>
        inline Image<T, KernelCount, ImageHeight / Stride, ImageWidth / Stride> ResNet50::ConvBlockAddProjection(
            Image<T, ImageChannels, ImageHeight / Stride, ImageWidth / Stride> image,
            Kernel<T, KernelCount, ImageChannels, KernelHeight, KernelWidth> kernel,
            BatchNorm<T, KernelCount> batchNorm,
            Image<T, KernelCount / ShortcutDimExpand, ImageHeight, ImageWidth> shortcut,
            Kernel<T, KernelCount, KernelCount / ShortcutDimExpand, 1, 1> projectionKernel,
            BatchNorm<T, KernelCount> projectionBatchNorm)
        {
            constexpr size_t strideImageChannels = ImageHeight * ImageWidth;
            constexpr size_t strideImageHeight = ImageWidth;
            constexpr size_t strideImageWidth = Stride;

            constexpr size_t strideOutputChannels = ImageHeight * ImageWidth / (Stride * Stride);
            constexpr size_t strideOutputHeight = ImageWidth / Stride;
            constexpr size_t strideOutputWidth = 1;

            constexpr size_t strideKernelCount = ImageChannels * KernelHeight * KernelWidth;
            constexpr size_t strideKernelChannels = KernelHeight * KernelWidth;
            constexpr int strideKernelHeight = static_cast<int>(KernelWidth);
            constexpr int strideKernelWidth = 1;

            constexpr size_t lengthHeightCount = ImageHeight / Stride;
            constexpr size_t lengthWidthCount = ImageWidth / Stride;

            constexpr int lowerHalfKernelHeight = static_cast<int>(KernelHeight) / 2;
            constexpr int upperHalfKernelHeight = static_cast<int>(KernelHeight - 1) / 2;
            constexpr int lowerHalfKernelWidth = static_cast<int>(KernelWidth) / 2;
            constexpr int upperHalfKernelWidth = static_cast<int>(KernelWidth - 1) / 2;

            auto output = Image<T, KernelCount, ImageHeight / Stride, ImageWidth / Stride>(outputBuffer);
            auto outputPtr = output.getPointer();

            auto imagePtr = image.getPointer();
            auto kernelPtr = kernel.getPointer();
            auto betaPtr = batchNorm.getBetaPointer();
            auto gammaPtr = batchNorm.getGammaPointer();
            auto shortcutPtr = shortcut.getPointer();
            auto projectionKernelPtr = projectionKernel.getPointer();
            auto projectionBetaPtr = projectionBatchNorm.getBetaPointer();
            auto projectionGammaPtr = projectionBatchNorm.getGammaPointer();

#pragma omp parallel for collapse(2)
            for (size_t kernel = 0; kernel < KernelCount; kernel++)
            {
                auto offsetKernelCount = kernel * strideKernelCount;
                auto offsetOutputChannels = kernel * strideOutputChannels;

                for (size_t height = 0; height < lengthHeightCount; height++)
                {
                    int lowerHeight = static_cast<int>(height);
                    int upperHeight = static_cast<int>(lengthHeightCount - height);
                    int lowerKernelHeight = -1 * (lowerHalfKernelHeight & (lowerHeight || (lowerHeight > lowerHalfKernelHeight)));
                    int upperKernelHeight = upperHalfKernelHeight & (upperHeight || (upperHeight > upperHalfKernelHeight));

                    for (size_t width = 0; width < lengthWidthCount; width++)
                    {
                        int lowerWidth = static_cast<int>(width);
                        int upperWidth = static_cast<int>(lengthHeightCount - width);
                        int lowerKernelWidth = -1 * (lowerHalfKernelWidth & (lowerWidth || (lowerWidth > lowerHalfKernelWidth)));
                        int upperKernelWidth = upperHalfKernelWidth & (upperWidth || (upperWidth > upperHalfKernelWidth));

                        auto offsetOutput = offsetOutputChannels + height * strideOutputHeight + width * strideOutputWidth;
                        auto offsetImage = offsetImageChannels + height * strideImageHeight + width * strideImageWidth;
                        auto offsetShortcut = offsetOutputChannels + height * strideImageHeight + width * strideImageWidth;
                        outputPtr[offsetOutput] = 0;
                        T shortcutValue = 0;

                        for (size_t channel = 0; channel < ImageChannels; channel++)
                        {
                            auto offsetKernel = offsetKernelCount + channel * strideKernelChannels - lowerKernelHeight * strideKernelHeight - lowerKernelWidth * strideKernelWidth;
                            auto offsetProjectionKernel = offsetKernelCount + channel;
                            auto offsetImageChannels = channel * strideImageChannels;

                            shortcutValue += shortcutPtr[offsetShortcut] * projectionKernelPtr[offsetProjectionKernel];

#pragma omp simd collapse(2) reduction(+ : outputPtr[offsetOutput])
                            for (int kernelHeight = lowerKernelHeight; kernelHeight < upperKernelHeight; kernelHeight += strideKernelHeight)
                            {
                                for (int kernelWidth = lowerKernelWidth; kernelWidth < upperKernelWidth; kernelWidth += strideKernelWidth)
                                {
                                    outputPtr[offsetOutput] +=
                                        imagePtr[offsetImage + kernelHeight * strideImageHeight + kernelWidth * strideImageWidth] *
                                        kernelPtr[offsetKernel + kernelHeight * strideKernelHeight + kernelWidth * strideKernelWidth];
                                }
                            }
                        }

                        outputPtr[offsetOutput] = relu<T>(gammaPtr[kernel] * outputPtr[offsetOutput] + betaPtr[kernel] +
                                                           projectionGammaPtr[kernel] * shortcutValue + projectionBetaPtr[kernel]);
                    }
                }
            }

            return output;
        }

        template <size_t Stride, typename T, size_t ImageChannels, size_t ImageHeight, size_t ImageWidth>
        inline Image<T, ImageChannels, ImageHeight / Stride, ImageWidth / Stride> ResNet50::MaxPool(Image<T, ImageChannels, ImageHeight, ImageWidth> image)
        {
            constexpr size_t strideImageWidth = Stride;

            constexpr size_t strideOutputChannels = ImageHeight * ImageWidth / (Stride * Stride);
            constexpr size_t strideOutputHeight = ImageWidth / Stride;
            constexpr size_t strideOutputWidth = 1;

            constexpr size_t lengthOutputChannels = ImageChannels * ImageHeight * ImageWidth / (Stride * Stride);
            constexpr size_t lengthOutputHeight = ImageHeight * ImageWidth / (Stride * Stride);
            constexpr size_t lengthOutputWidth = ImageWidth / Stride;

            auto output = Image<T, ImageChannels, ImageHeight / Stride, ImageWidth / Stride>(outputBuffer);
            auto outputPtr = output.getPointer();

            auto imagePtr = image.getPointer();

#pragma omp parallel for collapse(2)
            for (size_t channel = 0; channel < lengthOutputChannels; channel += strideOutputChannels)
            {
                for (size_t height = 0; height < lengthOutputHeight; height += strideOutputHeight)
                {
#pragma omp simd
                    for (size_t width = 0; width < lengthOutputWidth; width += strideOutputWidth)
                    {
                        size_t offset = channel + height * Stride + width * Stride;
                        outputPtr[channel + height + width] = std::max(
                            std::max(imagePtr[offset], imagePtr[offset + 1]),
                            std::max(imagePtr[offset + strideImageWidth], imagePtr[offset + strideImageWidth + 1]));
                    }
                }
            }

            return output;
        }

        template <typename T, size_t ImageChannels, size_t ImageHeight, size_t ImageWidth>
        inline Image<T, ImageChannels, 1, 1> ResNet50::GlobalAveragePool(Image<T, ImageChannels, ImageHeight, ImageWidth> image)
        {
            constexpr size_t strideOutputChannels = 1;

            constexpr size_t lengthOutputChannels = ImageChannels;

            constexpr size_t strideImageChannels = ImageHeight * ImageWidth;
            constexpr size_t strideImageHeight = ImageWidth;
            constexpr size_t strideImageWidth = 1;

            constexpr size_t lengthImageHeight = ImageHeight * ImageWidth;
            constexpr size_t lengthImageWidth = ImageWidth;

            auto output = Image<T, ImageChannels, 1, 1>(outputBuffer);
            auto outputPtr = output.getPointer();

            auto imagePtr = image.getPointer();

#pragma omp parallel for
            for (size_t channel = 0; channel < lengthOutputChannels; channel += strideOutputChannels)
            {
                auto imageChannel = channel * strideImageChannels;
                outputPtr[channel] = 0;

#pragma omp simd collapse(2) reduction(+ : outputPtr[channel])
                for (size_t height = 0; height < lengthImageHeight; height += strideImageHeight)
                {
                    for (size_t width = 0; width < lengthImageWidth; width += strideImageWidth)
                    {
                        outputPtr[channel] += imagePtr[imageChannel + height + width];
                    }
                }

                outputPtr[channel] /= lengthImageHeight * lengthImageWidth;
            }

            return output;
        }

        template <typename T>
        inline Array<T, 1000> ResNet50::fullyConnectedLayer(Array<T, 2048> input, Matrix<T, 1000, 2048> weights, Array<T, 1000> bias)
        {
            // TODO EFFICIENT IMPLEMENT

            auto output = Array<T, 1000>(outputBuffer);
            auto outputPtr = output.getPointer();

            auto inputPtr = input.getPointer();
            auto weightsPtr = weights.getPointer();
            auto biasPtr = bias.getPointer();

#pragma omp parallel for
            for (size_t column = 0; column < 1000; column++)
            {
#pragma omp simd reduction(+ : outputPtr[column])
                for (size_t row = 0; row < 2048; row++)
                {
                    outputPtr[column] += inputPtr[row] * weightsPtr[column * 2048 + row];
                }

                outputPtr[column] += biasPtr[column];
            }

            return output;
        }

        template <typename T>
        inline T *ResNet50::getWeight(size_t index)
        {
            return static_cast<T *>(weights[index]);
        }

        template <typename T>
        inline T ResNet50::relu(T value)
        {
            return (value > 0) * value;
        }
    } // namespace model
} // namespace ImageInference

#endif // IMAGEINFERENCE_RESNET50_H
