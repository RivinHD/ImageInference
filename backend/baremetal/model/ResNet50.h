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
#include <stdexcept>
#include <Fastor/Fastor.h>
#include <libxsmm.h>

#define MAX_RESNET50_SIZE 122 * 122 * 64 * 2 * 2 // 967936 additional 2x for zero padding
#define RESNET50_BLOCK_SIZE 32

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
            // TODO Libxsmm kernels with code dispatch see https://github.com/libxsmm/libxsmm/blob/main/documentation/libxsmm_mm.md#manual-code-dispatch
            // TODO use Libxsmm header only library.
            // TODO add zones { <code> } inside the block, so that no more needed kernels, images, batchNorms get deleted. 
            // All the blocks start with a 1x1 kernel. Therefore no padding is required.

            template <typename T, size_t BlockSize>
            void block0(
                ImageInference::types::Image<T, 0, BlockSize, 64, 56, 56> &input,
                ImageInference::types::Image<T, 0, BlockSize, 256, 56, 56> &output);

            template <typename T, size_t BlockSize>
            void block1(
                ImageInference::types::Image<T, 0, BlockSize, 256, 56, 56> &input,
                ImageInference::types::Image<T, 0, BlockSize, 512, 28, 28> &output);

            template <typename T, size_t BlockSize>
            void block2(
                ImageInference::types::Image<T, 0, BlockSize, 512, 28, 28> &input,
                ImageInference::types::Image<T, 0, BlockSize, 1024, 14, 14> &output);

            template <typename T, size_t BlockSize>
            void block3(
                ImageInference::types::Image<T, 0, BlockSize, 1024, 14, 14> &input,
                ImageInference::types::Image<T, 0, BlockSize, 2048, 7, 7> &output);

            template <size_t Stride, size_t OutPadding, size_t InPadding,
                      typename T, size_t BlockSizeCount, size_t BlockSizeChannel,
                      size_t ImageChannels, size_t ImageHeight, size_t ImageWidth,
                      size_t KernelCount, size_t KernelHeight, size_t KernelWidth>
            static void convBlock(
                ImageInference::types::Image<T, InPadding, BlockSizeChannel, ImageChannels, ImageHeight, ImageWidth> &image,
                ImageInference::types::Kernel<T, BlockSizeCount, BlockSizeChannel, KernelCount, ImageChannels, KernelHeight, KernelWidth> &kernel,
                ImageInference::types::BatchNorm<T, KernelCount> &batchNorm,
                ImageInference::types::Image<T, OutPadding, BlockSizeCount, KernelCount, ImageHeight / Stride, ImageWidth / Stride> &output);

            template <size_t OutPadding, size_t InPadding, size_t ShortcutPadding,
                      typename T, size_t BlockSizeCount, size_t BlockSizeChannel,
                      size_t ImageChannels, size_t ImageHeight, size_t ImageWidth,
                      size_t KernelCount, size_t KernelHeight, size_t KernelWidth>
            static void convBlockAddIdentity(
                ImageInference::types::Image<T, InPadding, BlockSizeChannel, ImageChannels, ImageHeight, ImageWidth> &image,
                ImageInference::types::Kernel<T, BlockSizeCount, BlockSizeChannel, KernelCount, ImageChannels, KernelHeight, KernelWidth> &kernel,
                ImageInference::types::BatchNorm<T, KernelCount> &batchNorm,
                ImageInference::types::Image<T, ShortcutPadding, BlockSizeCount, KernelCount, ImageHeight, ImageWidth> &shortcut,
                ImageInference::types::Image<T, OutPadding, BlockSizeCount, KernelCount, ImageHeight, ImageWidth> &output);

            template <size_t Stride, size_t ShortcutDimExpand,
                      size_t OutPadding, size_t InPadding, size_t ShortcutPadding,
                      typename T, size_t BlockSizeCount, size_t BlockSizeChannel,
                      size_t ImageChannels, size_t ImageHeight, size_t ImageWidth,
                      size_t KernelCount, size_t KernelHeight, size_t KernelWidth>
            static void convBlockAddProjection(
                ImageInference::types::Image<T, InPadding, BlockSizeChannel, ImageChannels, ImageHeight / Stride, ImageWidth / Stride> &image,
                ImageInference::types::Kernel<T, BlockSizeCount, BlockSizeChannel, KernelCount, ImageChannels, KernelHeight, KernelWidth> &kernel,
                ImageInference::types::BatchNorm<T, KernelCount> &batchNorm,
                ImageInference::types::Image<T, ShortcutPadding, BlockSizeCount, KernelCount / ShortcutDimExpand, ImageHeight, ImageWidth> &shortcut,
                ImageInference::types::Kernel<T, BlockSizeCount, BlockSizeCount, KernelCount, KernelCount / ShortcutDimExpand, 1, 1> &projectionKernel,
                ImageInference::types::BatchNorm<T, KernelCount> &projectionBatchNorm,
                ImageInference::types::Image<T, OutPadding, BlockSizeCount, KernelCount, ImageHeight / Stride, ImageWidth / Stride> &output);

            template <size_t Stride, size_t OutPadding, size_t InPadding,
                      typename T, size_t BlockSize,
                      size_t ImageChannels, size_t ImageHeight, size_t ImageWidth>
            static void maxPool(
                ImageInference::types::Image<T, InPadding, BlockSize, ImageChannels, ImageHeight, ImageWidth> &image,
                ImageInference::types::Image<T, OutPadding, BlockSize, ImageChannels, ImageHeight / Stride, ImageWidth / Stride> &output);

            template <size_t OutPadding, size_t InPadding, typename T, size_t BlockSize,
                      size_t ImageChannels, size_t ImageHeight, size_t ImageWidth>
            static void globalAveragePool(
                ImageInference::types::Image<T, InPadding, BlockSize, ImageChannels, ImageHeight, ImageWidth> &image,
                ImageInference::types::Image<T, OutPadding, BlockSize, ImageChannels, 1, 1> &output);

            template <size_t BlockSize, typename T, size_t Columns, size_t Rows>
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
            /// see file backend/baremetal/resnet50weights.txt for size information.
            ResNet50(const std::vector<void *> &modelWeights, ImageInference::types::ScalarType type);
            ~ResNet50();

            enum weightIndex
            {
                conv1_weight = 0,                         // [64, 3, 7, 7]
                bn1_weight = 1,                           // [64]
                bn1_bias = 2,                             // [64]
                layer1_0_conv1_weight = 3,                // [64, 64, 1, 1]
                layer1_0_bn1_weight = 4,                  // [64]
                layer1_0_bn1_bias = 5,                    // [64]
                layer1_0_conv2_weight = 6,                // [64, 64, 3, 3]
                layer1_0_bn2_weight = 7,                  // [64]
                layer1_0_bn2_bias = 8,                    // [64]
                layer1_0_conv3_weight = 9,                // [256, 64, 1, 1]
                layer1_0_bn3_weight = 10,                 // [256]
                layer1_0_bn3_bias = 11,                   // [256]
                layer1_0_downsample_0_weight = 12,        // [256, 64, 1, 1]
                layer1_0_downsample_1_weight = 13,        // [256]
                layer1_0_downsample_1_bias = 14,          // [256]
                layer1_1_conv1_weight = 15,               // [64, 256, 1, 1]
                layer1_1_bn1_weight = 16,                 // [64]
                layer1_1_bn1_bias = 17,                   // [64]
                layer1_1_conv2_weight = 18,               // [64, 64, 3, 3]
                layer1_1_bn2_weight = 19,                 // [64]
                layer1_1_bn2_bias = 20,                   // [64]
                layer1_1_conv3_weight = 21,               // [256, 64, 1, 1]
                layer1_1_bn3_weight = 22,                 // [256]
                layer1_1_bn3_bias = 23,                   // [256]
                layer1_2_conv1_weight = 24,               // [64, 256, 1, 1]
                layer1_2_bn1_weight = 25,                 // [64]
                layer1_2_bn1_bias = 26,                   // [64]
                layer1_2_conv2_weight = 27,               // [64, 64, 3, 3]
                layer1_2_bn2_weight = 28,                 // [64]
                layer1_2_bn2_bias = 29,                   // [64]
                layer1_2_conv3_weight = 30,               // [256, 64, 1, 1]
                layer1_2_bn3_weight = 31,                 // [256]
                layer1_2_bn3_bias = 32,                   // [256]
                layer2_0_conv1_weight = 33,               // [128, 256, 1, 1]
                layer2_0_bn1_weight = 34,                 // [128]
                layer2_0_bn1_bias = 35,                   // [128]
                layer2_0_conv2_weight = 36,               // [128, 128, 3, 3]
                layer2_0_bn2_weight = 37,                 // [128]
                layer2_0_bn2_bias = 38,                   // [128]
                layer2_0_conv3_weight = 39,               // [512, 128, 1, 1]
                layer2_0_bn3_weight = 40,                 // [512]
                layer2_0_bn3_bias = 41,                   // [512]
                layer2_0_downsample_0_weight = 42,        // [512, 256, 1, 1]
                layer2_0_downsample_1_weight = 43,        // [512]
                layer2_0_downsample_1_bias = 44,          // [512]
                layer2_1_conv1_weight = 45,               // [128, 512, 1, 1]
                layer2_1_bn1_weight = 46,                 // [128]
                layer2_1_bn1_bias = 47,                   // [128]
                layer2_1_conv2_weight = 48,               // [128, 128, 3, 3]
                layer2_1_bn2_weight = 49,                 // [128]
                layer2_1_bn2_bias = 50,                   // [128]
                layer2_1_conv3_weight = 51,               // [512, 128, 1, 1]
                layer2_1_bn3_weight = 52,                 // [512]
                layer2_1_bn3_bias = 53,                   // [512]
                layer2_2_conv1_weight = 54,               // [128, 512, 1, 1]
                layer2_2_bn1_weight = 55,                 // [128]
                layer2_2_bn1_bias = 56,                   // [128]
                layer2_2_conv2_weight = 57,               // [128, 128, 3, 3]
                layer2_2_bn2_weight = 58,                 // [128]
                layer2_2_bn2_bias = 59,                   // [128]
                layer2_2_conv3_weight = 60,               // [512, 128, 1, 1]
                layer2_2_bn3_weight = 61,                 // [512]
                layer2_2_bn3_bias = 62,                   // [512]
                layer2_3_conv1_weight = 63,               // [128, 512, 1, 1]
                layer2_3_bn1_weight = 64,                 // [128]
                layer2_3_bn1_bias = 65,                   // [128]
                layer2_3_conv2_weight = 66,               // [128, 128, 3, 3]
                layer2_3_bn2_weight = 67,                 // [128]
                layer2_3_bn2_bias = 68,                   // [128]
                layer2_3_conv3_weight = 69,               // [512, 128, 1, 1]
                layer2_3_bn3_weight = 70,                 // [512]
                layer2_3_bn3_bias = 71,                   // [512]
                layer3_0_conv1_weight = 72,               // [256, 512, 1, 1]
                layer3_0_bn1_weight = 73,                 // [256]
                layer3_0_bn1_bias = 74,                   // [256]
                layer3_0_conv2_weight = 75,               // [256, 256, 3, 3]
                layer3_0_bn2_weight = 76,                 // [256]
                layer3_0_bn2_bias = 77,                   // [256]
                layer3_0_conv3_weight = 78,               // [1024, 256, 1, 1]
                layer3_0_bn3_weight = 79,                 // [1024]
                layer3_0_bn3_bias = 80,                   // [1024]
                layer3_0_downsample_0_weight = 81,        // [1024, 512, 1, 1]
                layer3_0_downsample_1_weight = 82,        // [1024]
                layer3_0_downsample_1_bias = 83,          // [1024]
                layer3_1_conv1_weight = 84,               // [256, 1024, 1, 1]
                layer3_1_bn1_weight = 85,                 // [256]
                layer3_1_bn1_bias = 86,                   // [256]
                layer3_1_conv2_weight = 87,               // [256, 256, 3, 3]
                layer3_1_bn2_weight = 88,                 // [256]
                layer3_1_bn2_bias = 89,                   // [256]
                layer3_1_conv3_weight = 90,               // [1024, 256, 1, 1]
                layer3_1_bn3_weight = 91,                 // [1024]
                layer3_1_bn3_bias = 92,                   // [1024]
                layer3_2_conv1_weight = 93,               // [256, 1024, 1, 1]
                layer3_2_bn1_weight = 94,                 // [256]
                layer3_2_bn1_bias = 95,                   // [256]
                layer3_2_conv2_weight = 96,               // [256, 256, 3, 3]
                layer3_2_bn2_weight = 97,                 // [256]
                layer3_2_bn2_bias = 98,                   // [256]
                layer3_2_conv3_weight = 99,               // [1024, 256, 1, 1]
                layer3_2_bn3_weight = 100,                // [1024]
                layer3_2_bn3_bias = 101,                  // [1024]
                layer3_3_conv1_weight = 102,              // [256, 1024, 1, 1]
                layer3_3_bn1_weight = 103,                // [256]
                layer3_3_bn1_bias = 104,                  // [256]
                layer3_3_conv2_weight = 105,              // [256, 256, 3, 3]
                layer3_3_bn2_weight = 106,                // [256]
                layer3_3_bn2_bias = 107,                  // [256]
                layer3_3_conv3_weight = 108,              // [1024, 256, 1, 1]
                layer3_3_bn3_weight = 109,                // [1024]
                layer3_3_bn3_bias = 110,                  // [1024]
                layer3_4_conv1_weight = 111,              // [256, 1024, 1, 1]
                layer3_4_bn1_weight = 112,                // [256]
                layer3_4_bn1_bias = 113,                  // [256]
                layer3_4_conv2_weight = 114,              // [256, 256, 3, 3]
                layer3_4_bn2_weight = 115,                // [256]
                layer3_4_bn2_bias = 116,                  // [256]
                layer3_4_conv3_weight = 117,              // [1024, 256, 1, 1]
                layer3_4_bn3_weight = 118,                // [1024]
                layer3_4_bn3_bias = 119,                  // [1024]
                layer3_5_conv1_weight = 120,              // [256, 1024, 1, 1]
                layer3_5_bn1_weight = 121,                // [256]
                layer3_5_bn1_bias = 122,                  // [256]
                layer3_5_conv2_weight = 123,              // [256, 256, 3, 3]
                layer3_5_bn2_weight = 124,                // [256]
                layer3_5_bn2_bias = 125,                  // [256]
                layer3_5_conv3_weight = 126,              // [1024, 256, 1, 1]
                layer3_5_bn3_weight = 127,                // [1024]
                layer3_5_bn3_bias = 128,                  // [1024]
                layer4_0_conv1_weight = 129,              // [512, 1024, 1, 1]
                layer4_0_bn1_weight = 130,                // [512]
                layer4_0_bn1_bias = 131,                  // [512]
                layer4_0_conv2_weight = 132,              // [512, 512, 3, 3]
                layer4_0_bn2_weight = 133,                // [512]
                layer4_0_bn2_bias = 134,                  // [512]
                layer4_0_conv3_weight = 135,              // [2048, 512, 1, 1]
                layer4_0_bn3_weight = 136,                // [2048]
                layer4_0_bn3_bias = 137,                  // [2048]
                layer4_0_downsample_0_weight = 138,       // [2048, 1024, 1, 1]
                layer4_0_downsample_1_weight = 139,       // [2048]
                layer4_0_downsample_1_bias = 140,         // [2048]
                layer4_1_conv1_weight = 141,              // [512, 2048, 1, 1]
                layer4_1_bn1_weight = 142,                // [512]
                layer4_1_bn1_bias = 143,                  // [512]
                layer4_1_conv2_weight = 144,              // [512, 512, 3, 3]
                layer4_1_bn2_weight = 145,                // [512]
                layer4_1_bn2_bias = 146,                  // [512]
                layer4_1_conv3_weight = 147,              // [2048, 512, 1, 1]
                layer4_1_bn3_weight = 148,                // [2048]
                layer4_1_bn3_bias = 149,                  // [2048]
                layer4_2_conv1_weight = 150,              // [512, 2048, 1, 1]
                layer4_2_bn1_weight = 151,                // [512]
                layer4_2_bn1_bias = 152,                  // [512]
                layer4_2_conv2_weight = 153,              // [512, 512, 3, 3]
                layer4_2_bn2_weight = 154,                // [512]
                layer4_2_bn2_bias = 155,                  // [512]
                layer4_2_conv3_weight = 156,              // [2048, 512, 1, 1]
                layer4_2_bn3_weight = 157,                // [2048]
                layer4_2_bn3_bias = 158,                  // [2048]
                fc_weight = 159,                          // [1000, 2048]
                fc_bias = 160,                            // [1000]
                bn1_running_mean = 161,                   // [64]
                bn1_running_var = 162,                    // [64]
                layer1_0_bn1_running_mean = 163,          // [64]
                layer1_0_bn1_running_var = 164,           // [64]
                layer1_0_bn2_running_mean = 165,          // [64]
                layer1_0_bn2_running_var = 166,           // [64]
                layer1_0_bn3_running_mean = 167,          // [256]
                layer1_0_bn3_running_var = 168,           // [256]
                layer1_0_downsample_1_running_mean = 169, // [256]
                layer1_0_downsample_1_running_var = 170,  // [256]
                layer1_1_bn1_running_mean = 171,          // [64]
                layer1_1_bn1_running_var = 172,           // [64]
                layer1_1_bn2_running_mean = 173,          // [64]
                layer1_1_bn2_running_var = 174,           // [64]
                layer1_1_bn3_running_mean = 175,          // [256]
                layer1_1_bn3_running_var = 176,           // [256]
                layer1_2_bn1_running_mean = 177,          // [64]
                layer1_2_bn1_running_var = 178,           // [64]
                layer1_2_bn2_running_mean = 179,          // [64]
                layer1_2_bn2_running_var = 180,           // [64]
                layer1_2_bn3_running_mean = 181,          // [256]
                layer1_2_bn3_running_var = 182,           // [256]
                layer2_0_bn1_running_mean = 183,          // [128]
                layer2_0_bn1_running_var = 184,           // [128]
                layer2_0_bn2_running_mean = 185,          // [128]
                layer2_0_bn2_running_var = 186,           // [128]
                layer2_0_bn3_running_mean = 187,          // [512]
                layer2_0_bn3_running_var = 188,           // [512]
                layer2_0_downsample_1_running_mean = 189, // [512]
                layer2_0_downsample_1_running_var = 190,  // [512]
                layer2_1_bn1_running_mean = 191,          // [128]
                layer2_1_bn1_running_var = 192,           // [128]
                layer2_1_bn2_running_mean = 193,          // [128]
                layer2_1_bn2_running_var = 194,           // [128]
                layer2_1_bn3_running_mean = 195,          // [512]
                layer2_1_bn3_running_var = 196,           // [512]
                layer2_2_bn1_running_mean = 197,          // [128]
                layer2_2_bn1_running_var = 198,           // [128]
                layer2_2_bn2_running_mean = 199,          // [128]
                layer2_2_bn2_running_var = 200,           // [128]
                layer2_2_bn3_running_mean = 201,          // [512]
                layer2_2_bn3_running_var = 202,           // [512]
                layer2_3_bn1_running_mean = 203,          // [128]
                layer2_3_bn1_running_var = 204,           // [128]
                layer2_3_bn2_running_mean = 205,          // [128]
                layer2_3_bn2_running_var = 206,           // [128]
                layer2_3_bn3_running_mean = 207,          // [512]
                layer2_3_bn3_running_var = 208,           // [512]
                layer3_0_bn1_running_mean = 209,          // [256]
                layer3_0_bn1_running_var = 210,           // [256]
                layer3_0_bn2_running_mean = 211,          // [256]
                layer3_0_bn2_running_var = 212,           // [256]
                layer3_0_bn3_running_mean = 213,          // [1024]
                layer3_0_bn3_running_var = 214,           // [1024]
                layer3_0_downsample_1_running_mean = 215, // [1024]
                layer3_0_downsample_1_running_var = 216,  // [1024]
                layer3_1_bn1_running_mean = 217,          // [256]
                layer3_1_bn1_running_var = 218,           // [256]
                layer3_1_bn2_running_mean = 219,          // [256]
                layer3_1_bn2_running_var = 220,           // [256]
                layer3_1_bn3_running_mean = 221,          // [1024]
                layer3_1_bn3_running_var = 222,           // [1024]
                layer3_2_bn1_running_mean = 223,          // [256]
                layer3_2_bn1_running_var = 224,           // [256]
                layer3_2_bn2_running_mean = 225,          // [256]
                layer3_2_bn2_running_var = 226,           // [256]
                layer3_2_bn3_running_mean = 227,          // [1024]
                layer3_2_bn3_running_var = 228,           // [1024]
                layer3_3_bn1_running_mean = 229,          // [256]
                layer3_3_bn1_running_var = 230,           // [256]
                layer3_3_bn2_running_mean = 231,          // [256]
                layer3_3_bn2_running_var = 232,           // [256]
                layer3_3_bn3_running_mean = 233,          // [1024]
                layer3_3_bn3_running_var = 234,           // [1024]
                layer3_4_bn1_running_mean = 235,          // [256]
                layer3_4_bn1_running_var = 236,           // [256]
                layer3_4_bn2_running_mean = 237,          // [256]
                layer3_4_bn2_running_var = 238,           // [256]
                layer3_4_bn3_running_mean = 239,          // [1024]
                layer3_4_bn3_running_var = 240,           // [1024]
                layer3_5_bn1_running_mean = 241,          // [256]
                layer3_5_bn1_running_var = 242,           // [256]
                layer3_5_bn2_running_mean = 243,          // [256]
                layer3_5_bn2_running_var = 244,           // [256]
                layer3_5_bn3_running_mean = 245,          // [1024]
                layer3_5_bn3_running_var = 246,           // [1024]
                layer4_0_bn1_running_mean = 247,          // [512]
                layer4_0_bn1_running_var = 248,           // [512]
                layer4_0_bn2_running_mean = 249,          // [512]
                layer4_0_bn2_running_var = 250,           // [512]
                layer4_0_bn3_running_mean = 251,          // [2048]
                layer4_0_bn3_running_var = 252,           // [2048]
                layer4_0_downsample_1_running_mean = 253, // [2048]
                layer4_0_downsample_1_running_var = 254,  // [2048]
                layer4_1_bn1_running_mean = 255,          // [512]
                layer4_1_bn1_running_var = 256,           // [512]
                layer4_1_bn2_running_mean = 257,          // [512]
                layer4_1_bn2_running_var = 258,           // [512]
                layer4_1_bn3_running_mean = 259,          // [2048]
                layer4_1_bn3_running_var = 260,           // [2048]
                layer4_2_bn1_running_mean = 261,          // [512]
                layer4_2_bn1_running_var = 262,           // [512]
                layer4_2_bn2_running_mean = 263,          // [512]
                layer4_2_bn2_running_var = 264,           // [512]
                layer4_2_bn3_running_mean = 265,          // [2048]
                layer4_2_bn3_running_var = 266,           // [2048]
            };

            void inference(const float *input, float *output) override;
            ImageInference::types::ScalarType getType();

#ifdef IMAGEINFERENCE_TESTING
            friend class ImageInference::model::test::ResNet50Test;
#endif // IMAGEINFERENCE_TESTING
        };

        template <typename T, size_t BlockSize>
        inline void ResNet50::block0(
            ImageInference::types::Image<T, 0, BlockSize, 64, 56, 56> &input,
            ImageInference::types::Image<T, 0, BlockSize, 256, 56, 56> &output)
        {
            auto kernel_0_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 64, 64, 1, 1>(getWeight<T>(weightIndex::layer1_0_conv1_weight));
            auto batchNorm_0_0 = ImageInference::types::BatchNorm<T, 64>(
                getWeight<T>(weightIndex::layer1_0_bn1_weight), 
                getWeight<T>(weightIndex::layer1_0_bn1_bias),
                getWeight<T>(weightIndex::layer1_0_bn1_running_mean),
                getWeight<T>(weightIndex::layer1_0_bn1_running_var));
            auto image_0_0 = ImageInference::types::Image<T, 1, BlockSize, 64, 56, 56>(); // OutPadding of 1 is because a 3x3 kernel is coming next
            convBlock<1>(input, kernel_0_0, batchNorm_0_0, image_0_0);
            auto kernel_0_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 64, 64, 3, 3>(getWeight<T>(weightIndex::layer1_0_conv2_weight));
            auto batchNorm_0_1 = ImageInference::types::BatchNorm<T, 64>(
                getWeight<T>(weightIndex::layer1_0_bn2_weight), 
                getWeight<T>(weightIndex::layer1_0_bn2_bias),
                getWeight<T>(weightIndex::layer1_0_bn2_running_mean),
                getWeight<T>(weightIndex::layer1_0_bn2_running_var));
            auto image_0_1 = ImageInference::types::Image<T, 0, BlockSize, 64, 56, 56>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlock<1>(image_0_0, kernel_0_1, batchNorm_0_1, image_0_1);
            auto kernel_0_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 64, 1, 1>(getWeight<T>(weightIndex::layer1_0_conv3_weight));
            auto batchNorm_0_2 = ImageInference::types::BatchNorm<T, 256>(
                getWeight<T>(weightIndex::layer1_0_bn3_weight), 
                getWeight<T>(weightIndex::layer1_0_bn3_bias),
                getWeight<T>(weightIndex::layer1_0_bn3_running_mean),
                getWeight<T>(weightIndex::layer1_0_bn3_running_var));
            auto projectionKernel = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 64, 1, 1>(getWeight<T>(weightIndex::layer1_0_downsample_0_weight));
            auto projectionBatchNorm = ImageInference::types::BatchNorm<T, 256>(
                getWeight<T>(weightIndex::layer1_0_downsample_1_weight), 
                getWeight<T>(weightIndex::layer1_0_downsample_1_bias),
                getWeight<T>(weightIndex::layer1_0_downsample_1_running_mean),
                getWeight<T>(weightIndex::layer1_0_downsample_1_running_var));
            // OutPadding of 0 is because a 1x1 kernel is coming next
            auto image_0_2 = ImageInference::types::Image<T, 0, BlockSize, 256, 56, 56>();
            convBlockAddProjection<1, 4>(image_0_1, kernel_0_2, batchNorm_0_2, input, projectionKernel, projectionBatchNorm, image_0_2);

            auto kernel_1_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 64, 256, 1, 1>(getWeight<T>(weightIndex::layer1_1_conv1_weight));
            auto batchNorm_1_0 = ImageInference::types::BatchNorm<T, 64>(
                getWeight<T>(weightIndex::layer1_1_bn1_weight), 
                getWeight<T>(weightIndex::layer1_1_bn1_bias),
                getWeight<T>(weightIndex::layer1_1_bn1_running_mean),
                getWeight<T>(weightIndex::layer1_1_bn1_running_var));
            auto image_1_0 = ImageInference::types::Image<T, 1, BlockSize, 64, 56, 56>(); // OutPadding of 1 is because a 3x3 kernel is coming next
            convBlock<1>(image_0_2, kernel_1_0, batchNorm_1_0, image_1_0);                // OutPadding of 1 is because a 3x3 kernel is coming next
            auto kernel_1_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 64, 64, 3, 3>(getWeight<T>(weightIndex::layer1_1_conv2_weight));
            auto batchNorm_1_1 = ImageInference::types::BatchNorm<T, 64>(
                getWeight<T>(weightIndex::layer1_1_bn2_weight), 
                getWeight<T>(weightIndex::layer1_1_bn2_bias),
                getWeight<T>(weightIndex::layer1_1_bn2_running_mean),
                getWeight<T>(weightIndex::layer1_1_bn2_running_var));
            auto image_1_1 = ImageInference::types::Image<T, 0, BlockSize, 64, 56, 56>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlock<1>(image_1_0, kernel_1_1, batchNorm_1_1, image_1_1);
            auto kernel_1_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 64, 1, 1>(getWeight<T>(weightIndex::layer1_1_conv3_weight));
            auto batchNorm_1_2 = ImageInference::types::BatchNorm<T, 256>(
                getWeight<T>(weightIndex::layer1_1_bn3_weight), 
                getWeight<T>(weightIndex::layer1_1_bn3_bias),
                getWeight<T>(weightIndex::layer1_1_bn3_running_mean),
                getWeight<T>(weightIndex::layer1_1_bn3_running_var));
            // OutPadding of 0 is because a 1x1 kernel is coming next
            auto image_1_2 = ImageInference::types::Image<T, 0, BlockSize, 256, 56, 56>();
            convBlockAddIdentity(image_1_1, kernel_1_2, batchNorm_1_2, image_0_2, image_1_2);

            auto kernel_2_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 64, 256, 1, 1>(getWeight<T>(weightIndex::layer1_2_conv1_weight));
            auto batchNorm_2_0 = ImageInference::types::BatchNorm<T, 64>(
                getWeight<T>(weightIndex::layer1_2_bn1_weight), 
                getWeight<T>(weightIndex::layer1_2_bn1_bias),
                getWeight<T>(weightIndex::layer1_2_bn1_running_mean),
                getWeight<T>(weightIndex::layer1_2_bn1_running_var));
            auto image_2_0 = ImageInference::types::Image<T, 1, BlockSize, 64, 56, 56>(); // OutPadding of 1 is because a 3x3 kernel is coming next
            convBlock<1>(image_1_2, kernel_2_0, batchNorm_2_0, image_2_0);
            auto kernel_2_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 64, 64, 3, 3>(getWeight<T>(weightIndex::layer1_2_conv2_weight));
            auto batchNorm_2_1 = ImageInference::types::BatchNorm<T, 64>(
                getWeight<T>(weightIndex::layer1_2_bn2_weight), 
                getWeight<T>(weightIndex::layer1_2_bn2_bias),
                getWeight<T>(weightIndex::layer1_2_bn2_running_mean),
                getWeight<T>(weightIndex::layer1_2_bn2_running_var));
            auto image_2_1 = ImageInference::types::Image<T, 0, BlockSize, 64, 56, 56>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlock<1, 0>(image_2_0, kernel_2_1, batchNorm_2_1, image_2_1);
            auto kernel_2_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 64, 1, 1>(getWeight<T>(weightIndex::layer1_2_conv3_weight));
            auto batchNorm_2_2 = ImageInference::types::BatchNorm<T, 256>(
                getWeight<T>(weightIndex::layer1_2_bn3_weight), 
                getWeight<T>(weightIndex::layer1_2_bn3_bias),
                getWeight<T>(weightIndex::layer1_2_bn3_running_mean),
                getWeight<T>(weightIndex::layer1_2_bn3_running_var));
            convBlockAddIdentity(image_2_1, kernel_2_2, batchNorm_2_2, image_1_2, output);
        }

        template <typename T, size_t BlockSize>
        void ResNet50::block1(
            ImageInference::types::Image<T, 0, BlockSize, 256, 56, 56> &input,
            ImageInference::types::Image<T, 0, BlockSize, 512, 28, 28> &output)
        {
            auto kernel_0_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 128, 256, 1, 1>(getWeight<T>(weightIndex::layer2_0_conv1_weight));
            auto batchNorm_0_0 = ImageInference::types::BatchNorm<T, 128>(
                getWeight<T>(weightIndex::layer2_0_bn1_weight), 
                getWeight<T>(weightIndex::layer2_0_bn1_bias),
                getWeight<T>(weightIndex::layer2_0_bn1_running_mean),
                getWeight<T>(weightIndex::layer2_0_bn1_running_var));
            auto image_0_0 = ImageInference::types::Image<T, 1, BlockSize, 128, 56, 56>(); // OutPadding of 1 is because a 3x3 kernel is coming next
            convBlock<1>(input, kernel_0_0, batchNorm_0_0, image_0_0);
            auto kernel_0_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 128, 128, 3, 3>(getWeight<T>(weightIndex::layer2_0_conv2_weight));
            auto batchNorm_0_1 = ImageInference::types::BatchNorm<T, 128>(
                getWeight<T>(weightIndex::layer2_0_bn2_weight), 
                getWeight<T>(weightIndex::layer2_0_bn2_bias),
                getWeight<T>(weightIndex::layer2_0_bn2_running_mean),
                getWeight<T>(weightIndex::layer2_0_bn2_running_var));
            auto image_0_1 = ImageInference::types::Image<T, 0, BlockSize, 128, 28, 28>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlock<2>(image_0_0, kernel_0_1, batchNorm_0_1, image_0_1);
            auto kernel_0_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 512, 128, 1, 1>(getWeight<T>(weightIndex::layer2_0_conv3_weight));
            auto batchNorm_0_2 = ImageInference::types::BatchNorm<T, 512>(
                getWeight<T>(weightIndex::layer2_0_bn3_weight), 
                getWeight<T>(weightIndex::layer2_0_bn3_bias),
                getWeight<T>(weightIndex::layer2_0_bn3_running_mean),
                getWeight<T>(weightIndex::layer2_0_bn3_running_var));
            auto projectionKernel = ImageInference::types::Kernel<T, BlockSize, BlockSize, 512, 256, 1, 1>(getWeight<T>(weightIndex::layer2_0_downsample_0_weight));
            auto projectionBatchNorm = ImageInference::types::BatchNorm<T, 512>(
                getWeight<T>(weightIndex::layer2_0_downsample_1_weight), 
                getWeight<T>(weightIndex::layer2_0_downsample_1_bias),
                getWeight<T>(weightIndex::layer2_0_downsample_1_running_mean),
                getWeight<T>(weightIndex::layer2_0_downsample_1_running_var));
            auto image_0_2 = ImageInference::types::Image<T, 0, BlockSize, 512, 28, 28>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlockAddProjection<2, 2>(image_0_1, kernel_0_2, batchNorm_0_2, input, projectionKernel, projectionBatchNorm, image_0_2);

            auto kernel_1_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 128, 512, 1, 1>(getWeight<T>(weightIndex::layer2_1_conv1_weight));
            auto batchNorm_1_0 = ImageInference::types::BatchNorm<T, 128>(
                getWeight<T>(weightIndex::layer2_1_bn1_weight), 
                getWeight<T>(weightIndex::layer2_1_bn1_bias),
                getWeight<T>(weightIndex::layer2_1_bn1_running_mean),
                getWeight<T>(weightIndex::layer2_1_bn1_running_var));
            auto image_1_0 = ImageInference::types::Image<T, 1, BlockSize, 128, 28, 28>(); // OutPadding of 1 is because a 3x3 kernel is coming next
            convBlock<1>(image_0_2, kernel_1_0, batchNorm_1_0, image_1_0);
            auto kernel_1_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 128, 128, 3, 3>(getWeight<T>(weightIndex::layer2_1_conv2_weight));
            auto batchNorm_1_1 = ImageInference::types::BatchNorm<T, 128>(
                getWeight<T>(weightIndex::layer2_1_bn2_weight), 
                getWeight<T>(weightIndex::layer2_1_bn2_bias),
                getWeight<T>(weightIndex::layer2_1_bn2_running_mean),
                getWeight<T>(weightIndex::layer2_1_bn2_running_var));
            auto image_1_1 = ImageInference::types::Image<T, 0, BlockSize, 128, 28, 28>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlock<1>(image_1_0, kernel_1_1, batchNorm_1_1, image_1_1);
            auto kernel_1_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 512, 128, 1, 1>(getWeight<T>(weightIndex::layer2_1_conv3_weight));
            auto batchNorm_1_2 = ImageInference::types::BatchNorm<T, 512>(
                getWeight<T>(weightIndex::layer2_1_bn3_weight), 
                getWeight<T>(weightIndex::layer2_1_bn3_bias),
                getWeight<T>(weightIndex::layer2_1_bn3_running_mean),
                getWeight<T>(weightIndex::layer2_1_bn3_running_var));
            auto image_1_2 = ImageInference::types::Image<T, 0, BlockSize, 512, 28, 28>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlockAddIdentity(image_1_1, kernel_1_2, batchNorm_1_2, image_0_2, image_1_2);

            auto kernel_2_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 128, 512, 1, 1>(getWeight<T>(weightIndex::layer2_2_conv1_weight));
            auto batchNorm_2_0 = ImageInference::types::BatchNorm<T, 128>(
                getWeight<T>(weightIndex::layer2_2_bn1_weight), 
                getWeight<T>(weightIndex::layer2_2_bn1_bias),
                getWeight<T>(weightIndex::layer2_2_bn1_running_mean),
                getWeight<T>(weightIndex::layer2_2_bn1_running_var));
            auto image_2_0 = ImageInference::types::Image<T, 1, BlockSize, 128, 28, 28>(); // OutPadding of 1 is because a 3x3 kernel is coming next
            convBlock<1>(image_1_2, kernel_2_0, batchNorm_2_0, image_2_0);
            auto kernel_2_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 128, 128, 3, 3>(getWeight<T>(weightIndex::layer2_2_conv2_weight));
            auto batchNorm_2_1 = ImageInference::types::BatchNorm<T, 128>(
                getWeight<T>(weightIndex::layer2_2_bn2_weight), 
                getWeight<T>(weightIndex::layer2_2_bn2_bias),
                getWeight<T>(weightIndex::layer2_2_bn2_running_mean),
                getWeight<T>(weightIndex::layer2_2_bn2_running_var));
            auto image_2_1 = ImageInference::types::Image<T, 0, BlockSize, 128, 28, 28>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlock<1>(image_2_0, kernel_2_1, batchNorm_2_1, image_2_1);
            auto kernel_2_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 512, 128, 1, 1>(getWeight<T>(weightIndex::layer2_2_conv3_weight));
            auto batchNorm_2_2 = ImageInference::types::BatchNorm<T, 512>(
                getWeight<T>(weightIndex::layer2_2_bn3_weight), 
                getWeight<T>(weightIndex::layer2_2_bn3_bias),
                getWeight<T>(weightIndex::layer2_2_bn3_running_mean),
                getWeight<T>(weightIndex::layer2_2_bn3_running_var));
            auto image_2_2 = ImageInference::types::Image<T, 0, BlockSize, 512, 28, 28>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlockAddIdentity(image_2_1, kernel_2_2, batchNorm_2_2, image_1_2, image_2_2);

            auto kernel_3_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 128, 512, 1, 1>(getWeight<T>(weightIndex::layer2_3_conv1_weight));
            auto batchNorm_3_0 = ImageInference::types::BatchNorm<T, 128>(
                getWeight<T>(weightIndex::layer2_3_bn1_weight), 
                getWeight<T>(weightIndex::layer2_3_bn1_bias),
                getWeight<T>(weightIndex::layer2_3_bn1_running_mean),
                getWeight<T>(weightIndex::layer2_3_bn1_running_var));
            auto image_3_0 = ImageInference::types::Image<T, 1, BlockSize, 128, 28, 28>(); // OutPadding of 1 is because a 3x3 kernel is coming next
            convBlock<1>(image_2_2, kernel_3_0, batchNorm_3_0, image_3_0);
            auto kernel_3_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 128, 128, 3, 3>(getWeight<T>(weightIndex::layer2_3_conv2_weight));
            auto batchNorm_3_1 = ImageInference::types::BatchNorm<T, 128>(
                getWeight<T>(weightIndex::layer2_3_bn2_weight), 
                getWeight<T>(weightIndex::layer2_3_bn2_bias),
                getWeight<T>(weightIndex::layer2_3_bn2_running_mean),
                getWeight<T>(weightIndex::layer2_3_bn2_running_var));
            auto image_3_1 = ImageInference::types::Image<T, 0, BlockSize, 128, 28, 28>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlock<1>(image_3_0, kernel_3_1, batchNorm_3_1, image_3_1);
            auto kernel_3_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 512, 128, 1, 1>(getWeight<T>(weightIndex::layer2_3_conv3_weight));
            auto batchNorm_3_2 = ImageInference::types::BatchNorm<T, 512>(
                getWeight<T>(weightIndex::layer2_3_bn3_weight), 
                getWeight<T>(weightIndex::layer2_3_bn3_bias),
                getWeight<T>(weightIndex::layer2_3_bn3_running_mean),
                getWeight<T>(weightIndex::layer2_3_bn3_running_var));
            convBlockAddIdentity(image_3_1, kernel_3_2, batchNorm_3_2, image_2_2, output);
        }

        template <typename T, size_t BlockSize>
        void ResNet50::block2(
            ImageInference::types::Image<T, 0, BlockSize, 512, 28, 28> &input,
            ImageInference::types::Image<T, 0, BlockSize, 1024, 14, 14> &output)
        {
            auto kernel_0_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 512, 1, 1>(getWeight<T>(weightIndex::layer3_0_conv1_weight));
            auto batchNorm_0_0 = ImageInference::types::BatchNorm<T, 256>(
                getWeight<T>(weightIndex::layer3_0_bn1_weight), 
                getWeight<T>(weightIndex::layer3_0_bn1_bias),
                getWeight<T>(weightIndex::layer3_0_bn1_running_mean),
                getWeight<T>(weightIndex::layer3_0_bn1_running_var));
            auto image_0_0 = ImageInference::types::Image<T, 1, BlockSize, 256, 28, 28>(); // OutPadding of 1 is because a 3x3 kernel is coming next
            convBlock<1>(input, kernel_0_0, batchNorm_0_0, image_0_0);
            auto kernel_0_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 256, 3, 3>(getWeight<T>(weightIndex::layer3_0_conv2_weight));
            auto batchNorm_0_1 = ImageInference::types::BatchNorm<T, 256>(
                getWeight<T>(weightIndex::layer3_0_bn2_weight), 
                getWeight<T>(weightIndex::layer3_0_bn2_bias),
                getWeight<T>(weightIndex::layer3_0_bn2_running_mean),
                getWeight<T>(weightIndex::layer3_0_bn2_running_var));
            auto image_0_1 = ImageInference::types::Image<T, 0, BlockSize, 256, 14, 14>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlock<2>(image_0_0, kernel_0_1, batchNorm_0_1, image_0_1);
            auto kernel_0_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 1024, 256, 1, 1>(getWeight<T>(weightIndex::layer3_0_conv3_weight));
            auto batchNorm_0_2 = ImageInference::types::BatchNorm<T, 1024>(
                getWeight<T>(weightIndex::layer3_0_bn3_weight), 
                getWeight<T>(weightIndex::layer3_0_bn3_bias),
                getWeight<T>(weightIndex::layer3_0_bn3_running_mean),
                getWeight<T>(weightIndex::layer3_0_bn3_running_var));
            auto projectionKernel = ImageInference::types::Kernel<T, BlockSize, BlockSize, 1024, 512, 1, 1>(getWeight<T>(weightIndex::layer3_0_downsample_0_weight));
            auto projectionBatchNorm = ImageInference::types::BatchNorm<T, 1024>(
                getWeight<T>(weightIndex::layer3_0_downsample_1_weight), 
                getWeight<T>(weightIndex::layer3_0_downsample_1_bias),
                getWeight<T>(weightIndex::layer3_0_downsample_1_running_mean),
                getWeight<T>(weightIndex::layer3_0_downsample_1_running_var));
            auto image_0_2 = ImageInference::types::Image<T, 0, BlockSize, 1024, 14, 14>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlockAddProjection<2, 2>(image_0_1, kernel_0_2, batchNorm_0_2, input, projectionKernel, projectionBatchNorm, image_0_2);

            auto kernel_1_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 1024, 1, 1>(getWeight<T>(weightIndex::layer3_1_conv1_weight));
            auto batchNorm_1_0 = ImageInference::types::BatchNorm<T, 256>(
                getWeight<T>(weightIndex::layer3_1_bn1_weight), 
                getWeight<T>(weightIndex::layer3_1_bn1_bias),
                getWeight<T>(weightIndex::layer3_1_bn1_running_mean),
                getWeight<T>(weightIndex::layer3_1_bn1_running_var));
            auto image_1_0 = ImageInference::types::Image<T, 1, BlockSize, 256, 14, 14>(); // OutPadding of 1 is because a 3x3 kernel is coming next
            convBlock<1>(image_0_2, kernel_1_0, batchNorm_1_0, image_1_0);
            auto kernel_1_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 256, 3, 3>(getWeight<T>(weightIndex::layer3_1_conv2_weight));
            auto batchNorm_1_1 = ImageInference::types::BatchNorm<T, 256>(
                getWeight<T>(weightIndex::layer3_1_bn2_weight), 
                getWeight<T>(weightIndex::layer3_1_bn2_bias),
                getWeight<T>(weightIndex::layer3_1_bn2_running_mean),
                getWeight<T>(weightIndex::layer3_1_bn2_running_var));
            auto image_1_1 = ImageInference::types::Image<T, 0, BlockSize, 256, 14, 14>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlock<1>(image_1_0, kernel_1_1, batchNorm_1_1, image_1_1);
            auto kernel_1_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 1024, 256, 1, 1>(getWeight<T>(weightIndex::layer3_1_conv3_weight));
            auto batchNorm_1_2 = ImageInference::types::BatchNorm<T, 1024>(
                getWeight<T>(weightIndex::layer3_1_bn3_weight), 
                getWeight<T>(weightIndex::layer3_1_bn3_bias),
                getWeight<T>(weightIndex::layer3_1_bn3_running_mean),
                getWeight<T>(weightIndex::layer3_1_bn3_running_var));
            auto image_1_2 = ImageInference::types::Image<T, 0, BlockSize, 1024, 14, 14>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlockAddIdentity(image_1_1, kernel_1_2, batchNorm_1_2, image_0_2, image_1_2);

            auto kernel_2_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 1024, 1, 1>(getWeight<T>(weightIndex::layer3_2_conv1_weight));
            auto batchNorm_2_0 = ImageInference::types::BatchNorm<T, 256>(
                getWeight<T>(weightIndex::layer3_2_bn1_weight), 
                getWeight<T>(weightIndex::layer3_2_bn1_bias),
                getWeight<T>(weightIndex::layer3_2_bn1_running_mean),
                getWeight<T>(weightIndex::layer3_2_bn1_running_var));
            auto image_2_0 = ImageInference::types::Image<T, 1, BlockSize, 256, 14, 14>(); // OutPadding of 1 is because a 3x3 kernel is coming next
            convBlock<1>(image_1_2, kernel_2_0, batchNorm_2_0, image_2_0);
            auto kernel_2_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 256, 3, 3>(getWeight<T>(weightIndex::layer3_2_conv2_weight));
            auto batchNorm_2_1 = ImageInference::types::BatchNorm<T, 256>(
                getWeight<T>(weightIndex::layer3_2_bn2_weight), 
                getWeight<T>(weightIndex::layer3_2_bn2_bias),
                getWeight<T>(weightIndex::layer3_2_bn2_running_mean),
                getWeight<T>(weightIndex::layer3_2_bn2_running_var));
            auto image_2_1 = ImageInference::types::Image<T, 0, BlockSize, 256, 14, 14>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlock<1>(image_2_0, kernel_2_1, batchNorm_2_1, image_2_1);
            auto kernel_2_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 1024, 256, 1, 1>(getWeight<T>(weightIndex::layer3_2_conv3_weight));
            auto batchNorm_2_2 = ImageInference::types::BatchNorm<T, 1024>(
                getWeight<T>(weightIndex::layer3_2_bn3_weight), 
                getWeight<T>(weightIndex::layer3_2_bn3_bias),
                getWeight<T>(weightIndex::layer3_2_bn3_running_mean),
                getWeight<T>(weightIndex::layer3_2_bn3_running_var));
            auto image_2_2 = ImageInference::types::Image<T, 0, BlockSize, 1024, 14, 14>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlockAddIdentity(image_2_1, kernel_2_2, batchNorm_2_2, image_1_2, image_2_2);

            auto kernel_3_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 1024, 1, 1>(getWeight<T>(weightIndex::layer3_3_conv1_weight));
            auto batchNorm_3_0 = ImageInference::types::BatchNorm<T, 256>(
                getWeight<T>(weightIndex::layer3_3_bn1_weight), 
                getWeight<T>(weightIndex::layer3_3_bn1_bias),
                getWeight<T>(weightIndex::layer3_3_bn1_running_mean),
                getWeight<T>(weightIndex::layer3_3_bn1_running_var));
            auto image_3_0 = ImageInference::types::Image<T, 1, BlockSize, 256, 14, 14>(); // OutPadding of 1 is because a 3x3 kernel is coming next
            convBlock<1>(image_2_2, kernel_3_0, batchNorm_3_0, image_3_0);
            auto kernel_3_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 256, 3, 3>(getWeight<T>(weightIndex::layer3_3_conv2_weight));
            auto batchNorm_3_1 = ImageInference::types::BatchNorm<T, 256>(
                getWeight<T>(weightIndex::layer3_3_bn2_weight),
                 getWeight<T>(weightIndex::layer3_3_bn2_bias),
                 getWeight<T>(weightIndex::layer3_3_bn2_running_mean),
                 getWeight<T>(weightIndex::layer3_3_bn2_running_var));
            auto image_3_1 = ImageInference::types::Image<T, 0, BlockSize, 256, 14, 14>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlock<1>(image_3_0, kernel_3_1, batchNorm_3_1, image_3_1);
            auto kernel_3_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 1024, 256, 1, 1>(getWeight<T>(weightIndex::layer3_3_conv3_weight));
            auto batchNorm_3_2 = ImageInference::types::BatchNorm<T, 1024>(
                getWeight<T>(weightIndex::layer3_3_bn3_weight), 
                getWeight<T>(weightIndex::layer3_3_bn3_bias),
                getWeight<T>(weightIndex::layer3_3_bn3_running_mean),
                getWeight<T>(weightIndex::layer3_3_bn3_running_var));
            auto image_3_2 = ImageInference::types::Image<T, 0, BlockSize, 1024, 14, 14>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlockAddIdentity(image_3_1, kernel_3_2, batchNorm_3_2, image_2_2, image_3_2);

            auto kernel_4_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 1024, 1, 1>(getWeight<T>(weightIndex::layer3_4_conv1_weight));
            auto batchNorm_4_0 = ImageInference::types::BatchNorm<T, 256>(
                getWeight<T>(weightIndex::layer3_4_bn1_weight), 
                getWeight<T>(weightIndex::layer3_4_bn1_bias),
                getWeight<T>(weightIndex::layer3_4_bn1_running_mean),
                getWeight<T>(weightIndex::layer3_4_bn1_running_var));
            auto image_4_0 = ImageInference::types::Image<T, 1, BlockSize, 256, 14, 14>(); // OutPadding of 1 is because a 3x3 kernel is coming next
            convBlock<1>(image_3_2, kernel_4_0, batchNorm_4_0, image_4_0);
            auto kernel_4_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 256, 3, 3>(getWeight<T>(weightIndex::layer3_4_conv2_weight));
            auto batchNorm_4_1 = ImageInference::types::BatchNorm<T, 256>(
                getWeight<T>(weightIndex::layer3_4_bn2_weight), 
                getWeight<T>(weightIndex::layer3_4_bn2_bias),
                getWeight<T>(weightIndex::layer3_4_bn2_running_mean),
                getWeight<T>(weightIndex::layer3_4_bn2_running_var));
            auto image_4_1 = ImageInference::types::Image<T, 0, BlockSize, 256, 14, 14>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlock<1>(image_4_0, kernel_4_1, batchNorm_4_1, image_4_1);
            auto kernel_4_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 1024, 256, 1, 1>(getWeight<T>(weightIndex::layer3_4_conv3_weight));
            auto batchNorm_4_2 = ImageInference::types::BatchNorm<T, 1024>(
                getWeight<T>(weightIndex::layer3_4_bn3_weight), 
                getWeight<T>(weightIndex::layer3_4_bn3_bias),
                getWeight<T>(weightIndex::layer3_4_bn3_running_mean),
                getWeight<T>(weightIndex::layer3_4_bn3_running_var));
            auto image_4_2 = ImageInference::types::Image<T, 0, BlockSize, 1024, 14, 14>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlockAddIdentity(image_4_1, kernel_4_2, batchNorm_4_2, image_3_2, image_4_2);

            auto kernel_5_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 1024, 1, 1>(getWeight<T>(weightIndex::layer3_5_conv1_weight));
            auto batchNorm_5_0 = ImageInference::types::BatchNorm<T, 256>(
                getWeight<T>(weightIndex::layer3_5_bn1_weight), 
                getWeight<T>(weightIndex::layer3_5_bn1_bias),
                getWeight<T>(weightIndex::layer3_5_bn1_running_mean),
                getWeight<T>(weightIndex::layer3_5_bn1_running_var));
            auto image_5_0 = ImageInference::types::Image<T, 1, BlockSize, 256, 14, 14>(); // OutPadding of 1 is because a 3x3 kernel is coming next
            convBlock<1>(image_4_2, kernel_5_0, batchNorm_5_0, image_5_0);
            auto kernel_5_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 256, 256, 3, 3>(getWeight<T>(weightIndex::layer3_5_conv2_weight));
            auto batchNorm_5_1 = ImageInference::types::BatchNorm<T, 256>(
                getWeight<T>(weightIndex::layer3_5_bn2_weight), 
                getWeight<T>(weightIndex::layer3_5_bn2_bias),
                getWeight<T>(weightIndex::layer3_5_bn2_running_mean),
                getWeight<T>(weightIndex::layer3_5_bn2_running_var));
            auto image_5_1 = ImageInference::types::Image<T, 0, BlockSize, 256, 14, 14>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlock<1>(image_5_0, kernel_5_1, batchNorm_5_1, image_5_1);
            auto kernel_5_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 1024, 256, 1, 1>(getWeight<T>(weightIndex::layer3_5_conv3_weight));
            auto batchNorm_5_2 = ImageInference::types::BatchNorm<T, 1024>(
                getWeight<T>(weightIndex::layer3_5_bn3_weight), 
                getWeight<T>(weightIndex::layer3_5_bn3_bias),
                getWeight<T>(weightIndex::layer3_5_bn3_running_mean),
                getWeight<T>(weightIndex::layer3_5_bn3_running_var));
            convBlockAddIdentity(image_5_1, kernel_5_2, batchNorm_5_2, image_4_2, output);
        }

        template <typename T, size_t BlockSize>
        void ResNet50::block3(
            ImageInference::types::Image<T, 0, BlockSize, 1024, 14, 14> &input,
            ImageInference::types::Image<T, 0, BlockSize, 2048, 7, 7> &output)
        {
            auto kernel_0_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 512, 1024, 1, 1>(getWeight<T>(weightIndex::layer4_0_conv1_weight));
            auto batchNorm_0_0 = ImageInference::types::BatchNorm<T, 512>(
                getWeight<T>(weightIndex::layer4_0_bn1_weight), 
                getWeight<T>(weightIndex::layer4_0_bn1_bias),
                getWeight<T>(weightIndex::layer4_0_bn1_running_mean),
                getWeight<T>(weightIndex::layer4_0_bn1_running_var));
            auto image_0_0 = ImageInference::types::Image<T, 1, BlockSize, 512, 14, 14>(); // OutPadding of 1 is because a 3x3 kernel is coming next
            convBlock<1>(input, kernel_0_0, batchNorm_0_0, image_0_0);
            auto kernel_0_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 512, 512, 3, 3>(getWeight<T>(weightIndex::layer4_0_conv2_weight));
            auto batchNorm_0_1 = ImageInference::types::BatchNorm<T, 512>(
                getWeight<T>(weightIndex::layer4_0_bn2_weight), 
                getWeight<T>(weightIndex::layer4_0_bn2_bias),
                getWeight<T>(weightIndex::layer4_0_bn2_running_mean),
                getWeight<T>(weightIndex::layer4_0_bn2_running_var));
            auto image_0_1 = ImageInference::types::Image<T, 0, BlockSize, 512, 7, 7>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlock<2>(image_0_0, kernel_0_1, batchNorm_0_1, image_0_1);
            auto kernel_0_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 2048, 512, 1, 1>(getWeight<T>(weightIndex::layer4_0_conv3_weight));
            auto batchNorm_0_2 = ImageInference::types::BatchNorm<T, 2048>(
                getWeight<T>(weightIndex::layer4_0_bn3_weight), 
                getWeight<T>(weightIndex::layer4_0_bn3_bias),
                getWeight<T>(weightIndex::layer4_0_bn3_running_mean),
                getWeight<T>(weightIndex::layer4_0_bn3_running_var));
            auto projectionKernel = ImageInference::types::Kernel<T, BlockSize, BlockSize, 2048, 1024, 1, 1>(getWeight<T>(weightIndex::layer4_0_downsample_0_weight));
            auto projectionBatchNorm = ImageInference::types::BatchNorm<T, 2048>(
                getWeight<T>(weightIndex::layer4_0_downsample_1_weight), 
                getWeight<T>(weightIndex::layer4_0_downsample_1_bias),
                getWeight<T>(weightIndex::layer4_0_downsample_1_running_mean),
                getWeight<T>(weightIndex::layer4_0_downsample_1_running_var));
            auto image_0_2 = ImageInference::types::Image<T, 0, BlockSize, 2048, 7, 7>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlockAddProjection<2, 2>(image_0_1, kernel_0_2, batchNorm_0_2, input, projectionKernel, projectionBatchNorm, image_0_2);

            auto kernel_1_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 512, 2048, 1, 1>(getWeight<T>(weightIndex::layer4_1_conv1_weight));
            auto batchNorm_1_0 = ImageInference::types::BatchNorm<T, 512>(
                getWeight<T>(weightIndex::layer4_1_bn1_weight), 
                getWeight<T>(weightIndex::layer4_1_bn1_bias),
                getWeight<T>(weightIndex::layer4_1_bn1_running_mean),
                getWeight<T>(weightIndex::layer4_1_bn1_running_var));
            auto image_1_0 = ImageInference::types::Image<T, 1, BlockSize, 512, 7, 7>(); // OutPadding of 1 is because a 3x3 kernel is coming next
            convBlock<1>(image_0_2, kernel_1_0, batchNorm_1_0, image_1_0);
            auto kernel_1_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 512, 512, 3, 3>(getWeight<T>(weightIndex::layer4_1_conv2_weight));
            auto batchNorm_1_1 = ImageInference::types::BatchNorm<T, 512>(
                getWeight<T>(weightIndex::layer4_1_bn2_weight), 
                getWeight<T>(weightIndex::layer4_1_bn2_bias),
                getWeight<T>(weightIndex::layer4_1_bn2_running_mean),
                getWeight<T>(weightIndex::layer4_1_bn2_running_var));
            auto image_1_1 = ImageInference::types::Image<T, 0, BlockSize, 512, 7, 7>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlock<1>(image_1_0, kernel_1_1, batchNorm_1_1, image_1_1);
            auto kernel_1_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 2048, 512, 1, 1>(getWeight<T>(weightIndex::layer4_1_conv3_weight));
            auto batchNorm_1_2 = ImageInference::types::BatchNorm<T, 2048>(
                getWeight<T>(weightIndex::layer4_1_bn3_weight), 
                getWeight<T>(weightIndex::layer4_1_bn3_bias),
                getWeight<T>(weightIndex::layer4_1_bn3_running_mean),
                getWeight<T>(weightIndex::layer4_1_bn3_running_var));
            auto image_1_2 = ImageInference::types::Image<T, 0, BlockSize, 2048, 7, 7>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlockAddIdentity(image_1_1, kernel_1_2, batchNorm_1_2, image_0_2, image_1_2);

            auto kernel_2_0 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 512, 2048, 1, 1>(getWeight<T>(weightIndex::layer4_2_conv1_weight));
            auto batchNorm_2_0 = ImageInference::types::BatchNorm<T, 512>(
                getWeight<T>(weightIndex::layer4_2_bn1_weight), 
                getWeight<T>(weightIndex::layer4_2_bn1_bias),
                getWeight<T>(weightIndex::layer4_2_bn1_running_mean),
                getWeight<T>(weightIndex::layer4_2_bn1_running_var));
            auto image_2_0 = ImageInference::types::Image<T, 1, BlockSize, 512, 7, 7>(); // OutPadding of 1 is because a 3x3 kernel is coming next
            convBlock<1>(image_1_2, kernel_2_0, batchNorm_2_0, image_2_0);
            auto kernel_2_1 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 512, 512, 3, 3>(getWeight<T>(weightIndex::layer4_2_conv2_weight));
            auto batchNorm_2_1 = ImageInference::types::BatchNorm<T, 512>(
                getWeight<T>(weightIndex::layer4_2_bn2_weight), 
                getWeight<T>(weightIndex::layer4_2_bn2_bias),
                getWeight<T>(weightIndex::layer4_2_bn2_running_mean),
                getWeight<T>(weightIndex::layer4_2_bn2_running_var));
            auto image_2_1 = ImageInference::types::Image<T, 0, BlockSize, 512, 7, 7>(); // OutPadding of 0 is because a 1x1 kernel is coming next
            convBlock<1>(image_2_0, kernel_2_1, batchNorm_2_1, image_2_1);
            auto kernel_2_2 = ImageInference::types::Kernel<T, BlockSize, BlockSize, 2048, 512, 1, 1>(getWeight<T>(weightIndex::layer4_2_conv3_weight));
            auto batchNorm_2_2 = ImageInference::types::BatchNorm<T, 2048>(
                getWeight<T>(weightIndex::layer4_2_bn3_weight), 
                getWeight<T>(weightIndex::layer4_2_bn3_bias),
                getWeight<T>(weightIndex::layer4_2_bn3_running_mean),
                getWeight<T>(weightIndex::layer4_2_bn3_running_var));
            convBlockAddIdentity<0>(image_2_1, kernel_2_2, batchNorm_2_2, image_1_2, output);
        }

        template <size_t Stride, size_t OutPadding, size_t InPadding,
                  typename T, size_t BlockSizeCount, size_t BlockSizeChannel,
                  size_t ImageChannels, size_t ImageHeight, size_t ImageWidth,
                  size_t KernelCount, size_t KernelHeight, size_t KernelWidth>
        inline void ResNet50::convBlock(
            ImageInference::types::Image<T, InPadding, BlockSizeChannel, ImageChannels, ImageHeight, ImageWidth> &image,
            ImageInference::types::Kernel<T, BlockSizeCount, BlockSizeChannel, KernelCount, ImageChannels, KernelHeight, KernelWidth> &kernel,
            ImageInference::types::BatchNorm<T, KernelCount> &batchNorm,
            ImageInference::types::Image<T, OutPadding, BlockSizeCount, KernelCount, ImageHeight / Stride, ImageWidth / Stride> &output)
        {
            if constexpr (InPadding != KernelHeight / 2 || InPadding != KernelWidth / 2)
            {
                std::cerr << "ResNet50::convBlock: Padding is too small or to large for the kernel size. Padding: " << InPadding
                          << " KernelHeight: " << KernelHeight << " KernelWidth: " << KernelWidth << std::endl
                          << "Should be KernelHeight / 2 or KernelWidth / 2 = Padding." << std::endl;
                throw std::runtime_error("ResNet50::convBlock: Padding is too small or to large for the kernel size!");
            }

            constexpr const size_t countBlocks = KernelCount / BlockSizeCount;
            constexpr const size_t channelBlocks = ImageChannels / BlockSizeChannel;
            constexpr const size_t outputHeight = ImageHeight / Stride;
            constexpr const size_t outputWidth = ImageWidth / Stride;

            auto outputPtr = output.getPointer() + output.paddingOffset; // We skip the padding as we want to start at the data section.

            const auto imagePtr = image.getPointer();                         // ChannelBlocks x Height x Width x ChannelElements
            const auto kernelPtr = kernel.getPointer();                       // CountBlocks x ChannelBlocks x Height x Width x ChannelElements x CountElements
            const auto gammaPtr = batchNorm.getGammaPointer();                // Count = CountBlocks x CountElements
            const auto betaPtr = batchNorm.getBetaPointer();                  // Count = CountBlocks x CountElements
            const auto meanPtr = batchNorm.getMeanPointer();                  // Count = CountBlocks x CountElements
            const auto variancePtr = batchNorm.getProcessedVariancePointer(); // Count = CountBlocks x CountElements

#ifdef USE_OMP
#pragma omp parallel for collapse(2)
#endif // USE_OMP
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
                                const size_t imageOffset = image.getOffset(iBChannel, iHeight * Stride + kHeight, kWidth, 0);
                                const size_t kernelOffset = kernel.getOffset(iBCount, iBChannel, kHeight, kWidth, 0, 0);
                                const size_t outputOffset = output.getOffset(iBCount, iHeight, 0, 0);

#ifdef IMAGEINFERENCE_TESTING
                                // Get the last element touched by the matmul.
                                image.getOffset(iBChannel, iHeight * Stride + kHeight, ImageWidth - 1 + kWidth, BlockSizeChannel - 1);
                                kernel.getOffset(iBCount, iBChannel, kHeight, kWidth, BlockSizeChannel - 1, BlockSizeCount - 1);
                                // Adding padding offset as this is already applied at the output.
                                output.getOffset(iBCount, iHeight, outputWidth - 1, BlockSizeCount - 1 + output.paddingOffset);
#endif // IMAGEINFERENCE_TESTING

                                // Kernel of shape BlockSizeChannel x BlockSizeCount
                                // Input of shape ImageWidth x BlockSizeChannel
                                // Output of shape outputWidth x BlockSizeCount === ImageWidth / Stride x BlockSizeCount

                                // If we use libxsmm directly we don't need to do add separately!
                                // If we use the leading dimension on the image we can use it as stride.
                                // With the leading dimension we skip the next blocks as they should be skipped by the stride.
                                constexpr int MM = outputWidth;
                                constexpr int KK = BlockSizeChannel;
                                constexpr int NN = BlockSizeCount;
                                constexpr int ldImage = KK * Stride;
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
                                    imagePtr + imageOffset /*required*/,
                                    &ldImage /*ldb*/,
                                    &beta /*beta*/,
                                    outputPtr + outputOffset /*required*/,
                                    &NN /*ldc*/
                                );
                            }
                        }
                    }

                    // At this point we completed a complete row of the output.
                    // Now we apply the batch norm and relu.
                    for (size_t iWidth = 0; iWidth < outputWidth; iWidth++)
                    {
#ifdef USE_OMP
#pragma omp simd
#endif // USE_OMP
                        for (size_t iCount = 0; iCount < BlockSizeCount; iCount++)
                        {
                            const size_t offsetOutput = output.getOffset(iBCount, iHeight, iWidth, iCount);
                            const size_t offsetCount = iBCount * BlockSizeCount + iCount;
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
        }

        template <size_t OutPadding, size_t InPadding, size_t ShortcutPadding,
                  typename T, size_t BlockSizeCount, size_t BlockSizeChannel,
                  size_t ImageChannels, size_t ImageHeight, size_t ImageWidth,
                  size_t KernelCount, size_t KernelHeight, size_t KernelWidth>
        inline void ResNet50::convBlockAddIdentity(
            ImageInference::types::Image<T, InPadding, BlockSizeChannel, ImageChannels, ImageHeight, ImageWidth> &image,
            ImageInference::types::Kernel<T, BlockSizeCount, BlockSizeChannel, KernelCount, ImageChannels, KernelHeight, KernelWidth> &kernel,
            ImageInference::types::BatchNorm<T, KernelCount> &batchNorm,
            ImageInference::types::Image<T, ShortcutPadding, BlockSizeCount, KernelCount, ImageHeight, ImageWidth> &shortcut,
            ImageInference::types::Image<T, OutPadding, BlockSizeCount, KernelCount, ImageHeight, ImageWidth> &output)
        {
            if constexpr (InPadding != KernelHeight / 2 || InPadding != KernelWidth / 2)
            {
                std::cerr << "ResNet50::convBlockAddIdentity: Padding is too small or to large for the kernel size. Padding: " << InPadding
                          << " KernelHeight: " << KernelHeight << " KernelWidth: " << KernelWidth << std::endl
                          << "Should be KernelHeight / 2 or KernelWidth / 2 = Padding." << std::endl;
                throw std::runtime_error("ResNet50::convBlockAddIdentity: Padding is too small or to large for the kernel size!");
            }

            constexpr const size_t countBlocks = KernelCount / BlockSizeCount;
            constexpr const size_t channelBlocks = ImageChannels / BlockSizeChannel;
            constexpr const size_t outputHeight = ImageHeight;
            constexpr const size_t outputWidth = ImageWidth;

            auto outputPtr = output.getPointer() + output.paddingOffset; // We skip the padding as we want to start at the data section.

            const auto imagePtr = image.getPointer();                         // ChannelBlocks x Height x Width x ChannelElements
            const auto kernelPtr = kernel.getPointer();                       // CountBlocks x ChannelBlocks x Height x Width x ChannelElements x CountElements
            const auto gammaPtr = batchNorm.getGammaPointer();                // Count = CountBlocks x CountElements
            const auto betaPtr = batchNorm.getBetaPointer();                  // Count = CountBlocks x CountElements
            const auto meanPtr = batchNorm.getMeanPointer();                  // Count = CountBlocks x CountElements
            const auto variancePtr = batchNorm.getProcessedVariancePointer(); // Count = CountBlocks x CountElements
            const auto shortcutPtr = shortcut.getPointer();                   // ChannelBlocks x Height x Width x ChannelElements

#ifdef USE_OMP
#pragma omp parallel for collapse(2)
#endif // USE_OMP
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
                                const size_t imageOffset = image.getOffset(iBChannel, iHeight + kHeight, kWidth, 0);
                                const size_t kernelOffset = kernel.getOffset(iBCount, iBChannel, kHeight, kWidth, 0, 0);
                                const size_t outputOffset = output.getOffset(iBCount, iHeight, 0, 0);

                                // Kernel of shape BlockSizeChannel x BlockSizeCount
                                // Input of shape ImageWidth x BlockSizeChannel
                                // Output of shape outputWidth x BlockSizeCount === ImageWidth x BlockSizeCount

                                // If we use libxsmm directly we don't need to do add separately!
                                constexpr const int MM = outputWidth;
                                constexpr const int KK = BlockSizeChannel;
                                constexpr const int NN = BlockSizeCount;
                                constexpr const int ldImage = KK;
                                constexpr const float alpha = 1.0;
                                constexpr const float beta = 1.0;

                                constexpr const char transa = 'N';
                                constexpr const char transb = 'N';

                                libxsmm_sgemm(
                                    &transa /*transa*/,
                                    &transb /*transb*/,
                                    &NN /*required*/,
                                    &MM /*required*/,
                                    &KK /*required*/,
                                    &alpha /*alpha*/,
                                    kernelPtr + kernelOffset /*required*/,
                                    &NN /*lda*/,
                                    imagePtr + imageOffset /*required*/,
                                    &ldImage /*ldb*/,
                                    &beta /*beta*/,
                                    outputPtr + outputOffset /*required*/,
                                    &NN /*ldc*/
                                );
                            }
                        }
                    }

                    // At this point we completed a complete row of the output.
                    // Now we apply the batch norm and relu.
                    for (size_t iWidth = 0; iWidth < ImageWidth; iWidth++)
                    {
#ifdef USE_OMP
#pragma omp simd
#endif // USE_OMP
                        for (size_t iCount = 0; iCount < BlockSizeCount; iCount++)
                        {
                            const size_t offsetOutput = output.getOffset(iBCount, iHeight, iWidth, iCount);
                            const size_t offsetShortcut = shortcut.getOffset(iBCount, iHeight, iWidth, iCount);
                            const size_t offsetCount = iBCount * BlockSizeCount + iCount;
                            T batchNormValue = ResNet50::batchNorm<T>(
                                outputPtr[offsetOutput],
                                gammaPtr[offsetCount],
                                betaPtr[offsetCount],
                                meanPtr[offsetCount],
                                variancePtr[offsetCount]);
                            outputPtr[offsetOutput] = relu<T>(batchNormValue + shortcutPtr[offsetShortcut]);
                        }
                    }
                }
            }
        }

        template <size_t Stride, size_t ShortcutDimExpand, size_t OutPadding, size_t InPadding, size_t ShortcutPadding,
                  typename T, size_t BlockSizeCount, size_t BlockSizeChannel,
                  size_t ImageChannels, size_t ImageHeight, size_t ImageWidth,
                  size_t KernelCount, size_t KernelHeight, size_t KernelWidth>
        inline void ResNet50::convBlockAddProjection(
            ImageInference::types::Image<T, InPadding, BlockSizeChannel, ImageChannels, ImageHeight / Stride, ImageWidth / Stride> &image,
            ImageInference::types::Kernel<T, BlockSizeCount, BlockSizeChannel, KernelCount, ImageChannels, KernelHeight, KernelWidth> &kernel,
            ImageInference::types::BatchNorm<T, KernelCount> &batchNorm,
            ImageInference::types::Image<T, ShortcutPadding, BlockSizeCount, KernelCount / ShortcutDimExpand, ImageHeight, ImageWidth> &shortcut,
            ImageInference::types::Kernel<T, BlockSizeCount, BlockSizeCount, KernelCount, KernelCount / ShortcutDimExpand, 1, 1> &projectionKernel,
            ImageInference::types::BatchNorm<T, KernelCount> &projectionBatchNorm,
            ImageInference::types::Image<T, OutPadding, BlockSizeCount, KernelCount, ImageHeight / Stride, ImageWidth / Stride> &output)
        {
            if constexpr (InPadding != KernelHeight / 2 || InPadding != KernelWidth / 2)
            {
                std::cerr << "ResNet50::convBlockAddProjection: Padding is too small or to large for the kernel size. Padding: " << InPadding
                          << " KernelHeight: " << KernelHeight << " KernelWidth: " << KernelWidth << std::endl
                          << "Should be KernelHeight / 2 or KernelWidth / 2 = Padding." << std::endl;
                throw std::runtime_error("ResNet50::convBlockAddProjection: Padding is too small or to large for the kernel size!");
            }

            constexpr const size_t countBlocks = KernelCount / BlockSizeCount;
            constexpr const size_t channelBlocks = ImageChannels / BlockSizeChannel;
            constexpr const size_t outputHeight = ImageHeight / Stride;
            constexpr const size_t outputWidth = ImageWidth / Stride;
            constexpr const size_t shortcutChannelBlock = KernelCount / ShortcutDimExpand / BlockSizeCount;

            ImageInference::types::Image<T, 0, BlockSizeCount, KernelCount, ImageHeight / Stride, ImageWidth / Stride> projection;
            auto projectionPtr = projection.getPointer() + projection.paddingOffset; // We skip the padding as we want to start at the data section.

            auto outputPtr = output.getPointer() + output.paddingOffset; // We skip the padding as we want to start at the data section.

            const auto imagePtr = image.getPointer();                                             // ChannelBlocks x Height x Width x ChannelElements
            const auto kernelPtr = kernel.getPointer();                                           // CountBlocks x ChannelBlocks x Height x Width x ChannelElements x CountElements
            const auto gammaPtr = batchNorm.getGammaPointer();                                    // Count = CountBlocks x CountElements
            const auto betaPtr = batchNorm.getBetaPointer();                                      // Count = CountBlocks x CountElements
            const auto meanPtr = batchNorm.getMeanPointer();                                      // Count = CountBlocks x CountElements
            const auto variancePtr = batchNorm.getProcessedVariancePointer();                     // Count = CountBlocks x CountElements
            const auto shortcutPtr = shortcut.getPointer();                                       // ChannelBlocks x Height x Width x ChannelElements
            const auto projectionKernelPtr = projectionKernel.getPointer();                       // CountBlocks x CountBlocks x 1 x 1 x CountElements x CountElements
            const auto projectionGammaPtr = projectionBatchNorm.getGammaPointer();                // Count = CountBlocks x CountElements
            const auto projectionBetaPtr = projectionBatchNorm.getBetaPointer();                  // Count = CountBlocks x CountElements
            const auto projectionMeanPtr = projectionBatchNorm.getMeanPointer();                  // Count = CountBlocks x CountElements
            const auto projectionVariancePtr = projectionBatchNorm.getProcessedVariancePointer(); // Count = CountBlocks x CountElements

#ifdef USE_OMP
#pragma omp parallel for collapse(2)
#endif // USE_OMP
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
                                const size_t imageOffset = image.getOffset(iBChannel, iHeight + kHeight, kWidth, 0);
                                const size_t kernelOffset = kernel.getOffset(iBCount, iBChannel, kHeight, kWidth, 0, 0);
                                const size_t outputOffset = output.getOffset(iBCount, iHeight, 0, 0);

                                // Kernel of shape BlockSizeChannel x BlockSizeCount
                                // Input of shape ImageWidth x BlockSizeChannel
                                // Output of shape outputWidth x BlockSizeCount === ImageWidth / Stride x BlockSizeCount

                                // If we use libxsmm directly we don't need to do add separately!
                                // Both input and output have the same stride therefore we can do a norma matrix multiplication.
                                constexpr const int MM = outputWidth;
                                constexpr const int KK = BlockSizeChannel;
                                constexpr const int NN = BlockSizeCount;
                                constexpr const int ldImage = KK;
                                constexpr const float alpha = 1.0;
                                constexpr const float beta = 1.0;

                                constexpr const char transa = 'N';
                                constexpr const char transb = 'N';

                                libxsmm_sgemm(
                                    &transa /*transa*/,
                                    &transb /*transb*/,
                                    &NN /*required*/,
                                    &MM /*required*/,
                                    &KK /*required*/,
                                    &alpha /*alpha*/,
                                    kernelPtr + kernelOffset /*required*/,
                                    &NN /*lda*/,
                                    imagePtr + imageOffset /*required*/,
                                    &ldImage /*ldb*/,
                                    &beta /*beta*/,
                                    outputPtr + outputOffset /*required*/,
                                    &NN /*ldc*/
                                );
                            }
                        }
                    }

                    // Calculate the shortcut projection
                    for (size_t iBChannel = 0; iBChannel < shortcutChannelBlock; iBChannel++)
                    {
                        const size_t offsetShortcut = shortcut.getOffset(iBChannel, iHeight * Stride, 0, 0);
                        const size_t offsetProjectionKernel = projectionKernel.getOffset(iBCount, iBChannel, 0, 0, 0, 0);
                        const size_t offsetProjection = projection.getOffset(iBCount, iHeight, 0, 0);

                        // Kernel of shape BlockSizeChannel x BlockSizeCount
                        // Input of shape ImageWidth x BlockSizeChannel
                        // Output of shape outputWidth x BlockSizeCount === ImageWidth / Stride x BlockSizeCount

                        // If we use libxsmm directly we don't need to do add separately!
                        // If we use the leading dimension on the image we can use it as stride.
                        // With the leading dimension we skip the next blocks as they should be skipped by the stride.
                        constexpr const int MM = outputWidth;
                        constexpr const int KK = BlockSizeChannel;
                        constexpr const int NN = BlockSizeCount;
                        constexpr const int ldImage = KK * Stride;
                        constexpr const float alpha = 1.0;
                        constexpr const float beta = 0.0;

                        constexpr const char transa = 'N';
                        constexpr const char transb = 'N';

                        libxsmm_sgemm(
                            &transa /*transa*/,
                            &transb /*transb*/,
                            &NN /*required*/,
                            &MM /*required*/,
                            &KK /*required*/,
                            &alpha /*alpha*/,
                            projectionKernelPtr + offsetProjectionKernel /*required*/,
                            &NN /*lda*/,
                            shortcutPtr + offsetShortcut /*required*/,
                            &ldImage /*ldb*/,
                            &beta /*beta*/,
                            projectionPtr + offsetProjection /*required*/,
                            &NN /*ldc*/
                        );
                    }

                    // At this point we completed a complete row of the projection.
                    // Now we apply the batch norm.
                    for (size_t iWidth = 0; iWidth < outputWidth; iWidth++)
                    {
#ifdef USE_OMP
#pragma omp simd
#endif // USE_OMP
                        for (size_t iCount = 0; iCount < BlockSizeCount; iCount++)
                        {
                            const size_t offsetProject = projection.getOffset(iBCount, iHeight, iWidth, iCount);
                            const size_t offsetOutput = output.getOffset(iBCount, iHeight, iWidth, iCount);
                            const size_t offsetCount = iBCount * BlockSizeCount + iCount;

                            const T batchNormValue = ResNet50::batchNorm<T>(
                                outputPtr[offsetOutput],
                                gammaPtr[offsetCount],
                                betaPtr[offsetCount],
                                meanPtr[offsetCount],
                                variancePtr[offsetCount]);

                            const T projectedValue = ResNet50::batchNorm<T>(
                                projectionPtr[offsetProject],
                                projectionGammaPtr[offsetCount],
                                projectionBetaPtr[offsetCount],
                                projectionMeanPtr[offsetCount],
                                projectionVariancePtr[offsetCount]);

                            outputPtr[offsetOutput] = relu<T>(batchNormValue + projectedValue);
                        }
                    }
                }
            }
        }

        template <size_t Stride, size_t OutPadding, size_t InPadding,
                  typename T, size_t BlockSize,
                  size_t ImageChannels, size_t ImageHeight, size_t ImageWidth>
        inline void ResNet50::maxPool(
            ImageInference::types::Image<T, InPadding, BlockSize, ImageChannels, ImageHeight, ImageWidth> &image,
            ImageInference::types::Image<T, OutPadding, BlockSize, ImageChannels, ImageHeight / Stride, ImageWidth / Stride> &output)
        {
            if constexpr (InPadding != 1)
            {
                std::cerr << "ResNet50::maxPool: Padding is too small or to large for 3x3 Max Pooling. Padding is " << InPadding
                          << " but should be 1." << std::endl;
                throw std::runtime_error("ResNet50::maxPool: Padding is too small or to large for 3x3 Max Pooling!");
            }

            constexpr const size_t channelBlocks = ImageChannels / BlockSize;
            constexpr const size_t outputHeight = ImageHeight / Stride;
            constexpr const size_t outputWidth = ImageWidth / Stride;

            auto outputPtr = output.getPointer() + output.paddingOffset; // We skip the padding as we want to start at the data section.

            const auto imagePtr = image.getPointer();

// 3x3 Stencil that gets the max value
#ifdef USE_OMP // We can parallelize the channel blocks as they are independent of each other for this max operation.
#pragma omp parallel for
#endif // USE_OMP
            for (size_t iBChannel = 0; iBChannel < channelBlocks; iBChannel++)
            {
                for (size_t iHeight = 0; iHeight < outputHeight; iHeight++)
                {
                    for (size_t iWidth = 0; iWidth < outputWidth; iWidth++)
                    {
                        const size_t preOffsetOutput = output.getOffset(iBChannel, iHeight, iWidth, 0);

#ifdef USE_OMP
#pragma omp simd
#endif // USE_OMP
                        for (size_t iChannel = 0; iChannel < BlockSize; iChannel++)
                        {
                            const size_t offsetOutput = preOffsetOutput + iChannel * output.strideChannel;
                            outputPtr[offsetOutput] = std::numeric_limits<T>::lowest();
                        }

                        for (size_t kHeight = 0; kHeight < 3; kHeight++)
                        {
                            for (size_t kWidth = 0; kWidth < 3; kWidth++)
                            {
#ifdef USE_OMP // We can apply simd because the elements are independent of each other.
#pragma omp simd
#endif // USE_OMP
                                for (size_t iChannel = 0; iChannel < BlockSize; iChannel++)
                                {
                                    const size_t offsetOutput = preOffsetOutput + iChannel * output.strideChannel;
                                    const size_t offsetImage = image.getOffset(iBChannel, iHeight * Stride + kHeight, iWidth * Stride + kWidth, iChannel);
                                    outputPtr[offsetOutput] = std::max(outputPtr[offsetOutput], imagePtr[offsetImage]);
                                }
                            }
                        }
                    }
                }
            }
        }

        template <size_t OutPadding, size_t InPadding, typename T, size_t BlockSize,
                  size_t ImageChannels, size_t ImageHeight, size_t ImageWidth>
        inline void ResNet50::globalAveragePool(
            ImageInference::types::Image<T, InPadding, BlockSize, ImageChannels, ImageHeight, ImageWidth> &image,
            ImageInference::types::Image<T, OutPadding, BlockSize, ImageChannels, 1, 1> &output)
        {
            constexpr const size_t channelBlocks = ImageChannels / BlockSize;

            auto outputPtr = output.getPointer() + output.paddingOffset; // We skip the padding as we want to start at the data section.

            auto imagePtr = image.getPointer() + image.paddingOffset; // We skip the padding as padding should not be averaged.
            constexpr const float scale = 1.0f / (ImageHeight * ImageWidth);

// We can do this reduction because the ouput has only one element per channel.
// The channel are larger (2048). Therefore we have many blocks to parallelize on.
// Otherwise the loops should be split and collapsed.
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : outputPtr[ : ImageChannels])
#endif // USE_OMP
            for (size_t iBChannel = 0; iBChannel < channelBlocks; iBChannel++)
            {
                for (size_t iHeight = 0; iHeight < ImageHeight; iHeight++)
                {
                    for (size_t iWidth = 0; iWidth < ImageWidth; iWidth++)
                    {
#ifdef USE_OMP // We can apply simd because the elements are independent of each other.
#pragma omp simd
#endif // USE_OMP
                        for (size_t iChannel = 0; iChannel < BlockSize; iChannel++)
                        {
                            const size_t offsetOutput = output.getOffset(iBChannel, 0, 0, iChannel);
                            const size_t offsetImage = image.getOffset(iBChannel, iHeight, iWidth, iChannel);
                            outputPtr[offsetOutput] += imagePtr[offsetImage];
                        }
                    }
                }

#ifdef USE_OMP // We can apply simd because the elements are independent of each other.
#pragma omp simd
#endif // USE_OMP
                for (size_t iChannel = 0; iChannel < BlockSize; iChannel++)
                {
                    const size_t offsetOutput = output.getOffset(iBChannel, 0, 0, iChannel);
                    outputPtr[offsetOutput] = static_cast<T>(outputPtr[offsetOutput] * scale);
                }
            }
        }

        template <size_t BlockSize, typename T, size_t Columns, size_t Rows>
        void ResNet50::fullyConnectedLayer(
            ImageInference::types::Array<T, Rows> &input,
            ImageInference::types::Matrix<T, Columns, Rows> &weight,
            ImageInference::types::Array<T, Columns> &biasAccumulator)
        {
            constexpr const size_t ColumnsBlocks = Columns / BlockSize;
            constexpr const size_t processableColumns = ColumnsBlocks * BlockSize;
            constexpr const size_t remainderColumns = Columns % BlockSize;

            auto inputPtr = input.getPointer();
            auto weightPtr = weight.getPointer();
            auto biasPtr = biasAccumulator.getPointer();

#ifdef USE_OMP // The accumulation on biasMap are independent of each other.
#pragma omp parallel for
#endif // USE_OMP
            for (size_t iBColumn = 0; iBColumn < processableColumns; iBColumn += BlockSize)
            {
                size_t weightOffset = weight.getOffset(iBColumn, 0);
                size_t biasOffset = biasAccumulator.getOffset(iBColumn);

                Fastor::TensorMap<T, Rows> inputMap(inputPtr);
                Fastor::TensorMap<T, BlockSize, Rows> weightMap(weightPtr + weightOffset);
                Fastor::TensorMap<T, BlockSize> biasMap(biasPtr + biasOffset);
                biasMap += Fastor::matmul(weightMap, inputMap);
            }

            // Handle the remainder
            size_t weightOffset = weight.getOffset(processableColumns, 0);
            size_t biasOffset = biasAccumulator.getOffset(processableColumns);

            Fastor::TensorMap<T, Rows> inputMap(inputPtr);
            Fastor::TensorMap<T, remainderColumns, Rows> weightMap(weightPtr + weightOffset);
            Fastor::TensorMap<T, remainderColumns> biasMap(biasPtr + biasOffset);
            biasMap += Fastor::matmul(weightMap, inputMap);
        }

        template <typename T>
        inline T *ResNet50::getWeight(const size_t index)
        {
            return static_cast<T *>(modelWeights[index]);
        }

#ifdef USE_OMP
#pragma omp declare simd
#endif // USE_OMP
        template <typename T>
        inline T ResNet50::relu(const T value)
        {
            return (value > 0) * value;
        }

#ifdef USE_OMP
#pragma omp declare simd
#endif // USE_OMP
        template <typename T>
        inline T ResNet50::batchNorm(const T value, const T gamma, const T beta, const T mean, const T processedVariance)
        {
            return gamma * (value - mean) * processedVariance + beta;
        }
    } // namespace model
} // namespace ImageInference

#endif // IMAGEINFERENCE_RESNET50_H
