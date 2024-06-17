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
#include <stddef.h>
#include "math/Image.h"
#include "math/Kernel.h"
#include "math/Array.h"

#define MAX_RESNET50_SIZE 122 * 122 * 64

/// @brief The resnet50 v1.5 model from https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch
class ResNet50 : public IModel<float>
{
private:
    float inputBuffer[MAX_RESNET50_SIZE]{0};
    float outputBuffer[MAX_RESNET50_SIZE]{0};

    Image<float, 256, 61, 61> block0(Image<float, 64, 61, 61> &input);
    Image<float, 512, 30, 30> block1(Image<float, 256, 61, 61> &input);
    Image<float, 1024, 15, 15> block2(Image<float, 512, 30, 30> &input);
    Image<float, 2048, 7, 7> block3(Image<float, 1024, 15, 15> &input);

public:
    ResNet50(/* args */);
    ~ResNet50();

    const float *inference(const float *input) override;

    template <size_t Stride, typename T,
              size_t ImageChannels, size_t ImageHeight, size_t ImageWidth,
              size_t KernelCount, size_t KernelHeight, size_t KernelWidth>
    Image<T, KernelCount, ImageHeight / Stride, ImageWidth / Stride> ConvBlock(
        Image<T, ImageChannels, ImageHeight, ImageWidth> image,
        Kernel<T, KernelCount, ImageChannels, KernelHeight, KernelWidth> kernel);

    template <typename T,
              size_t ImageChannels, size_t ImageHeight, size_t ImageWidth,
              size_t KernelCount, size_t KernelHeight, size_t KernelWidth>
    Image<T, KernelCount, ImageHeight, ImageWidth> ConvBlockAddIdentity(
        Image<T, ImageChannels, ImageHeight, ImageWidth> image,
        Kernel<T, KernelCount, ImageChannels, KernelHeight, KernelWidth> kernel,
        Image<T, KernelCount, ImageHeight, ImageWidth> shortcut);

    template <size_t ShortcutDimExpand, typename T,
              size_t ImageChannels, size_t ImageHeight, size_t ImageWidth,
              size_t KernelCount, size_t KernelHeight, size_t KernelWidth>
    Image<T, KernelCount, ImageHeight, ImageWidth> ConvBlockAddIdentity(
        Image<T, ImageChannels, ImageHeight, ImageWidth> image,
        Kernel<T, KernelCount, ImageChannels, KernelHeight, KernelWidth> kernel,
        Image<T, KernelCount / ShortcutDimExpand, ImageHeight, ImageWidth> shortcut);

    template <size_t Stride, size_t ShortcutDimExpand, typename T,
              size_t ImageChannels, size_t ImageHeight, size_t ImageWidth,
              size_t KernelCount, size_t KernelHeight, size_t KernelWidth>
    Image<T, KernelCount, ImageHeight / Stride, ImageWidth / Stride> ConvBlockAddProjection(
        Image<T, ImageChannels, ImageHeight / Stride, ImageWidth / Stride> image,
        Kernel<T, KernelCount, ImageChannels, KernelHeight, KernelWidth> kernel,
        Image<T, KernelCount / ShortcutDimExpand, ImageHeight, ImageWidth> shortcut);

    template <size_t Stride, typename T,
              size_t ImageChannels, size_t ImageHeight, size_t ImageWidth>
    Image<T, ImageChannels, ImageHeight / Stride, ImageWidth / Stride> MaxPool(
        Image<T, ImageChannels, ImageHeight, ImageWidth> image);

    template <typename T,
              size_t ImageChannels, size_t ImageHeight, size_t ImageWidth>
    Image<T, ImageChannels, 1, 1> GlobalAveragePool(Image<T, ImageChannels, ImageHeight, ImageWidth> image);

    template <typename T>
    Array<T, 1000> fullyConnectedLayer(Array<T, 2048> input);
};

#endif // IMAGEINFERENCE_RESNET50_H
