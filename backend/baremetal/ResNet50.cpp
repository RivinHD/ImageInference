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

#include "ResNet50.h"
#include "math/Image.h"
#include <cstring>

Image<float, 256, 61, 61> ResNet50::block0(Image<float, 64, 61, 61> &input)
{
    auto kernel_0_0 = Kernel<float, 64, 64, 1, 1>(NULL);
    auto image_0_0 = ConvBlock<1>(input, kernel_0_0);
    auto kernel_0_1 = Kernel<float, 64, 64, 3, 3>(NULL);
    auto image_0_1 = ConvBlock<1>(image_0_0, kernel_0_1);
    auto kernel_0_2 = Kernel<float, 256, 64, 1, 1>(NULL);
    auto image_0_2 = ConvBlockAddIdentity<4>(image_0_1, kernel_0_2, input);

    auto kernel_1_0 = Kernel<float, 64, 256, 1, 1>(NULL);
    auto image_1_0 = ConvBlock<1>(image_0_2, kernel_1_0);
    auto kernel_1_1 = Kernel<float, 64, 64, 3, 3>(NULL);
    auto image_1_1 = ConvBlock<1>(image_1_0, kernel_1_1);
    auto kerne_1_2 = Kernel<float, 256, 64, 1, 1>(NULL);
    auto image_1_2 = ConvBlockAddIdentity(image_1_1, kerne_1_2, image_0_2);

    auto kernel_2_0 = Kernel<float, 64, 256, 1, 1>(NULL);
    auto image_2_0 = ConvBlock<1>(image_1_2, kernel_2_0);
    auto kernel_2_1 = Kernel<float, 64, 64, 3, 3>(NULL);
    auto image_2_1 = ConvBlock<1>(image_2_0, kernel_2_1);
    auto kerne_2_2 = Kernel<float, 256, 64, 1, 1>(NULL);
    auto image_2_2 = ConvBlockAddIdentity(image_2_1, kerne_2_2, image_1_2);

    return image_2_2;
}

Image<float, 512, 30, 30> ResNet50::block1(Image<float, 256, 61, 61> &input)
{
    auto kernel_0_0 = Kernel<float, 128, 256, 1, 1>(NULL);
    auto image_0_0 = ConvBlock<1>(input, kernel_0_0);
    auto kernel_0_1 = Kernel<float, 128, 128, 3, 3>(NULL);
    auto image_0_1 = ConvBlock<2>(image_0_0, kernel_0_1);
    auto kerne_0_2 = Kernel<float, 512, 128, 1, 1>(NULL);
    auto image_0_2 = ConvBlockAddProjection<2, 2>(image_0_1, kerne_0_2, input);

    auto kernel_1_0 = Kernel<float, 128, 512, 1, 1>(NULL);
    auto image_1_0 = ConvBlock<1>(image_0_2, kernel_1_0);
    auto kernel_1_1 = Kernel<float, 128, 128, 3, 3>(NULL);
    auto image_1_1 = ConvBlock<1>(image_1_0, kernel_1_1);
    auto kerne_1_2 = Kernel<float, 512, 128, 1, 1>(NULL);
    auto image_1_2 = ConvBlockAddIdentity(image_1_1, kerne_1_2, image_0_2);

    auto kernel_2_0 = Kernel<float, 128, 512, 1, 1>(NULL);
    auto image_2_0 = ConvBlock<1>(image_1_2, kernel_2_0);
    auto kernel_2_1 = Kernel<float, 128, 128, 3, 3>(NULL);
    auto image_2_1 = ConvBlock<1>(image_2_0, kernel_2_1);
    auto kerne_2_2 = Kernel<float, 512, 128, 1, 1>(NULL);
    auto image_2_2 = ConvBlockAddIdentity(image_2_1, kerne_2_2, image_1_2);

    auto kernel_3_0 = Kernel<float, 128, 512, 1, 1>(NULL);
    auto image_3_0 = ConvBlock<1>(image_2_2, kernel_3_0);
    auto kernel_3_1 = Kernel<float, 128, 128, 3, 3>(NULL);
    auto image_3_1 = ConvBlock<1>(image_3_0, kernel_3_1);
    auto kerne_3_2 = Kernel<float, 512, 128, 1, 1>(NULL);
    auto image_3_2 = ConvBlockAddIdentity(image_3_1, kerne_3_2, image_2_2);

    auto kernel_4_0 = Kernel<float, 128, 512, 1, 1>(NULL);
    auto image_4_0 = ConvBlock<1>(image_3_2, kernel_4_0);
    auto kernel_4_1 = Kernel<float, 128, 128, 3, 3>(NULL);
    auto image_4_1 = ConvBlock<1>(image_4_0, kernel_4_1);
    auto kerne_4_2 = Kernel<float, 512, 128, 1, 1>(NULL);
    auto image_4_2 = ConvBlockAddIdentity(image_4_1, kerne_4_2, image_3_2);

    return image_4_2;
}

Image<float, 1024, 15, 15> ResNet50::block2(Image<float, 512, 30, 30> &input)
{
    auto kernel_0_0 = Kernel<float, 256, 512, 1, 1>(NULL);
    auto image_0_0 = ConvBlock<1>(input, kernel_0_0);
    auto kernel_0_1 = Kernel<float, 256, 256, 3, 3>(NULL);
    auto image_0_1 = ConvBlock<2>(image_0_0, kernel_0_1);
    auto kernel_0_2 = Kernel<float, 1024, 256, 1, 1>(NULL);
    auto image_0_2 = ConvBlockAddProjection<2, 2>(image_0_1, kernel_0_2, input);

    auto kernel_1_0 = Kernel<float, 256, 1024, 1, 1>(NULL);
    auto image_1_0 = ConvBlock<1>(image_0_2, kernel_1_0);
    auto kernel_1_1 = Kernel<float, 256, 256, 3, 3>(NULL);
    auto image_1_1 = ConvBlock<1>(image_1_0, kernel_1_1);
    auto kernel_1_2 = Kernel<float, 1024, 256, 1, 1>(NULL);
    auto image_1_2 = ConvBlockAddIdentity(image_1_1, kernel_1_2, image_0_2);

    auto kernel_2_0 = Kernel<float, 256, 1024, 1, 1>(NULL);
    auto image_2_0 = ConvBlock<1>(image_1_2, kernel_2_0);
    auto kernel_2_1 = Kernel<float, 256, 256, 3, 3>(NULL);
    auto image_2_1 = ConvBlock<1>(image_2_0, kernel_2_1);
    auto kernel_2_2 = Kernel<float, 1024, 256, 1, 1>(NULL);
    auto image_2_2 = ConvBlockAddIdentity(image_2_1, kernel_2_2, image_1_2);

    auto kernel_3_0 = Kernel<float, 256, 1024, 1, 1>(NULL);
    auto image_3_0 = ConvBlock<1>(image_2_2, kernel_3_0);
    auto kernel_3_1 = Kernel<float, 256, 256, 3, 3>(NULL);
    auto image_3_1 = ConvBlock<1>(image_3_0, kernel_3_1);
    auto kernel_3_2 = Kernel<float, 1024, 256, 1, 1>(NULL);
    auto image_3_2 = ConvBlockAddIdentity(image_3_1, kernel_3_2, image_2_2);

    auto kernel_4_0 = Kernel<float, 256, 1024, 1, 1>(NULL);
    auto image_4_0 = ConvBlock<1>(image_3_2, kernel_4_0);
    auto kernel_4_1 = Kernel<float, 256, 256, 3, 3>(NULL);
    auto image_4_1 = ConvBlock<1>(image_4_0, kernel_4_1);
    auto kernel_4_2 = Kernel<float, 1024, 256, 1, 1>(NULL);
    auto image_4_2 = ConvBlockAddIdentity(image_4_1, kernel_4_2, image_3_2);

    auto kernel_5_0 = Kernel<float, 256, 1024, 1, 1>(NULL);
    auto image_5_0 = ConvBlock<1>(image_4_2, kernel_5_0);
    auto kernel_5_1 = Kernel<float, 256, 256, 3, 3>(NULL);
    auto image_5_1 = ConvBlock<1>(image_5_0, kernel_5_1);
    auto kernel_5_2 = Kernel<float, 1024, 256, 1, 1>(NULL);
    auto image_5_2 = ConvBlockAddIdentity(image_5_1, kernel_5_2, image_4_2);

    return image_5_2;
}

Image<float, 2048, 7, 7> ResNet50::block3(Image<float, 1024, 15, 15> &input)
{
    auto kernel_0_0 = Kernel<float, 512, 1024, 1, 1>(NULL);
    auto image_0_0 = ConvBlock<1>(input, kernel_0_0);
    auto kernel_0_1 = Kernel<float, 512, 512, 3, 3>(NULL);
    auto image_0_1 = ConvBlock<2>(image_0_0, kernel_0_1);
    auto kerne_0_2 = Kernel<float, 2048, 512, 1, 1>(NULL);
    auto image_0_2 = ConvBlockAddProjection<2, 2>(image_0_1, kerne_0_2, input);

    auto kernel_1_0 = Kernel<float, 512, 2048, 1, 1>(NULL);
    auto image_1_0 = ConvBlock<1>(image_0_2, kernel_1_0);
    auto kernel_1_1 = Kernel<float, 512, 512, 3, 3>(NULL);
    auto image_1_1 = ConvBlock<1>(image_1_0, kernel_1_1);
    auto kerne_1_2 = Kernel<float, 2048, 512, 1, 1>(NULL);
    auto image_1_2 = ConvBlockAddIdentity(image_1_1, kerne_1_2, image_0_2);

    auto kernel_2_0 = Kernel<float, 512, 2048, 1, 1>(NULL);
    auto image_2_0 = ConvBlock<1>(image_1_2, kernel_2_0);
    auto kernel_2_1 = Kernel<float, 512, 512, 3, 3>(NULL);
    auto image_2_1 = ConvBlock<1>(image_2_0, kernel_2_1);
    auto kerne_2_2 = Kernel<float, 2048, 512, 1, 1>(NULL);
    auto image_2_2 = ConvBlockAddIdentity(image_2_1, kerne_2_2, image_1_2);

    return image_2_2;
}

const float *ResNet50::inference(const float *input)
{
    std::memcpy(inputBuffer, input, 3 * 244 * 244 * sizeof(float));

    /// Input
    auto image = Image<float, 3, 244, 244>(inputBuffer);

    auto kernel0 = Kernel<float, 64, 3, 7, 7>(NULL);
    auto imagePreConv = ConvBlock<2>(image, kernel0);

    auto imageMax0 = MaxPool<2>(imagePreConv);

    // Blocks
    auto imageB0 = block0(imageMax0);
    auto imageB1 = block1(imageB0);
    auto imageB2 = block2(imageB1);
    auto imageB3 = block3(imageB2);

    // Output
    auto imageGAP = GlobalAveragePool(imageB3);
    auto array = fullyConnectedLayer(imageGAP.flatten());
    return array.raw_pointer;
}