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
#include <cstring>
#include <new>
#include <algorithm>

ImageInference::model::ResNet50::ResNet50(const std::vector<void *> &modelWeights, ImageInference::types::ScalarType type)
    : modelWeights(modelWeights), type(type)
{
    libxsmm_init();
}

ImageInference::model::ResNet50::~ResNet50()
{
    libxsmm_finalize();
}

void ImageInference::model::ResNet50::inference(const float *input, float* output)
{
    // For a 7x7 kernel we need to add padding of 3.
    auto image = ImageInference::types::Image<float, 3, 3, 3, 224, 224>(input);

    auto kernel0 = ImageInference::types::Kernel<float, RESNET50_BLOCK_SIZE, 3, 64, 3, 7, 7>(getWeight<float>(weightIndex::conv1_weight));
    auto batchNorm0 = ImageInference::types::BatchNorm<float, 64>(getWeight<float>(weightIndex::bn1_weight), getWeight<float>(weightIndex::bn1_bias));
    // For Max Pooling we need padding of 1 as it applies a 3x3 kernel.
    auto imagePreConv = convBlock<2, 1>(image, kernel0, batchNorm0);
    
    // Next is a 1x1 Kernel. Therefore no padding required.
    auto imageMax0 = maxPool<2, 0>(imagePreConv);

    // Blocks
    auto imageB0 = block0(imageMax0);
    auto imageB1 = block1(imageB0);
    auto imageB2 = block2(imageB1);
    auto imageB3 = block3(imageB2);

    // // Output
    auto imageGAP = globalAveragePool<0>(imageB3);  // We don't need padding for a fully connected layer.
    auto weight = ImageInference::types::Matrix<float, 1000, 2048>(getWeight<float>(weightIndex::fc_weight));
    auto biasAccumulator = ImageInference::types::Array<float, 1000>(getWeight<float>(weightIndex::fc_bias));
    auto flatten = imageGAP.flatten();
    fullyConnectedLayer(flatten, weight, biasAccumulator);
    std::copy(biasAccumulator.getPointer(), biasAccumulator.getPointer() + biasAccumulator.size, output);
}

ImageInference::types::ScalarType ImageInference::model::ResNet50::getType()
{
    return type;
}
