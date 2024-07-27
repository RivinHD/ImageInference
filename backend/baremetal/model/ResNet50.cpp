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

ImageInference::model::ResNet50::ResNet50(const std::vector<void*> &weights, ScalarType type)
    : weights(weights), type(type)
{
    if (type == ScalarType::Float)
    {
        inputBuffer = new float[MAX_RESNET50_SIZE];
        outputBuffer = new float[MAX_RESNET50_SIZE];
    }
    else if (type == ScalarType::Int8)
    {
        inputBuffer = new int8_t[MAX_RESNET50_SIZE];
        outputBuffer = new int8_t[MAX_RESNET50_SIZE];
    }
}

ImageInference::model::ResNet50::~ResNet50()
{
    if (type == ScalarType::Float)
    {
        delete[] static_cast<float *>(inputBuffer);
        delete[] static_cast<float *>(outputBuffer);
    }
    else if (type == ScalarType::Int8)
    {
        delete[] static_cast<int8_t *>(inputBuffer);
        delete[] static_cast<int8_t *>(outputBuffer);
    }
}

const float *ImageInference::model::ResNet50::inference(const float *input)
{
    std::memcpy(inputBuffer, input, 3 * 244 * 244 * sizeof(float));

    /// Input
    auto image = Image<float, 3, 244, 244>(static_cast<float *>(inputBuffer));

    auto kernel0 = Kernel<float, 64, 3, 7, 7>(getWeight<float>(weightIndex::conv1_weight));
    auto batchNorm0 = BatchNorm<float, 64>(getWeight<float>(weightIndex::bn1_weight), getWeight<float>(weightIndex::bn1_bias));
    auto imagePreConv = ConvBlock<2>(image, kernel0, batchNorm0);

    auto imageMax0 = MaxPool<2>(imagePreConv);

    // Blocks
    auto imageB0 = block0(imageMax0);
    auto imageB1 = block1(imageB0);
    auto imageB2 = block2(imageB1);
    auto imageB3 = block3(imageB2);

    // Output
    auto imageGAP = GlobalAveragePool(imageB3);
    auto weights = Matrix<float, 1000, 2048>(getWeight<float>(weightIndex::fc_weight));
    auto biases = Array<float, 1000>(getWeight<float>(weightIndex::fc_bias));
    auto array = fullyConnectedLayer(imageGAP.flatten(), weights, biases);
    return array.raw_pointer;
}

const int8_t *ImageInference::model::ResNet50::inference(const int8_t *input)
{
    std::memcpy(inputBuffer, input, 3 * 244 * 244 * sizeof(int8_t));

    /// Input
    auto image = Image<int8_t, 3, 244, 244>(static_cast<int8_t *>(inputBuffer));

    auto kernel0 = Kernel<int8_t, 64, 3, 7, 7>(getWeight<int8_t>(weightIndex::conv1_weight));
    auto batchNorm0 = BatchNorm<int8_t, 64>(getWeight<int8_t>(weightIndex::bn1_weight), getWeight<int8_t>(weightIndex::bn1_bias));
    auto imagePreConv = ConvBlock<2>(image, kernel0, batchNorm0);

    auto imageMax0 = MaxPool<2>(imagePreConv);

    // Blocks
    auto imageB0 = block0(imageMax0);
    auto imageB1 = block1(imageB0);
    auto imageB2 = block2(imageB1);
    auto imageB3 = block3(imageB2);

    // Output
    auto imageGAP = GlobalAveragePool(imageB3);
    auto weights = Matrix<int8_t, 1000, 2048>(getWeight<int8_t>(weightIndex::fc_weight));
    auto biases = Array<int8_t, 1000>(getWeight<int8_t>(weightIndex::fc_bias));
    auto array = fullyConnectedLayer(imageGAP.flatten(), weights, biases);
    return array.raw_pointer;
}

ImageInference::types::ScalarType ImageInference::model::ResNet50::getType()
{
    return type;
}
