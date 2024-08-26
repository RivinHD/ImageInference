// SPDX-FileCopyrightText: © 2024 Vincent Gerlach
//
// SPDX-License-Identifier: MIT

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

void ImageInference::model::ResNet50::inference(const float *input, float *output)
{
    // For a 7x7 kernel we need to add padding of 3.
    auto image = ImageInference::types::Image<float, 3, 3, 3, 224, 224>(input);

    auto kernel0 = ImageInference::types::Kernel<float, RESNET50_BLOCK_SIZE, 3, 64, 3, 7, 7>(getWeight<float>(weightIndex::conv1_weight));
    auto batchNorm0 = ImageInference::types::BatchNorm<float, 64>(
        getWeight<float>(weightIndex::bn1_weight),
        getWeight<float>(weightIndex::bn1_bias),
        getWeight<float>(weightIndex::bn1_running_mean),
        getWeight<float>(weightIndex::bn1_running_var));
    // For Max Pooling we need padding of 1 as it applies a 3x3 kernel.
    auto imagePreConv = ImageInference::types::Image<float, 1, RESNET50_BLOCK_SIZE, 64, 112, 112>();
    convBlock<2>(image, kernel0, batchNorm0, imagePreConv);

    // Next is a 1x1 Kernel. Therefore no padding required.
    auto imageMax0 = ImageInference::types::Image<float, 0, RESNET50_BLOCK_SIZE, 64, 56, 56>();
    maxPool<2>(imagePreConv, imageMax0);

    // Blocks
    auto imageB0 = ImageInference::types::Image<float, 0, RESNET50_BLOCK_SIZE, 256, 56, 56>();
    block0(imageMax0, imageB0);
    auto imageB1 = ImageInference::types::Image<float, 0, RESNET50_BLOCK_SIZE, 512, 28, 28>();
    block1(imageB0, imageB1);
    auto imageB2 = ImageInference::types::Image<float, 0, RESNET50_BLOCK_SIZE, 1024, 14, 14>();
    block2(imageB1, imageB2);
    auto imageB3 = ImageInference::types::Image<float, 0, RESNET50_BLOCK_SIZE, 2048, 7, 7>();
    block3(imageB2, imageB3);

    // // Output
    auto imageGAP = ImageInference::types::Image<float, 0, RESNET50_BLOCK_SIZE, 2048, 1, 1>(); // We don't need padding for a fully connected layer.
    globalAveragePool(imageB3, imageGAP);
    auto weight = ImageInference::types::Matrix<float, 1000, 2048>(getWeight<float>(weightIndex::fc_weight));
    auto biasAccumulator = ImageInference::types::Array<float, 1000>(getWeight<float>(weightIndex::fc_bias));
    auto flatten = imageGAP.flatten();
    fullyConnectedLayer<RESNET50_BLOCK_SIZE>(flatten, weight, biasAccumulator);
    std::copy(biasAccumulator.getPointer(), biasAccumulator.getPointer() + biasAccumulator.size, output);
}

ImageInference::types::ScalarType ImageInference::model::ResNet50::getType()
{
    return type;
}
