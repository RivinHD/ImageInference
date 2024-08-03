#include "ResNet50Test.h"

ImageInference::model::test::ResNet50Test::ResNet50Test()
{
}

ImageInference::model::test::ResNet50Test::~ResNet50Test()
{
}

void ImageInference::model::test::ResNet50Test::fullyConnectedLayer(const float *input, const float *weight, const float *bias, float *output)
{
    Array<float, 2048> inputVector(input);
    Matrix<float, 1000, 2048> weightMatrix(weight);
    Array<float, 1000> biasVector(bias);

    auto outputVector = ResNet50::fullyConnectedLayer<float>(inputVector, weightMatrix, biasVector);
    std::copy(outputVector.getPointer(), outputVector.getPointer() + outputVector.size, output);
}

float ImageInference::model::test::ResNet50Test::relu(float input)
{
    return ResNet50::relu<float>(input);
}

float ImageInference::model::test::ResNet50Test::batchNorm(float input, float gamma, float beta, float mean, float variance)
{
    return ResNet50::batchNorm<float>(input, gamma, beta, mean, variance);
}
