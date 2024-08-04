#include "ResNet50Test.h"

ImageInference::model::test::ResNet50Test::ResNet50Test()
{
}

ImageInference::model::test::ResNet50Test::~ResNet50Test()
{
}

float ImageInference::model::test::ResNet50Test::relu(float input)
{
    return ImageInference::model::ResNet50::relu<float>(input);
}

float ImageInference::model::test::ResNet50Test::batchNorm(float input, float gamma, float beta, float mean, float variance)
{
    return ImageInference::model::ResNet50::batchNorm<float>(input, gamma, beta, mean, variance);
}
