#include "ResNet50Test.h"

ImageInference::model::test::ResNet50Test::ResNet50Test()
{
}

ImageInference::model::test::ResNet50Test::~ResNet50Test()
{
}

void ImageInference::model::test::ResNet50Test::block0(ImageInference::model::ResNet50 &resnet50, const float *input, float *output)
{
    ImageInference::types::Image<float, 0, 16UL, 64UL, 56UL, 56UL> inputImage(input);
    auto outputImage = resnet50.block0(inputImage);
    auto flatten = outputImage.flatten(); // Get the data order of Channel x Height x Width
    std::copy(flatten.getPointer(), flatten.getPointer() + flatten.size, output);
}

void ImageInference::model::test::ResNet50Test::block1(ImageInference::model::ResNet50 &resnet50, const float *input, float *output)
{
    ImageInference::types::Image<float, 0, 16UL, 256UL, 56UL, 56UL> inputImage(input);
    auto outputImage = resnet50.block1(inputImage);
    auto flatten = outputImage.flatten(); // Get the data order of Channel x Height x Width
    std::copy(flatten.getPointer(), flatten.getPointer() + flatten.size, output);
}

void ImageInference::model::test::ResNet50Test::block2(ImageInference::model::ResNet50 &resnet50, const float *input, float *output)
{
    ImageInference::types::Image<float, 0, 16UL, 512UL, 28UL, 28UL> inputImage(input);
    auto outputImage = resnet50.block2(inputImage);
    auto flatten = outputImage.flatten(); // Get the data order of Channel x Height x Width
    std::copy(flatten.getPointer(), flatten.getPointer() + flatten.size, output);
}

void ImageInference::model::test::ResNet50Test::block3(ImageInference::model::ResNet50 &resnet50, const float *input, float *output)
{
    ImageInference::types::Image<float, 0, 16UL, 1024UL, 14UL, 14UL> inputImage(input);
    auto outputImage = resnet50.block3(inputImage);
    auto flatten = outputImage.flatten(); // Get the data order of Channel x Height x Width
    std::copy(flatten.getPointer(), flatten.getPointer() + flatten.size, output);
}

float ImageInference::model::test::ResNet50Test::relu(float input)
{
    return ImageInference::model::ResNet50::relu<float>(input);
}

float ImageInference::model::test::ResNet50Test::batchNorm(float input, float gamma, float beta, float mean, float variance)
{
    return ImageInference::model::ResNet50::batchNorm<float>(input, gamma, beta, mean, variance);
}

float *ImageInference::model::test::ResNet50Test::getWeight(ImageInference::model::ResNet50 resnet50, size_t index)
{
    return resnet50.getWeight<float>(index);
}
