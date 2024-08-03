#ifndef IMAGEINFERENCE_RESNET50TEST_H
#define IMAGEINFERENCE_RESNET50TEST_H

#include <algorithm>
#include "../ResNet50.h"

namespace ImageInference
{
    namespace model
    {
        namespace test
        {
            using ImageInference::model::ResNet50;
            using ImageInference::types::BatchNorm;
            using ImageInference::types::Image;
            using ImageInference::types::Kernel;

            class ResNet50Test
            {
            public:
                ResNet50Test();
                ~ResNet50Test();

                template <size_t TStride, size_t TInPadding, size_t TBlockSize,
                          size_t TOutChannels, size_t TInChannels,
                          size_t THeight, size_t TWidth,
                          size_t TKernelHeight, size_t TKernelWidth>
                static void convRelu(const float *input, const float *kernel, float *output)
                {
                    Image<float, TInPadding, TBlockSize, TInChannels, THeight, TWidth> inputImage(input);
                    Kernel<float, TBlockSize, TBlockSize, TOutChannels, TInChannels, TKernelHeight, TKernelWidth> inputKernel(kernel);

                    float gamma[TOutChannels] = {0};
                    float beta[TOutChannels] = {0};
                    std::fill(gamma, gamma + TOutChannels, 1.0f);
                    BatchNorm<float, TOutChannels> batchNorm(gamma, beta);

                    auto outputImage = ResNet50::convBlock<TStride, 0>(inputImage, inputKernel, batchNorm);
                    auto flatten = outputImage.flatten(); // Get the data order of Channel x Height x Width
                    std::copy(flatten.getPointer(), flatten.getPointer() + flatten.size, output);
                }

                template <size_t TStride, size_t TInPadding, size_t TBlockSize,
                          size_t TInChannels, size_t THeight, size_t TWidth>
                static void maxPool(const float *input, float *output)
                {
                    Image<float, TInPadding, TBlockSize, TInChannels, THeight, TWidth> inputImage(input);
                    auto outputImage = ResNet50::maxPool<TStride, 0>(inputImage);
                    auto flatten = outputImage.flatten(); // Get the data order of Channel x Height x Width
                    std::copy(flatten.getPointer(), flatten.getPointer() + flatten.size, output);
                }

                template <size_t TInPadding, size_t TBlockSize,
                          size_t TInChannels, size_t THeight, size_t TWidth>
                static void globalAveragePool(const float *input, float *output)
                {
                    Image<float, TInPadding, TBlockSize, TInChannels, THeight, TWidth> inputImage(input);
                    auto outputImage = ResNet50::globalAveragePool<0>(inputImage);
                    auto flatten = outputImage.flatten(); // Get the data order of Channel x Height x Width
                    std::copy(flatten.getPointer(), flatten.getPointer() + flatten.size, output);
                }

                static void fullyConnectedLayer(const float *input, const float *weight, const float *bias, float *output);

                static float relu(float input);

                static float batchNorm(float input, float gamma, float beta, float mean, float variance);
            };
        }
    }
}

#endif // IMAGEINFERENCE_RESNET50TEST_H