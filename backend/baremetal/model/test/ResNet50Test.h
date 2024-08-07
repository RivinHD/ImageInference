#ifndef IMAGEINFERENCE_RESNET50TEST_H
#define IMAGEINFERENCE_RESNET50TEST_H

#include <algorithm>
#include "../../types/Image.h"
#include "../../types/Kernel.h"
#include "../../types/Array.h"
#include "../../types/BatchNorm.h"
#include "../../types/Matrix.h"
#include "../ResNet50.h"

namespace ImageInference
{
    namespace model
    {
        namespace test
        {
            class ResNet50Test
            {
            public:
                ResNet50Test();
                ~ResNet50Test();

                static void block0(ImageInference::model::ResNet50 &resnet50, const float *input, float *output);
                static void block1(ImageInference::model::ResNet50 &resnet50, const float *input, float *output);
                static void block2(ImageInference::model::ResNet50 &resnet50, const float *input, float *output);
                static void block3(ImageInference::model::ResNet50 &resnet50, const float *input, float *output);

                template <size_t TStride, size_t TInPadding, size_t TBlockSize,
                          size_t TOutChannels, size_t TInChannels,
                          size_t THeight, size_t TWidth,
                          size_t TKernelHeight, size_t TKernelWidth>
                static void convBlock(const float *input, const float *kernel, const float *batchGamma, const float *batchBeta,const float *batchMean, const float *batchVariance, float *output)
                {
                    ImageInference::types::Image<float, TInPadding, TBlockSize, TInChannels, THeight, TWidth> inputImage(input);
                    ImageInference::types::Kernel<float, TBlockSize, TBlockSize, TOutChannels, TInChannels, TKernelHeight, TKernelWidth> inputKernel(kernel);
                    ImageInference::types::BatchNorm<float, TOutChannels> batchNorm(batchGamma, batchBeta, batchMean, batchVariance);

                    auto outputImage = ImageInference::types::Image<float, 0, TBlockSize, TOutChannels, THeight / TStride, TWidth / TStride>();
                    ImageInference::model::ResNet50::convBlock<TStride>(inputImage, inputKernel, batchNorm, outputImage);
                    auto flatten = outputImage.flatten(); // Get the data order of Channel x Height x Width
                    std::copy(flatten.getPointer(), flatten.getPointer() + flatten.size, output);
                }

                template <size_t TInPadding, size_t TBlockSize,
                          size_t TOutChannels, size_t TInChannels,
                          size_t THeight, size_t TWidth,
                          size_t TKernelHeight, size_t TKernelWidth>
                static void convBlockShortcut(
                    const float *input,
                    const float *kernel,
                    const float *batchGamma,
                    const float *batchBeta,
                    const float *batchMean,
                    const float *batchVariance,
                    const float *shortcut,
                    float *output)
                {
                    ImageInference::types::Image<float, TInPadding, TBlockSize, TInChannels, THeight, TWidth> inputImage(input);
                    ImageInference::types::Kernel<float, TBlockSize, TBlockSize, TOutChannels, TInChannels, TKernelHeight, TKernelWidth> inputKernel(kernel);
                    ImageInference::types::BatchNorm<float, TOutChannels> batchNorm(batchGamma, batchBeta, batchMean, batchVariance);
                    ImageInference::types::Image<float, 0, TBlockSize, TOutChannels, THeight, TWidth> shortcutImage(shortcut);

                    auto outputImage = ImageInference::types::Image<float, 0, TBlockSize, TOutChannels, THeight, TWidth>();
                    ImageInference::model::ResNet50::convBlockAddIdentity(inputImage, inputKernel, batchNorm, shortcutImage, outputImage);
                    auto flatten = outputImage.flatten(); // Get the data order of Channel x Height x Width
                    std::copy(flatten.getPointer(), flatten.getPointer() + flatten.size, output);
                }

                template <size_t TStride, size_t ShortcutDimExpand, size_t TInPadding, size_t TBlockSize,
                          size_t TOutChannels, size_t TInChannels,
                          size_t THeight, size_t TWidth,
                          size_t TKernelHeight, size_t TKernelWidth>
                static void convBlockProjection(
                    const float *input,
                    const float *kernel,
                    const float *batchGamma,
                    const float *batchBeta,
                    const float *batchMean,
                    const float *batchVariance,
                    const float *shortcut,
                    const float *projectionKernel,
                    const float *projectionBatchGamma,
                    const float *projectionBatchBeta,
                    const float *projectionBatchMean,
                    const float *projectionBatchVariance,
                    float *output)
                {
                    ImageInference::types::Image<float, TInPadding, TBlockSize, TInChannels, THeight / TStride, TWidth / TStride> inputImage(input);
                    ImageInference::types::Kernel<float, TBlockSize, TBlockSize, TOutChannels, TInChannels, TKernelHeight, TKernelWidth> inputKernel(kernel);
                    ImageInference::types::BatchNorm<float, TOutChannels> batchNorm(batchGamma, batchBeta, batchMean, batchVariance);
                    ImageInference::types::Image<float, 0, TBlockSize, TOutChannels / ShortcutDimExpand, THeight, TWidth> shortcutImage(shortcut);
                    ImageInference::types::Kernel<float, TBlockSize, TBlockSize, TOutChannels, TOutChannels / ShortcutDimExpand, 1, 1> projectionKernelImage(projectionKernel);
                    ImageInference::types::BatchNorm<float, TOutChannels> projectionBatchNorm(projectionBatchGamma, projectionBatchBeta, projectionBatchMean, projectionBatchVariance);

                    auto outputImage = ImageInference::types::Image<float, 0, TBlockSize, TOutChannels, THeight / TStride, TWidth / TStride>();
                    ImageInference::model::ResNet50::convBlockAddProjection<TStride, ShortcutDimExpand>(inputImage, inputKernel, batchNorm, shortcutImage, projectionKernelImage, projectionBatchNorm, outputImage);
                    auto flatten = outputImage.flatten(); // Get the data order of Channel x Height x Width
                    std::copy(flatten.getPointer(), flatten.getPointer() + flatten.size, output);
                }

                template <size_t TStride, size_t TInPadding, size_t TBlockSize,
                          size_t TInChannels, size_t THeight, size_t TWidth>
                static void maxPool(const float *input, float *output)
                {
                    ImageInference::types::Image<float, TInPadding, TBlockSize, TInChannels, THeight, TWidth> inputImage(input);
                    auto outputImage = ImageInference::types::Image<float, 0, TBlockSize, TInChannels, THeight / TStride, TWidth / TStride>();
                    ImageInference::model::ResNet50::maxPool<TStride>(inputImage, outputImage);
                    auto flatten = outputImage.flatten(); // Get the data order of Channel x Height x Width
                    std::copy(flatten.getPointer(), flatten.getPointer() + flatten.size, output);
                }

                template <size_t TInPadding, size_t TBlockSize,
                          size_t TInChannels, size_t THeight, size_t TWidth>
                static void globalAveragePool(const float *input, float *output)
                {
                    ImageInference::types::Image<float, TInPadding, TBlockSize, TInChannels, THeight, TWidth> inputImage(input);
                    auto outputImage = ImageInference::types::Image<float, 0, TBlockSize, TInChannels, 1, 1>();
                    ImageInference::model::ResNet50::globalAveragePool(inputImage, outputImage);
                    auto flatten = outputImage.flatten(); // Get the data order of Channel x Height x Width
                    std::copy(flatten.getPointer(), flatten.getPointer() + flatten.size, output);
                }

                template <size_t TColumns, size_t TRows>
                static void fullyConnectedLayer(const float *input, const float *weight, const float *bias, float *output)
                {
                    ImageInference::types::Array<float, TRows> inputVector(input);
                    ImageInference::types::Matrix<float, TColumns, TRows> weightMatrix(weight);
                    ImageInference::types::Array<float, TColumns> biasAccumulator(bias);

                    ResNet50::fullyConnectedLayer<64>(inputVector, weightMatrix, biasAccumulator);
                    std::copy(biasAccumulator.getPointer(), biasAccumulator.getPointer() + biasAccumulator.size, output);
                }

                static float relu(float input);

                static float batchNorm(float input, float gammaVariance, float beta, float mean);

                static float *getWeight(ImageInference::model::ResNet50 resnet50, size_t index);
            };
        }
    }
}

#endif // IMAGEINFERENCE_RESNET50TEST_H