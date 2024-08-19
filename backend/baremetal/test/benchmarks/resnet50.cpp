#ifndef USE_ATEN_LIB
#define USE_ATEN_LIB
#endif // !USE_ATEN_LIB

#include <ATen/ATen.h>
#include <torch/library.h>
#include <iostream>
#include <iomanip>
#include "../../model/test/ResNet50Test.h"
#include <benchmark/benchmark.h>

namespace ImageInference
{
    namespace test
    {
        using at::Tensor;

        template <size_t TStride, size_t TInPadding, size_t TBlockSize,
                  size_t TOutChannels, size_t TInChannels,
                  size_t THeight, size_t TWidth,
                  size_t TKernelHeight, size_t TKernelWidth>
        class ConvolutionFixture : public benchmark::Fixture
        {
        public:
            constexpr static size_t stride = TStride;
            constexpr static size_t inPadding = TInPadding;
            constexpr static size_t blockSize = TBlockSize;
            constexpr static size_t outChannels = TOutChannels;
            constexpr static size_t inChannels = TInChannels;
            constexpr static size_t height = THeight;
            constexpr static size_t width = TWidth;
            constexpr static size_t kernelHeight = TKernelHeight;
            constexpr static size_t kernelWidth = TKernelWidth;

            Tensor in;
            Tensor weight;
            Tensor batchGamma;
            Tensor batchBeta;
            Tensor batchMean;
            Tensor batchVar;

            float *inPtr;
            float *weightPtr;
            float *batchGammaPtr;
            float *batchBetaPtr;
            float *batchMeanPtr;
            float *batchVarPtr;

            ImageInference::types::Image<float, inPadding, blockSize, inChannels, height, width> *inputImage;
            ImageInference::types::Image<float, 0, blockSize, outChannels, height / stride, width / stride> *outputImage;

            void SetUp(::benchmark::State &state)
            {
                in = at::rand({1, inChannels, height, width});
                weight = at::rand({outChannels, inChannels, kernelHeight, kernelWidth});
                batchGamma = at::rand({outChannels});
                batchBeta = at::rand({outChannels});
                batchMean = at::rand({outChannels});
                batchVar = at::rand({outChannels});

                inPtr = in.mutable_data_ptr<float>();
                weightPtr = weight.mutable_data_ptr<float>();
                batchGammaPtr = batchGamma.mutable_data_ptr<float>();
                batchBetaPtr = batchBeta.mutable_data_ptr<float>();
                batchMeanPtr = batchMean.mutable_data_ptr<float>();
                batchVarPtr = batchVar.mutable_data_ptr<float>();

                inputImage = new ImageInference::types::Image<float, inPadding, blockSize, inChannels, height, width>(inPtr);
                outputImage = new ImageInference::types::Image<float, 0, blockSize, outChannels, height / stride, width / stride>();
            }

            void TearDown(::benchmark::State &state)
            {
                delete inputImage;
                delete outputImage;
            }
        };

        template <size_t TInPadding, size_t TBlockSize,
                  size_t TOutChannels, size_t TInChannels,
                  size_t THeight, size_t TWidth,
                  size_t TKernelHeight, size_t TKernelWidth>
        class ConvolutionShortcutFixture : public benchmark::Fixture
        {
        public:
            constexpr static size_t inPadding = TInPadding;
            constexpr static size_t blockSize = TBlockSize;
            constexpr static size_t outChannels = TOutChannels;
            constexpr static size_t inChannels = TInChannels;
            constexpr static size_t height = THeight;
            constexpr static size_t width = TWidth;
            constexpr static size_t kernelHeight = TKernelHeight;
            constexpr static size_t kernelWidth = TKernelWidth;

            Tensor in;
            Tensor weight;
            Tensor batchGamma;
            Tensor batchBeta;
            Tensor batchMean;
            Tensor batchVar;
            Tensor shortcut;

            float *inPtr;
            float *weightPtr;
            float *batchGammaPtr;
            float *batchBetaPtr;
            float *batchMeanPtr;
            float *batchVarPtr;
            float *shortcutPtr;

            ImageInference::types::Image<float, inPadding, blockSize, inChannels, height, width> *inputImage;
            ImageInference::types::Image<float, 0, blockSize, outChannels, height, width> *shortcutImage;
            ImageInference::types::Image<float, 0, blockSize, outChannels, height, width> *outputImage;

            void SetUp(::benchmark::State &state)
            {

                in = at::rand({1, inChannels, height, width});
                weight = at::rand({outChannels, inChannels, kernelHeight, kernelWidth});
                batchGamma = at::rand({outChannels});
                batchBeta = at::rand({outChannels});
                batchMean = at::rand({outChannels});
                batchVar = at::rand({outChannels});
                shortcut = at::rand({outChannels, height, width});

                inPtr = in.mutable_data_ptr<float>();
                weightPtr = weight.mutable_data_ptr<float>();
                batchGammaPtr = batchGamma.mutable_data_ptr<float>();
                batchBetaPtr = batchBeta.mutable_data_ptr<float>();
                batchMeanPtr = batchMean.mutable_data_ptr<float>();
                batchVarPtr = batchVar.mutable_data_ptr<float>();
                shortcutPtr = shortcut.mutable_data_ptr<float>();

                inputImage = new ImageInference::types::Image<float, inPadding, blockSize, inChannels, height, width>(inPtr);
                shortcutImage = new ImageInference::types::Image<float, 0, blockSize, outChannels, height, width>(shortcutPtr);
                outputImage = new ImageInference::types::Image<float, 0, blockSize, outChannels, height, width>();
            }

            void TearDown(::benchmark::State &state)
            {
                delete inputImage;
                delete shortcutImage;
                delete outputImage;
            }
        };

        template <size_t TStride, size_t TInPadding, size_t TBlockSize,
                  size_t TOutChannels, size_t TInChannels, size_t TShortcutChannels,
                  size_t THeight, size_t TWidth,
                  size_t TKernelHeight, size_t TKernelWidth>
        class ConvolutionProjectionFixture : public benchmark::Fixture
        {
        public:
            constexpr static size_t stride = TStride;
            constexpr static size_t inPadding = TInPadding;
            constexpr static size_t blockSize = TBlockSize;
            constexpr static size_t outChannels = TOutChannels;
            constexpr static size_t inChannels = TInChannels;
            constexpr static size_t shortcutChannels = TShortcutChannels;
            constexpr static size_t height = THeight;
            constexpr static size_t width = TWidth;
            constexpr static size_t kernelHeight = TKernelHeight;
            constexpr static size_t kernelWidth = TKernelWidth;
            constexpr static size_t shortcutDimExpand = outChannels / shortcutChannels;

            Tensor in;
            Tensor weight;
            Tensor batchGamma;
            Tensor batchBeta;
            Tensor batchMean;
            Tensor batchVar;
            Tensor shortcut;
            Tensor projectionWeight;
            Tensor projectionBatchGamma;
            Tensor projectionBatchBeta;
            Tensor projectionBatchMean;
            Tensor projectionBatchVar;

            float *inPtr;
            float *weightPtr;
            float *batchGammaPtr;
            float *batchBetaPtr;
            float *batchMeanPtr;
            float *batchVarPtr;
            float *shortcutPtr;
            float *projectionWeightPtr;
            float *projectionBatchGammaPtr;
            float *projectionBatchBetaPtr;
            float *projectionBatchMeanPtr;
            float *projectionBatchVarPtr;

            ImageInference::types::Image<float, inPadding, blockSize, inChannels, height / stride, width / stride> *inputImage;
            ImageInference::types::Image<float, 0, blockSize, outChannels / shortcutDimExpand, height, width> *shortcutImage;
            ImageInference::types::Image<float, 0, blockSize, outChannels, height / stride, width / stride> *outputImage;

            void SetUp(::benchmark::State &state)
            {
                in = at::rand({1, inChannels, height / stride, width / stride});
                weight = at::rand({outChannels, inChannels, kernelHeight, kernelWidth});
                batchGamma = at::rand({outChannels});
                batchBeta = at::rand({outChannels});
                batchMean = at::rand({outChannels});
                batchVar = at::rand({outChannels});
                shortcut = at::rand({1, shortcutChannels, height, width});
                projectionWeight = at::rand({outChannels, shortcutChannels, 1, 1});
                projectionBatchGamma = at::rand({outChannels});
                projectionBatchBeta = at::rand({outChannels});
                projectionBatchMean = at::rand({outChannels});
                projectionBatchVar = at::rand({outChannels});

                inPtr = in.mutable_data_ptr<float>();
                weightPtr = weight.mutable_data_ptr<float>();
                batchGammaPtr = batchGamma.mutable_data_ptr<float>();
                batchBetaPtr = batchBeta.mutable_data_ptr<float>();
                batchMeanPtr = batchMean.mutable_data_ptr<float>();
                batchVarPtr = batchVar.mutable_data_ptr<float>();
                shortcutPtr = shortcut.mutable_data_ptr<float>();
                projectionWeightPtr = projectionWeight.mutable_data_ptr<float>();
                projectionBatchGammaPtr = projectionBatchGamma.mutable_data_ptr<float>();
                projectionBatchBetaPtr = projectionBatchBeta.mutable_data_ptr<float>();
                projectionBatchMeanPtr = projectionBatchMean.mutable_data_ptr<float>();
                projectionBatchVarPtr = projectionBatchVar.mutable_data_ptr<float>();

                inputImage = new ImageInference::types::Image<float, inPadding, blockSize, inChannels, height / stride, width / stride>(inPtr);
                shortcutImage = new ImageInference::types::Image<float, 0, blockSize, outChannels / shortcutDimExpand, height, width>(shortcutPtr);
                outputImage = new ImageInference::types::Image<float, 0, blockSize, outChannels, height / stride, width / stride>();
            }

            void TearDown(::benchmark::State &state)
            {
                delete inputImage;
                delete shortcutImage;
                delete outputImage;
            }
        };

        template <size_t TStride, size_t TInPadding, size_t TBlockSize,
                  size_t TChannels, size_t THeight, size_t TWidth>
        class MaxPoolFixture : public benchmark::Fixture
        {
        public:
            constexpr static size_t stride = TStride;
            constexpr static size_t inPadding = TInPadding;
            constexpr static size_t blockSize = TBlockSize;
            constexpr static size_t channels = TChannels;
            constexpr static size_t height = THeight;
            constexpr static size_t width = TWidth;

            Tensor in;

            float *inPtr;

            ImageInference::types::Image<float, inPadding, blockSize, channels, height, width> *inputImage;
            ImageInference::types::Image<float, 0, blockSize, channels, height / stride, width / stride> *outputImage;

            MaxPoolFixture() {}
            ~MaxPoolFixture() {}

            void SetUp(::benchmark::State &state)
            {
                in = at::rand({channels, height, width});

                inPtr = in.mutable_data_ptr<float>();

                inputImage = new ImageInference::types::Image<float, inPadding, blockSize, channels, height, width>(inPtr);
                outputImage = new ImageInference::types::Image<float, 0, blockSize, channels, height / stride, width / stride>();
            }

            void TearDown(::benchmark::State &state)
            {
                delete inputImage;
                delete outputImage;
            }
        };

        template <size_t TInPadding, size_t TBlockSize,
                  size_t TChannels, size_t THeight, size_t TWidth>
        class GlobalAveragePoolFixture : public benchmark::Fixture
        {
        public:
            constexpr static size_t inPadding = TInPadding;
            constexpr static size_t blockSize = TBlockSize;
            constexpr static size_t channels = TChannels;
            constexpr static size_t height = THeight;
            constexpr static size_t width = TWidth;

            Tensor in;

            float *inPtr;

            ImageInference::types::Image<float, inPadding, blockSize, channels, height, width> *inputImage;
            ImageInference::types::Image<float, 0, blockSize, channels, 1, 1> *outputImage;

            GlobalAveragePoolFixture() {}
            ~GlobalAveragePoolFixture() {}

            void SetUp(::benchmark::State &state)
            {

                in = at::randn({channels, height, width});

                inPtr = in.mutable_data_ptr<float>();

                inputImage = new ImageInference::types::Image<float, inPadding, blockSize, channels, height, width>(inPtr);
                outputImage = new ImageInference::types::Image<float, 0, blockSize, channels, 1, 1>();
            }

            void TearDown(::benchmark::State &state)
            {
                delete inputImage;
                delete outputImage;
            }
        };

        template <size_t TInDim, size_t TOutDim>
        class FullyConnectedFixture : public benchmark::Fixture
        {
        public:
            constexpr static size_t inDim = TInDim;
            constexpr static size_t outDim = TOutDim;

            Tensor in;
            Tensor weight;
            Tensor bias;
            float *inPtr;
            float *weightPtr;
            float *biasPtr;
            float *outPtr;

            ImageInference::types::Array<float, inDim> *inputVector;
            ImageInference::types::Array<float, outDim> *biasAccumulator;

            void SetUp(::benchmark::State &state) override
            {
                in = at::randn({TInDim});
                weight = at::randn({TOutDim, TInDim});
                bias = at::randn({TOutDim});

                inPtr = in.mutable_data_ptr<float>();
                weightPtr = weight.mutable_data_ptr<float>();
                biasPtr = bias.mutable_data_ptr<float>();

                inputVector = new ImageInference::types::Array<float, inDim>(inPtr);
                biasAccumulator = new ImageInference::types::Array<float, outDim>(biasPtr);
            }

            void TearDown(::benchmark::State &state) override
            {
                delete inputVector;
                delete biasAccumulator;
            }
        };

        // Args: TStride, TInPadding, TBlockSize, TOutChannels, TInChannels, THeight, TWidth, TKernelHeight, TKernelWidth
        BENCHMARK_TEMPLATE_F(ConvolutionFixture, Convolution_Custom, 1, 1, 32, 64, 64, 224, 224, 3, 3)
        (benchmark::State &st)
        {
            for (auto _ : st)
            {
                ImageInference::types::Kernel<float, blockSize, blockSize, outChannels, inChannels, kernelHeight, kernelWidth> inputKernel(weightPtr);
                ImageInference::types::BatchNorm<float, outChannels> batchNorm(batchGammaPtr, batchBetaPtr, batchMeanPtr, batchVarPtr);
                ImageInference::model::ResNet50::convBlock<stride, 0>(*inputImage, inputKernel, batchNorm, *outputImage);
            }
        };

        // Args: TStride, TInPadding, TBlockSize, TOutChannels, TInChannels, THeight, TWidth, TKernelHeight, TKernelWidth
        BENCHMARK_TEMPLATE_F(ConvolutionFixture, Convolution_ATen, 1, 1, 32, 64, 64, 224, 224, 3, 3)
        (benchmark::State &st)
        {
            for (auto _ : st)
            {
                Tensor expected = at::conv2d(in, weight, {}, stride, inPadding);
                expected = at::batch_norm(expected, batchGamma, batchBeta, batchMean, batchVar, false, 0.1, 1e-5, false);
                expected = at::relu(expected);
            }
        };

        // Args: TInPadding, TBlockSize, TOutChannels, TInChannels, THeight, TWidth, TKernelHeight, TKernelWidth
        BENCHMARK_TEMPLATE_F(ConvolutionShortcutFixture, Convolution_Shortcut_Custom, 1, 32, 64, 64, 224, 224, 3, 3)
        (benchmark::State &st)
        {
            for (auto _ : st)
            {
                ImageInference::types::Kernel<float, blockSize, blockSize, outChannels, inChannels, kernelHeight, kernelWidth> inputKernel(weightPtr);
                ImageInference::types::BatchNorm<float, outChannels> batchNorm(batchGammaPtr, batchBetaPtr, batchMeanPtr, batchVarPtr);
                ImageInference::model::ResNet50::convBlockAddIdentity<0>(*inputImage, inputKernel, batchNorm, *shortcutImage, *outputImage);
            }
        };

        // Args: TInPadding, TBlockSize, TOutChannels, TInChannels, THeight, TWidth, TKernelHeight, TKernelWidth
        BENCHMARK_TEMPLATE_F(ConvolutionShortcutFixture, Convolution_Shortcut_ATen, 1, 32, 64, 64, 224, 224, 3, 3)
        (benchmark::State &st)
        {
            for (auto _ : st)
            {
                Tensor expected = at::conv2d(in, weight, {}, 1, inPadding);
                expected = at::batch_norm(expected, batchGamma, batchBeta, batchMean, batchVar, false, 0.1, 1e-5, false);
                expected += shortcut;
                expected = at::relu(expected);
            }
        };

        // Args: TStride, TInPadding, TBlockSize, TOutChannels, TInChannels, TShortcutChannels, THeight, TWidth, TKernelHeight, TKernelWidth
        BENCHMARK_TEMPLATE_F(ConvolutionProjectionFixture, Convolution_Projection_Custom, 1, 1, 32, 64, 64, 32, 224, 224, 3, 3)
        (benchmark::State &st)
        {
            for (auto _ : st)
            {
                ImageInference::types::Kernel<float, blockSize, blockSize, outChannels, inChannels, kernelHeight, kernelWidth> inputKernel(weightPtr);
                ImageInference::types::BatchNorm<float, outChannels> batchNorm(batchGammaPtr, batchBetaPtr, batchMeanPtr, batchVarPtr);
                ImageInference::types::Kernel<float, blockSize, blockSize, outChannels, outChannels / shortcutDimExpand, 1, 1> projectionKernel(projectionWeightPtr);
                ImageInference::types::BatchNorm<float, outChannels> projectionBatchNorm(projectionBatchGammaPtr, projectionBatchBetaPtr, projectionBatchMeanPtr, projectionBatchVarPtr);

                ImageInference::model::ResNet50::convBlockAddProjection<stride, shortcutDimExpand>(*inputImage, inputKernel, batchNorm, *shortcutImage, projectionKernel, projectionBatchNorm, *outputImage);
            }
        };

        // Args: TStride, TInPadding, TBlockSize, TOutChannels, TInChannels, TShortcutChannels, THeight, TWidth, TKernelHeight, TKernelWidth
        BENCHMARK_TEMPLATE_F(ConvolutionProjectionFixture, Convolution_Projection_ATen, 1, 1, 32, 64, 64, 32, 224, 224, 3, 3)
        (benchmark::State &st)
        {
            for (auto _ : st)
            {
                Tensor expected = at::conv2d(in, weight, {}, 1, inPadding);
                expected = at::batch_norm(expected, batchGamma, batchBeta, batchMean, batchVar, false, 0.1, 1e-5, false);
                Tensor projection = at::conv2d(shortcut, projectionWeight, {}, stride);
                projection = at::batch_norm(projection, projectionBatchGamma, projectionBatchBeta, projectionBatchMean, projectionBatchVar, false, 0.1, 1e-5, false);
                expected += projection;
                expected = at::relu(expected);
            }
        };

        // Args: TStride, TInPadding, TBlockSize, TChannels, THeight, TWidth
        BENCHMARK_TEMPLATE_F(MaxPoolFixture, MaxPool_Custom, 1, 1, 32, 64, 224, 224)
        (benchmark::State &st)
        {
            for (auto _ : st)
            {
                ImageInference::model::ResNet50::maxPool<stride>(*inputImage, *outputImage);
            }
        };

        // Args: TStride, TInPadding, TBlockSize, TChannels, THeight, TWidth
        BENCHMARK_TEMPLATE_F(MaxPoolFixture, MaxPool_ATen, 1, 1, 32, 64, 224, 224)
        (benchmark::State &st)
        {
            for (auto _ : st)
            {
                Tensor expected = at::max_pool2d(in, {3, 3}, stride, inPadding);
            }
        };

        // Args: TInPadding, TBlockSize, TChannels, THeight, TWidth
        BENCHMARK_TEMPLATE_F(GlobalAveragePoolFixture, GlobalAveragePool_Custom, 1, 32, 64, 224, 224)
        (benchmark::State &st)
        {
            for (auto _ : st)
            {
                ImageInference::model::ResNet50::globalAveragePool(*inputImage, *outputImage);
            }
        };

        // Args: TInPadding, TBlockSize, TChannels, THeight, TWidth
        BENCHMARK_TEMPLATE_F(GlobalAveragePoolFixture, GlobalAveragePool_ATen, 1, 32, 64, 224, 224)
        (benchmark::State &st)
        {
            for (auto _ : st)
            {
                Tensor expected = at::adaptive_avg_pool2d(in, {1, 1});
            }
        };

        BENCHMARK_TEMPLATE_F(FullyConnectedFixture, FullyConnected_Custom, 2048, 1000)
        (benchmark::State &st)
        {
            for (auto _ : st)
            {
                ImageInference::types::Matrix<float, outDim, inDim> weightMatrix(weightPtr);
                ImageInference::model::ResNet50::fullyConnectedLayer<32>(*inputVector, weightMatrix, *biasAccumulator);
            }
        };

        BENCHMARK_TEMPLATE_F(FullyConnectedFixture, FullyConnected_ATen, 2048, 1000)
        (benchmark::State &st)
        {
            for (auto _ : st)
            {
                Tensor expected = at::linear(in, weight, bias);
            }
        };
    }
}