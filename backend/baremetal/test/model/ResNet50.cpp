#ifndef USE_ATEN_LIB
#define USE_ATEN_LIB
#endif // !USE_ATEN_LIB
#include <ATen/ATen.h>
#include <torch/library.h>
#include <iostream>
#include <iomanip>
#include <catch2/catch_test_macros.hpp>
#include "../../model/test/ResNet50Test.h"
#include <Fastor/Fastor.h>
#include "../utils/Reader.h"

namespace ImageInference
{
    namespace test
    {
        using at::Tensor;

        void printMismatchedValues(bool success, const Tensor &out, const Tensor &expected, size_t stride, size_t channels, size_t height, size_t width)
        {
            if (!success)
            {
                for (size_t iChannel = 0; iChannel < channels; iChannel++)
                {
                    for (size_t iHeight = 0; iHeight < height / stride; iHeight++)
                    {
                        for (size_t iWidth = 0; iWidth < width / stride; iWidth++)
                        {
                            auto got = out[iChannel][iHeight][iWidth].item<float>();
                            auto exp = expected[iChannel][iHeight][iWidth].item<float>();
                            if (got != exp)
                            {
                                std::cerr << std::setprecision(20)
                                          << "Expected " << exp << " but got " << got << std::endl
                                          << "Indices: Channel:= " << iChannel << " Height:= " << iHeight << " Width:= " << iWidth << std::endl
                                          << std::endl;
                            }
                        }
                    }
                }
            }
        }

        void printMismatchedValues(bool success, const Tensor &out, const Tensor &expected, size_t channels)
        {
            if (!success)
            {
                for (size_t iChannel = 0; iChannel < channels; iChannel++)
                {
                    auto got = out[iChannel].item<float>();
                    auto exp = expected[iChannel].item<float>();
                    if (got != exp)
                    {
                        std::cerr << std::setprecision(20)
                                  << "Expected " << exp << " but got " << got << std::endl
                                  << "Indices: Channel:= " << iChannel << std::endl
                                  << std::endl;
                    }
                }
            }
        }

        TEST_CASE("test_resnet50_conv1x1_channels16x16", "[resnet50][convolution]")
        {
            constexpr size_t stride = 1;
            constexpr size_t inPadding = 0;
            constexpr size_t blockSize = 16;
            constexpr size_t outChannels = 16;
            constexpr size_t inChannels = 16;
            constexpr size_t height = 10;
            constexpr size_t width = 10;
            constexpr size_t kernelHeight = 1;
            constexpr size_t kernelWidth = 1;

            Tensor in = at::rand({1, inChannels, height, width});
            Tensor weight = at::rand({outChannels, inChannels, kernelHeight, kernelWidth});
            Tensor batchGamma = at::rand({outChannels});
            Tensor batchBeta = at::rand({outChannels});
            Tensor batchMean = at::rand({outChannels});
            Tensor batchVar = at::rand({outChannels});

            Tensor out = at::zeros({outChannels, height, width});

            float *inPtr = in.mutable_data_ptr<float>();
            float *weightPtr = weight.mutable_data_ptr<float>();
            float *batchGammaPtr = batchGamma.mutable_data_ptr<float>();
            float *batchBetaPtr = batchBeta.mutable_data_ptr<float>();
            float *batchMeanPtr = batchMean.mutable_data_ptr<float>();
            float *batchVarPtr = batchVar.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            ImageInference::model::test::ResNet50Test::convBlock<
                stride, inPadding, blockSize, outChannels, inChannels,
                height, width, kernelHeight, kernelWidth>(inPtr, weightPtr, batchGammaPtr, batchBetaPtr, batchMeanPtr, batchVarPtr, outPtr);

            Tensor expected = at::conv2d(in, weight, {}, stride, inPadding);
            expected = at::batch_norm(expected, batchGamma, batchBeta, batchMean, batchVar, false, 0.1, 1e-5, false);
            expected = at::relu(expected);

            // //printMismatchedValues(at::allclose(out, expected[0], 1.0e-4, 1.0e-5), out, expected[0], stride, outChannels, height, width);
            REQUIRE(at::allclose(out, expected[0], 1.0e-4, 1.0e-5));
        }

        TEST_CASE("test_resnet50_conv1x1_channels64x32", "[resnet50][convolution]")
        {
            constexpr size_t stride = 1;
            constexpr size_t inPadding = 0;
            constexpr size_t blockSize = 32;
            constexpr size_t outChannels = 64;
            constexpr size_t inChannels = 64;
            constexpr size_t height = 56;
            constexpr size_t width = 56;
            constexpr size_t kernelHeight = 1;
            constexpr size_t kernelWidth = 1;

            Tensor in = at::rand({1, inChannels, height, width});
            Tensor weight = at::rand({outChannels, inChannels, kernelHeight, kernelWidth});
            Tensor batchGamma = at::rand({outChannels});
            Tensor batchBeta = at::rand({outChannels});
            Tensor batchMean = at::rand({outChannels});
            Tensor batchVar = at::rand({outChannels});

            Tensor out = at::zeros({outChannels, height, width});

            float *inPtr = in.mutable_data_ptr<float>();
            float *weightPtr = weight.mutable_data_ptr<float>();
            float *batchGammaPtr = batchGamma.mutable_data_ptr<float>();
            float *batchBetaPtr = batchBeta.mutable_data_ptr<float>();
            float *batchMeanPtr = batchMean.mutable_data_ptr<float>();
            float *batchVarPtr = batchVar.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            ImageInference::model::test::ResNet50Test::convBlock<
                stride, inPadding, blockSize, outChannels, inChannels,
                height, width, kernelHeight, kernelWidth>(inPtr, weightPtr, batchGammaPtr, batchBetaPtr, batchMeanPtr, batchVarPtr, outPtr);

            Tensor expected = at::conv2d(in, weight, {}, stride, inPadding);
            expected = at::batch_norm(expected, batchGamma, batchBeta, batchMean, batchVar, false, 0.1, 1e-5, false);
            expected = at::relu(expected);

            // //printMismatchedValues(at::allclose(out, expected[0], 1.0e-3, 1.0e-4), out, expected[0], stride, outChannels, height, width);
            REQUIRE(at::allclose(out, expected[0], 1.0e-3, 1.0e-4));
        }

        TEST_CASE("test_resnet50_conv1x1_channels16x32", "[resnet50][convolution]")
        {
            constexpr size_t stride = 1;
            constexpr size_t inPadding = 0;
            constexpr size_t blockSize = 16;
            constexpr size_t outChannels = 32;
            constexpr size_t inChannels = 16;
            constexpr size_t height = 10;
            constexpr size_t width = 10;
            constexpr size_t kernelHeight = 1;
            constexpr size_t kernelWidth = 1;

            Tensor in = at::rand({1, inChannels, height, width});
            Tensor weight = at::rand({outChannels, inChannels, kernelHeight, kernelWidth});
            Tensor batchGamma = at::rand({outChannels});
            Tensor batchBeta = at::rand({outChannels});
            Tensor batchMean = at::rand({outChannels});
            Tensor batchVar = at::rand({outChannels});

            Tensor out = at::zeros({outChannels, height, width});

            float *inPtr = in.mutable_data_ptr<float>();
            float *weightPtr = weight.mutable_data_ptr<float>();
            float *batchGammaPtr = batchGamma.mutable_data_ptr<float>();
            float *batchBetaPtr = batchBeta.mutable_data_ptr<float>();
            float *batchMeanPtr = batchMean.mutable_data_ptr<float>();
            float *batchVarPtr = batchVar.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            ImageInference::model::test::ResNet50Test::convBlock<
                stride, inPadding, blockSize, outChannels, inChannels,
                height, width, kernelHeight, kernelWidth>(inPtr, weightPtr, batchGammaPtr, batchBetaPtr, batchMeanPtr, batchVarPtr, outPtr);

            Tensor expected = at::conv2d(in, weight, {}, stride, inPadding);
            expected = at::batch_norm(expected, batchGamma, batchBeta, batchMean, batchVar, false, 0.1, 1e-5, false);
            expected = at::relu(expected);

            // //printMismatchedValues(at::allclose(out, expected[0], 1.0e-4, 1.0e-5), out, expected[0], stride, outChannels, height, width);
            REQUIRE(at::allclose(out, expected[0], 1.0e-4, 1.0e-5));
        }

        TEST_CASE("test_resnet50_conv3x3_channels16x16", "[resnet50][convolution]")
        {
            constexpr size_t stride = 1;
            constexpr size_t inPadding = 1;
            constexpr size_t blockSize = 16;
            constexpr size_t outChannels = 16;
            constexpr size_t inChannels = 16;
            constexpr size_t height = 10;
            constexpr size_t width = 10;
            constexpr size_t kernelHeight = 3;
            constexpr size_t kernelWidth = 3;

            Tensor in = at::rand({1, inChannels, height, width});
            Tensor weight = at::rand({outChannels, inChannels, kernelHeight, kernelWidth});
            Tensor batchGamma = at::rand({outChannels});
            Tensor batchBeta = at::rand({outChannels});
            Tensor batchMean = at::rand({outChannels});
            Tensor batchVar = at::rand({outChannels});

            Tensor out = at::zeros({outChannels, height, width});

            float *inPtr = in.mutable_data_ptr<float>();
            float *weightPtr = weight.mutable_data_ptr<float>();
            float *batchGammaPtr = batchGamma.mutable_data_ptr<float>();
            float *batchBetaPtr = batchBeta.mutable_data_ptr<float>();
            float *batchMeanPtr = batchMean.mutable_data_ptr<float>();
            float *batchVarPtr = batchVar.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            ImageInference::model::test::ResNet50Test::convBlock<
                stride, inPadding, blockSize, outChannels, inChannels,
                height, width, kernelHeight, kernelWidth>(inPtr, weightPtr, batchGammaPtr, batchBetaPtr, batchMeanPtr, batchVarPtr, outPtr);

            Tensor expected = at::conv2d(in, weight, {}, stride, inPadding);
            expected = at::batch_norm(expected, batchGamma, batchBeta, batchMean, batchVar, false, 0.1, 1e-5, false);
            expected = at::relu(expected);

            // //printMismatchedValues(at::allclose(out, expected[0], 1.0e-4, 1.0e-5), out, expected[0], stride, outChannels, height, width);
            REQUIRE(at::allclose(out, expected[0], 1.0e-4, 1.0e-5));
        }

        TEST_CASE("test_resnet50_conv3x3_channels16x32", "[resnet50][convolution]")
        {
            constexpr size_t stride = 1;
            constexpr size_t inPadding = 1;
            constexpr size_t blockSize = 16;
            constexpr size_t outChannels = 32;
            constexpr size_t inChannels = 16;
            constexpr size_t height = 10;
            constexpr size_t width = 10;
            constexpr size_t kernelHeight = 3;
            constexpr size_t kernelWidth = 3;

            Tensor in = at::rand({1, inChannels, height, width});
            Tensor weight = at::rand({outChannels, inChannels, kernelHeight, kernelWidth});
            Tensor batchGamma = at::rand({outChannels});
            Tensor batchBeta = at::rand({outChannels});
            Tensor batchMean = at::rand({outChannels});
            Tensor batchVar = at::rand({outChannels});

            Tensor out = at::zeros({outChannels, height, width});

            float *inPtr = in.mutable_data_ptr<float>();
            float *weightPtr = weight.mutable_data_ptr<float>();
            float *batchGammaPtr = batchGamma.mutable_data_ptr<float>();
            float *batchBetaPtr = batchBeta.mutable_data_ptr<float>();
            float *batchMeanPtr = batchMean.mutable_data_ptr<float>();
            float *batchVarPtr = batchVar.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            ImageInference::model::test::ResNet50Test::convBlock<
                stride, inPadding, blockSize, outChannels, inChannels,
                height, width, kernelHeight, kernelWidth>(inPtr, weightPtr, batchGammaPtr, batchBetaPtr, batchMeanPtr, batchVarPtr, outPtr);

            Tensor expected = at::conv2d(in, weight, {}, stride, inPadding);
            expected = at::batch_norm(expected, batchGamma, batchBeta, batchMean, batchVar, false, 0.1, 1e-5, false);
            expected = at::relu(expected);

            // //printMismatchedValues(at::allclose(out, expected[0], 1.0e-4, 1.0e-5), out, expected[0], stride, outChannels, height, width);
            REQUIRE(at::allclose(out, expected[0], 1.0e-4, 1.0e-5));
        }

        TEST_CASE("test_resnet50_conv3x3_channels16x16_stride2", "[resnet50][convolution]")
        {
            constexpr size_t stride = 2;
            constexpr size_t inPadding = 1;
            constexpr size_t blockSize = 16;
            constexpr size_t outChannels = 16;
            constexpr size_t inChannels = 16;
            constexpr size_t height = 10;
            constexpr size_t width = 10;
            constexpr size_t kernelHeight = 3;
            constexpr size_t kernelWidth = 3;

            Tensor in = at::rand({1, inChannels, height, width});
            Tensor weight = at::rand({outChannels, inChannels, kernelHeight, kernelWidth});
            Tensor batchGamma = at::rand({outChannels});
            Tensor batchBeta = at::rand({outChannels});
            Tensor batchMean = at::rand({outChannels});
            Tensor batchVar = at::rand({outChannels});

            Tensor out = at::zeros({outChannels, height / stride, width / stride});

            float *inPtr = in.mutable_data_ptr<float>();
            float *weightPtr = weight.mutable_data_ptr<float>();
            float *batchGammaPtr = batchGamma.mutable_data_ptr<float>();
            float *batchBetaPtr = batchBeta.mutable_data_ptr<float>();
            float *batchMeanPtr = batchMean.mutable_data_ptr<float>();
            float *batchVarPtr = batchVar.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            ImageInference::model::test::ResNet50Test::convBlock<
                stride, inPadding, blockSize, outChannels, inChannels,
                height, width, kernelHeight, kernelWidth>(inPtr, weightPtr, batchGammaPtr, batchBetaPtr, batchMeanPtr, batchVarPtr, outPtr);

            Tensor expected = at::conv2d(in, weight, {}, stride, inPadding);
            expected = at::batch_norm(expected, batchGamma, batchBeta, batchMean, batchVar, false, 0.1, 1e-5, false);
            expected = at::relu(expected);

            // //printMismatchedValues(at::allclose(out, expected[0], 1.0e-4, 1.0e-5), out, expected[0], stride, outChannels, height, width);
            REQUIRE(at::allclose(out, expected[0], 1.0e-4, 1.0e-5));
        }

        TEST_CASE("test_resnet50_conv3x3_channels16x32_stride2", "[resnet50][convolution]")
        {
            constexpr size_t stride = 2;
            constexpr size_t inPadding = 1;
            constexpr size_t blockSize = 16;
            constexpr size_t outChannels = 32;
            constexpr size_t inChannels = 16;
            constexpr size_t height = 10;
            constexpr size_t width = 10;
            constexpr size_t kernelHeight = 3;
            constexpr size_t kernelWidth = 3;

            Tensor in = at::rand({1, inChannels, height, width});
            Tensor weight = at::rand({outChannels, inChannels, kernelHeight, kernelWidth});
            Tensor batchGamma = at::rand({outChannels});
            Tensor batchBeta = at::rand({outChannels});
            Tensor batchMean = at::rand({outChannels});
            Tensor batchVar = at::rand({outChannels});

            Tensor out = at::zeros({outChannels, height / stride, width / stride});

            float *inPtr = in.mutable_data_ptr<float>();
            float *weightPtr = weight.mutable_data_ptr<float>();
            float *batchGammaPtr = batchGamma.mutable_data_ptr<float>();
            float *batchBetaPtr = batchBeta.mutable_data_ptr<float>();
            float *batchMeanPtr = batchMean.mutable_data_ptr<float>();
            float *batchVarPtr = batchVar.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            ImageInference::model::test::ResNet50Test::convBlock<
                stride, inPadding, blockSize, outChannels, inChannels,
                height, width, kernelHeight, kernelWidth>(inPtr, weightPtr, batchGammaPtr, batchBetaPtr, batchMeanPtr, batchVarPtr, outPtr);

            Tensor expected = at::conv2d(in, weight, {}, stride, inPadding);
            expected = at::batch_norm(expected, batchGamma, batchBeta, batchMean, batchVar, false, 0.1, 1e-5, false);
            expected = at::relu(expected);

            // //printMismatchedValues(at::allclose(out, expected[0], 1.0e-4, 1.0e-5), out, expected[0], stride, outChannels, height, width);
            REQUIRE(at::allclose(out, expected[0], 1.0e-4, 1.0e-5));
        }

        TEST_CASE("test_resnet50_conv3x3_channels16x32_blockSize1", "[resnet50][convolution]")
        {
            constexpr size_t stride = 1;
            constexpr size_t inPadding = 1;
            constexpr size_t blockSize = 1;
            constexpr size_t outChannels = 32;
            constexpr size_t inChannels = 16;
            constexpr size_t height = 10;
            constexpr size_t width = 10;
            constexpr size_t kernelHeight = 3;
            constexpr size_t kernelWidth = 3;

            Tensor in = at::rand({1, inChannels, height, width});
            Tensor weight = at::rand({outChannels, inChannels, kernelHeight, kernelWidth});
            Tensor batchGamma = at::rand({outChannels});
            Tensor batchBeta = at::rand({outChannels});
            Tensor batchMean = at::rand({outChannels});
            Tensor batchVar = at::rand({outChannels});

            Tensor out = at::zeros({outChannels, height, width});

            float *inPtr = in.mutable_data_ptr<float>();
            float *weightPtr = weight.mutable_data_ptr<float>();
            float *batchGammaPtr = batchGamma.mutable_data_ptr<float>();
            float *batchBetaPtr = batchBeta.mutable_data_ptr<float>();
            float *batchMeanPtr = batchMean.mutable_data_ptr<float>();
            float *batchVarPtr = batchVar.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            ImageInference::model::test::ResNet50Test::convBlock<
                stride, inPadding, blockSize, outChannels, inChannels,
                height, width, kernelHeight, kernelWidth>(inPtr, weightPtr, batchGammaPtr, batchBetaPtr,batchMeanPtr, batchVarPtr, outPtr);

            Tensor expected = at::conv2d(in, weight, {}, stride, inPadding);
            expected = at::batch_norm(expected, batchGamma, batchBeta, batchMean, batchVar, false, 0.1, 1e-5, false);
            expected = at::relu(expected);

            // //printMismatchedValues(at::allclose(out, expected[0], 1.0e-4, 1.0e-5), out, expected[0], stride, outChannels, height, width);
            REQUIRE(at::allclose(out, expected[0], 1.0e-4, 1.0e-5));
        }

        TEST_CASE("test_resnet50_conv7x7_channels16x32_blockSize1", "[resnet50][convolution]")
        {
            constexpr size_t stride = 1;
            constexpr size_t inPadding = 3;
            constexpr size_t blockSize = 1;
            constexpr size_t outChannels = 32;
            constexpr size_t inChannels = 16;
            constexpr size_t height = 10;
            constexpr size_t width = 10;
            constexpr size_t kernelHeight = 7;
            constexpr size_t kernelWidth = 7;

            Tensor in = at::rand({1, inChannels, height, width});
            Tensor weight = at::rand({outChannels, inChannels, kernelHeight, kernelWidth});
            Tensor batchGamma = at::rand({outChannels});
            Tensor batchBeta = at::rand({outChannels});
            Tensor batchMean = at::rand({outChannels});
            Tensor batchVar = at::rand({outChannels});

            Tensor out = at::zeros({outChannels, height, width});

            float *inPtr = in.mutable_data_ptr<float>();
            float *weightPtr = weight.mutable_data_ptr<float>();
            float *batchGammaPtr = batchGamma.mutable_data_ptr<float>();
            float *batchBetaPtr = batchBeta.mutable_data_ptr<float>();
            float *batchMeanPtr = batchMean.mutable_data_ptr<float>();
            float *batchVarPtr = batchVar.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            ImageInference::model::test::ResNet50Test::convBlock<
                stride, inPadding, blockSize, outChannels, inChannels,
                height, width, kernelHeight, kernelWidth>(inPtr, weightPtr, batchGammaPtr, batchBetaPtr, batchMeanPtr, batchVarPtr, outPtr);

            Tensor expected = at::conv2d(in, weight, {}, stride, inPadding);
            expected = at::batch_norm(expected, batchGamma, batchBeta, batchMean, batchVar, false, 0.1, 1e-5, false);
            expected = at::relu(expected);

            // //printMismatchedValues(at::allclose(out, expected[0], 1.0e-4, 1.0e-5), out, expected[0], stride, outChannels, height, width);
            REQUIRE(at::allclose(out, expected[0], 1.0e-4, 1.0e-5));
        }

        TEST_CASE("test_resnet50_conv7x7_channels3x64_blockSize1", "[resnet50][convolution]")
        {
            constexpr size_t stride = 1;
            constexpr size_t inPadding = 3;
            constexpr size_t blockSize = 1;
            constexpr size_t outChannels = 64;
            constexpr size_t inChannels = 3;
            constexpr size_t height = 244;
            constexpr size_t width = 244;
            constexpr size_t kernelHeight = 7;
            constexpr size_t kernelWidth = 7;

            Tensor in = at::rand({1, inChannels, height, width});
            Tensor weight = at::rand({outChannels, inChannels, kernelHeight, kernelWidth});
            Tensor batchGamma = at::rand({outChannels});
            Tensor batchBeta = at::rand({outChannels});
            Tensor batchMean = at::rand({outChannels});
            Tensor batchVar = at::rand({outChannels});

            Tensor out = at::zeros({outChannels, height, width});

            float *inPtr = in.mutable_data_ptr<float>();
            float *weightPtr = weight.mutable_data_ptr<float>();
            float *batchGammaPtr = batchGamma.mutable_data_ptr<float>();
            float *batchBetaPtr = batchBeta.mutable_data_ptr<float>();
            float *batchMeanPtr = batchMean.mutable_data_ptr<float>();
            float *batchVarPtr = batchVar.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            ImageInference::model::test::ResNet50Test::convBlock<
                stride, inPadding, blockSize, outChannels, inChannels,
                height, width, kernelHeight, kernelWidth>(inPtr, weightPtr, batchGammaPtr, batchBetaPtr, batchMeanPtr, batchVarPtr, outPtr);

            Tensor expected = at::conv2d(in, weight, {}, stride, inPadding);
            expected = at::batch_norm(expected, batchGamma, batchBeta, batchMean, batchVar, false, 0.1, 1e-5, false);
            expected = at::relu(expected);

            //printMismatchedValues(at::allclose(out, expected[0], 1.0e-2, 1.0e-3), out, expected[0], stride, outChannels, height, width);
            REQUIRE(at::allclose(out, expected[0], 1.0e-2, 1.0e-3));
        }

        TEST_CASE("test_resnet50_conv3x3_shortcut_channels16x16", "[resnet50][convolution][shortcut]")
        {
            constexpr size_t inPadding = 1;
            constexpr size_t blockSize = 16;
            constexpr size_t outChannels = 16;
            constexpr size_t inChannels = 16;
            constexpr size_t height = 10;
            constexpr size_t width = 10;
            constexpr size_t kernelHeight = 3;
            constexpr size_t kernelWidth = 3;

            Tensor in = at::rand({1, inChannels, height, width});
            Tensor weight = at::rand({outChannels, inChannels, kernelHeight, kernelWidth});
            Tensor batchGamma = at::rand({outChannels});
            Tensor batchBeta = at::rand({outChannels});
            Tensor batchMean = at::rand({outChannels});
            Tensor batchVar = at::rand({outChannels});
            Tensor shortcut = at::rand({outChannels, height, width});

            Tensor out = at::zeros({outChannels, height, width});

            float *inPtr = in.mutable_data_ptr<float>();
            float *weightPtr = weight.mutable_data_ptr<float>();
            float *batchGammaPtr = batchGamma.mutable_data_ptr<float>();
            float *batchBetaPtr = batchBeta.mutable_data_ptr<float>();
            float *batchMeanPtr = batchMean.mutable_data_ptr<float>();
            float *batchVarPtr = batchVar.mutable_data_ptr<float>();
            float *shortcutPtr = shortcut.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            ImageInference::model::test::ResNet50Test::convBlockShortcut<
                inPadding, blockSize, outChannels, inChannels,
                height, width, kernelHeight, kernelWidth>(inPtr, weightPtr, batchGammaPtr, batchBetaPtr,batchMeanPtr, batchVarPtr, shortcutPtr, outPtr);

            Tensor expected = at::conv2d(in, weight, {}, 1, inPadding);
            expected = at::batch_norm(expected, batchGamma, batchBeta, batchMean, batchVar, false, 0.1, 1e-5, false);
            expected += shortcut;
            expected = at::relu(expected);

            // //printMismatchedValues(at::allclose(out, expected[0], 1.0e-4, 1.0e-5), out, expected[0], 1, outChannels, height, width);
            REQUIRE(at::allclose(out, expected[0], 1.0e-4, 1.0e-5));
        }

        TEST_CASE("test_resnet50_conv3x3_shortcut_channels16x32", "[resnet50][convolution][shortcut]")
        {
            constexpr size_t inPadding = 1;
            constexpr size_t blockSize = 16;
            constexpr size_t outChannels = 32;
            constexpr size_t inChannels = 16;
            constexpr size_t height = 10;
            constexpr size_t width = 10;
            constexpr size_t kernelHeight = 3;
            constexpr size_t kernelWidth = 3;

            Tensor in = at::rand({1, inChannels, height, width});
            Tensor weight = at::rand({outChannels, inChannels, kernelHeight, kernelWidth});
            Tensor batchGamma = at::rand({outChannels});
            Tensor batchBeta = at::rand({outChannels});
            Tensor batchMean = at::rand({outChannels});
            Tensor batchVar = at::rand({outChannels});
            Tensor shortcut = at::rand({outChannels, height, width});

            Tensor out = at::zeros({outChannels, height, width});
            float *inPtr = in.mutable_data_ptr<float>();
            float *weightPtr = weight.mutable_data_ptr<float>();
            float *batchGammaPtr = batchGamma.mutable_data_ptr<float>();
            float *batchBetaPtr = batchBeta.mutable_data_ptr<float>();
            float *batchMeanPtr = batchMean.mutable_data_ptr<float>();
            float *batchVarPtr = batchVar.mutable_data_ptr<float>();
            float *shortcutPtr = shortcut.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            ImageInference::model::test::ResNet50Test::convBlockShortcut<
                inPadding, blockSize, outChannels, inChannels,
                height, width, kernelHeight, kernelWidth>(inPtr, weightPtr, batchGammaPtr, batchBetaPtr,batchMeanPtr, batchVarPtr, shortcutPtr, outPtr);

            Tensor expected = at::conv2d(in, weight, {}, 1, inPadding);
            expected = at::batch_norm(expected, batchGamma, batchBeta, batchMean, batchVar, false, 0.1, 1e-5, false);
            expected += shortcut;
            expected = at::relu(expected);

            // //printMismatchedValues(at::allclose(out, expected[0], 1.0e-4, 1.0e-5), out, expected[0], 1, outChannels, height, width);
            REQUIRE(at::allclose(out, expected[0], 1.0e-4, 1.0e-5));
        }

        TEST_CASE("test_resnet50_conv3x3_projection_channels32x32", "[resnet50][convolution][projection]")
        {
            constexpr size_t stride = 1;
            constexpr size_t inPadding = 1;
            constexpr size_t blockSize = 16;
            constexpr size_t outChannels = 32;
            constexpr size_t inChannels = 32;
            constexpr size_t shortcutChannels = 16;
            constexpr size_t height = 10;
            constexpr size_t width = 10;
            constexpr size_t kernelHeight = 3;
            constexpr size_t kernelWidth = 3;

            Tensor in = at::rand({1, inChannels, height / stride, width / stride});
            Tensor weight = at::rand({outChannels, inChannels, kernelHeight, kernelWidth});
            Tensor batchGamma = at::rand({outChannels});
            Tensor batchBeta = at::rand({outChannels});
            Tensor batchMean = at::rand({outChannels});
            Tensor batchVar = at::rand({outChannels});
            Tensor shortcut = at::rand({1, shortcutChannels, height, width});
            Tensor projectionWeight = at::rand({outChannels, shortcutChannels, 1, 1});
            Tensor projectionBatchGamma = at::rand({outChannels});
            Tensor projectionBatchBeta = at::rand({outChannels});
            Tensor projectionBatchMean = at::rand({outChannels});
            Tensor projectionBatchVar = at::rand({outChannels});

            Tensor out = at::zeros({outChannels, height / stride, width / stride});

            float *inPtr = in.mutable_data_ptr<float>();
            float *weightPtr = weight.mutable_data_ptr<float>();
            float *batchGammaPtr = batchGamma.mutable_data_ptr<float>();
            float *batchBetaPtr = batchBeta.mutable_data_ptr<float>();
            float *batchMeanPtr = batchMean.mutable_data_ptr<float>();
            float *batchVarPtr = batchVar.mutable_data_ptr<float>();
            float *shortcutPtr = shortcut.mutable_data_ptr<float>();
            float *projectionWeightPtr = projectionWeight.mutable_data_ptr<float>();
            float *projectionBatchGammaPtr = projectionBatchGamma.mutable_data_ptr<float>();
            float *projectionBatchBetaPtr = projectionBatchBeta.mutable_data_ptr<float>();
            float *projectionBatchMeanPtr = projectionBatchMean.mutable_data_ptr<float>();
            float *projectionBatchVarPtr = projectionBatchVar.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            ImageInference::model::test::ResNet50Test::convBlockProjection<
                stride, outChannels / shortcutChannels, inPadding, blockSize, outChannels, inChannels,
                height, width, kernelHeight, kernelWidth>(inPtr, weightPtr, batchGammaPtr, batchBetaPtr, batchMeanPtr, batchVarPtr,
                                                          shortcutPtr, projectionWeightPtr, projectionBatchGammaPtr, projectionBatchBetaPtr, projectionBatchMeanPtr, projectionBatchVarPtr,
                                                          outPtr);

            Tensor expected = at::conv2d(in, weight, {}, 1, inPadding);
            expected = at::batch_norm(expected, batchGamma, batchBeta, batchMean, batchVar, false, 0.1, 1e-5, false);
            Tensor projection = at::conv2d(shortcut, projectionWeight, {}, stride);
            projection = at::batch_norm(projection, projectionBatchGamma, projectionBatchBeta, projectionBatchMean, projectionBatchVar, false, 0.1, 1e-5, false);
            expected += projection;
            expected = at::relu(expected);

            // //printMismatchedValues(at::allclose(out, expected[0], 1.0e-4, 1.0e-5), out, expected[0], 1, outChannels, height, width);
            REQUIRE(at::allclose(out, expected[0], 1.0e-4, 1.0e-5));
        }

        TEST_CASE("test_resnet50_conv3x3_projection_channels32x64", "[resnet50][convolution][projection]")
        {
            constexpr size_t stride = 1;
            constexpr size_t inPadding = 1;
            constexpr size_t blockSize = 16;
            constexpr size_t outChannels = 64;
            constexpr size_t inChannels = 32;
            constexpr size_t shortcutChannels = 16;
            constexpr size_t height = 10;
            constexpr size_t width = 10;
            constexpr size_t kernelHeight = 3;
            constexpr size_t kernelWidth = 3;

            Tensor in = at::rand({1, inChannels, height / stride, width / stride});
            Tensor weight = at::rand({outChannels, inChannels, kernelHeight, kernelWidth});
            Tensor batchGamma = at::rand({outChannels});
            Tensor batchBeta = at::rand({outChannels});
            Tensor batchMean = at::rand({outChannels});
            Tensor batchVar = at::rand({outChannels});
            Tensor shortcut = at::rand({1, shortcutChannels, height, width});
            Tensor projectionWeight = at::rand({outChannels, shortcutChannels, 1, 1});
            Tensor projectionBatchGamma = at::rand({outChannels});
            Tensor projectionBatchBeta = at::rand({outChannels});
            Tensor projectionBatchMean = at::rand({outChannels});
            Tensor projectionBatchVar = at::rand({outChannels});

            Tensor out = at::zeros({outChannels, height / stride, width / stride});

            float *inPtr = in.mutable_data_ptr<float>();
            float *weightPtr = weight.mutable_data_ptr<float>();
            float *batchGammaPtr = batchGamma.mutable_data_ptr<float>();
            float *batchBetaPtr = batchBeta.mutable_data_ptr<float>();
            float *batchMeanPtr = batchMean.mutable_data_ptr<float>();
            float *batchVarPtr = batchVar.mutable_data_ptr<float>();
            float *shortcutPtr = shortcut.mutable_data_ptr<float>();
            float *projectionWeightPtr = projectionWeight.mutable_data_ptr<float>();
            float *projectionBatchGammaPtr = projectionBatchGamma.mutable_data_ptr<float>();
            float *projectionBatchBetaPtr = projectionBatchBeta.mutable_data_ptr<float>();
            float *projectionBatchMeanPtr = projectionBatchMean.mutable_data_ptr<float>();
            float *projectionBatchVarPtr = projectionBatchVar.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            ImageInference::model::test::ResNet50Test::convBlockProjection<
                stride, outChannels / shortcutChannels, inPadding, blockSize, outChannels, inChannels,
                height, width, kernelHeight, kernelWidth>(inPtr, weightPtr, batchGammaPtr, batchBetaPtr, batchMeanPtr, batchVarPtr,
                                                          shortcutPtr, projectionWeightPtr, projectionBatchGammaPtr, projectionBatchBetaPtr, projectionBatchMeanPtr, projectionBatchVarPtr,
                                                          outPtr);

            Tensor expected = at::conv2d(in, weight, {}, 1, inPadding);
            expected = at::batch_norm(expected, batchGamma, batchBeta, batchMean, batchVar, false, 0.1, 1e-5, false);
            Tensor projection = at::conv2d(shortcut, projectionWeight, {}, stride);
            projection = at::batch_norm(projection, projectionBatchGamma, projectionBatchBeta, projectionBatchMean, projectionBatchVar, false, 0.1, 1e-5, false);
            expected += projection;
            expected = at::relu(expected);

            // //printMismatchedValues(at::allclose(out, expected[0], 1.0e-4, 1.0e-5), out, expected[0], 1, outChannels, height, width);
            REQUIRE(at::allclose(out, expected[0], 1.0e-4, 1.0e-5));
        }

        TEST_CASE("test_resnet50_conv3x3_projection_channels32x32_stride2", "[resnet50][convolution][projection]")
        {
            constexpr size_t stride = 2;
            constexpr size_t inPadding = 1;
            constexpr size_t blockSize = 16;
            constexpr size_t outChannels = 32;
            constexpr size_t inChannels = 32;
            constexpr size_t shortcutChannels = 16;
            constexpr size_t height = 10;
            constexpr size_t width = 10;
            constexpr size_t kernelHeight = 3;
            constexpr size_t kernelWidth = 3;

            Tensor in = at::rand({1, inChannels, height / stride, width / stride});
            Tensor weight = at::rand({outChannels, inChannels, kernelHeight, kernelWidth});
            Tensor batchGamma = at::rand({outChannels});
            Tensor batchBeta = at::rand({outChannels});
            Tensor batchMean = at::rand({outChannels});
            Tensor batchVar = at::rand({outChannels});
            Tensor shortcut = at::rand({1, shortcutChannels, height, width});
            Tensor projectionWeight = at::rand({outChannels, shortcutChannels, 1, 1});
            Tensor projectionBatchGamma = at::rand({outChannels});
            Tensor projectionBatchBeta = at::rand({outChannels});
            Tensor projectionBatchMean = at::rand({outChannels});
            Tensor projectionBatchVar = at::rand({outChannels});

            Tensor out = at::zeros({outChannels, height / stride, width / stride});

            float *inPtr = in.mutable_data_ptr<float>();
            float *weightPtr = weight.mutable_data_ptr<float>();
            float *batchGammaPtr = batchGamma.mutable_data_ptr<float>();
            float *batchBetaPtr = batchBeta.mutable_data_ptr<float>();
            float *batchMeanPtr = batchMean.mutable_data_ptr<float>();
            float *batchVarPtr = batchVar.mutable_data_ptr<float>();
            float *shortcutPtr = shortcut.mutable_data_ptr<float>();
            float *projectionWeightPtr = projectionWeight.mutable_data_ptr<float>();
            float *projectionBatchGammaPtr = projectionBatchGamma.mutable_data_ptr<float>();
            float *projectionBatchBetaPtr = projectionBatchBeta.mutable_data_ptr<float>();
            float *projectionBatchMeanPtr = projectionBatchMean.mutable_data_ptr<float>();
            float *projectionBatchVarPtr = projectionBatchVar.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            ImageInference::model::test::ResNet50Test::convBlockProjection<
                stride, outChannels / shortcutChannels, inPadding, blockSize, outChannels, inChannels,
                height, width, kernelHeight, kernelWidth>(inPtr, weightPtr, batchGammaPtr, batchBetaPtr, batchMeanPtr, batchVarPtr,
                                                          shortcutPtr, projectionWeightPtr, projectionBatchGammaPtr, projectionBatchBetaPtr, projectionBatchMeanPtr, projectionBatchVarPtr,
                                                          outPtr);

            Tensor expected = at::conv2d(in, weight, {}, 1, inPadding);
            expected = at::batch_norm(expected, batchGamma, batchBeta, batchMean, batchVar, false, 0.1, 1e-5, false);
            Tensor projection = at::conv2d(shortcut, projectionWeight, {}, stride);
            projection = at::batch_norm(projection, projectionBatchGamma, projectionBatchBeta, projectionBatchMean, projectionBatchVar, false, 0.1, 1e-5, false);
            expected += projection;
            expected = at::relu(expected);

            // //printMismatchedValues(at::allclose(out, expected[0], 1.0e-4, 1.0e-5), out, expected[0], 1, outChannels, height, width);
            REQUIRE(at::allclose(out, expected[0], 1.0e-4, 1.0e-5));
        }

        TEST_CASE("test_resnet50_conv3x3_projection_channels32x64_stride2", "[resnet50][convolution][projection]")
        {
            constexpr size_t stride = 2;
            constexpr size_t inPadding = 1;
            constexpr size_t blockSize = 16;
            constexpr size_t outChannels = 64;
            constexpr size_t inChannels = 32;
            constexpr size_t shortcutChannels = 16;
            constexpr size_t height = 10;
            constexpr size_t width = 10;
            constexpr size_t kernelHeight = 3;
            constexpr size_t kernelWidth = 3;

            Tensor in = at::rand({1, inChannels, height / stride, width / stride});
            Tensor weight = at::rand({outChannels, inChannels, kernelHeight, kernelWidth});
            Tensor batchGamma = at::rand({outChannels});
            Tensor batchBeta = at::rand({outChannels});
            Tensor batchMean = at::rand({outChannels});
            Tensor batchVar = at::rand({outChannels});
            Tensor shortcut = at::rand({1, shortcutChannels, height, width});
            Tensor projectionWeight = at::rand({outChannels, shortcutChannels, 1, 1});
            Tensor projectionBatchGamma = at::rand({outChannels});
            Tensor projectionBatchBeta = at::rand({outChannels});
            Tensor projectionBatchMean = at::rand({outChannels});
            Tensor projectionBatchVar = at::rand({outChannels});

            Tensor out = at::zeros({outChannels, height / stride, width / stride});

            float *inPtr = in.mutable_data_ptr<float>();
            float *weightPtr = weight.mutable_data_ptr<float>();
            float *batchGammaPtr = batchGamma.mutable_data_ptr<float>();
            float *batchBetaPtr = batchBeta.mutable_data_ptr<float>();
            float *batchMeanPtr = batchMean.mutable_data_ptr<float>();
            float *batchVarPtr = batchVar.mutable_data_ptr<float>();
            float *shortcutPtr = shortcut.mutable_data_ptr<float>();
            float *projectionWeightPtr = projectionWeight.mutable_data_ptr<float>();
            float *projectionBatchGammaPtr = projectionBatchGamma.mutable_data_ptr<float>();
            float *projectionBatchBetaPtr = projectionBatchBeta.mutable_data_ptr<float>();
            float *projectionBatchMeanPtr = projectionBatchMean.mutable_data_ptr<float>();
            float *projectionBatchVarPtr = projectionBatchVar.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            ImageInference::model::test::ResNet50Test::convBlockProjection<
                stride, outChannels / shortcutChannels, inPadding, blockSize, outChannels, inChannels,
                height, width, kernelHeight, kernelWidth>(inPtr, weightPtr, batchGammaPtr, batchBetaPtr, batchMeanPtr, batchVarPtr,
                                                          shortcutPtr, projectionWeightPtr, projectionBatchGammaPtr, projectionBatchBetaPtr, projectionBatchMeanPtr, projectionBatchVarPtr,
                                                          outPtr);

            Tensor expected = at::conv2d(in, weight, {}, 1, inPadding);
            expected = at::batch_norm(expected, batchGamma, batchBeta, batchMean, batchVar, false, 0.1, 1e-5, false);
            Tensor projection = at::conv2d(shortcut, projectionWeight, {}, stride);
            projection = at::batch_norm(projection, projectionBatchGamma, projectionBatchBeta, projectionBatchMean, projectionBatchVar, false, 0.1, 1e-5, false);
            expected += projection;
            expected = at::relu(expected);

            // //printMismatchedValues(at::allclose(out, expected[0], 1.0e-4, 1.0e-5), out, expected[0], 1, outChannels, height, width);
            REQUIRE(at::allclose(out, expected[0], 1.0e-4, 1.0e-5));
        }

        TEST_CASE("test_resnet50_maxpool", "[resnet50][maxpool]")
        {
            constexpr size_t stride = 1;
            constexpr size_t inPadding = 1;
            constexpr size_t blockSize = 16;
            constexpr size_t channels = 16;
            constexpr size_t height = 10;
            constexpr size_t width = 10;

            Tensor in = at::rand({channels, height, width});

            Tensor out = at::zeros({channels, height / stride, width / stride});

            float *inPtr = in.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            ImageInference::model::test::ResNet50Test::maxPool<
                stride, inPadding, blockSize, channels, height, width>(inPtr, outPtr);

            Tensor expected = at::max_pool2d(in, {3, 3}, stride, inPadding);

            bool success = at::allclose(out, expected);
            // //printMismatchedValues(success, out, expected, stride, channels, height, width);
            REQUIRE(success);
        }

        TEST_CASE("test_resnet50_maxpool_channels", "[resnet50][maxpool]")
        {

            constexpr size_t stride = 1;
            constexpr size_t inPadding = 1;
            constexpr size_t blockSize = 16;
            constexpr size_t channels = 16;
            constexpr size_t height = 10;
            constexpr size_t width = 10;

            Tensor in = at::ones({channels, height, width});
            for (size_t i = 0; i < channels; i++)
            {
                in[i] *= i;
            }

            Tensor out = at::zeros({channels, height / stride, width / stride});

            float *inPtr = in.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            ImageInference::model::test::ResNet50Test::maxPool<
                stride, inPadding, blockSize, channels, height, width>(inPtr, outPtr);

            Tensor expected = at::max_pool2d(in, {3, 3}, stride, inPadding);

            bool success = at::allclose(out, expected);
            // //printMismatchedValues(success, out, expected, stride, channels, height, width);
            REQUIRE(success);
        }

        TEST_CASE("test_resnet50_global_average", "[resnet50][globalAverage]")
        {
            Tensor in = at::randn({16, 10, 10});

            constexpr size_t inPadding = 0;
            constexpr size_t blockSize = 16;
            constexpr size_t channels = 16;
            constexpr size_t height = 10;
            constexpr size_t width = 10;

            Tensor out = at::zeros({channels, 1, 1});

            float *inPtr = in.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            ImageInference::model::test::ResNet50Test::globalAveragePool<
                inPadding, blockSize, channels, height, width>(inPtr, outPtr);

            Tensor expected = at::adaptive_avg_pool2d(in, {1, 1});

            REQUIRE(at::allclose(out, expected, 1.0e-4, 1.0e-6));
        }

        TEST_CASE("test_resnet50_fully_connected", "[resnet50][fullyConnected]")
        {
            Tensor in = at::randn({2048});
            Tensor weight = at::randn({1000, 2048});
            Tensor bias = at::randn({1000});
            Tensor biasCopy = bias.clone().detach();

            Tensor out = at::zeros({1000});

            float *inPtr = in.mutable_data_ptr<float>();
            float *weightPtr = weight.mutable_data_ptr<float>();
            float *biasPtr = bias.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            // Fastor::TensorMap<float, 2048> inFastor(inPtr);
            // Fastor::TensorMap<float, 1000, 2048> weightFastor(weightPtr);
            // Fastor::TensorMap<float, 1000> biasFastor(biasPtr);
            // biasFastor += Fastor::matmul(weightFastor, inFastor);

            ImageInference::model::test::ResNet50Test::fullyConnectedLayer<1000, 2048>(inPtr, weightPtr, biasPtr, outPtr);

            Tensor expected = at::linear(in, weight, biasCopy);

            if (!at::allclose(out, expected, 1.0e-3, 1.0e-4))
            {
                for (size_t i = 0; i < 1000; i++)
                {
                    if (out[i].item<float>() != expected[i].item<float>())
                    {
                        std::cerr << "Expected " << expected[i].item<float>() << " but got " << out[i].item<float>() << " at index " << i << std::endl;
                    }
                }
            }

            REQUIRE(at::allclose(out, expected, 1.0e-3, 1.0e-4));
        }

        TEST_CASE("test_resnet50_relu", "[resnet50][relu]")
        {
            Tensor in = at::randn({100});
            Tensor out = at::zeros({100});

            for (size_t i = 0; i < 100; i++)
            {
                out[i] = ImageInference::model::test::ResNet50Test::relu(in[i].item<float>());
            }

            Tensor expected = at::relu(in);

            REQUIRE(at::allclose(out, expected));
        }

        TEST_CASE("test_resnet50_batchnorm", "[resnet50][batchNorm]")
        {
            Tensor in = at::randn({1, 16, 16, 16});
            Tensor gamma = at::randn({16});
            Tensor beta = at::randn({16});
            Tensor out = at::zeros({1, 16, 16, 16});

            Tensor mean = at::mean(in, {0, 2, 3});
            Tensor var = at::var(in, {0, 2, 3}, false);

            for (size_t i = 0; i < 16; i++)
            {
                for (size_t j = 0; j < 16; j++)
                {
                    for (size_t k = 0; k < 16; k++)
                    {
                        float gammaVar = gamma[i].item<float>() / std::sqrt(var[i].item<float>() + 1e-05);
                        out[0][i][j][k] = ImageInference::model::test::ResNet50Test::batchNorm(
                            in[0][i][j][k].item<float>(),
                            gammaVar,
                            beta[i].item<float>(),
                            mean[i].item<float>());
                    }
                }
            }

            Tensor expected = at::batch_norm(in, gamma, beta, mean, var, false, 0.1, 1e-05, false);

            if (!at::allclose(out, expected, 1.0e-4, 1.0e-6))
            {
                for (size_t i = 0; i < 16; i++)
                {
                    for (size_t j = 0; j < 16; j++)
                    {
                        for (size_t k = 0; k < 16; k++)
                        {
                            if (out[0][i][j][k].item<float>() != expected[0][i][j][k].item<float>())
                            {
                                std::cerr << "Expected " << expected[0][i][j][k].item<float>() << " but got " << out[0][i][j][k].item<float>() << std::endl
                                          << "In: " << in[0][i][j][k].item<float>() << " Mean: " << mean[i].item<float>() << " Variance: " << var[i].item<float>() << std::endl
                                          << "Gamma: " << gamma[i].item<float>() << " Beta: " << beta[i].item<float>() << std::endl
                                          << std::endl;
                            }
                        }
                    }
                }
            }

            REQUIRE(at::allclose(out, expected, 1.0e-4, 1.0e-6));
        }

        void testWholeResnet50(ImageInference::model::ResNet50 &resnet50, const std::string &compareFilepath)
        {
            // Read the input and comparison output
            const char *projectDirectory = std::getenv("PROJECT_ROOT");
            if (projectDirectory == nullptr)
            {
                throw std::runtime_error("PROJECT_ROOT environment variable is not set");
            }

            std::string inputPath = std::string(projectDirectory) + "/test_data/" + compareFilepath;
            ImageInference::test::utils::Reader reader(inputPath);

            std::vector<int64_t> sizes;
            float *readTensorPtr = reader.getNextTensor(sizes);
            Tensor in = at::from_blob(readTensorPtr, sizes);
            readTensorPtr = reader.getNextTensor(sizes);
            Tensor outExpected = at::from_blob(readTensorPtr, sizes);
            Tensor out = at::zeros({1, 1000});

            float *inPtr = in.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            REQUIRE((in.size(0) == 1));
            REQUIRE((in.size(1) == 3));
            REQUIRE((in.size(2) == 224));
            REQUIRE((in.size(3) == 224));

            REQUIRE((outExpected.size(0) == 1));
            REQUIRE((outExpected.size(1) == 1000));

            // std::cerr << std::endl
            //           << "Tested File: " << compareFilepath << std::endl
            //           << "Input [" << inPtr << ", " << inPtr + in.numel() << ") size: " << in.sizes() << " with " << in.numel() << " elements." << std::endl
            //           << "Expected size: " << outExpected.sizes() << std::endl;

            resnet50.inference(inPtr, outPtr);

            // //printMismatchedValues(at::allclose(out, outExpected, 70.0, 4), out[0], outExpected[0], 1000);
            std::cout << "Relative Error: " << (out - outExpected / (outExpected + 1e-5 * (outExpected == 0))).abs().max().item<float>() << std::endl
                      << "Absolute Error: " << (out - outExpected).abs().max().item<float>() << std::endl;
            REQUIRE(at::allclose(out, outExpected, 15.0, 12));
        }

        TEST_CASE("test_resnet50_whole_model", "[resnet50][inference]")
        {
            // Read the weights from the file
            const char *projectDirectory = std::getenv("PROJECT_ROOT");
            if (projectDirectory == nullptr)
            {
                throw std::runtime_error("PROJECT_ROOT environment variable is not set");
            }

            std::string weightsPath = std::string(projectDirectory) + "/test_data/resnet50_weights_v2.bin";
            ImageInference::test::utils::Reader reader(weightsPath);
            std::vector<at::Tensor> weights;
            std::vector<void *> weightPtrs;
            while (reader.hasNext())
            {
                std::vector<int64_t> sizes;
                float *readTensorPtr = reader.getNextTensor(sizes);
                auto tensor = at::from_blob(readTensorPtr, sizes);
                weights.push_back(tensor);
                weightPtrs.push_back(tensor.mutable_data_ptr<float>());
            }

            // std::cerr << "Weights size: " << weights.size() << std::endl;
            // for (size_t i = 0; i < weights.size(); i++)
            // {
            //     std::cerr << "Weight at Index " << i << " with size " << weights[i].sizes() << std::endl;
            // }

            ImageInference::model::ResNet50 resnet50(weightPtrs, ImageInference::types::ScalarType::Float);

            testWholeResnet50(resnet50, "resnet50_test_ones.bin");
            testWholeResnet50(resnet50, "resnet50_test0.bin");
            testWholeResnet50(resnet50, "resnet50_test1.bin");
            testWholeResnet50(resnet50, "resnet50_test2.bin");
            testWholeResnet50(resnet50, "resnet50_test3.bin");
            testWholeResnet50(resnet50, "resnet50_test4.bin");
            testWholeResnet50(resnet50, "resnet50_test5.bin");
            testWholeResnet50(resnet50, "resnet50_test6.bin");
            testWholeResnet50(resnet50, "resnet50_test7.bin");
            testWholeResnet50(resnet50, "resnet50_test8.bin");
            testWholeResnet50(resnet50, "resnet50_test9.bin");
        }

        void testResnet50Block0(ImageInference::model::ResNet50 &resnet50, const std::string &compareFilepath)
        {
            // Read the input and comparison output
            const char *projectDirectory = std::getenv("PROJECT_ROOT");
            if (projectDirectory == nullptr)
            {
                throw std::runtime_error("PROJECT_ROOT environment variable is not set");
            }

            std::string inputPath = std::string(projectDirectory) + "/test_data/" + compareFilepath;
            ImageInference::test::utils::Reader reader(inputPath);

            std::vector<int64_t> sizes;
            float *readTensorPtr = reader.getNextTensor(sizes);
            Tensor in = at::from_blob(readTensorPtr, sizes);
            readTensorPtr = reader.getNextTensor(sizes);
            Tensor outExpected = at::from_blob(readTensorPtr, sizes);
            Tensor out = at::zeros({1, 256, 56, 56});

            float *inPtr = in.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            REQUIRE((in.size(0) == 1));
            REQUIRE((in.size(1) == 64));
            REQUIRE((in.size(2) == 56));
            REQUIRE((in.size(3) == 56));

            REQUIRE((outExpected.size(0) == 1));
            REQUIRE((outExpected.size(1) == 256));
            REQUIRE((outExpected.size(2) == 56));
            REQUIRE((outExpected.size(3) == 56));

            ImageInference::model::test::ResNet50Test::block0(resnet50, inPtr, outPtr);

            //printMismatchedValues(at::allclose(out, outExpected, 1.0e-1, 1.0e-2), out[0], outExpected[0], 1, 256, 56, 56);
            std::cout << "Relative Error: " << (out - outExpected / (outExpected + 1e-5 * (outExpected == 0))).abs().max().item<float>() << std::endl
                      << "Absolute Error: " << (out - outExpected).abs().max().item<float>() << std::endl;
            REQUIRE(at::allclose(out, outExpected, 5.0, 5));
        }

        TEST_CASE("test_resnet50_block0", "[resnet50][block0]")
        {

            // Read the weights from the file
            const char *projectDirectory = std::getenv("PROJECT_ROOT");
            if (projectDirectory == nullptr)
            {
                throw std::runtime_error("PROJECT_ROOT environment variable is not set");
            }

            std::string weightsPath = std::string(projectDirectory) + "/test_data/resnet50_weights_v2.bin";
            ImageInference::test::utils::Reader reader(weightsPath);
            std::vector<at::Tensor> weights;
            std::vector<void *> weightPtrs;
            while (reader.hasNext())
            {
                std::vector<int64_t> sizes;
                float *readTensorPtr = reader.getNextTensor(sizes);
                auto tensor = at::from_blob(readTensorPtr, sizes);
                weights.push_back(tensor);
                weightPtrs.push_back(tensor.mutable_data_ptr<float>());
            }

            // std::cerr << "Weights size: " << weights.size() << std::endl;
            // for (size_t i = 0; i < weights.size(); i++)
            // {
            //     std::cerr << "Weight at Index " << i << " with size " << weights[i].sizes() << std::endl;
            // }

            ImageInference::model::ResNet50 resnet50(weightPtrs, ImageInference::types::ScalarType::Float);

            testResnet50Block0(resnet50, "resnet50_block0_test_ones.bin");
            testResnet50Block0(resnet50, "resnet50_block0_test0.bin");
            testResnet50Block0(resnet50, "resnet50_block0_test1.bin");
            testResnet50Block0(resnet50, "resnet50_block0_test2.bin");
            testResnet50Block0(resnet50, "resnet50_block0_test3.bin");
            testResnet50Block0(resnet50, "resnet50_block0_test4.bin");
            testResnet50Block0(resnet50, "resnet50_block0_test5.bin");
            testResnet50Block0(resnet50, "resnet50_block0_test6.bin");
            testResnet50Block0(resnet50, "resnet50_block0_test7.bin");
            testResnet50Block0(resnet50, "resnet50_block0_test8.bin");
            testResnet50Block0(resnet50, "resnet50_block0_test9.bin");
        }

        void testResnet50Block1(ImageInference::model::ResNet50 &resnet50, const std::string &compareFilepath)
        {
            // Read the input and comparison output
            const char *projectDirectory = std::getenv("PROJECT_ROOT");
            if (projectDirectory == nullptr)
            {
                throw std::runtime_error("PROJECT_ROOT environment variable is not set");
            }

            std::string inputPath = std::string(projectDirectory) + "/test_data/" + compareFilepath;
            ImageInference::test::utils::Reader reader(inputPath);

            std::vector<int64_t> sizes;
            float *readTensorPtr = reader.getNextTensor(sizes);
            Tensor in = at::from_blob(readTensorPtr, sizes);
            readTensorPtr = reader.getNextTensor(sizes);
            Tensor outExpected = at::from_blob(readTensorPtr, sizes);
            Tensor out = at::zeros({1, 512, 28, 28});

            float *inPtr = in.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            REQUIRE((in.size(0) == 1));
            REQUIRE((in.size(1) == 256));
            REQUIRE((in.size(2) == 56));
            REQUIRE((in.size(3) == 56));

            REQUIRE((outExpected.size(0) == 1));
            REQUIRE((outExpected.size(1) == 512));
            REQUIRE((outExpected.size(2) == 28));
            REQUIRE((outExpected.size(3) == 28));

            ImageInference::model::test::ResNet50Test::block1(resnet50, inPtr, outPtr);

            //printMismatchedValues(at::allclose(out, outExpected, 50.0, 3), out[0], outExpected[0], 1, 512, 28, 28);
            std::cout << "Relative Error: " << (out - outExpected / (outExpected + 1e-5 * (outExpected == 0))).abs().max().item<float>() << std::endl
                      << "Absolute Error: " << (out - outExpected).abs().max().item<float>() << std::endl;
            REQUIRE(at::allclose(out, outExpected, 8.0, 5.5));
        }

        TEST_CASE("test_resnet50_block1", "[resnet50][block1]")
        {

            // Read the weights from the file
            const char *projectDirectory = std::getenv("PROJECT_ROOT");
            if (projectDirectory == nullptr)
            {
                throw std::runtime_error("PROJECT_ROOT environment variable is not set");
            }

            std::string weightsPath = std::string(projectDirectory) + "/test_data/resnet50_weights_v2.bin";
            ImageInference::test::utils::Reader reader(weightsPath);
            std::vector<at::Tensor> weights;
            std::vector<void *> weightPtrs;
            while (reader.hasNext())
            {
                std::vector<int64_t> sizes;
                float *readTensorPtr = reader.getNextTensor(sizes);
                auto tensor = at::from_blob(readTensorPtr, sizes);
                weights.push_back(tensor);
                weightPtrs.push_back(tensor.mutable_data_ptr<float>());
            }

            // std::cerr << "Weights size: " << weights.size() << std::endl;
            // for (size_t i = 0; i < weights.size(); i++)
            // {
            //     std::cerr << "Weight at Index " << i << " with size " << weights[i].sizes() << std::endl;
            // }

            ImageInference::model::ResNet50 resnet50(weightPtrs, ImageInference::types::ScalarType::Float);

            testResnet50Block1(resnet50, "resnet50_block1_test_ones.bin");
            testResnet50Block1(resnet50, "resnet50_block1_test0.bin");
            testResnet50Block1(resnet50, "resnet50_block1_test1.bin");
            testResnet50Block1(resnet50, "resnet50_block1_test2.bin");
            testResnet50Block1(resnet50, "resnet50_block1_test3.bin");
            testResnet50Block1(resnet50, "resnet50_block1_test4.bin");
            testResnet50Block1(resnet50, "resnet50_block1_test5.bin");
            testResnet50Block1(resnet50, "resnet50_block1_test6.bin");
            testResnet50Block1(resnet50, "resnet50_block1_test7.bin");
            testResnet50Block1(resnet50, "resnet50_block1_test8.bin");
            testResnet50Block1(resnet50, "resnet50_block1_test9.bin");
        }

        void testResnet50Block2(ImageInference::model::ResNet50 &resnet50, const std::string &compareFilepath)
        {
            // Read the input and comparison output
            const char *projectDirectory = std::getenv("PROJECT_ROOT");
            if (projectDirectory == nullptr)
            {
                throw std::runtime_error("PROJECT_ROOT environment variable is not set");
            }

            std::string inputPath = std::string(projectDirectory) + "/test_data/" + compareFilepath;
            ImageInference::test::utils::Reader reader(inputPath);

            std::vector<int64_t> sizes;
            float *readTensorPtr = reader.getNextTensor(sizes);
            Tensor in = at::from_blob(readTensorPtr, sizes);
            readTensorPtr = reader.getNextTensor(sizes);
            Tensor outExpected = at::from_blob(readTensorPtr, sizes);
            Tensor out = at::zeros({1, 1024, 14, 14});

            float *inPtr = in.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            REQUIRE((in.size(0) == 1));
            REQUIRE((in.size(1) == 512));
            REQUIRE((in.size(2) == 28));
            REQUIRE((in.size(3) == 28));

            REQUIRE((outExpected.size(0) == 1));
            REQUIRE((outExpected.size(1) == 1024));
            REQUIRE((outExpected.size(2) == 14));
            REQUIRE((outExpected.size(3) == 14));

            ImageInference::model::test::ResNet50Test::block2(resnet50, inPtr, outPtr);

            //printMismatchedValues(at::allclose(out, outExpected, 50.0, 3), out[0], outExpected[0], 1, 1024, 14, 14);
            std::cout << "Relative Error: " << (out - outExpected / (outExpected + 1e-5 * (outExpected == 0))).abs().max().item<float>() << std::endl
                      << "Absolute Error: " << (out - outExpected).abs().max().item<float>() << std::endl;
            REQUIRE(at::allclose(out, outExpected, 12.0, 10));
        }

        TEST_CASE("test_resnet50_block2", "[resnet50][block2]")
        {

            // Read the weights from the file
            const char *projectDirectory = std::getenv("PROJECT_ROOT");
            if (projectDirectory == nullptr)
            {
                throw std::runtime_error("PROJECT_ROOT environment variable is not set");
            }

            std::string weightsPath = std::string(projectDirectory) + "/test_data/resnet50_weights_v2.bin";
            ImageInference::test::utils::Reader reader(weightsPath);
            std::vector<at::Tensor> weights;
            std::vector<void *> weightPtrs;
            while (reader.hasNext())
            {
                std::vector<int64_t> sizes;
                float *readTensorPtr = reader.getNextTensor(sizes);
                auto tensor = at::from_blob(readTensorPtr, sizes);
                weights.push_back(tensor);
                weightPtrs.push_back(tensor.mutable_data_ptr<float>());
            }

            // std::cerr << "Weights size: " << weights.size() << std::endl;
            // for (size_t i = 0; i < weights.size(); i++)
            // {
            //     std::cerr << "Weight at Index " << i << " with size " << weights[i].sizes() << std::endl;
            // }

            ImageInference::model::ResNet50 resnet50(weightPtrs, ImageInference::types::ScalarType::Float);

            testResnet50Block2(resnet50, "resnet50_block2_test_ones.bin");
            testResnet50Block2(resnet50, "resnet50_block2_test0.bin");
            testResnet50Block2(resnet50, "resnet50_block2_test1.bin");
            testResnet50Block2(resnet50, "resnet50_block2_test2.bin");
            testResnet50Block2(resnet50, "resnet50_block2_test3.bin");
            testResnet50Block2(resnet50, "resnet50_block2_test4.bin");
            testResnet50Block2(resnet50, "resnet50_block2_test5.bin");
            testResnet50Block2(resnet50, "resnet50_block2_test6.bin");
            testResnet50Block2(resnet50, "resnet50_block2_test7.bin");
            testResnet50Block2(resnet50, "resnet50_block2_test8.bin");
            testResnet50Block2(resnet50, "resnet50_block2_test9.bin");
        }

        void testResnet50Block3(ImageInference::model::ResNet50 &resnet50, const std::string &compareFilepath)
        {
            // Read the input and comparison output
            const char *projectDirectory = std::getenv("PROJECT_ROOT");
            if (projectDirectory == nullptr)
            {
                throw std::runtime_error("PROJECT_ROOT environment variable is not set");
            }

            std::string inputPath = std::string(projectDirectory) + "/test_data/" + compareFilepath;
            ImageInference::test::utils::Reader reader(inputPath);

            std::vector<int64_t> sizes;
            float *readTensorPtr = reader.getNextTensor(sizes);
            Tensor in = at::from_blob(readTensorPtr, sizes);
            readTensorPtr = reader.getNextTensor(sizes);
            Tensor outExpected = at::from_blob(readTensorPtr, sizes);
            Tensor out = at::zeros({1, 2048, 7, 7});

            float *inPtr = in.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            REQUIRE((in.size(0) == 1));
            REQUIRE((in.size(1) == 1024));
            REQUIRE((in.size(2) == 14));
            REQUIRE((in.size(3) == 14));

            REQUIRE((outExpected.size(0) == 1));
            REQUIRE((outExpected.size(1) == 2048));
            REQUIRE((outExpected.size(2) == 7));
            REQUIRE((outExpected.size(3) == 7));

            ImageInference::model::test::ResNet50Test::block3(resnet50, inPtr, outPtr);

            //printMismatchedValues(at::allclose(out, outExpected, 50.0, 3), out[0], outExpected[0], 1, 2048, 7, 7);
            std::cout << "Relative Error: " << (out - outExpected / (outExpected + 1e-5 * (outExpected == 0))).abs().max().item<float>() << std::endl
                      << "Absolute Error: " << (out - outExpected).abs().max().item<float>() << std::endl;
            REQUIRE(at::allclose(out, outExpected, 9.0, 7));
        }

        TEST_CASE("test_resnet50_block3", "[resnet50][block3]")
        {
            // Read the weights from the file
            const char *projectDirectory = std::getenv("PROJECT_ROOT");
            if (projectDirectory == nullptr)
            {
                throw std::runtime_error("PROJECT_ROOT environment variable is not set");
            }

            std::string weightsPath = std::string(projectDirectory) + "/test_data/resnet50_weights_v2.bin";
            ImageInference::test::utils::Reader reader(weightsPath);
            std::vector<at::Tensor> weights;
            std::vector<void *> weightPtrs;
            while (reader.hasNext())
            {
                std::vector<int64_t> sizes;
                float *readTensorPtr = reader.getNextTensor(sizes);
                auto tensor = at::from_blob(readTensorPtr, sizes);

                weights.push_back(tensor);
                weightPtrs.push_back(tensor.mutable_data_ptr<float>());
            }

            // std::cerr << "Weights size: " << weights.size() << std::endl;
            // for (size_t i = 0; i < weights.size(); i++)
            // {
            //     std::cerr << "Weight at Index " << i << " with size " << weights[i].sizes() << std::endl;
            // }

            ImageInference::model::ResNet50 resnet50(weightPtrs, ImageInference::types::ScalarType::Float);

            testResnet50Block3(resnet50, "resnet50_block3_test_ones.bin");
            testResnet50Block3(resnet50, "resnet50_block3_test0.bin");
            testResnet50Block3(resnet50, "resnet50_block3_test0.bin");
            testResnet50Block3(resnet50, "resnet50_block3_test1.bin");
            testResnet50Block3(resnet50, "resnet50_block3_test2.bin");
            testResnet50Block3(resnet50, "resnet50_block3_test3.bin");
            testResnet50Block3(resnet50, "resnet50_block3_test4.bin");
            testResnet50Block3(resnet50, "resnet50_block3_test5.bin");
            testResnet50Block3(resnet50, "resnet50_block3_test6.bin");
            testResnet50Block3(resnet50, "resnet50_block3_test7.bin");
            testResnet50Block3(resnet50, "resnet50_block3_test8.bin");
            testResnet50Block3(resnet50, "resnet50_block3_test9.bin");
        }

        at::Tensor atenConvBlock(
            std::vector<at::Tensor> &weights,
            at::Tensor &in,
            size_t kernelIndex,
            size_t gammaIndex,
            size_t betaIndex,
            size_t meanIndex,
            size_t varIndex,
            size_t stride,
            size_t inPadding)
        {
            Tensor weight = weights[kernelIndex];
            Tensor batchGamma = weights[gammaIndex];
            Tensor batchBeta = weights[betaIndex];
            Tensor mean = weights[meanIndex];
            Tensor var = weights[varIndex];

            Tensor expected = at::conv2d(in, weight, {}, stride, inPadding);
            expected = at::batch_norm(expected, batchGamma, batchBeta, mean, var, false, 0.1, 1e-5, false);
            expected = at::relu(expected);
            return expected;
        }

        at::Tensor atenConvBlockShortcut(
            std::vector<at::Tensor> &weights,
            at::Tensor &in,
            size_t kernelIndex,
            size_t gammaIndex,
            size_t betaIndex,
            size_t meanIndex,
            size_t varIndex,
            size_t inPadding,
            at::Tensor &shortcut)
        {
            Tensor weight = weights[kernelIndex];
            Tensor batchGamma = weights[gammaIndex];
            Tensor batchBeta = weights[betaIndex];
            Tensor mean = weights[meanIndex];
            Tensor var = weights[varIndex];

            Tensor expected = at::conv2d(in, weight, {}, 1, inPadding);
            expected = at::batch_norm(expected, batchGamma, batchBeta, mean, var, false, 0.1, 1e-5, false);
            expected += shortcut;
            expected = at::relu(expected);
            return expected;
        }

        at::Tensor atenConvBlockProjection(
            std::vector<at::Tensor> &weights,
            at::Tensor &in,
            size_t kernelIndex,
            size_t gammaIndex,
            size_t betaIndex,
            size_t meanIndex,
            size_t varIndex,
            size_t stride,
            size_t inPadding,
            at::Tensor &shortcut,
            size_t projectionKernelIndex,
            size_t projectionGammaIndex,
            size_t projectionBetaIndex,
            size_t projectionMeanIndex,
            size_t projectionVarIndex)
        {
            Tensor weight = weights[kernelIndex];
            Tensor batchGamma = weights[gammaIndex];
            Tensor batchBeta = weights[betaIndex];
            Tensor mean = weights[meanIndex];
            Tensor var = weights[varIndex];
            Tensor projectionWeight = weights[projectionKernelIndex];
            Tensor projectionBatchGamma = weights[projectionGammaIndex];
            Tensor projectionBatchBeta = weights[projectionBetaIndex];
            Tensor projectionMean = weights[projectionMeanIndex];
            Tensor projectionVar = weights[projectionVarIndex];

            Tensor expected = at::conv2d(in, weight, {}, 1, inPadding);
            expected = at::batch_norm(expected, batchGamma, batchBeta, mean, var, false, 0.1, 1e-5, false);
            Tensor projection = at::conv2d(shortcut, projectionWeight, {}, stride);
            projection = at::batch_norm(projection, projectionBatchGamma, projectionBatchBeta, projectionMean, projectionVar, false, 0.1, 1e-5, false);
            expected += projection;
            expected = at::relu(expected);
            return expected;
        }

        TEST_CASE("test_resnet50_block0_aten_implementation", "[resnet50][block0]")
        {
            // Read the weights from the file
            const char *projectDirectory = std::getenv("PROJECT_ROOT");
            if (projectDirectory == nullptr)
            {
                throw std::runtime_error("PROJECT_ROOT environment variable is not set");
            }

            std::string weightsPath = std::string(projectDirectory) + "/test_data/resnet50_weights_v2.bin";
            ImageInference::test::utils::Reader reader(weightsPath);
            std::vector<at::Tensor> weights;
            std::vector<void *> weightPtrs;
            while (reader.hasNext())
            {
                std::vector<int64_t> sizes;
                float *readTensorPtr = reader.getNextTensor(sizes);
                auto tensor = at::from_blob(readTensorPtr, sizes);
                // if (tensor.sizes().size() >= 2)
                // {
                //     tensor = tensor.transpose(-2, -1).contiguous();
                // }
                weights.push_back(tensor);
                weightPtrs.push_back(tensor.mutable_data_ptr<float>());
            }

            // std::cerr << "Weights size: " << weights->size() << std::endl;
            // for (size_t i = 0; i < weights->size(); i++)
            // {
            //     std::cerr << "Weight at Index " << i << " with size " << (*weights)[i].sizes() << std::endl;
            // }

            ImageInference::model::ResNet50 resnet50(weightPtrs, ImageInference::types::ScalarType::Float);

            std::string inputPath = std::string(projectDirectory) + "/test_data/" + "resnet50_block0_test_ones.bin";
            ImageInference::test::utils::Reader readerInput(inputPath);

            std::vector<int64_t> sizes;
            float *readTensorPtr = readerInput.getNextTensor(sizes);
            Tensor in = at::from_blob(readTensorPtr, sizes);
            readTensorPtr = readerInput.getNextTensor(sizes);
            Tensor outExpected = at::from_blob(readTensorPtr, sizes);
            Tensor out;
            Tensor shortcut;
            Tensor outCustom = at::zeros({1, 256, 56, 56});
            float *inPtr = in.mutable_data_ptr<float>();
            float *outCustomPtr = outCustom.mutable_data_ptr<float>();

            REQUIRE((in.size(0) == 1));
            REQUIRE((in.size(1) == 64));
            REQUIRE((in.size(2) == 56));
            REQUIRE((in.size(3) == 56));

            REQUIRE((outExpected.size(0) == 1));
            REQUIRE((outExpected.size(1) == 256));
            REQUIRE((outExpected.size(2) == 56));
            REQUIRE((outExpected.size(3) == 56));

            // Do aten implementation of block0
            shortcut = in;
            out = atenConvBlock(
                weights, in,
                ImageInference::model::ResNet50::layer1_0_conv1_weight,
                ImageInference::model::ResNet50::layer1_0_bn1_weight,
                ImageInference::model::ResNet50::layer1_0_bn1_bias,
                ImageInference::model::ResNet50::layer1_0_bn1_running_mean,
                ImageInference::model::ResNet50::layer1_0_bn1_running_var,
                1, 0);

            out = atenConvBlock(
                weights, out,
                ImageInference::model::ResNet50::layer1_0_conv2_weight,
                ImageInference::model::ResNet50::layer1_0_bn2_weight,
                ImageInference::model::ResNet50::layer1_0_bn2_bias,
                ImageInference::model::ResNet50::layer1_0_bn2_running_mean,
                ImageInference::model::ResNet50::layer1_0_bn2_running_var,
                1, 1);

            out = atenConvBlockProjection(
                weights, out,
                ImageInference::model::ResNet50::layer1_0_conv3_weight,
                ImageInference::model::ResNet50::layer1_0_bn3_weight,
                ImageInference::model::ResNet50::layer1_0_bn3_bias,
                ImageInference::model::ResNet50::layer1_0_bn3_running_mean,
                ImageInference::model::ResNet50::layer1_0_bn3_running_var,
                1, 0, shortcut,
                ImageInference::model::ResNet50::layer1_0_downsample_0_weight,
                ImageInference::model::ResNet50::layer1_0_downsample_1_weight,
                ImageInference::model::ResNet50::layer1_0_downsample_1_bias,
                ImageInference::model::ResNet50::layer1_0_downsample_1_running_mean,
                ImageInference::model::ResNet50::layer1_0_downsample_1_running_var);

            shortcut = out;
            out = atenConvBlock(
                weights, out,
                ImageInference::model::ResNet50::layer1_1_conv1_weight,
                ImageInference::model::ResNet50::layer1_1_bn1_weight,
                ImageInference::model::ResNet50::layer1_1_bn1_bias,
                ImageInference::model::ResNet50::layer1_1_bn1_running_mean,
                ImageInference::model::ResNet50::layer1_1_bn1_running_var,
                1, 0);

            out = atenConvBlock(
                weights, out,
                ImageInference::model::ResNet50::layer1_1_conv2_weight,
                ImageInference::model::ResNet50::layer1_1_bn2_weight,
                ImageInference::model::ResNet50::layer1_1_bn2_bias,
                ImageInference::model::ResNet50::layer1_1_bn2_running_mean,
                ImageInference::model::ResNet50::layer1_1_bn2_running_var,
                1, 1);

            out = atenConvBlockShortcut(
                weights, out,
                ImageInference::model::ResNet50::layer1_1_conv3_weight,
                ImageInference::model::ResNet50::layer1_1_bn3_weight,
                ImageInference::model::ResNet50::layer1_1_bn3_bias,
                ImageInference::model::ResNet50::layer1_1_bn3_running_mean,
                ImageInference::model::ResNet50::layer1_1_bn3_running_var,
                0, shortcut);

            shortcut = out;
            out = atenConvBlock(
                weights, out,
                ImageInference::model::ResNet50::layer1_2_conv1_weight,
                ImageInference::model::ResNet50::layer1_2_bn1_weight,
                ImageInference::model::ResNet50::layer1_2_bn1_bias,
                ImageInference::model::ResNet50::layer1_2_bn1_running_mean,
                ImageInference::model::ResNet50::layer1_2_bn1_running_var,
                1, 0);

            out = atenConvBlock(
                weights, out,
                ImageInference::model::ResNet50::layer1_2_conv2_weight,
                ImageInference::model::ResNet50::layer1_2_bn2_weight,
                ImageInference::model::ResNet50::layer1_2_bn2_bias,
                ImageInference::model::ResNet50::layer1_2_bn2_running_mean,
                ImageInference::model::ResNet50::layer1_2_bn2_running_var,
                1, 1);

            out = atenConvBlockShortcut(
                weights, out,
                ImageInference::model::ResNet50::layer1_2_conv3_weight,
                ImageInference::model::ResNet50::layer1_2_bn3_weight,
                ImageInference::model::ResNet50::layer1_2_bn3_bias,
                ImageInference::model::ResNet50::layer1_2_bn3_running_mean,
                ImageInference::model::ResNet50::layer1_2_bn3_running_var,
                0, shortcut);

            ImageInference::model::test::ResNet50Test::block0(resnet50, inPtr, outCustomPtr);

            REQUIRE((out.size(0) == 1));
            REQUIRE((out.size(1) == 256));
            REQUIRE((out.size(2) == 56));
            REQUIRE((out.size(3) == 56));

            //printMismatchedValues(at::allclose(outCustom, out, 50.0, 3), outCustom[0], out[0], 1, 256, 56, 56);
            std::cout << "Relative Error: " << (outCustom - out).abs().max().item<float>() / out.abs().max().item<float>() << std::endl
                      << "Absolute Error: " << (outCustom - out).abs().max().item<float>() << std::endl;
            CHECK(at::allclose(outCustom, out, 50.0, 3));

            //printMismatchedValues(at::allclose(out, outExpected, 1.0e-1, 1.0e-2), out[0], outExpected[0], 1, 256, 56, 56);
            REQUIRE(at::allclose(out, outExpected, 1.0e-1, 1.0e-2));
        }

        void testResnet50conv7x7(ImageInference::model::ResNet50 &resnet50, const std::string &compareFilepath)
        {
            constexpr size_t stride = 2;
            constexpr size_t inPadding = 3;
            constexpr size_t blockSize = 1;
            constexpr size_t outChannels = 64;
            constexpr size_t inChannels = 3;
            constexpr size_t height = 224;
            constexpr size_t width = 224;
            constexpr size_t kernelHeight = 7;
            constexpr size_t kernelWidth = 7;

            // Read the input and comparison output
            const char *projectDirectory = std::getenv("PROJECT_ROOT");
            if (projectDirectory == nullptr)
            {
                throw std::runtime_error("PROJECT_ROOT environment variable is not set");
            }

            std::string inputPath = std::string(projectDirectory) + "/test_data/" + compareFilepath;
            ImageInference::test::utils::Reader reader(inputPath);

            std::vector<int64_t> sizes;
            float *readTensorPtr = reader.getNextTensor(sizes);
            Tensor in = at::from_blob(readTensorPtr, sizes);
            readTensorPtr = reader.getNextTensor(sizes);
            Tensor outExpected = at::from_blob(readTensorPtr, sizes);
            Tensor out = at::zeros({1, 64, 112, 112});

            float *inPtr = in.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            REQUIRE((in.size(0) == 1));
            REQUIRE((in.size(1) == 3));
            REQUIRE((in.size(2) == 224));
            REQUIRE((in.size(3) == 224));

            REQUIRE((outExpected.size(0) == 1));
            REQUIRE((outExpected.size(1) == 64));
            REQUIRE((outExpected.size(2) == 112));
            REQUIRE((outExpected.size(3) == 112));

            float *weightPtr = ImageInference::model::test::ResNet50Test::getWeight(resnet50, ImageInference::model::ResNet50::conv1_weight);
            float *gammaPtr = ImageInference::model::test::ResNet50Test::getWeight(resnet50, ImageInference::model::ResNet50::bn1_weight);
            float *betaPtr = ImageInference::model::test::ResNet50Test::getWeight(resnet50, ImageInference::model::ResNet50::bn1_bias);
            float *meanPtr = ImageInference::model::test::ResNet50Test::getWeight(resnet50, ImageInference::model::ResNet50::bn1_running_mean);
            float *varPtr = ImageInference::model::test::ResNet50Test::getWeight(resnet50, ImageInference::model::ResNet50::bn1_running_var);

            ImageInference::model::test::ResNet50Test::convBlock<
                stride, inPadding, blockSize, outChannels, inChannels,
                height, width, kernelHeight, kernelWidth>(inPtr, weightPtr, gammaPtr, betaPtr, meanPtr, varPtr, outPtr);

            // //printMismatchedValues(at::allclose(out, outExpected, 1.0e-1, 1.0e-2), out[0], outExpected[0], 1, 2048, 7, 7);
            std::cout << "Relative Error: " << (out - outExpected).abs().max().item<float>() / outExpected.abs().max().item<float>() << std::endl
                      << "Absolute Error: " << (out - outExpected).abs().max().item<float>() << std::endl;
            REQUIRE(at::allclose(out, outExpected, 1.0e-1, 1.0e-2));
        }

        TEST_CASE("test_resnet50_first_conv7x7", "[resnet50][conv7x7]")
        {
            // Read the weights from the file
            const char *projectDirectory = std::getenv("PROJECT_ROOT");
            if (projectDirectory == nullptr)
            {
                throw std::runtime_error("PROJECT_ROOT environment variable is not set");
            }

            std::string weightsPath = std::string(projectDirectory) + "/test_data/resnet50_weights_v2.bin";
            ImageInference::test::utils::Reader reader(weightsPath);
            std::vector<at::Tensor> weights;
            std::vector<void *> weightPtrs;
            while (reader.hasNext())
            {
                std::vector<int64_t> sizes;
                float *readTensorPtr = reader.getNextTensor(sizes);
                auto tensor = at::from_blob(readTensorPtr, sizes);
                weights.push_back(tensor);
                weightPtrs.push_back(tensor.mutable_data_ptr<float>());
            }

            // std::cerr << "Weights size: " << weights.size() << std::endl;
            // for (size_t i = 0; i < weights.size(); i++)
            // {
            //     std::cerr << "Weight at Index " << i << " with size " << weights[i].sizes() << std::endl;
            // }

            ImageInference::model::ResNet50 resnet50(weightPtrs, ImageInference::types::ScalarType::Float);

            return;  // There is a error deep down in this test setup witch leads to a segfault

            testResnet50conv7x7(resnet50, "resnet50_conv7x7_ones.bin");
            testResnet50conv7x7(resnet50, "resnet50_conv7x7_test0.bin");
            testResnet50conv7x7(resnet50, "resnet50_conv7x7_test0.bin");
            testResnet50conv7x7(resnet50, "resnet50_conv7x7_test1.bin");
            testResnet50conv7x7(resnet50, "resnet50_conv7x7_test2.bin");
            testResnet50conv7x7(resnet50, "resnet50_conv7x7_test3.bin");
            testResnet50conv7x7(resnet50, "resnet50_conv7x7_test4.bin");
            testResnet50conv7x7(resnet50, "resnet50_conv7x7_test5.bin");
            testResnet50conv7x7(resnet50, "resnet50_conv7x7_test6.bin");
            testResnet50conv7x7(resnet50, "resnet50_conv7x7_test7.bin");
            testResnet50conv7x7(resnet50, "resnet50_conv7x7_test8.bin");
            testResnet50conv7x7(resnet50, "resnet50_conv7x7_test9.bin");
        }

        void testResnet50bn1(ImageInference::model::ResNet50 &resnet50, const std::string &compareFilepath)
        {
            constexpr size_t inPadding = 0;
            constexpr size_t blockSize = 32;
            constexpr size_t inChannels = 64;
            constexpr size_t height = 1;
            constexpr size_t width = 1;

            // Read the input and comparison output
            const char *projectDirectory = std::getenv("PROJECT_ROOT");
            if (projectDirectory == nullptr)
            {
                throw std::runtime_error("PROJECT_ROOT environment variable is not set");
            }

            std::string inputPath = std::string(projectDirectory) + "/test_data/" + compareFilepath;
            ImageInference::test::utils::Reader reader(inputPath);

            std::vector<int64_t> sizes;
            float *readTensorPtr = reader.getNextTensor(sizes);
            Tensor in = at::from_blob(readTensorPtr, sizes);
            readTensorPtr = reader.getNextTensor(sizes);
            Tensor outExpected = at::from_blob(readTensorPtr, sizes);

            float *inPtr = in.mutable_data_ptr<float>();

            REQUIRE((in.size(0) == 1));
            REQUIRE((in.size(1) == 64));
            REQUIRE((in.size(2) == 1));
            REQUIRE((in.size(3) == 1));

            REQUIRE((outExpected.size(0) == 1));
            REQUIRE((outExpected.size(1) == 64));
            REQUIRE((outExpected.size(2) == 1));
            REQUIRE((outExpected.size(3) == 1));

            float *gammaPtr = ImageInference::model::test::ResNet50Test::getWeight(resnet50, ImageInference::model::ResNet50::bn1_weight);
            float *betaPtr = ImageInference::model::test::ResNet50Test::getWeight(resnet50, ImageInference::model::ResNet50::bn1_bias);
            float *meanPtr = ImageInference::model::test::ResNet50Test::getWeight(resnet50, ImageInference::model::ResNet50::bn1_running_mean);
            float *varPtr = ImageInference::model::test::ResNet50Test::getWeight(resnet50, ImageInference::model::ResNet50::bn1_running_var);

            ImageInference::types::Image<float, inPadding, blockSize, inChannels, height, width> image(inPtr);
            ImageInference::types::BatchNorm<float, inChannels> batchNorm(gammaPtr, betaPtr, meanPtr, varPtr);
            ImageInference::types::Image<float, inPadding, blockSize, inChannels, height, width> out;
            float *imagePtr = image.getPointer();
            float *outputPtr = out.getPointer();

            constexpr size_t inChannelsBlocks = inChannels / blockSize;
            for (size_t iBChannel = 0; iBChannel < inChannelsBlocks; iBChannel++)
            {
                for (size_t iHeight = 0; iHeight < height; iHeight++)
                {
                    for (size_t iWidth = 0; iWidth < width; iWidth++)
                    {
                        for (size_t iChannel = 0; iChannel < blockSize; iChannel++)
                        {
                            size_t imageOffset = image.getOffset(iBChannel, iHeight, iWidth, iChannel);
                            size_t outOffset = out.getOffset(iBChannel, iHeight, iWidth, iChannel);
                            size_t channelOffset = iBChannel * blockSize + iChannel;
                            outputPtr[outOffset] = ImageInference::model::test::ResNet50Test::batchNorm(
                                imagePtr[imageOffset],
                                batchNorm.getGammaVariancePointer()[channelOffset],
                                batchNorm.getBetaPointer()[channelOffset],
                                batchNorm.getMeanPointer()[channelOffset]);
                            std::cerr << "Image: " << imagePtr[imageOffset] << " Gamma: " << gammaPtr[channelOffset] << " Beta: " << betaPtr[channelOffset] << " Mean: " << meanPtr[channelOffset] << " Var: " << varPtr[channelOffset] << " Output: " << outputPtr[outOffset] << std::endl
                                      << "Indices: " << iBChannel << " " << iHeight << " " << iWidth << " " << iChannel << std::endl;
                            // outputPtr[outOffset] = imagePtr[imageOffset] * gammaPtr[channelOffset] + betaPtr[channelOffset];
                        }
                    }
                }
            }
            auto flatten = out.flatten();
            Tensor outTensor = at::from_blob(flatten.getPointer(), {1, inChannels, height, width});

            std::cerr << "OUTPUT" << std::endl;
            //printMismatchedValues(at::allclose(outTensor, outExpected, 1.0e-1, 1.0e-2), outTensor[0], outExpected[0], 1, inChannels, height, width);
            REQUIRE(at::allclose(outTensor, outExpected, 1.0e-1, 1.0e-2));
        }

        TEST_CASE("test_resnet50_batchNorm1", "[resnet50][batchNorm]")
        {
            // Read the weights from the file
            const char *projectDirectory = std::getenv("PROJECT_ROOT");
            if (projectDirectory == nullptr)
            {
                throw std::runtime_error("PROJECT_ROOT environment variable is not set");
            }

            std::string weightsPath = std::string(projectDirectory) + "/test_data/resnet50_weights_v2.bin";
            ImageInference::test::utils::Reader reader(weightsPath);
            std::vector<at::Tensor> weights;
            std::vector<void *> weightPtrs;
            while (reader.hasNext())
            {
                std::vector<int64_t> sizes;
                float *readTensorPtr = reader.getNextTensor(sizes);
                auto tensor = at::from_blob(readTensorPtr, sizes);
                weights.push_back(tensor);
                weightPtrs.push_back(tensor.mutable_data_ptr<float>());
            }

            REQUIRE((weights[ImageInference::model::ResNet50::bn1_weight].size(0) == 64));
            REQUIRE((weights[ImageInference::model::ResNet50::bn1_bias].size(0) == 64));

            // std::cerr << "Weights size: " << weights.size() << std::endl;
            // for (size_t i = 0; i < weights.size(); i++)
            // {
            //     std::cerr << "Weight at Index " << i << " with size " << weights[i].sizes() << std::endl;
            // }

            ImageInference::model::ResNet50 resnet50(weightPtrs, ImageInference::types::ScalarType::Float);

            testResnet50bn1(resnet50, "resnet50_batchNorm1_ones.bin");
            testResnet50bn1(resnet50, "resnet50_batchNorm1_test0.bin");
            testResnet50bn1(resnet50, "resnet50_batchNorm1_test0.bin");
            testResnet50bn1(resnet50, "resnet50_batchNorm1_test1.bin");
            testResnet50bn1(resnet50, "resnet50_batchNorm1_test2.bin");
            testResnet50bn1(resnet50, "resnet50_batchNorm1_test3.bin");
            testResnet50bn1(resnet50, "resnet50_batchNorm1_test4.bin");
            testResnet50bn1(resnet50, "resnet50_batchNorm1_test5.bin");
            testResnet50bn1(resnet50, "resnet50_batchNorm1_test6.bin");
            testResnet50bn1(resnet50, "resnet50_batchNorm1_test7.bin");
            testResnet50bn1(resnet50, "resnet50_batchNorm1_test8.bin");
            testResnet50bn1(resnet50, "resnet50_batchNorm1_test9.bin");
        }
    }
}