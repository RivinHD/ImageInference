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

namespace ImageInference
{
    namespace test
    {
        using at::Tensor;

        void printMismatchedValues(const Tensor &out, const Tensor &expected, size_t stride, size_t channels, size_t height, size_t width)
        {
            if (!at::allclose(out, expected))
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

        TEST_CASE("test_resnet50_conv3x3_channels16x16", "[resnet50][convolution]")
        {
            Tensor in = at::rand({1, 16, 10, 10});
            Tensor weight = at::rand({16, 16, 3, 3});
            Tensor batchGamma = at::rand({16});
            Tensor batchBeta = at::rand({16});

            // The actual testing is done in python.
            constexpr size_t stride = 1;
            constexpr size_t inPadding = 1;
            constexpr size_t blockSize = 16;
            constexpr size_t outChannels = 16;
            constexpr size_t inChannels = 16;
            constexpr size_t height = 10;
            constexpr size_t width = 10;
            constexpr size_t kernelHeight = 3;
            constexpr size_t kernelWidth = 3;

            Tensor out = at::zeros({outChannels, height, width});

            float *inPtr = in.mutable_data_ptr<float>();
            float *weightPtr = weight.mutable_data_ptr<float>();
            float *batchGammaPtr = batchGamma.mutable_data_ptr<float>();
            float *batchBetaPtr = batchBeta.mutable_data_ptr<float>();
            float *outPtr = out.mutable_data_ptr<float>();

            ImageInference::model::test::ResNet50Test::convBlock<
                stride, inPadding, blockSize, outChannels, inChannels,
                height, width, kernelHeight, kernelWidth>(inPtr, weightPtr, batchGammaPtr, batchBetaPtr, outPtr);

            Tensor mean = at::mean(in, {0, 2, 3});
            Tensor var = at::var(in, {0, 2, 3}, false);

            Tensor expected = at::conv2d(in, weight, {}, stride, inPadding);
            // expected = at::batch_norm(expected, batchGamma, batchBeta, mean, var, false, 0.1, 1e-5, false);
            // expected = at::relu(expected);

            printMismatchedValues(out, expected[0], stride, outChannels, height, width);
            REQUIRE(at::allclose(out, expected[0]));
        }

        TEST_CASE("test_resnet50_maxpool", "[resnet50][maxpool]")
        {

            // The actual testing is done in python.
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

            printMismatchedValues(out, expected, stride, channels, height, width);

            REQUIRE(at::allclose(out, expected));
        }

        TEST_CASE("test_resnet50_maxpool_channels", "[resnet50][maxpool]")
        {

            // The actual testing is done in python.
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

            printMismatchedValues(out, expected, stride, channels, height, width);

            REQUIRE(at::allclose(out, expected));
        }

        TEST_CASE("test_resnet50_global_average", "[resnet50][globalAverage]")
        {
            Tensor in = at::randn({16, 10, 10});

            // The actual testing is done in python.
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

            if (!at::allclose(out, expected, 1.0e-3, 1.0e-5))
            {
                for (size_t i = 0; i < 1000; i++)
                {
                    if (out[i].item<float>() != expected[i].item<float>())
                    {
                        std::cerr << "Expected " << expected[i].item<float>() << " but got " << out[i].item<float>() << " at index " << i << std::endl;
                    }
                }
            }

            REQUIRE(at::allclose(out, expected, 1.0e-3, 1.0e-5));
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
                        float batchVar = 1 / std::sqrt(var[i].item<float>() + 1e-05);
                        out[0][i][j][k] = ImageInference::model::test::ResNet50Test::batchNorm(
                            in[0][i][j][k].item<float>(),
                            gamma[i].item<float>(),
                            beta[i].item<float>(),
                            mean[i].item<float>(),
                            batchVar);
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
    }
}