#ifndef USE_ATEN_LIB
#define USE_ATEN_LIB
#endif // !USE_ATEN_LIB
#include <ATen/ATen.h>
#include <torch/library.h>
#include <catch2/catch_test_macros.hpp>
#include "../../types/Image.h"

namespace ImageInference
{
    namespace test
    {
        namespace types
        {
            using at::Tensor;
            using ImageInference::types::Image;

            TEST_CASE("test_types_image_initialization", "[types][image][init]")
            {
                constexpr size_t padding = 0;
                constexpr size_t blockSize = 16;
                constexpr size_t channels = 16;
                constexpr size_t height = 10;
                constexpr size_t width = 10;

                Tensor input = at::randn({channels, height, width});
                Image<float, padding, blockSize, channels, height, width> image(input.const_data_ptr<float>());
                Tensor out = at::from_blob(image.getPointer(), {channels / blockSize, height, width, blockSize});

                Tensor expected = input.view({channels / blockSize, blockSize, height, width}).permute({0, 2, 3, 1}).contiguous();

                // Evaluating with values that of constexpr require additional brackets
                REQUIRE((image.paddingOffset == 0));
                REQUIRE((image.size == (channels * height * width)));

                REQUIRE(at::allclose(out, expected));
            }

            TEST_CASE("test_types_image_initialization_padded", "[types][image][init][padding]")
            {
                constexpr size_t padding = 3;
                constexpr size_t blockSize = 16;
                constexpr size_t channels = 16;
                constexpr size_t height = 10;
                constexpr size_t width = 10;

                Tensor input = at::randn({channels, height, width});
                Image<float, padding, blockSize, channels, height, width> image(input.const_data_ptr<float>());
                Tensor out = at::from_blob(image.getPointer(), {channels / blockSize, height + 2 * padding, width + 2 * padding, blockSize});

                Tensor padded = at::pad(input, {padding, padding, padding, padding}, "constant", 0);
                Tensor expected = padded.view({channels / blockSize, blockSize, height + 2 * padding, width + 2 * padding}).permute({0, 2, 3, 1}).contiguous();

                // Evaluating with values that of constexpr require additional brackets
                REQUIRE((image.size == (channels * (height + 2 * padding) * (width + 2 * padding))));

                REQUIRE(at::allclose(out, expected));
            }

            TEST_CASE("test_types_image_initialization_batch_norm", "[types][image][batchNorm]")
            {
                constexpr size_t padding = 3;
                constexpr size_t blockSize = 16;
                constexpr size_t channels = 16;
                constexpr size_t height = 10;
                constexpr size_t width = 10;

                Tensor input = at::randn({channels, height, width});
                Image<float, padding, blockSize, channels, height, width> image(input.const_data_ptr<float>());
                Tensor outMean = at::from_blob(image.getMeanPointer(), {channels});
                Tensor outBatchVar = at::from_blob(image.getBatchVariancePointer(), {channels});

                Tensor expectedMean = at::mean(input, {1, 2});
                Tensor expectedBatchVar =  1 / at::sqrt(at::var(input, {1, 2}, false) + 1e-5);

                // Evaluating with values that of constexpr require additional brackets
                REQUIRE((image.size == (channels * (height + 2 * padding) * (width + 2 * padding))));

                REQUIRE(at::allclose(outMean, expectedMean));
                REQUIRE(at::allclose(outBatchVar, expectedBatchVar));
            }

            TEST_CASE("test_types_image_correct_padding", "[types][image][padding]")
            {
                constexpr size_t padding = 3;
                constexpr size_t blockSize = 16;
                constexpr size_t channels = 16;
                constexpr size_t height = 10;
                constexpr size_t width = 10;

                Tensor input = at::ones({channels, height, width});
                Image<float, padding, blockSize, channels, height, width> image(input.const_data_ptr<float>());
                auto imagePrt = image.getPointer();

                for (size_t i = 0; i < image.paddingOffset; i++)
                {
                    REQUIRE((imagePrt[i] == 0));
                }

                for (size_t iBChannel = 0; iBChannel < channels / blockSize; iBChannel++)
                {
                    for (size_t iHeight = 0; iHeight < height; iHeight++)
                    {
                        for (size_t iWidth = 0; iWidth < width; iWidth++)
                        {
                            for (size_t iChannel = 0; iChannel < blockSize; iChannel++)
                            {
                                size_t offset = image.getOffset(iBChannel, iHeight, iWidth, iChannel);
                                REQUIRE((imagePrt[image.paddingOffset + offset] == 1));
                            }
                        }
                    }
                }
            }

            TEST_CASE("test_types_image_init_flatten", "[types][image][init][flatten]")
            {
                constexpr size_t padding = 0;
                constexpr size_t blockSize = 16;
                constexpr size_t channels = 16;
                constexpr size_t height = 10;
                constexpr size_t width = 10;

                Tensor input = at::randn({channels, height, width});
                Image<float, padding, blockSize, channels, height, width> image(input.const_data_ptr<float>());
                auto flatten = image.flatten();
                Tensor out = at::from_blob(flatten.getPointer(), {channels * height * width});

                Tensor expected = input.view({channels * height * width});

                // Evaluating with values that of constexpr require additional brackets
                REQUIRE((image.paddingOffset == 0));
                REQUIRE((image.size == (channels * height * width)));

                REQUIRE(at::allclose(out, expected));
            }

            TEST_CASE("test_types_image_init_flatten_padded", "[types][image][init][flatten][padding]")
            {
                constexpr size_t padding = 3;
                constexpr size_t blockSize = 16;
                constexpr size_t channels = 16;
                constexpr size_t height = 10;
                constexpr size_t width = 10;

                Tensor input = at::randn({channels, height, width});
                Image<float, padding, blockSize, channels, height, width> image(input.const_data_ptr<float>());
                auto flatten = image.flatten(); // Padding should be removed during the flatten process
                Tensor out = at::from_blob(flatten.getPointer(), {channels * height * width});

                Tensor expected = input.view({channels * height * width});

                // Evaluating with values that of constexpr require additional brackets
                REQUIRE((image.size == (channels * (height + 2 * padding) * (width + 2 * padding))));
                REQUIRE((flatten.size == (channels * height * width)));

                REQUIRE(at::allclose(out, expected));
            }

            TEST_CASE("test_types_image_correct_batch_norm", "[types][image][batchNorm]")
            {
                constexpr size_t padding = 3;
                constexpr size_t blockSize = 16;
                constexpr size_t channels = 16;
                constexpr size_t height = 10;
                constexpr size_t width = 10;

                Tensor input = at::randn({channels, height, width});
                Image<float, padding, blockSize, channels, height, width> transfer(input.const_data_ptr<float>());
                Image<float, padding, blockSize, channels, height, width> image;

                auto transferPrt = transfer.getPointer() + transfer.paddingOffset;
                auto imagePrt = image.getPointer() + image.paddingOffset;

                for (size_t iBChannel = 0; iBChannel < channels / blockSize; iBChannel++)
                {
                    for (size_t iHeight = 0; iHeight < height; iHeight++)
                    {
                        for (size_t iWidth = 0; iWidth < width; iWidth++)
                        {
                            for (size_t iChannel = 0; iChannel < blockSize; iChannel++)
                            {
                                size_t offset = image.getOffset(iBChannel, iHeight, iWidth, iChannel);
                                size_t transferOffset = transfer.getOffset(iBChannel, iHeight, iWidth, iChannel);
                                imagePrt[offset] = transferPrt[transferOffset];
                            }
                        }
                    }
                }

                // Calculate mean and variance separately
                image.calculateMeanVariance();

                Tensor outMean = at::from_blob(image.getMeanPointer(), {channels});
                Tensor outBatchVar = at::from_blob(image.getBatchVariancePointer(), {channels});

                Tensor expectedMean = at::mean(input, {1, 2});
                Tensor expectedBatchVar =  1 / at::sqrt(at::var(input, {1, 2}, false) + 1e-5);

                // Evaluating with values that of constexpr require additional brackets
                REQUIRE((image.size == (channels * (height + 2 * padding) * (width + 2 * padding))));

                REQUIRE(at::allclose(outMean, expectedMean, 1e-5, 1e-7));
                REQUIRE(at::allclose(outBatchVar, expectedBatchVar));
            }
        }
    }
}