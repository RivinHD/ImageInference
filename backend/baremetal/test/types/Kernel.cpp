#ifndef USE_ATEN_LIB
#define USE_ATEN_LIB
#endif // !USE_ATEN_LIB
#include <ATen/ATen.h>
#include <torch/library.h>
#include <catch2/catch_test_macros.hpp>
#include "../../types/Kernel.h"

namespace ImageInference
{
    namespace test
    {
        namespace types
        {
            using at::Tensor;
            using ImageInference::types::Kernel;

            TEST_CASE("test_types_kernel_initialization_16x16", "[types][kernel]")
            {
                constexpr size_t blockSize = 16;
                constexpr size_t inChannels = 16;
                constexpr size_t outChannels = 16;
                constexpr size_t height = 3;
                constexpr size_t width = 3;

                Tensor input = at::randn({outChannels, inChannels, height, width});
                Kernel<float, blockSize, blockSize, outChannels, inChannels, height, width> kernel(input.const_data_ptr<float>());
                Tensor out = at::from_blob(kernel.getPointer(), {outChannels / blockSize, inChannels / blockSize, height, width, blockSize, blockSize});

                Tensor expected = input.view({outChannels / blockSize, blockSize, inChannels / blockSize, blockSize, height, width})
                                      .permute({0, 2, 4, 5, 3, 1})
                                      .contiguous();

                // Evaluating with values that of constexpr require additional brackets
                REQUIRE((kernel.size == (inChannels * outChannels * height * width)));

                REQUIRE(at::allclose(out, expected));
            }

            TEST_CASE("test_types_kernel_initialization_128x64", "[types][kernel]")
            {
                constexpr size_t blockSize = 16;
                constexpr size_t inChannels = 128;
                constexpr size_t outChannels = 64;
                constexpr size_t height = 3;
                constexpr size_t width = 3;

                Tensor input = at::randn({outChannels, inChannels, height, width});
                Kernel<float, blockSize, blockSize, outChannels, inChannels, height, width> kernel(input.const_data_ptr<float>());
                Tensor out = at::from_blob(kernel.getPointer(), {outChannels / blockSize, inChannels / blockSize, height, width, blockSize, blockSize});

                Tensor expected = input.view({outChannels / blockSize, blockSize, inChannels / blockSize, blockSize, height, width})
                                      .permute({0, 2, 4, 5, 3, 1})
                                      .contiguous();

                // Evaluating with values that of constexpr require additional brackets

                REQUIRE((kernel.size == (inChannels * outChannels * height * width)));
                auto outSizes = out.sizes();
                auto expectedSizes = expected.sizes();

                REQUIRE(at::allclose(out, expected));
            }

            TEST_CASE("test_types_kernel_initialization_128x64_blockSize32x16", "[types][kernel]")
            {
                constexpr size_t inblockSize = 32;
                constexpr size_t outBlockSize = 16;
                constexpr size_t inChannels = 128;
                constexpr size_t outChannels = 64;
                constexpr size_t height = 3;
                constexpr size_t width = 3;

                Tensor input = at::randn({outChannels, inChannels, height, width});
                Kernel<float, outBlockSize, inblockSize, outChannels, inChannels, height, width> kernel(input.const_data_ptr<float>());
                Tensor out = at::from_blob(kernel.getPointer(), {outChannels / outBlockSize, inChannels / inblockSize, height, width, inblockSize, outBlockSize});

                Tensor expected = input.view({outChannels / outBlockSize, outBlockSize, inChannels / inblockSize, inblockSize, height, width})
                                      .permute({0, 2, 4, 5, 3, 1})
                                      .contiguous();

                // Evaluating with values that of constexpr require additional brackets

                REQUIRE((kernel.size == (inChannels * outChannels * height * width)));

                REQUIRE(at::allclose(out, expected));
            }
        }
    }
}