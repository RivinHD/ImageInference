#ifndef USE_ATEN_LIB
#define USE_ATEN_LIB
#endif // !USE_ATEN_LIB
#include <ATen/ATen.h>
#include <torch/library.h>
#include <catch2/catch_test_macros.hpp>
#include "../../types/Matrix.h"

namespace ImageInference
{
    namespace test
    {
        namespace types
        {
            using at::Tensor;
            using ImageInference::types::Matrix;

            TEST_CASE("test_types_matrix_initialization", "[types][matrix]")
            {
                constexpr size_t rows = 16;
                constexpr size_t cols = 16;

                Tensor input = at::randn({cols, rows});
                Matrix<float, cols, rows> matrix1(input.const_data_ptr<float>());
                Matrix<float, cols, rows> matrix2(input.const_data_ptr<float>());
                Matrix<float, cols, rows> matrix3(input.const_data_ptr<float>());

                REQUIRE((matrix1.size == (rows * cols)));
            }
        }
    }
}