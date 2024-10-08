// SPDX-FileCopyrightText: © 2024 Vincent Gerlach
//
// SPDX-License-Identifier: MIT

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

namespace ImageInference
{
    namespace test
    {
        namespace utils
        {
            class Reader
            {
            private:
                std::ifstream fileStream;
                std::vector<float *> readFloatData;
                inline static const std::string HEADER_TENSOR = "Tensor";

            public:
                /// Reads in binary tensor in the format of:
                /// Tensor<countSizes><sizes><data>
                /// Tensor is a raw ascii text, which indicates that a new Tensor starts
                /// <countSizes> is in binary int64 and indicates the number of elements in the <sizes>
                /// <sizes> is in binary int64 and indicates the size of the tensor
                /// <data> is in binary float32 and contains the data of the tensor
                ///
                /// @param filepath The filepath to the binary file.
                Reader(std::string filepath);

                ~Reader();

                bool hasNext();
                float *getNextTensor(std::vector<int64_t> &outSizes);
            };
        }
    }
}
