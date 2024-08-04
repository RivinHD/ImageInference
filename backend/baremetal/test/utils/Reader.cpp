#include "Reader.h"

ImageInference::test::utils::Reader::Reader(std::string filepath)
{
    fileStream = std::ifstream(filepath, std::ios::binary);
    fileStream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
}

ImageInference::test::utils::Reader::~Reader()
{
    fileStream.close();
}

bool ImageInference::test::utils::Reader::hasNext()
{
    return fileStream.peek() != EOF;
}

at::Tensor ImageInference::test::utils::Reader::getNextTensor()
{
    if (!hasNext())
    {
        throw std::runtime_error("No more tensors to read. End of file reached.");
    }

    char header[7];
    fileStream.read(header, 6);
    header[6] = '\0';
    if (std::string(header) != HEADER_TENSOR)
    {
        std::cerr << "Invalid header. Expected ascii chars that represent 'Tensor' but got " << header << std::endl;
        throw std::runtime_error("Invalid header. Expected ascii chars that represent 'Tensor'.");
    }

    int64_t countSizes;
    fileStream.read(reinterpret_cast<char *>(&countSizes), sizeof(int64_t));

    std::vector<int64_t> sizes(countSizes);
    fileStream.read(reinterpret_cast<char *>(sizes.data()), countSizes * sizeof(int64_t));

    at::Tensor tensor = at::empty(sizes, at::ScalarType::Float);
    fileStream.read(reinterpret_cast<char *>(tensor.mutable_data_ptr()), tensor.numel() * sizeof(float));

    return tensor;
}