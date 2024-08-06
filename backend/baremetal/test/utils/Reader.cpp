#include "Reader.h"
#include <numeric>

ImageInference::test::utils::Reader::Reader(std::string filepath)
{
    fileStream = std::ifstream(filepath, std::ios::binary);
    fileStream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
}

ImageInference::test::utils::Reader::~Reader()
{
    fileStream.close();

    for (auto array : readFloatData)
    {
        delete[] array;
    }
}

bool ImageInference::test::utils::Reader::hasNext()
{
    return fileStream.peek() != EOF;
}

float *ImageInference::test::utils::Reader::getNextTensor(std::vector<int64_t> &outSizes)
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

    outSizes.clear();
    outSizes.resize(countSizes);
    fileStream.read(reinterpret_cast<char *>(outSizes.data()), countSizes * sizeof(int64_t));

    int64_t size = std::accumulate(outSizes.begin(), outSizes.end(), 1, std::multiplies<int64_t>());
    float *tensor = new float[size];
    fileStream.read(reinterpret_cast<char *>(tensor), size * sizeof(float));
    readFloatData.push_back(tensor);

    return tensor;
}