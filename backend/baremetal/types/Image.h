// SPDX-FileCopyrightText: © 2024 Vincent Gerlach
//
// SPDX-License-Identifier: MIT

#ifndef IMAGEINFERENCE_IMAGE_H
#define IMAGEINFERENCE_IMAGE_H

#include "Macros.h"
#include <stddef.h>
#include <cmath>
#include "Array.h"
#include <new>
#include <stdexcept>
#include <iostream>

namespace ImageInference
{
    namespace types
    {
        template <typename T, size_t TPadding, size_t TBlockSize, size_t TChannels, size_t THeight, size_t TWidth>
        class Image
        {
        private:
            T *data;

        public:
            static constexpr const size_t strideChannelBlock = (THeight + 2 * TPadding) * (TWidth + 2 * TPadding) * TBlockSize;
            static constexpr const size_t strideHeight = (TWidth + 2 * TPadding) * TBlockSize;
            static constexpr const size_t strideWidth = TBlockSize;
            static constexpr const size_t strideChannel = 1;
            static constexpr const size_t paddingOffset = TPadding * (TWidth + 2 * TPadding) * TBlockSize + TPadding * TBlockSize; // TPadding * strideHeight + TPadding * strideWidth;
            static constexpr const size_t size = TChannels * (THeight + 2 * TPadding) * (TWidth + 2 * TPadding);

            Image();

            Image(const T *data);

            ~Image();

            T *getPointer();

            size_t getOffset(size_t iBlockChannel, size_t iHeight, size_t iWidth, size_t iChannel);

            Array<T, TChannels * THeight * TWidth> flatten();
        };

        /// Creates an image from already blocked data in the format ChannelBlocks x Height x Width x ChannelElements.
        /// During this mapping the mean and batch_variance are calculated too.
        ///
        /// @tparam T The type of the Image.
        /// @tparam TBlockSize The size of the block that is used.
        /// @tparam TChannels The total number of channels used.
        /// @tparam THeight The dimensions height wise.
        /// @tparam TWidth The dimensions width wise.
        ///
        /// @param data The pointer to the data.
        template <typename T, size_t TPadding, size_t TBlockSize, size_t TChannels, size_t THeight, size_t TWidth>
        inline Image<T, TPadding, TBlockSize, TChannels, THeight, TWidth>::Image()
        {
            if constexpr (TChannels % TBlockSize != 0)
            {
                std::cerr << "Image (" << this << "):The number of channels is not a multiple of the block size. Channels: " << TChannels
                          << " BlockSize: " << TBlockSize << std::endl;
                throw std::runtime_error("Image: The number of channels should be a multiple of the block size!");
            }

            data = new (std::align_val_t(PAGE_CACHE_ALIGN(T, size))) T[size]{0};
        }

        /// Converts the input data in format Channel x Height x Width to the blocked format ChannelBlocks x Height x Width x ChannelElements.
        ///
        /// @tparam T The type of the Image.
        /// @tparam TPadding The padding that is used.
        /// @tparam TBlockSize The size of the block that is used.
        /// @tparam TChannels The total number of channels used.
        /// @tparam THeight The dimensions height wise.
        /// @tparam TWidth The dimensions width wise.
        ///
        /// @param input The data to be converted.
        template <typename T, size_t TPadding, size_t TBlockSize, size_t TChannels, size_t THeight, size_t TWidth>
        inline Image<T, TPadding, TBlockSize, TChannels, THeight, TWidth>::Image(const T *input)
        {
            if constexpr (TChannels % TBlockSize != 0)
            {
                std::cerr << "Image (" << this << "):The number of channels is not a multiple of the block size." << std::endl
                          << "Channels: " << TChannels << " BlockSize: " << TBlockSize << std::endl
                          << std::endl;
                throw std::runtime_error("Image: The number of channels should be a multiple of the block size!");
            }

            data = new (std::align_val_t(PAGE_CACHE_ALIGN(T, size))) T[size]{0};

            if (data == nullptr)
            {
                std::cerr << "Could not allocate memory for Image (" << this << ") on member 'data'" << std::endl;
                throw std::runtime_error("Could not allocate memory for Image on member 'data'");
            }

            constexpr size_t channelBlocks = TChannels / TBlockSize;

            constexpr size_t strideInputChannel = THeight * TWidth;
            constexpr size_t strideInputHeight = TWidth;
            constexpr size_t strideInputWidth = 1;

            auto dataPtr = data + paddingOffset;
#ifdef USE_OMP
#pragma omp parallel for collapse(2)
#endif // USE_OMP
            for (size_t iBChannel = 0; iBChannel < channelBlocks; iBChannel++)
            {
                for (size_t iHeight = 0; iHeight < THeight; iHeight++)
                {
                    for (size_t iChannel = 0; iChannel < TBlockSize; iChannel++)
                    {
                        for (size_t iWidth = 0; iWidth < TWidth; iWidth++)
                        {
                            T in = input[(iBChannel * TBlockSize + iChannel) * strideInputChannel + iHeight * strideInputHeight + iWidth * strideInputWidth];

                            size_t offset = getOffset(iBChannel, iHeight, iWidth, iChannel);
                            dataPtr[offset] = in;
                        }
                    }
                }
            }
        }

        /// Get the pointer of the data.
        /// The image is stored in a blocked Format.
        /// ChannelBlocks x Height x Width x ChannelElements
        ///
        /// @tparam T The type of the Image.
        /// @tparam TBlockSize The size of the block that is used.
        /// @tparam TChannels The total number of channels used.
        /// @tparam THeight The dimensions height wise.
        /// @tparam TWidth The dimensions width wise.
        /// @return The pointer to the data.
        template <typename T, size_t TPadding, size_t TBlockSize, size_t TChannels, size_t THeight, size_t TWidth>
        inline T *Image<T, TPadding, TBlockSize, TChannels, THeight, TWidth>::getPointer()
        {
            return data;
        }

        template <typename T, size_t TPadding, size_t TBlockSize, size_t TChannels, size_t THeight, size_t TWidth>
        inline size_t Image<T, TPadding, TBlockSize, TChannels, THeight, TWidth>::getOffset(size_t iBlockChannel, size_t iHeight, size_t iWidth, size_t iChannel)
        {
            size_t offset = iBlockChannel * strideChannelBlock +
                            iHeight * strideHeight +
                            iWidth * strideWidth +
                            iChannel * strideChannel;

#ifdef IMAGEINFERENCE_TESTING
            if (offset >= size)
            {
                std::cerr << "Image (" << this << "): Offset is out of bounds: " << offset << " >= " << size << std::endl
                          << "Indices: ChannelBlock:= " << iBlockChannel << " Height:= " << iHeight
                          << " Width:= " << iWidth << " Channel:= " << iChannel << std::endl
                          << "Strides: ChannelBlock:= " << strideChannelBlock << " Height:= " << strideHeight
                          << " Width:= " << strideWidth << " Channel:= " << strideChannel << std::endl
                          << "Sizes: ChannelBlock:= " << TChannels / TBlockSize << " Height:= " << THeight
                          << " Width:= " << TWidth << " Channel:= " << TBlockSize << std::endl
                          << std::endl;

                throw std::runtime_error("Image: Offset is out of bounds!");
            }
#endif // IMAGEINFERENCE_TESTING

            return offset;
        }

        template <typename T, size_t TPadding, size_t TBlockSize, size_t TChannels, size_t THeight, size_t TWidth>
        inline Array<T, TChannels * THeight * TWidth> Image<T, TPadding, TBlockSize, TChannels, THeight, TWidth>::flatten()
        {
            constexpr size_t channelBlocks = TChannels / TBlockSize;

            constexpr size_t strideOutputChannel = THeight * TWidth;
            constexpr size_t strideOutputHeight = TWidth;
            constexpr size_t strideOutputWidth = 1;

            auto output = Array<T, TChannels * THeight * TWidth>();
            auto outputPrt = output.getPointer();

            auto dataPtr = data + paddingOffset;
#ifdef USE_OMP
#pragma omp parallel for collapse(2)
#endif // USE_OMP
            for (size_t iBChannel = 0; iBChannel < channelBlocks; iBChannel++)
            {
                for (size_t iHeight = 0; iHeight < THeight; iHeight++)
                {
                    for (size_t iChannel = 0; iChannel < TBlockSize; iChannel++)
                    {
                        for (size_t iWidth = 0; iWidth < TWidth; iWidth++)
                        {
                            size_t offsetOutput = (iBChannel * TBlockSize + iChannel) * strideOutputChannel +
                                                  iHeight * strideOutputHeight +
                                                  iWidth * strideOutputWidth;

                            size_t offsetData = getOffset(iBChannel, iHeight, iWidth, iChannel);

                            outputPrt[offsetOutput] = dataPtr[offsetData];
                        }
                    }
                }
            }

            return output;
        }

        template <typename T, size_t TPadding, size_t TBlockSize, size_t TChannels, size_t THeight, size_t TWidth>
        inline Image<T, TPadding, TBlockSize, TChannels, THeight, TWidth>::~Image()
        {
            operator delete[](data, std::align_val_t(PAGE_CACHE_ALIGN(T, size)));
        }
    } // namespace types
} // namespace ImageInference

#endif // IMAGEINFERENCE_IMAGE_H