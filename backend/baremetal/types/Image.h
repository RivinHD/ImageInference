//  Copyright (c) 2024 by Vincent Gerlach. All rights reserved.
//
//  SPDX-License-Identifier: GPL-3.0-or-later
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  in the root folder of this project with the name LICENSE. If not, see <http://www.gnu.org/licenses/>.

#ifndef IMAGEINFERENCE_IMAGE_H
#define IMAGEINFERENCE_IMAGE_H

#ifdef IMAGEINFERENCE_TESTING
#include <iostream>
#endif // IMAGEINFERENCE_TESTING

#include "Macros.h"
#include <stddef.h>
#include <cmath>
#include "Array.h"

namespace ImageInference
{
    namespace types
    {
        template <typename T, size_t TPadding, size_t TBlockSize, size_t TChannels, size_t THeight, size_t TWidth>
        class Image
        {
        private:
            // https://github.com/Mozilla-Ocho/llamafile/blob/05493179b70429ff32fb55942ee7bdb76367cfba/llamafile/tinyblas_cpu.h#L54
            // Following the alignment of llamafile i.e Matrix alignment is 4096 bytes.
            PAGE_CACHE_ALIGN(sizeof(T), TChannels *(THeight + 2 * TPadding) * (TWidth + 2 * TPadding))
            T data[TChannels * (THeight + 2 * TPadding) * (TWidth + 2 * TPadding)]{0};

            PAGE_CACHE_ALIGN(sizeof(T), TChannels)
            T mean[TChannels]{0};

            PAGE_CACHE_ALIGN(sizeof(T), TChannels)
            T batch_variance[TChannels]{0};

            bool isMeanVarianceCalculated = false;

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

            T *getMeanPointer();

            T *getBatchVariancePointer();

            bool getIsMeanVarianceCalculated();

            // based on Welford's online algorithm
            void updateMeanVariance(T value, size_t iChannel, size_t count);

            void finalizeMeanVariance(size_t iChannel, size_t count);

            size_t getOffset(size_t iBlockChannel, size_t iHeight, size_t iWidth, size_t iChannel);

            Array<T, TChannels * THeight * TWidth> flatten();

            void calculateMeanVariance();
        };

        template <typename T, size_t TPadding, size_t TBlockSize, size_t TChannels, size_t THeight, size_t TWidth>
        inline void Image<T, TPadding, TBlockSize, TChannels, THeight, TWidth>::updateMeanVariance(T value, size_t iChannel, size_t count)
        {
#ifdef IMAGEINFERENCE_TESTING
            if (iChannel >= TChannels)
            {
                std::cerr << "Image (" << this << "): Channel index is out of bounds: " << iChannel << " >= " << TChannels << std::endl
                          << std::endl;
            }
#endif // IMAGEINFERENCE_TESTING

            T delta = value - mean[iChannel];
            mean[iChannel] += delta / count;
            batch_variance[iChannel] += delta * (value - mean[iChannel]);
        }

        template <typename T, size_t TPadding, size_t TBlockSize, size_t TChannels, size_t THeight, size_t TWidth>
        inline void Image<T, TPadding, TBlockSize, TChannels, THeight, TWidth>::finalizeMeanVariance(size_t iChannel, size_t count)
        {
#ifdef IMAGEINFERENCE_TESTING
            if (iChannel >= TChannels)
            {
                std::cerr << "Image (" << this << "): Channel index is out of bounds: " << iChannel << " >= " << TChannels << std::endl
                          << std::endl;
            }
#endif // IMAGEINFERENCE_TESTING

            batch_variance[iChannel] /= count;
            batch_variance[iChannel] = static_cast<T>(1 / std::sqrt(batch_variance[iChannel] + 1e-05));
            isMeanVarianceCalculated = true;
        }

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

            constexpr size_t channelBlocks = TChannels / TBlockSize;

            constexpr size_t strideInputChannel = THeight * TWidth;
            constexpr size_t strideInputHeight = TWidth;
            constexpr size_t strideInputWidth = 1;

            auto dataPtr = data + paddingOffset;
            size_t count[TChannels]{0};
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
                            T in = input[iChannel * strideInputChannel + iHeight * strideInputHeight + iWidth * strideInputWidth];
                            size_t offset = getOffset(iBChannel, iHeight, iWidth, iChannel);
                            dataPtr[offset] = in;

                            updateMeanVariance(in,
                                               iBChannel * TBlockSize + iChannel,
                                               ++count[iBChannel * TBlockSize + iChannel]);
                        }
                    }
                }
            }
#ifdef USE_OMP
#pragma omp parallel for
#endif // USE_OMP
            for (size_t iBChannel = 0; iBChannel < channelBlocks; iBChannel++)
            {
                for (size_t iChannel = 0; iChannel < TBlockSize; iChannel++)
                {
                    finalizeMeanVariance(iBChannel * TBlockSize + iChannel, count[iBChannel * TBlockSize + iChannel]);
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

        /// Get the pointer of the mean.
        /// The mean is stored in a blocked Format.
        /// ChannelBlocks x ChannelElements
        ///
        /// @tparam T The type of the Image.
        /// @tparam TBlockSize The size of the block that is used.
        /// @tparam TChannels The total number of channels used.
        /// @tparam THeight The dimensions height wise.
        /// @tparam TWidth The dimensions width wise.
        /// @return The pointer to the mean.
        template <typename T, size_t TPadding, size_t TBlockSize, size_t TChannels, size_t THeight, size_t TWidth>
        inline T *Image<T, TPadding, TBlockSize, TChannels, THeight, TWidth>::getMeanPointer()
        {
            return mean;
        }

        /// Get the pointer of the batch variance.
        /// The mean is stored in a blocked Format.
        /// ChannelBlocks x ChannelElements.
        /// The batch variance is calculate with 1 / sqrt(variance + epsilon).
        ///
        /// @tparam T The type of the Image.
        /// @tparam TBlockSize The size of the block that is used.
        /// @tparam TChannels The total number of channels used.
        /// @tparam THeight The dimensions height wise.
        /// @tparam TWidth The dimensions width wise.
        /// @return The pointer to the batch variance.
        template <typename T, size_t TPadding, size_t TBlockSize, size_t TChannels, size_t THeight, size_t TWidth>
        inline T *Image<T, TPadding, TBlockSize, TChannels, THeight, TWidth>::getBatchVariancePointer()
        {
            return batch_variance;
        }

        template <typename T, size_t TPadding, size_t TBlockSize, size_t TChannels, size_t THeight, size_t TWidth>
        inline bool Image<T, TPadding, TBlockSize, TChannels, THeight, TWidth>::getIsMeanVarianceCalculated()
        {
            return isMeanVarianceCalculated;
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
        inline void Image<T, TPadding, TBlockSize, TChannels, THeight, TWidth>::calculateMeanVariance()
        {
            if (isMeanVarianceCalculated)
            {
                return;
            }

            constexpr size_t channelBlocks = TChannels / TBlockSize;

            T *dataPtr = data + paddingOffset;
            size_t count[TChannels]{0};

            for (size_t iBChannel = 0; iBChannel < channelBlocks; iBChannel++)
            {
                for (size_t iHeight = 0; iHeight < THeight; iHeight++)
                {
                    for (size_t iWidth = 0; iWidth < TWidth; iWidth++)
                    {
                        for (size_t iChannel = 0; iChannel < TBlockSize; iChannel++)
                        {
                            size_t offset = getOffset(iBChannel, iHeight, iWidth, iChannel);

                            updateMeanVariance(dataPtr[offset],
                                               iBChannel * TBlockSize + iChannel,
                                               ++count[iBChannel * TBlockSize + iChannel]);
                        }
                    }
                }
            }

            for (size_t iBChannel = 0; iBChannel < channelBlocks; iBChannel++)
            {
                for (size_t iChannel = 0; iChannel < TBlockSize; iChannel++)
                {
                    finalizeMeanVariance(iBChannel * TBlockSize + iChannel, count[iBChannel * TBlockSize + iChannel]);
                }
            }
        }

        template <typename T, size_t TPadding, size_t TBlockSize, size_t TChannels, size_t THeight, size_t TWidth>
        inline Image<T, TPadding, TBlockSize, TChannels, THeight, TWidth>::~Image()
        {
        }
    } // namespace types
} // namespace ImageInference

#endif // IMAGEINFERENCE_IMAGE_H