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

#ifndef IMAGEINFERENCE_KERNEL_H
#define IMAGEINFERENCE_KERNEL_H

#include "Macros.h"
#include <stddef.h>
#include <stdexcept>
#include <iostream>

namespace ImageInference
{
    namespace types
    {
        template <typename T, size_t TBlockSizeCount, size_t TBlockSizeChannel, size_t TCount, size_t TChannels, size_t THeight, size_t TWidth>
        class Kernel
        {
        private:
             T* data;

        public:
            static constexpr const size_t strideCountBlock = TChannels * THeight * TWidth * TBlockSizeCount;
            static constexpr const size_t strideChannelBlock = THeight * TWidth * TBlockSizeCount * TBlockSizeChannel;
            static constexpr const size_t strideHeight = TWidth * TBlockSizeCount * TBlockSizeChannel;
            static constexpr const size_t strideWidth = TBlockSizeCount * TBlockSizeChannel;
            static constexpr const size_t strideChannel = TBlockSizeCount;
            static constexpr const size_t strideCount = 1;
            static constexpr const size_t size = TCount * TChannels * THeight * TWidth;

            Kernel(const T *input);
            ~Kernel();

            T *getPointer();
            size_t getOffset(size_t iBlockCount, size_t iBlockChannel, size_t iHeight, size_t iWidth, size_t iChannel, size_t iCount);
        };

        /// Converts the input data in format TCount x Channel x Height x Width
        /// to the blocked format CountBlocks x ChannelBlocks x Height x Width x ChannelElements x CountElements.
        ///
        /// @tparam T The type of the Kernel.
        /// @tparam TBlockSizeCount The size of the block that is used for the Count dimension.
        /// @tparam TBlockSizeChannel The size of the block that is used for the Channel dimension.
        /// @tparam TCount The total number of kernels i.e. the output channel dimension.
        /// @tparam TChannels The total number of channels used i.e. the input channel dimension.
        /// @tparam THeight The dimensions height wise.
        /// @tparam TWidth The dimensions width wise.
        /// @param input The input to convert.
        template <typename T, size_t TBlockSizeCount, size_t TBlockSizeChannel, size_t TCount, size_t TChannels, size_t THeight, size_t TWidth>
        inline Kernel<T, TBlockSizeCount, TBlockSizeChannel, TCount, TChannels, THeight, TWidth>::Kernel(const T *input)
        {
            if constexpr (TCount % TBlockSizeCount != 0)
            {
                std::cerr << "Kernel (" << this << "):The number of count is not a multiple of the count block size." << std::endl
                          << "Channels: " << TChannels << " BlockSize: " << TBlockSizeCount << std::endl
                          << std::endl;
                throw std::runtime_error("Image: The number of count should be a multiple of the count block size!");
            }

            if constexpr (TChannels % TBlockSizeChannel != 0)
            {
                std::cerr << "Kernel (" << this << "):The number of channels is not a multiple of the channel block size." << std::endl
                          << "Channels: " << TChannels << " BlockSize: " << TBlockSizeChannel << std::endl
                          << std::endl;
                throw std::runtime_error("Image: The number of channels should be a multiple of the channel block size!");
            }

            data = new (std::align_val_t(PAGE_CACHE_ALIGN(T, size))) T[size]{0};

            if (data == nullptr)
            {
                std::cerr << "Could not allocate memory for Kernel (" << this << ")" << std::endl;
                throw std::runtime_error("Could not allocate memory for Kernel");
            }

            constexpr size_t countBlocks = TCount / TBlockSizeCount;
            constexpr size_t channelBlocks = TChannels / TBlockSizeChannel;

            constexpr size_t strideInputCount = TChannels * THeight * TWidth;
            constexpr size_t strideInputChannel = THeight * TWidth;
            constexpr size_t strideInputHeight = TWidth;
            constexpr size_t strideInputWidth = 1;
#ifdef USE_OMP
#pragma omp parallel for collapse(3)
#endif
            for (size_t iBCount = 0; iBCount < countBlocks; iBCount++)
            {
                for (size_t iBChannel = 0; iBChannel < channelBlocks; iBChannel++)
                {
                    for (size_t iHeight = 0; iHeight < THeight; iHeight++)
                    {
                        for (size_t iWidth = 0; iWidth < TWidth; iWidth++)
                        {
                            for (size_t iChannel = 0; iChannel < TBlockSizeChannel; iChannel++)
                            {
                                for (size_t iCount = 0; iCount < TBlockSizeCount; iCount++)
                                {
                                    size_t iInput = (iBCount * TBlockSizeCount + iCount) * strideInputCount +
                                                    (iBChannel * TBlockSizeChannel + iChannel) * strideInputChannel +
                                                    iHeight * strideInputHeight +
                                                    iWidth * strideInputWidth;
                                    size_t iData = iBCount * strideCountBlock +
                                                   iBChannel * strideChannelBlock +
                                                   iHeight * strideHeight +
                                                   iWidth * strideWidth +
                                                   iChannel * strideChannel +
                                                   iCount * strideCount;
                                    data[iData] = input[iInput];
                                }
                            }
                        }
                    }
                }
            }
        }

        template <typename T, size_t TBlockSizeCount, size_t TBlockSizeChannel, size_t TCount, size_t TChannels, size_t THeight, size_t TWidth>
        inline Kernel<T, TBlockSizeCount, TBlockSizeChannel, TCount, TChannels, THeight, TWidth>::~Kernel()
        {
            operator delete[](data, std::align_val_t(PAGE_CACHE_ALIGN(T, size)));
        }

        /// Get the pointer of the data.
        /// The Kernel is in blocked format CountBlocks x ChannelBlocks x Height x Width x ChannelElements x CountElements.
        ///
        /// @tparam T The type of the Kernel.
        /// @tparam TCount The total number of kernels i.e. the output channel dimension.
        /// @tparam TChannels The total number of channels used i.e. the input channel dimension.
        /// @tparam THeight The dimensions height wise.
        /// @tparam TWidth The dimensions width wise.
        /// @tparam TBlockSizeCount The size of the block that is used for the Count dimension.
        /// @tparam TBlockSizeChannel The size of the block that is used for the Channel dimension.
        template <typename T, size_t TBlockSizeCount, size_t TBlockSizeChannel, size_t TCount, size_t TChannels, size_t THeight, size_t TWidth>
        inline T *Kernel<T, TBlockSizeCount, TBlockSizeChannel, TCount, TChannels, THeight, TWidth>::getPointer()
        {
            return data;
        }

        template <typename T, size_t TBlockSizeCount, size_t TBlockSizeChannel, size_t TCount, size_t TChannels, size_t THeight, size_t TWidth>
        inline size_t Kernel<T, TBlockSizeCount, TBlockSizeChannel, TCount, TChannels, THeight, TWidth>::getOffset(
            size_t iBlockCount, size_t iBlockChannel, size_t iHeight, size_t iWidth, size_t iChannel, size_t iCount)
        {
            size_t offset = iBlockCount * strideCountBlock +
                            iBlockChannel * strideChannelBlock +
                            iHeight * strideHeight +
                            iWidth * strideWidth +
                            iChannel * strideChannel +
                            iCount * strideCount;

#ifdef IMAGEINFERENCE_TESTING
            if (offset >= size)
            {
                std::cerr << "Kernel (" << this << "):Offset is out of bounds: " << offset << " >= " << size << std::endl
                          << "Indices: CountBlock:= " << iBlockCount << " ChannelBlock:= " << iBlockChannel
                          << " Height:= " << iHeight << " Width:= " << iWidth
                          << " Channel:= " << iChannel << " Count:= " << iCount << std::endl
                          << "Stride: CountBlock:= " << strideCountBlock << " ChannelBlock:= " << strideChannelBlock
                          << " Height:= " << strideHeight << " Width:= " << strideWidth
                          << " Channel:= " << strideChannel << " Count:= " << strideCount << std::endl
                          << "Sizes: CountBlock:= " << TCount / TBlockSizeCount << " ChannelBlock:= " << TChannels / TBlockSizeChannel
                          << " Height:= " << THeight << " Width:= " << TWidth
                          << " Channel:= " << TBlockSizeChannel << " Count:= " << TBlockSizeCount << std::endl
                          << std::endl;
                
                throw std::runtime_error("Kernel: Offset is out of bounds!");
            }
#endif // IMAGEINFERENCE_TESTING

            return offset;
        }
    } // namespace types
} // namespace ImageInference

#endif // IMAGEINFERENCE_KERNEL_H