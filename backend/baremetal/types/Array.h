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

#ifndef IMAGEINFERENCE_ARRAY_H
#define IMAGEINFERENCE_ARRAY_H

#include "Macros.h"
#include <stddef.h>
#include <algorithm>
#include <new>
#include <execution>

namespace ImageInference
{
    namespace types
    {
        template <typename T, size_t TSize>
        class Array
        {
        private:
            T *data;

        public:
            static constexpr const size_t size = TSize;

            Array();
            Array(const T *input);
            ~Array();

            T *getPointer();
            size_t getOffset(size_t iRow);
        };

        template <typename T, size_t TSize>
        inline Array<T, TSize>::Array()
        {
            data = new (std::align_val_t(PAGE_CACHE_ALIGN(T, TSize))) T[TSize];
            if (data == nullptr)
            {
                std::cerr << "Could not allocate memory for Array (" << this << ")" << std::endl;
                throw std::runtime_error("Could not allocate memory for Array");
            }
            
        }

        template <typename T, size_t TSize>
        inline Array<T, TSize>::Array(const T *input)
        {
            data = new (std::align_val_t(PAGE_CACHE_ALIGN(T, TSize))) T[TSize];
            constexpr const size_t iterBlockSize = 64;
            constexpr const size_t iterBlocks = TSize / iterBlockSize;
            constexpr const size_t processableBlocks = iterBlockSize * iterBlocks;

#ifdef USE_OMP
#pragma omp parallel for
#endif // USE_OMP
            for (size_t i = 0; i < processableBlocks; i += iterBlockSize)
            {
#ifdef USE_OMP
#pragma omp simd
#endif // USE_OMP
                for (size_t j = 0; j < iterBlockSize; j++)
                {
                    data[i + j] = input[i + j];
                }
            }

#ifdef USE_OMP
#pragma omp simd
#endif // USE_OMP
            for (size_t i = processableBlocks; i < TSize; i++)
            {
                data[i] = input[i];
            }
        }

        template <typename T, size_t TSize>
        inline Array<T, TSize>::~Array()
        {
            operator delete[](data, std::align_val_t(PAGE_CACHE_ALIGN(T, TSize)));
        }

        template <typename T, size_t TSize>
        inline T *Array<T, TSize>::getPointer()
        {
            return data;
        }

        template <typename T, size_t TSize>
        inline size_t Array<T, TSize>::getOffset(size_t iRow)
        {
            size_t offset = iRow;

#ifdef IMAGEINFERENCE_TESTING
            if (offset >= TSize)
            {
                std::cerr << "Array (" << this << "): Offset is out of bounds: " << offset << " >= " << TSize << std::endl
                          << "Indices: Row:= " << iRow << std::endl
                          << "Stride: Row:= " << 1 << std::endl
                          << std::endl;

                throw std::runtime_error(" Array: Offset is out of bounds.");
            }
#endif // IMAGEINFERENCE_TESTING

            return iRow;
        }
    } // namespace types
} // namespace ImageInference

#endif // IMAGEINFERENCE_ARRAY_H
