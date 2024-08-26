// SPDX-FileCopyrightText: © 2024 Vincent Gerlach
//
// SPDX-License-Identifier: MIT

#ifndef IMAGEINFERENCE_ARRAY_H
#define IMAGEINFERENCE_ARRAY_H

#include "Macros.h"
#include <stddef.h>
#include <algorithm>
#include <new>
#include <execution>
#include <iostream>

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

            return offset;
        }
    } // namespace types
} // namespace ImageInference

#endif // IMAGEINFERENCE_ARRAY_H
