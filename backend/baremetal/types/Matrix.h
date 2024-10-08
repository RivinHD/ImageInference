// SPDX-FileCopyrightText: © 2024 Vincent Gerlach
//
// SPDX-License-Identifier: MIT

#ifndef IMAGEINFERENCE_MATRIX_H
#define IMAGEINFERENCE_MATRIX_H

#include "Macros.h"
#include <stddef.h>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace ImageInference
{
    namespace types
    {
        /// The Matrix is in RowMajor format therefore Row is the last template argument as it is the faster dimension.
        /// @tparam T The type of the Matrix.
        /// @tparam TColumns The number of columns in the Matrix.
        /// @tparam TRows The number of rows in the Matrix.
        template <typename T, size_t TColumns, size_t TRows>
        class Matrix
        {
        private:
            T *data;

        public:
            static constexpr const size_t strideColumn = TRows;
            static constexpr const size_t strideRow = 1;
            static constexpr const size_t size = TColumns * TRows;

            Matrix();
            Matrix(const T *input);
            ~Matrix();

            T *getPointer();
            size_t getOffset(size_t iColumn, size_t iRow);
        };

        template <typename T, size_t TColumns, size_t TRows>
        inline Matrix<T, TColumns, TRows>::Matrix()
        {
            data = new (std::align_val_t(PAGE_CACHE_ALIGN(T, size))) T[size]{0};
            if (data == nullptr)
            {
                std::cerr << "Could not allocate memory for Matrix (" << this << ")" << std::endl;
                throw std::runtime_error("Could not allocate memory for Matrix");
            }
            
        }

        template <typename T, size_t TColumns, size_t TRows>
        inline Matrix<T, TColumns, TRows>::Matrix(const T *input)
        {
            data = new (std::align_val_t(PAGE_CACHE_ALIGN(T, size))) T[size]{0};
            constexpr const size_t iterBlockSize = 64;
            constexpr const size_t iterBlocks = size / iterBlockSize;
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
            for (size_t i = processableBlocks; i < size; i++)
            {
                data[i] = input[i];
            }
        }

        template <typename T, size_t TColumns, size_t TRows>
        inline Matrix<T, TColumns, TRows>::~Matrix()
        {
            operator delete[](data, std::align_val_t(PAGE_CACHE_ALIGN(T, size)));
        }

        template <typename T, size_t TColumns, size_t TRows>
        inline T *Matrix<T, TColumns, TRows>::getPointer()
        {
            return data;
        }

        template <typename T, size_t TColumns, size_t TRows>
        inline size_t Matrix<T, TColumns, TRows>::getOffset(size_t iColumn, size_t iRow)
        {
            size_t offset = iColumn * strideColumn + iRow * strideRow;

#ifdef IMAGEINFERENCE_TESTING
            if (offset >= size)
            {
                std::cerr << "Matrix (" << this << "): Offset is out of bounds: " << offset << " >= " << size << std::endl
                          << "Indices:= Column" << iColumn << " Row:= " << iRow << std::endl
                          << "Stride:= Column" << strideColumn << " Row:= " << strideRow << std::endl
                          << std::endl;

                throw std::runtime_error("Matrix: Offset is out of bounds.");
            }
#endif // IMAGEINFERENCE_TESTING

            return offset;
        }
    } // namespace types
} // namespace ImageInference

#endif // IMAGEINFERENCE_MATRIX_H
