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

#ifndef IMAGEINFERENCE_MATRIX_H
#define IMAGEINFERENCE_MATRIX_H

#ifdef IMAGEINFERENCE_TESTING
#include <iostream>
#endif // IMAGEINFERENCE_TESTING

#include "Macros.h"
#include <stddef.h>
#include <algorithm>

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
            // https://github.com/Mozilla-Ocho/llamafile/blob/05493179b70429ff32fb55942ee7bdb76367cfba/llamafile/tinyblas_cpu.h#L54
            // Following the alignment of llamafile i.e Matrix alignment is 4096 bytes.
            PAGE_CACHE_ALIGN(sizeof(T), TColumns *TRows)
            T data[TColumns * TRows] { 0 };

#ifdef IMAGEINFERENCE_TESTING
            T overflow_test[TColumns * TRows]{0};
#endif // IMAGEINFERENCE_TESTING

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
        }

        template <typename T, size_t TColumns, size_t TRows>
        inline Matrix<T, TColumns, TRows>::Matrix(const T *input)
        {
            std::copy(input, input + TColumns * TRows, data);
        }

        template <typename T, size_t TColumns, size_t TRows>
        inline Matrix<T, TColumns, TRows>::~Matrix()
        {
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
                std::cerr << "Matrix (" << this << "):Offset is out of bounds: " << offset << " >= " << size << std::endl
                          << "Indices:= Column" << iColumn << " Row:= " << iRow << std::endl
                          << "Stride:= Column" << strideColumn << " Row:= " << strideRow << std::endl
                          << std::endl;
            }
#endif // IMAGEINFERENCE_TESTING

            return iColumn * strideColumn + iRow * strideRow;
        }
    } // namespace types
} // namespace ImageInference

#endif // IMAGEINFERENCE_MATRIX_H
