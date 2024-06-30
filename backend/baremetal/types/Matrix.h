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

#include <stddef.h>

namespace ImageInference
{
    namespace types
    {
        template <typename T, size_t TRows, size_t TColumns>
        class Matrix
        {
        private:
            T *data;

        public:
            const size_t Rows = TRows;
            const size_t Columns = TColumns;
            const T *raw_pointer = data;

            Matrix(T *data);
            ~Matrix();
        };

        template <typename T, size_t TRows, size_t TColumns>
        inline Matrix<T, TRows, TColumns>::Matrix(T *data)
            : data(data)
        {
        }

        template <typename T, size_t TRows, size_t TColumns>
        inline Matrix<T, TRows, TColumns>::~Matrix()
        {
        }
    } // namespace types
} // namespace ImageInference

#endif // IMAGEINFERENCE_MATRIX_H
