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

namespace ImageInference
{
    namespace types
    {
        template <typename T, size_t TSize>
        class Array
        {
        private:
            // https://github.com/Mozilla-Ocho/llamafile/blob/05493179b70429ff32fb55942ee7bdb76367cfba/llamafile/tinyblas_cpu.h#L53
            // Following the alingment of llamafile i.e Row alignment is 64 bytes.
            CACHE_ALIGN(sizeof(T), TSize) T data[TSize]{0};

        public:
            static constexpr const size_t size = TSize;

            Array();
            Array(const T* input);
            ~Array();

            T* getPointer();
        };

        template <typename T, size_t TSize>
        inline Array<T, TSize>::Array()
        {
        }

        template <typename T, size_t TSize>
        inline Array<T, TSize>::Array(const T *input)
        {
            std::copy(input, input + TSize, data);
        }

        template <typename T, size_t TSize>
        inline Array<T, TSize>::~Array()
        {
        }

        template <typename T, size_t TSize>
        inline T *Array<T, TSize>::getPointer()
        {
            return data;
        }
    } // namespace types
} // namespace ImageInference

#endif // IMAGEINFERENCE_ARRAY_H
