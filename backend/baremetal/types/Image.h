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

#include <stddef.h>
#include "Array.h"

namespace ImageInference
{
    namespace types
    {
        template <typename T, size_t TChannels, size_t THeight, size_t TWidth>
        class Image
        {
        private:
            T *data;

        public:
            const size_t Channels = TChannels;
            const size_t Height = THeight;
            const size_t Width = TWidth;
            const T *raw_pointer = data;

            Image(T *data);
            ~Image();

            Array<T, TChannels * THeight * TWidth> flatten();
        };

        template <typename T, size_t TChannels, size_t THeight, size_t TWidth>
        inline Image<T, TChannels, THeight, TWidth>::Image(T *data)
            : data(data)
        {
        }

        template <typename T, size_t TChannels, size_t THeight, size_t TWidth>
        inline Array<T, TChannels * THeight * TWidth> Image<T, TChannels, THeight, TWidth>::flatten()
        {
            return Array<T, TChannels * THeight * TWidth>(this->data);
        }

        template <typename T, size_t TChannels, size_t THeight, size_t TWidth>
        inline Image<T, TChannels, THeight, TWidth>::~Image()
        {
        }
    } // namespace types
} // namespace ImageInference

#endif // IMAGEINFERENCE_IMAGE_H