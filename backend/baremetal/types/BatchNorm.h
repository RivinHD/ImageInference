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

#ifndef IMAGEINFERENCE_BATCH_NORM_H
#define IMAGEINFERENCE_BATCH_NORM_H

#include <stddef.h>

namespace ImageInference
{
    namespace types
    {
        template <typename T, size_t TChannels>
        class BatchNorm
        {
        private:
            const T *gamma;
            const T *beta;

        public:
            /// @brief Initialize a batch normalization container.
            /// @param gamma The value gamma that is used in batch normalization.
            /// @param beta The value beta that is used in batch normalization.
            BatchNorm(const void *gamma, const void *beta);
            ~BatchNorm();

            const T *getGammaPointer();
            const T *getBetaPointer();
        };

        template <typename T, size_t TChannels>
        inline BatchNorm<T, TChannels>::BatchNorm(const void *gamma, const void *beta)
            : gamma(static_cast<const T *>(gamma)), beta(static_cast<const T *>(beta))
        {
        }

        template <typename T, size_t TChannels>
        inline BatchNorm<T, TChannels>::~BatchNorm()
        {
        }

        template <typename T, size_t TChannels>
        inline const T *BatchNorm<T, TChannels>::getGammaPointer()
        {
            return gamma;
        }

        template <typename T, size_t TChannels>
        inline const T *BatchNorm<T, TChannels>::getBetaPointer()
        {
            return beta;
        }
    } // namespace types
} // namespace ImageInference

#endif // IMAGEINFERENCE_BATCH_NORM_H