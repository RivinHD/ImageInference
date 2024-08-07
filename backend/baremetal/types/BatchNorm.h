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
#include "Macros.h"

namespace ImageInference
{
    namespace types
    {
        template <typename T, size_t TChannels>
        class BatchNorm
        {
        private:
            const T *beta;
            const T *mean;

            /// @brief Combination of gamma and variance.
            T *gammaVariance;

        public:
            /// @brief Initialize a batch normalization container.
            /// @param gamma The value gamma that is used in batch normalization.
            /// @param beta The value beta that is used in batch normalization.
            BatchNorm(const void *gamma, const void *beta, const void *mean, const void *variance);
            ~BatchNorm();

            const T *getGammaVariancePointer();
            const T *getBetaPointer();
            const T *getMeanPointer();
        };

        template <typename T, size_t TChannels>
        inline BatchNorm<T, TChannels>::BatchNorm(const void *gamma, const void *beta, const void *mean, const void *variance)
            : beta(static_cast<const T *>(beta)), mean(static_cast<const T *>(mean))
        {
            const T *inVariance = static_cast<const T *>(variance);
            const T *inGamma = static_cast<const T *>(gamma);

            gammaVariance = new (std::align_val_t(PAGE_CACHE_ALIGN(T, TChannels))) T[TChannels];

            constexpr const size_t iterBlockSize = 64;
            constexpr const size_t iterBlocks = TChannels / iterBlockSize;
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
                    gammaVariance[i + j] = inGamma[i + j] / std::sqrt(inVariance[i + j] + 1e-5);
                }
            }

#ifdef USE_OMP
#pragma omp simd
#endif // USE_OMP
            for (size_t i = processableBlocks; i < TChannels; i++)
            {
                gammaVariance[i] = inGamma[i] / std::sqrt(inVariance[i] + 1e-5);
            }
        }

        template <typename T, size_t TChannels>
        inline BatchNorm<T, TChannels>::~BatchNorm()
        {
            operator delete[](gammaVariance, std::align_val_t(PAGE_CACHE_ALIGN(T, TChannels)));
        }

        template <typename T, size_t TChannels>
        inline const T *BatchNorm<T, TChannels>::getGammaVariancePointer()
        {
            return gammaVariance;
        }

        template <typename T, size_t TChannels>
        inline const T *BatchNorm<T, TChannels>::getBetaPointer()
        {
            return beta;
        }

        template <typename T, size_t TChannels>
        inline const T *BatchNorm<T, TChannels>::getMeanPointer()
        {
            return mean;
        }
    } // namespace types
} // namespace ImageInference

#endif // IMAGEINFERENCE_BATCH_NORM_H