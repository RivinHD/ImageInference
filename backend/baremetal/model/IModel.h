// SPDX-FileCopyrightText: © 2024 Vincent Gerlach
//
// SPDX-License-Identifier: MIT

#ifndef IMAGEINFERENCE_IMODEL_H
#define IMAGEINFERENCE_IMODEL_H

#include <stddef.h>

namespace ImageInference
{
    namespace model
    {
        /// @brief Interface for a model.
        /// @tparam T The data type used by the model.
        template <typename T>
        class IModel
        {
        public:
            virtual ~IModel() {}

            /// @brief
            /// @param input
            /// @return

            /// @brief Do a forward pass through the model without any evaluation for training.
            /// @param input The input data to the model.
            /// @return The output data from the model.
            virtual void inference(const T *input, T* output) = 0;
        };
    } // namespace model
} // namespace ImageInference
#endif // IMAGEINFERENCE_IMODEL_H