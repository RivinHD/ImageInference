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

#ifndef IMAGEINFERENCE_IMODEL_H
#define IMAGEINFERENCE_IMODEL_H

#include <stddef.h>

/// @brief Interface for a model.
/// @tparam T The data type used by the model.
template <typename T>
class IModel
{
public:
    IModel();
    ~IModel();

    /// @brief
    /// @param input
    /// @return

    /// @brief Do a forward pass through the model without any evaluation for training.
    /// @param input The input data to the model.
    /// @return The output data from the model.
    virtual const T *inference(const T *input) = 0;
};

#endif // IMAGEINFERENCE_IMODEL_H