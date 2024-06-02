/*
 *  Copyright (c) 2024 by Vincent Gerlach. All rights reserved.
 *
 *  SPDX-License-Identifier: GPL-3.0-or-later
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  in the root folder of this project with the name LICENSE. If not, see <http://www.gnu.org/licenses/>.
 */

package com.neuralnetwork.imageinference.ui.details

/**
 * The state a model can be in.
 */
enum class ModelState {
    /**
     * The state when a model is created but not used yet.
     */
    INITIAL,

    /**
     * The state when a model is running inference.
     */
    RUNNING,

    /**
     * The state when a model has finished inference successfully.
     */
    SUCCESS,

    /**
     * The state when a model has failed inference.
     * This state is also used if the requirements for running the model are not met.
     */
    FAILED
}
