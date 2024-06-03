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

package com.neuralnetwork.imageinference.model

/**
 * Provides a connection to the selected model from a fragment.
 */
interface ModelConnector {

    /**
     * Get the current model.
     *
     * @return The selected model.
     */
    fun getModel(): Model?

    /**
     * Get the name of the current model.
     *
     * @return The name of the selected model.
     */
    fun getModelName(): String

    /**
     * Set a callback that is triggered when the model changes.
     *
     * @param callback The callback to be triggered.
     */
    fun setOnModelChangedListener(callback : (m: Model?) -> Unit)

    /**
     * Removes the set callback only if it still set as the current callback.
     *
     * @param callback The callback to be removed.
     */
    fun removeOnModelChangeListener(callback: (m: Model?) -> Unit)
}
