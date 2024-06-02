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

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel

/**
 * The view model for the details fragment.
 *
 * @constructor Creates an details view model filled with the given details and success state.
 *
 * @param modelDetail (optional) The detail object that will be shown.
 * @param modelState (optional) The model state object that will be used.
 */
class DetailsViewModel(modelDetail : MutableLiveData<ModelDetails>? = null, modelState : MutableLiveData<ModelState>? = null) : ViewModel() {

    /**
     * Holds the details that are update by the model and displayed by the details fragment.
     */
    private val _details = modelDetail ?: MutableLiveData<ModelDetails>()

    /**
     * Holds the state of the model.
     */
    private val _state = modelState ?: MutableLiveData<ModelState>()

    /**
     * Gets the details of the model.
     */
    val details : LiveData<ModelDetails> = _details

    /**
     * Gets the model state.
     */
    val state : LiveData<ModelState> = _state

}
