/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
 */

package com.neuralnetwork.visioninference.ui.details

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import com.neuralnetwork.visioninference.model.ModelDetails
import com.neuralnetwork.visioninference.model.ModelState

/**
 * The view model for the details fragment.
 *
 * @constructor Creates an details view model filled with the given details and success state.
 *
 * @param modelDetail (optional) The detail object that will be shown.
 * @param modelState (optional) The model state object that will be used.
 */
class DetailsViewModel(
    modelDetail: MutableLiveData<ModelDetails>? = null,
    modelState: MutableLiveData<ModelState>? = null
) : ViewModel() {

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
    val details: LiveData<ModelDetails> = _details

    /**
     * Gets the model state.
     */
    val state: LiveData<ModelState> = _state

}
