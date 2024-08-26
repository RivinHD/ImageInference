/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
 */

package com.neuralnetwork.imageinference.model

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
    FAILED,

    /**
     * The state when no model is selected.
     */
    NO_MODEL_SELECTED,

    /**
     * The state when no data is selected for the model to do inference on.
     */
    NO_DATA_SELECTED,

    /**
     * The state when the model is cancelled.
     */
    CANCELLED

}
