/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
 */

package com.neuralnetwork.visioninference.model

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
