/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
 */

package com.neuralnetwork.visioninference.ui.details.containers

/**
 * Container to store a single result of a model.
 *
 * @property name The name/label/class of the result.
 * @property accuracy The accuracy of the given label.
 * @constructor Create an result with the given name and accuracy.
 */
data class ModelResult(val name: String, val accuracy: Float){
    companion object{

        val Default = ModelResult("None", 0.0f)
    }
}
