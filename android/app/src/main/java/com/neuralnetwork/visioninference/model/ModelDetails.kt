/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
 */

package com.neuralnetwork.visioninference.model

import com.neuralnetwork.visioninference.ui.details.containers.ModelInputType
import com.neuralnetwork.visioninference.ui.details.containers.ModelResult
import java.util.Locale
import kotlin.math.min

/**
 * Holds all details a model may generate during its inference.
 *
 * @property modelInputType The type of input the model was evaluated with.
 * @constructor Creates an empty Model details
 */
data class ModelDetails(val modelInputType: ModelInputType) {
    /**
     * Holds the results of the model inference.
     */
    private var _results: Array<ModelResult> = Array(0) { ModelResult.Default }

    /**
     * Holds the evaluation time of the model in nanoseconds.
     */
    private var _evaluationTimeNano: Long? = null

    /**
     * Gets or sets the evaluation time of the model in nanoseconds.
     */
    var evaluationTimeNano: Long?
        get() = _evaluationTimeNano
        set(value) {
            if (value != null) {
                _evaluationTimeNano = value
            }
        }

    /**
     * Gets or sets the evaluation time of the model in milliseconds.
     */
    var evaluationTimeMillisecond: Long?
        get() = _evaluationTimeNano?.div(1_000_000)
        set(value) {
            if (value != null) {
                _evaluationTimeNano = value * 1_000_000
            }
        }

    /**
     * Gets the evaluation time of the model as a string in milliseconds.
     */
    val evaluationTimeString: String get() {
        val time = evaluationTimeMillisecond
        return if (time != null) {
            when (time){
                in 0..<500 -> {
                    "${time} ms"
                }
                in 500..<1000 -> {
                    "0.${time} s"
                }
                else -> {
                    String.format(Locale.ROOT,"%.3d s", time / 1000.0)
                }
            }
        } else {
            "N/A"
        }
    }

    /**
     * Gets the evaluation time of the model as a string in nanoseconds.
     */
    val evaluationTimeNanoString: String get() {
        val time = _evaluationTimeNano
        return if (time != null) {  
            "$time ns"
        } else {
            "N/A"
        }
    }

    /**
     * Gets the results of the model inference.
     */
    var results: Array<ModelResult>
        get() = _results
        set(value) {
            value.sortByDescending { it.accuracy }
            _results = value
        }

    /**
     * Get the top results of the model inference.
     *
     * @param count The amount of top results to return.
     * @return The [count] count top results of the model inference.
     */
    fun getTopResults(count: Int): Array<ModelResult> {
        if (count < 0) {
            throw RuntimeException("Cannot return a negative amount of results.")
        }
        val transferArray: Array<ModelResult> = Array(count) { ModelResult.Default }
        for (i in 0..<min(count, _results.size)) {
            transferArray[i] = _results[i].copy()
        }
        return transferArray
    }

    /**
     * Combine the parameters with the other details where the values
     * of this ModelDetails are used as the primary/base values.
     *
     * @param other The ModelDetails to fill the this ModelDetails.
     * @return The combined ModelDetails.
     */
    fun combine(other: ModelDetails): ModelDetails {
        val result = ModelDetails(this.modelInputType)

        result.update(this)
        result.merge(other)

        return result
    }

    /**
     * Merges the parameters with the other details where the values
     * of this ModelDetails are used as the primary/base values.
     * Note: That this operation is done on the current object.
     *
     * @param other The ModelDetails to fill the this ModelDetails.
     * @return The combined ModelDetails as this object.
     */
    fun merge(other: ModelDetails): ModelDetails {
        if (this._results.isEmpty()) {
            this._results = other._results
        }
        if (this._evaluationTimeNano == null){
            this._evaluationTimeNano = other._evaluationTimeNano
        }
        return this
    }

    /**
     * Updates the parameters with the other details by overwriting.
     * Note: That this operation is done on the current object.
     *
     * @param other The ModelDetails to overwrite the this ModelDetails.
     * @return The update ModelDetails as this object.
     */
    fun update(other: ModelDetails): ModelDetails {
        this._results = other._results
        this._evaluationTimeNano = other._evaluationTimeNano
        return this
    }


}
