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

import com.neuralnetwork.imageinference.ui.details.containers.ModelInputType
import com.neuralnetwork.imageinference.ui.details.containers.ModelResult
import java.text.DateFormat
import kotlin.math.min

data class ModelDetails(val modelInputType: ModelInputType) {
    private var _results: Array<ModelResult> = Array(0) { ModelResult.Default }

    private var _evaluationTimeNano: Long? = null

    var evaluationTimeNano: Long?
        get() = _evaluationTimeNano
        set(value) {
            if (value != null) {
                _evaluationTimeNano = value
            }
        }

    var evaluationTimeMillisecond: Long?
        get() = _evaluationTimeNano?.div(1_000_000)
        set(value) {
            if (value != null) {
                _evaluationTimeNano = value * 1_000_000
            }
        }

    val evaluationTimeString: String get() = DateFormat.getInstance().format(_evaluationTimeNano)

    var results: Array<ModelResult>
        get() = _results
        set(value) {
            value.sortBy { it.accuracy }
            _results = value
        }

    /**
     * Get top results
     *
     * @param count
     * @return
     */
    fun getTopResults(count: Int): Array<ModelResult> {
        if (count < 0) {
            throw RuntimeException("Cannot return a negative amount of results.")
        }
        val transferArray: Array<ModelResult> = Array(count) { ModelResult.Default }
        for (i in 0..<min(count, _results.count())) {
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
