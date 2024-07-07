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

package com.neuralnetwork.imageinference.ui.menu.options.benchmark

import com.neuralnetwork.imageinference.model.ModelDetails
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlin.math.max
import kotlin.math.min

/**
 * Holds the details of a benchmark.
 *
 * @property _collectionName The name of the collection that is benchmarked.
 * @property _modelName The name of the model that is benchmarked.
 * @constructor Creates a new benchmark details object.
 */
@Serializable
data class BenchmarkDetails(
    @SerialName("collectionName") private val _collectionName: String,
    @SerialName("modelName") private val _modelName: String
) {
    /**
     * Holds the details of a benchmark for a specific model.
     *
     * @property T The type of the properties to store.
     * @property average The average of all added measurements.
     * @property min The minimum of all added measurements.
     * @property max The maximum of all added measurements.
     * @constructor Create a new AverageMinMax object.
     */
    @Serializable
    data class AverageMinMax<T>(
        var average: T,
        var min: T,
        var max: T,
    ) {
        /**
         * The count of details that are stored in this object.
         */
        var count: Long = 0
    }

    /**
     * The average, min, max evaluation time of all details of different models in nanoseconds.
     */
    @SerialName("evaluationTimeNano")
    private var _evaluationTimeNano = AverageMinMax(0, Long.MAX_VALUE, Long.MIN_VALUE)

    /**
     * Get the name of the collection this benchmark is based on.
     */
    val collectionName get() = _collectionName

    /**
     * Get the name of the model this benchmark is based on.
     */
    val modelName get() = _modelName

    /**
     * Get the average, min, max evaluation time of all details of different models in nanoseconds.
     */
    val evaluationTimeNano get() = _evaluationTimeNano

    /**
     * Adds details from one inference run to the benchmark.
     *
     * @param details
     */
    fun addDetails(details: ModelDetails) {
        val timeNano = details.evaluationTimeNano
        if (timeNano != null) {
            val time = _evaluationTimeNano
            time.average = (time.average * time.count + timeNano) / (time.count + 1)
            time.min = min(time.min, timeNano)
            time.max = max(time.max, timeNano)
            time.count++
        }
    }
}
