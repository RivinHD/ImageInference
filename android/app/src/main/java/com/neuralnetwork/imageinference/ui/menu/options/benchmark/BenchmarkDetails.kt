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
 * @property collectionName The name of the collection that is benchmarked.
 * @property modelName The name of the model that is benchmarked.
 * @constructor Creates a new benchmark details object.
 */
@Serializable
data class BenchmarkDetails(
    val collectionName: String,
    val modelName: String
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
     * The top1 accuracy of ths benchmark.
     */
    private var _top1: Float = 0.0f

    /**
     * The top5 accuracy of ths benchmark.
     */
    private var _top5: Float = 0.0f

    /**
     * Indicates if labeled details where added.
     */
    private var _labeled: Boolean = false

    /**
     * Get the average, min, max evaluation time of all details of different models in nanoseconds.
     */
    val evaluationTimeNano get() = _evaluationTimeNano

    /**
     * Gets the Top 1 accuracy of the benchmark.
     */
    val top1 get() = _top1

    /**
     * Gets the Top 5 accuracy of the benchmark.
     */
    val top5 get() = _top5

    /**
     * Gets the indication if labeled data was added.
     */
    val labeled get() = _labeled

    /**
     * Adds details from one inference run to the benchmark.
     *
     * @param details The details object that gets added.
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

    /**
     * Adds details from one inference run to the benchmark.
     *
     * @param details The details object that gets added.
     * @param label The label of the image the details evaluated.
     */
    fun addDetails(details: ModelDetails, label: String) {
        _labeled = true
        val results = details.getTopResults(5)
        val count = _evaluationTimeNano.count
        if (results.isNotEmpty()) {
            val top1Correct = if (results[0].name == label) 1.0f else 0.0f
            _top1 = (_top1 * count + top1Correct) / (count + 1)
            val top5Correct = if (results.any { it.name == label }) 1.0f else 0.0f
            _top5 = (_top5 * count + top5Correct) / (count + 1)
        }

        addDetails(details)
    }
}
