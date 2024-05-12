package com.example.imageinference.ui.details

import com.example.imageinference.ui.details.containers.ModelInputType
import com.example.imageinference.ui.details.containers.ModelResult
import java.lang.RuntimeException
import kotlin.math.min

data class ModelDetails(val modelInputType: ModelInputType) {
    private var _results: Array<ModelResult> = Array(0) { ModelResult.default() }

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
        val transferArray: Array<ModelResult> = Array(count) { ModelResult.default() }
        for (i in 0..min(count, _results.count())) {
            transferArray[i] = _results[i].copy()
        }
        return transferArray
    }

    /**
     * Caches the given array and sort it based on the accuracy.
     *
     * @param results The results to cache and sort.
     */
    fun setResults(results: Array<ModelResult>) {
        results.sortBy { it.accuracy }
        _results = results
    }


}
