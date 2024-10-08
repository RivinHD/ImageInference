/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
 */

package com.neuralnetwork.visioninference.ui.menu.options.benchmark

import android.annotation.SuppressLint
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.neuralnetwork.visioninference.databinding.ItemBenchmarkDetailsBinding

class BenchmarkDetailsAdapter(private val dataset: Array<BenchmarkDetails>) :
    RecyclerView.Adapter<BenchmarkDetailsAdapter.ViewHolder>() {

    /**
     * Provide a reference to the type of views that you are using
     * (custom ViewHolder)
     */
    class ViewHolder(private val binding: ItemBenchmarkDetailsBinding) :
        RecyclerView.ViewHolder(binding.root) {

        /**
         * Binds the benchmark details object to the view.
         *
         * @param benchmarkDetails The benchmark details to bind.
         */
        @SuppressLint("SetTextI18n")
        fun bind(benchmarkDetails: BenchmarkDetails) {
            binding.modelName.text = benchmarkDetails.modelName
            binding.maxTimeValue.text = "${benchmarkDetails.evaluationTimeNano.max} ns"
            binding.minTimeValue.text = "${benchmarkDetails.evaluationTimeNano.min} ns"
            binding.averageTimeValue.text = "${benchmarkDetails.evaluationTimeNano.average} ns"
            binding.counterValue.text = "${benchmarkDetails.evaluationTimeNano.count}"
            if (benchmarkDetails.labeled) {
                binding.top1.visibility = View.VISIBLE
                binding.top5.visibility = View.VISIBLE
                binding.top1Value.text = "${"%.2f".format(benchmarkDetails.top1 * TO_PERCENTAGE)} %"
                binding.top5Value.text = "${"%.2f".format(benchmarkDetails.top5 * TO_PERCENTAGE)} %"
            } else {
                binding.top1.visibility = View.GONE
                binding.top5.visibility = View.GONE
            }
        }
    }

    // Create new views (invoked by the layout manager)
    override fun onCreateViewHolder(viewGroup: ViewGroup, viewType: Int): ViewHolder {
        // Create a new view, which defines the UI of the list item
        val binding = ItemBenchmarkDetailsBinding.inflate(
            LayoutInflater.from(viewGroup.context),
            viewGroup,
            false
        )

        return ViewHolder(binding)
    }

    // Replace the contents of a view (invoked by the layout manager)
    override fun onBindViewHolder(viewHolder: ViewHolder, position: Int) {
        val item = dataset[position]
        viewHolder.bind(item)
    }

    // Return the size of your dataset (invoked by the layout manager)
    override fun getItemCount() = dataset.size

    companion object {
        const val TO_PERCENTAGE = 100
    }
}
