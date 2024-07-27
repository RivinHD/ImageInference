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

import android.annotation.SuppressLint
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.neuralnetwork.imageinference.databinding.ItemBenchmarkDetailsBinding
import com.neuralnetwork.imageinference.ui.details.DetailsFragment
import com.neuralnetwork.imageinference.ui.details.DetailsFragment.Companion

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
