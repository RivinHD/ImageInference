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

import android.annotation.SuppressLint
import android.content.Context
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.neuralnetwork.imageinference.R
import com.neuralnetwork.imageinference.databinding.FragmentDetailsBinding

/**
 * Fragment that shows the details of the model.
 *
 * @constructor Create empty details fragment.
 */
class DetailsFragment : Fragment() {

    /**
     * The binding that gets updated from the fragment.
     */
    private var _binding: FragmentDetailsBinding? = null

    /**
     * The binding that holds the view of this fragment.
     * This property is only valid between onCreateView and onDestroyView.
     */
    private val binding get() = _binding!!

    /**
     * The binding that holds the view of this fragment.
     * This property is only valid between onCreateView and onDestroyView.
     */
    private lateinit var _context: Context

    override fun onAttach(context: Context) {
        super.onAttach(context)
        _context = context
    }

    @SuppressLint("SetTextI18n")
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentDetailsBinding.inflate(inflater, container, false)
        val root: View = binding.root

        val vm: DetailsViewModel = when (parentFragment) {
            is DetailsConnector -> {
                (requireParentFragment() as DetailsConnector).getDetailViewModel()
            }

            else -> {
                ViewModelProvider(this)[DetailsViewModel::class.java]
            }
        }

        observeDetails(vm)
        observeState(vm)

        return root
    }

    /**
     * Setup the observe of the view model property details LiveData.
     *
     * @param vm The view model of this fragment.
     */
    @SuppressLint("SetTextI18n")
    private fun observeDetails(vm: DetailsViewModel) {
        val classes = arrayOf(
            binding.option1Class,
            binding.option2Class,
            binding.option3Class,
            binding.option4Class,
            binding.option5Class
        )
        val accuracies = arrayOf(
            binding.option1Percentage,
            binding.option2Percentage,
            binding.option3Percentage,
            binding.option4Percentage,
            binding.option5Percentage
        )

        vm.details.observe(viewLifecycleOwner) {
            val results = it.getTopResults(5)

            val applier = classes.zip(accuracies).zip(results) { (a, b), c -> Triple(a, b, c) }
            for ((name, accuracy, result) in applier) {
                name.text = result.name
                accuracy.text = "${"%.2f".format(result.accuracy * 100)} %"
            }
        }
    }

    /**
     * Setup the observe on the view model property state LiveData.
     *
     * @param vm The view model of this fragment.
     */
    private fun observeState(
        vm: DetailsViewModel
    ) {
        val information = binding.informationText
        val detailsView = binding.detailsView

        vm.state.observe(viewLifecycleOwner) {
            information.visibility = when (it) {
                null, ModelState.INITIAL, ModelState.RUNNING, ModelState.FAILED -> View.VISIBLE
                ModelState.SUCCESS -> View.INVISIBLE
            }
            information.text = getString(
                when (it) {
                    null, ModelState.INITIAL -> R.string.no_data
                    ModelState.RUNNING -> R.string.inference_running
                    ModelState.SUCCESS -> R.string.inference_successful
                    ModelState.FAILED -> R.string.inference_failed
                }
            )
            detailsView.visibility = when (it) {
                null, ModelState.INITIAL, ModelState.RUNNING, ModelState.FAILED -> View.INVISIBLE
                ModelState.SUCCESS -> View.VISIBLE
            }
        }
    }
}
