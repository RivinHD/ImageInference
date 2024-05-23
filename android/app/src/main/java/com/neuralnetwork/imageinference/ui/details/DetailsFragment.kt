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

class DetailsFragment : Fragment() {


    private var _binding: FragmentDetailsBinding? = null

    // This property is only valid between onCreateView and
    // onDestroyView.
    private val binding get() = _binding!!
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

        val information = binding.informationText
        val detailsView = binding.detailsView
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

        vm.success.observe(viewLifecycleOwner) {
            information.visibility = if (it != true) View.VISIBLE else View.INVISIBLE
            information.text = getString(
                if (it == null) R.string.no_data else R.string.inference_failed
            )
            detailsView.visibility = if (it == true) View.VISIBLE else View.INVISIBLE
        }

        vm.details.observe(viewLifecycleOwner) {
            val results = it.getTopResults(5)

            val applier = classes.zip(accuracies).zip(results) { (a, b), c -> Triple(a, b, c) }
            for ((name, accuracy, result) in applier) {
                name.text = result.name
                accuracy.text = "${"%.2f".format(result.accuracy * 100)} %"
            }
        }

        return root
    }
}
