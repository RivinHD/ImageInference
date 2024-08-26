/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
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
import com.neuralnetwork.imageinference.model.ModelState
import com.neuralnetwork.imageinference.ui.details.containers.ModelInputType

/**
 * Details fragment
 *
 * @constructor Create empty Details fragment
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

        setupCorrectDetails(vm)

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
            val results = it.getTopResults(TOP_RESULTS_COUNT)

            val applier = classes.zip(accuracies).zip(results) { (a, b), c -> Triple(a, b, c) }
            for ((name, accuracy, result) in applier) {
                name.text = result.name
                accuracy.text = "${"%.2f".format(result.accuracy * TO_PERCENTAGE)} %"
            }

            val time = binding.optionTimeValue
            time.text = it.evaluationTimeString
            val frames = binding.optionFrameValue
            frames.text =
                (SECOND_TO_NANOSECOND / (it.evaluationTimeNano ?: SECOND_TO_NANOSECOND)).toString()
            val timeNano = binding.optionTimeNanoValue
            timeNano.text = it.evaluationTimeNanoString
        }
    }

    /**
     * Setup the observe on the view model property state LiveData.
     *
     * @param vm The view model of this fragment.
     */
    private fun observeState(vm: DetailsViewModel) {
        val informationView = binding.informationView
        val information = binding.informationText
        val detailsView = binding.detailsView

        vm.state.observe(viewLifecycleOwner) {
            val details = vm.details.value ?: return@observe
            val type = details.modelInputType

            informationView.visibility = when (type) {
                ModelInputType.VIDEO -> {
                    View.VISIBLE
                }

                ModelInputType.PHOTO, ModelInputType.IMAGE -> {
                    when (it) {
                        ModelState.SUCCESS -> View.GONE
                        else -> View.VISIBLE
                    }
                }
            }

            information.visibility = when (type) {
                ModelInputType.VIDEO -> {
                    when (it) {
                        ModelState.RUNNING, ModelState.SUCCESS -> View.GONE
                        else -> View.VISIBLE
                    }
                }

                ModelInputType.PHOTO, ModelInputType.IMAGE -> {
                    View.VISIBLE
                }
            }

            information.text = getString(
                when (it) {
                    null, ModelState.INITIAL, ModelState.NO_DATA_SELECTED -> R.string.no_data
                    ModelState.RUNNING -> R.string.inference_running
                    ModelState.SUCCESS -> R.string.inference_successful
                    ModelState.FAILED -> R.string.inference_failed
                    ModelState.NO_MODEL_SELECTED -> R.string.select_a_model_to_start_inference
                    ModelState.CANCELLED -> R.string.inference_was_cancelled
                }
            )

            detailsView.visibility = when (type) {
                ModelInputType.VIDEO -> {
                    when (it) {
                        ModelState.RUNNING, ModelState.SUCCESS -> View.VISIBLE
                        else -> View.GONE
                    }
                }

                ModelInputType.PHOTO, ModelInputType.IMAGE -> {
                    when (it) {
                        ModelState.SUCCESS -> View.VISIBLE
                        else -> View.INVISIBLE
                    }
                }
            }
        }
    }

    /**
     * Setup the correct details by hiding none important details.
     *
     * @param vm The view model of this fragment.
     */
    private fun setupCorrectDetails(vm: DetailsViewModel) {
        val details = vm.details.value ?: return
        val type = details.modelInputType

        when (type) {
            ModelInputType.VIDEO -> {
            }

            ModelInputType.PHOTO -> {
                binding.optionFrames.visibility = View.GONE
            }

            ModelInputType.IMAGE -> {
                binding.optionFrames.visibility = View.GONE
            }
        }
    }

    companion object {
        private const val TOP_RESULTS_COUNT = 5
        private const val SECOND_TO_NANOSECOND: Long = 1_000_000_000
        private const val TO_PERCENTAGE = 100
    }
}
