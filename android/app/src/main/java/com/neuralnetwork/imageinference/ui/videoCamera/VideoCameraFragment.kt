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

package com.neuralnetwork.imageinference.ui.videoCamera

import android.content.Context
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.graphics.drawable.DrawableCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.lifecycle.ViewModelProvider
import com.google.common.util.concurrent.ListenableFuture
import com.neuralnetwork.imageinference.MainActivity
import com.neuralnetwork.imageinference.R
import com.neuralnetwork.imageinference.databinding.FragmentVideoCameraBinding
import com.neuralnetwork.imageinference.model.ModelConnector
import com.neuralnetwork.imageinference.ui.details.DetailsConnector
import com.neuralnetwork.imageinference.ui.details.DetailsViewModel
import com.neuralnetwork.imageinference.ui.details.ModelState

/**
 * Fragment that uses a video camera for the model input.
 *
 * @constructor Create empty Video camera fragment.
 */
class VideoCameraFragment : Fragment(), DetailsConnector {
    /**
     * Holds the camera provider future.
     */
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>

    /**
     * The binding that gets updated from the fragment.
     */
    private var _binding: FragmentVideoCameraBinding? = null

    /**
     * The binding that holds the view of this fragment.
     * This property is only valid between onCreateView and onDestroyView.
     */
    private val binding get() = _binding!!

    /**
     * The context object of the fragment.
     * This property is only valid between onAttach and onDetach.
     */
    private lateinit var _context: Context

    override fun onAttach(context: Context) {
        super.onAttach(context)
        _context = context
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val vm: VideoCameraViewModel by viewModels()
        _binding = FragmentVideoCameraBinding.inflate(inflater, container, false)
        val root: View = binding.root
        val modelConnector = (activity as ModelConnector)
        vm.model = modelConnector.getModel()
        modelConnector.setOnModelChangedListener(vm.onModelChangedCallback)

        setupCamera(vm)
        setupVideoRecord(vm)

        observeIsRecording(vm)
        observeModelState(vm)

        return root
    }

    /**
     * Setup the observe on the view model property isRecording LiveData.
     *
     * @param vm The view model of this fragment.
     */
    private fun observeIsRecording(vm: VideoCameraViewModel) {
        val videoRecord = binding.videoRecord
        vm.isRecording.observe(viewLifecycleOwner) {
            if (it) {
                videoRecord.text = getString(R.string.stop)
            } else {
                videoRecord.text = getString(R.string.video_record)
            }
        }
    }

    /**
     * Setup the observe on the view model property models state LiveData.
     *
     * @param vm The view model of this fragment.
     */
    private fun observeModelState(vm: VideoCameraViewModel) {
        val videoRecord = binding.videoRecord

        vm.modelState.observe(viewLifecycleOwner) {
            if (it == ModelState.RUNNING) {
                val colorID = R.color.loading
                val drawableID = R.drawable.inference

                val drawable = ContextCompat.getDrawable(_context, drawableID)
                val color = ContextCompat.getColor(_context, colorID)

                if (drawable != null) {
                    DrawableCompat.setTint(drawable, color)
                }

                videoRecord.setCompoundDrawablesRelativeWithIntrinsicBounds(
                    drawable, null, drawable, null
                )
            } else {
                videoRecord.setCompoundDrawablesRelativeWithIntrinsicBounds(
                    null, null, null, null
                )
            }
        }
    }

    /**
     * Setup the video record button with the OnClickListener.
     *
     * @param vm The view model of this fragment.
     */
    private fun setupVideoRecord(
        vm: VideoCameraViewModel
    ) {
        val videoRecord = binding.videoRecord
        videoRecord.setOnClickListener {
            if (vm.isRecording.value == true) {
                vm.stopRecording()
            } else {
                vm.startRecording(ContextCompat.getMainExecutor(_context))
            }
        }
    }

    /**
     * Setup the camera on the preview.
     *
     * @param vm The view model of this fragment.
     */
    private fun setupCamera(
        vm: VideoCameraViewModel
    ) {
        val videoPreview = binding.videoPreview
        // Request camera permission
        MainActivity.checkCameraPermission(_context)

        // Setup camera preview and capture
        cameraProviderFuture = ProcessCameraProvider.getInstance(_context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview: Preview = Preview.Builder()
                .build()
            preview.setSurfaceProvider(videoPreview.getSurfaceProvider())

            cameraProvider.bindToLifecycle(
                viewLifecycleOwner,
                vm.cameraSelector,
                vm.imageAnalysis,
                preview
            )
        }, ContextCompat.getMainExecutor(_context))
    }

    override fun onDestroyView() {
        super.onDestroyView()
        val modelConnector = (activity as ModelConnector)
        val vm = ViewModelProvider(this)[VideoCameraViewModel::class.java]
        modelConnector.removeOnModelChangeListener(vm.onModelChangedCallback)
        vm.stopRecording()
        _binding = null
    }

    override fun getDetailViewModel(): DetailsViewModel {
        val vm: VideoCameraViewModel by viewModels()
        return vm.detailsViewModel
    }
}
