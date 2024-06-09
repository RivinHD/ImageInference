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

package com.neuralnetwork.imageinference.ui.photoCamera


import android.content.Context
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.camera.core.ExperimentalZeroShutterLag
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.graphics.drawable.DrawableCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import com.google.common.util.concurrent.ListenableFuture
import com.neuralnetwork.imageinference.MainActivity
import com.neuralnetwork.imageinference.R
import com.neuralnetwork.imageinference.databinding.FragmentPhotoCameraBinding
import com.neuralnetwork.imageinference.model.ModelConnector
import com.neuralnetwork.imageinference.model.ModelState
import com.neuralnetwork.imageinference.ui.details.DetailsConnector
import com.neuralnetwork.imageinference.ui.details.DetailsViewModel
import kotlinx.coroutines.CancellationException

/**
 * Fragment that uses a photo camera for the model input.
 *
 * @constructor Create empty photo camera fragment.
 */
class PhotoCameraFragment : Fragment(), DetailsConnector {
    /**
     * Holds the camera provider future.
     */
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>

    /**
     * The binding that gets updated from the fragment.
     */
    private var _binding: FragmentPhotoCameraBinding? = null

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

    @ExperimentalZeroShutterLag
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val vm : PhotoCameraViewModel by viewModels()
        _binding = FragmentPhotoCameraBinding.inflate(inflater, container, false)
        val root: View = binding.root
        val modelConnector = (activity as ModelConnector)
        vm.model = modelConnector.getModel()
        modelConnector.setOnModelChangedListener(vm.onModelChangedCallback)

        setupCamera(vm)
        setupPhotoCapture(vm)

        observeModelState(vm)
        observeImage(vm)

        return root
    }

    /**
     * Setup the observe on the view model property image LiveData.
     *
     * @param vm The view model of this fragment.
     */
    private fun observeImage(
        vm: PhotoCameraViewModel
    ) {
        val photoCapture = binding.photoCapture
        val photoView = binding.photoInferenceImage
        val photoPreview = binding.photoPreview

        vm.image.observe(viewLifecycleOwner) {
            if (it == null) {
                photoPreview.visibility = View.VISIBLE
                photoCapture.text = getString(R.string.photo_capture)
                photoCapture.setCompoundDrawablesRelativeWithIntrinsicBounds(
                    null, null, null, null
                )
            } else {
                photoView.setImageBitmap(it)
                photoPreview.visibility = View.INVISIBLE
                photoCapture.text = getString(R.string.photo_clear)
            }
        }
    }

    /**
     * Setup the observe on the view model property models state LiveData.
     *
     * @param vm The view model of this fragment.
     */
    private fun observeModelState(
        vm: PhotoCameraViewModel
    ) {
        val photoCapture = binding.photoCapture
        val progressBar = binding.photoProgressbar

        vm.modelState.observe(viewLifecycleOwner) {
            photoCapture.isEnabled = (it != ModelState.RUNNING)
            setButtonDrawable(it)

            progressBar.visibility = when(it){
                ModelState.RUNNING -> View.VISIBLE
                else -> View.GONE
            }
        }
    }

    /**
     * Set the drawable of the photo capture button.
     *
     * @param it The model state that was observed.
     */
    private  fun setButtonDrawable(it: ModelState){
        val photoCapture = binding.photoCapture

        val colorID = when (it) {
            ModelState.RUNNING -> R.color.loading
            ModelState.SUCCESS -> R.color.success_green
            ModelState.FAILED -> R.color.fail_red
            else -> return
        }

        val drawableID = when (it) {
            ModelState.RUNNING -> R.drawable.inference
            ModelState.SUCCESS -> R.drawable.ic_check_google
            ModelState.FAILED -> R.drawable.ic_close_google
            else -> return
        }

        val drawable = ContextCompat.getDrawable(_context, drawableID)
        val color = ContextCompat.getColor(_context, colorID)
        if (drawable != null) {
            DrawableCompat.setTint(drawable, color)
        }

        photoCapture.setCompoundDrawablesRelativeWithIntrinsicBounds(
            drawable,
            null,
            drawable,
            null
        )
    }

    /**
     * Setup the photo capture button with the OnClickListener.
     *
     * @param vm The view model of this fragment.
     */
    private fun setupPhotoCapture(
        vm: PhotoCameraViewModel
    ) {
        val photoCapture = binding.photoCapture

        photoCapture.setOnClickListener {
            if (vm.image.value == null) {
                // Take the current photo from the camera preview
                vm.imageCapture.takePicture(
                    ContextCompat.getMainExecutor(_context),
                    vm.onImageCaptureCallback
                )
            } else {
                vm.clearImage()
            }
        }
    }

    /**
     * Setup the camera on the preview.
     *
     * @param vm The view model of this fragment.
     */
    private fun setupCamera(
        vm: PhotoCameraViewModel
    ) {
        val photoPreview = binding.photoPreview
        // Request camera permission
        MainActivity.checkCameraPermission(_context)

        // Setup camera preview and capture
        cameraProviderFuture = ProcessCameraProvider.getInstance(_context)
        cameraProviderFuture.addListener(
            {
                try {
                    val cameraProvider = cameraProviderFuture.get()
                    vm.preview.setSurfaceProvider(photoPreview.getSurfaceProvider())

                    cameraProvider.bindToLifecycle(
                        viewLifecycleOwner,
                        vm.cameraSelector,
                        vm.imageCapture,
                        vm.preview
                    )
                } catch (e: CancellationException) {
                    Log.e("PhotoCapture", e.toString())
                }
            },
            ContextCompat.getMainExecutor(_context)
        )

    }

    override fun onDestroyView() {
        super.onDestroyView()
        val modelConnector = (activity as ModelConnector)
        val vm : PhotoCameraViewModel by viewModels()
        modelConnector.removeOnModelChangeListener(vm.onModelChangedCallback)
        cameraProviderFuture.cancel(true)
        _binding = null
    }

    override fun getDetailViewModel(): DetailsViewModel {
        val vm : PhotoCameraViewModel by viewModels()
        return vm.detailsViewModel
    }
}
