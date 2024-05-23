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
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.camera.core.ExperimentalZeroShutterLag
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.graphics.drawable.DrawableCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.google.common.util.concurrent.ListenableFuture
import com.neuralnetwork.imageinference.MainActivity
import com.neuralnetwork.imageinference.R
import com.neuralnetwork.imageinference.databinding.FragmentPhotoCameraBinding
import com.neuralnetwork.imageinference.ui.details.DetailsConnector
import com.neuralnetwork.imageinference.ui.details.DetailsViewModel

class PhotoCameraFragment : Fragment(), DetailsConnector {

    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>

    private var _binding: FragmentPhotoCameraBinding? = null

    // This property is only valid between onCreateView and
    // onDestroyView.
    private val binding get() = _binding!!

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
        val vm = ViewModelProvider(this)[PhotoCameraViewModel::class.java]
        _binding = FragmentPhotoCameraBinding.inflate(inflater, container, false)
        val root: View = binding.root

        val photoCapture = binding.photoCapture
        val photoView = binding.photoInferenceImage
        val photoPreview = binding.photoPreview

        // Request camera permission
        MainActivity.checkCameraPermission(_context)

        // Setup camera preview and capture
        cameraProviderFuture = ProcessCameraProvider.getInstance(_context)
        cameraProviderFuture.addListener(
            {
                val cameraProvider = cameraProviderFuture.get()

                vm.preview.setSurfaceProvider(photoPreview.getSurfaceProvider())

                cameraProvider.bindToLifecycle(
                    viewLifecycleOwner,
                    vm.cameraSelector,
                    vm.imageCapture,
                    vm.preview
                )
            },
            ContextCompat.getMainExecutor(_context)
        )

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

        vm.modelSuccess.observe(viewLifecycleOwner) {
            val colorID = if (it) {
                R.color.success_green
            } else {
                R.color.fail_red
            }

            val drawableID = if (it) {
                R.drawable.ic_check_google
            } else {
                R.drawable.ic_close_google
            }

            val drawable = ContextCompat.getDrawable(_context, drawableID)
            val color = ContextCompat.getColor(_context, colorID)
            if (drawable != null) {
                DrawableCompat.setTint(drawable, color)
            }
            photoCapture.setCompoundDrawablesRelativeWithIntrinsicBounds(drawable, null, drawable, null)
        }

        vm.image.observe(viewLifecycleOwner) {
            if (it == null) {
                photoPreview.visibility = View.VISIBLE
                photoCapture.text = getString(R.string.photo_capture)
                photoCapture.setCompoundDrawablesRelativeWithIntrinsicBounds(null, null, null, null)
            } else {
                photoView.setImageBitmap(it)
                photoPreview.visibility = View.INVISIBLE
                photoCapture.text = getString(R.string.photo_clear)
            }
        }

        return root
    }

    override fun onDestroyView() {
        super.onDestroyView()
        cameraProviderFuture.cancel(true)
        _binding = null
    }

    override fun getDetailViewModel(): DetailsViewModel {
        val photoCameraViewModel = ViewModelProvider(this)[PhotoCameraViewModel::class.java]
        return photoCameraViewModel.detailsViewModel
    }
}
