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
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalZeroShutterLag
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
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
        val photoCameraViewModel = ViewModelProvider(this)[PhotoCameraViewModel::class.java]
        _binding = FragmentPhotoCameraBinding.inflate(inflater, container, false)
        val root: View = binding.root

        val photoCapture = binding.photoCapture
        val photoView = binding.photoInferenceImage
        val photoPreview = binding.photoPreview

        // Request camera permission
        MainActivity.checkCameraPermission(_context)

        // Setup camera preview and capture
        cameraProviderFuture = ProcessCameraProvider.getInstance(_context)
        val imageCapture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_ZERO_SHUTTER_LAG)
            .build()
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview: Preview = Preview.Builder()
                .build()

            val cameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            preview.setSurfaceProvider(photoPreview.getSurfaceProvider())

            cameraProvider.bindToLifecycle(
                viewLifecycleOwner,
                cameraSelector,
                imageCapture,
                preview
            )
        }, ContextCompat.getMainExecutor(_context))


        photoCapture.setOnClickListener {
            if (photoPreview.visibility == View.VISIBLE) {
                // Take the current photo from the camera preview
                var photoSuccess = true
                imageCapture.takePicture(
                    ContextCompat.getMainExecutor(_context),
                    object : ImageCapture.OnImageCapturedCallback() {
                        override fun onCaptureSuccess(image: ImageProxy) {
                            super.onCaptureSuccess(image)
                            photoView.setImageBitmap(image.toBitmap())
                            photoCameraViewModel.runModel(image.toBitmap())
                        }

                        override fun onError(exception: ImageCaptureException) {
                            super.onError(exception)
                            photoSuccess = false
                        }
                    }
                )

                if (photoSuccess) {
                    photoPreview.visibility = View.INVISIBLE
                    photoCapture.text = getString(R.string.photo_clear)
                }

            } else {
                // Delete the current photo and activate the preview again
                photoPreview.visibility = View.VISIBLE
                photoCapture.text = getString(R.string.photo_capture)
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
        return photoCameraViewModel.getDetailViewModel()
    }
}
