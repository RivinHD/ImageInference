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

import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalZeroShutterLag
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.neuralnetwork.imageinference.model.Model
import com.neuralnetwork.imageinference.ui.details.DetailsViewModel
import com.neuralnetwork.imageinference.ui.details.ModelDetails
import com.neuralnetwork.imageinference.ui.details.ModelState
import com.neuralnetwork.imageinference.ui.details.containers.ModelInputType
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * The view model for the photo camera fragment.
 *
 * @constructor Creates an empty photo camera view model.
 */
class PhotoCameraViewModel : ViewModel() {
    /**
     * Holds the details that are update by the model and displayed by the details fragment.
     */
    private val _details = MutableLiveData<ModelDetails>().apply {
        this.value = ModelDetails(ModelInputType.PHOTO)
    }

    /**
     * Holds the state of the model.
     */
    private val _modelState = MutableLiveData<ModelState>().apply {
        value = ModelState.INITIAL
    }

    /**
     * Holds the details view model that is provided to the details fragment.
     */
    private val _detailsViewModel = DetailsViewModel(_details, _modelState)

    /**
     * Holds the image capture object that states how the image is captured.
     */
    @ExperimentalZeroShutterLag
    private val _imageCapture = ImageCapture.Builder()
        .setCaptureMode(ImageCapture.CAPTURE_MODE_ZERO_SHUTTER_LAG)
        .build()

    /**
     * Holds the preview object that displays the current camera content.
     */
    private val _preview: Preview = Preview.Builder()
        .build()

    /**
     * Holds the selected camera for the image camera.
     */
    private val _cameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

    /**
     * Holds the image that is captured by the camera and where inference is run on.
     */
    private val _image = MutableLiveData<Bitmap?>()

    /**
     * Get the detail view model for the details fragment.
     */
    val detailsViewModel get() = _detailsViewModel

    /**
     * Gets the built image capture object that states how the image is captured.
     */
    val imageCapture
        @ExperimentalZeroShutterLag
        get() = _imageCapture

    /**
     * Gets the built preview object that displays the current camera content.
     */
    val preview get() = _preview

    /**
     * Gets the selected camera for the image camera.
     */
    val cameraSelector get() = _cameraSelector

    /**
     * The model that is used for the inference.
     */
    var model: Model? = null

    /**
     * Gets the image that is captured by the camera and where inference is run on.
     */
    val image: LiveData<Bitmap?> = _image

    /**
     * Gets the state of the model.
     */
    val modelState: LiveData<ModelState> = _modelState

    /**
     * Gets the callback how the image capture is handled.
     */
    val onImageCaptureCallback = object : ImageCapture.OnImageCapturedCallback() {
        override fun onCaptureSuccess(image: ImageProxy) {
            Log.d("PhotoCapture", "Got an image!")
            super.onCaptureSuccess(image)
            val currentImage = image.toBitmap()
            image.close()
            _image.value = currentImage
            runModel(currentImage)
        }

        override fun onError(exception: ImageCaptureException) {
            super.onError(exception)
            _modelState.value = ModelState.FAILED
        }
    }

    /**
     * Callback how the model change is handled.
     */
    val onModelChangedCallback : ((Model?) -> Unit) = {
            model = it
            _modelState.value = ModelState.INITIAL
        }

    /**
     * Run the current model on the given image.
     *
     * @param image The image to run inference on.
     */
    fun runModel(image: Bitmap) {
        if (_modelState.value == ModelState.RUNNING){
            Log.d("PhotoCapture", "The model is already running.")
            return
        }

        Log.d("PhotoCapture", "Try running the model!")
        val fixedModel: Model? = model
        if (fixedModel == null) {
            Log.e("PhotoCapture", "Failed to get the model.")
            _modelState.value = ModelState.FAILED
            return
        }

        val details: ModelDetails? = _details.value
        if (details == null) {
            Log.e("PhotoCapture", "Failed to get the details.")
            _modelState.value = ModelState.FAILED
            return
        }

        Log.d("PhotoCapture", "Running the model!")
        _modelState.value = ModelState.RUNNING
        viewModelScope.launch(Dispatchers.Default) {
            val outputDetails = fixedModel.run(image, details)
            withContext(Dispatchers.Main) {
                _details.value = outputDetails
                _modelState.value = ModelState.SUCCESS
            }
        }
    }

    /**
     * Clears the image that is captured by the camera.
     */
    fun clearImage() {
        _image.value = null
    }
}
