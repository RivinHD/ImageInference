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

import android.graphics.Bitmap
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.neuralnetwork.imageinference.model.Model
import com.neuralnetwork.imageinference.model.ModelDetails
import com.neuralnetwork.imageinference.model.ModelState
import com.neuralnetwork.imageinference.ui.details.DetailsViewModel
import com.neuralnetwork.imageinference.ui.details.containers.ModelInputType
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executor

/**
 * The view model for the video camera fragment.
 *
 * @constructor Creates an empty video camera view model.
 */
class VideoCameraViewModel : ViewModel(), ImageAnalysis.Analyzer {
    /**
     * Holds the details that are update by the model and displayed by the details fragment.
     */
    private val _details = MutableLiveData<ModelDetails>().apply {
        value = ModelDetails(ModelInputType.VIDEO)
    }

    /**
     * Holds the state of the model.
     */
    private val _modelState = MutableLiveData<ModelState>().apply {
        value = ModelState.INITIAL
    }

    /**
     * Holds the state of the recording button.
     */
    private val _isRecording = MutableLiveData<Boolean>().apply {
        value = false
    }

    /**
     * Holds the details view model that is provided to the details fragment.
     */
    private val _detailsViewModel = DetailsViewModel(_details, _modelState)

    /**
     * Holds the resolution setup for the video camera.
     */
    private val _resolutionSelector = ResolutionSelector.Builder()
        .setResolutionStrategy(
            ResolutionStrategy(
                Size(256, 256),
                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER
            )
        )
        .build()

    /**
     * Holds the image analysis for the video camera.
     */
    private val _imageAnalysis = ImageAnalysis.Builder()
        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .setResolutionSelector(_resolutionSelector)
        .build()

    /**
     * Holds the selected camera for the video camera.
     */
    private val _cameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

    /**
     * Gets the detail view model for the details fragment.
     */
    val detailsViewModel get() = _detailsViewModel

    /**
     * Gets or sets the model for the inference.
     */
    var model: Model? = null

    /**
     * Gets the image analysis for the video camera.
     */
    val imageAnalysis get() = _imageAnalysis

    /**
     * Gets the selected camera for the video camera.
     */
    val cameraSelector get() = _cameraSelector

    /**
     * Gets the recording state of the video camera.
     */
    val isRecording: LiveData<Boolean> = _isRecording

    /**
     * Gets the state of the model.
     */
    val modelState: LiveData<ModelState> = _modelState

    /**
     * Callback how the model change is handled.
     */
    val onModelChangedCallback : ((Model?) -> Unit) = {
        this.model = it
        _modelState.value = ModelState.INITIAL
    }

    /**
     * Starts the recording of the camera with running inference.
     *
     * @param executor The executor where the analysis runs on.
     */
    fun startRecording(executor: Executor) {
        _isRecording.value = true
        imageAnalysis.setAnalyzer(executor, this)
    }

    /**
     * Stops the recording of the camera with stopped inference.
     */
    fun stopRecording() {
        _isRecording.value = false
        imageAnalysis.clearAnalyzer()
    }

    override fun analyze(image: ImageProxy) {
        Log.d("VideoCapture", "Got an image!")
        val rotation = image.imageInfo.rotationDegrees.toFloat()
        runModel(image.toBitmap().rotate(rotation))
        image.close()
    }

    /**
     * Run the current model on the given image.
     *
     * @param image The image to run inference on.
     */
    private fun runModel(image: Bitmap) {
        if (_modelState.value == ModelState.RUNNING){
            Log.d("VideoCapture", "The model is already running.")
            return
        }

        Log.d("VideoCapture", "Try running the model.")
        val fixedModel: Model? = model
        if (fixedModel == null) {
            Log.e("VideoCapture", "Failed to get the model.")
            _modelState.value = ModelState.FAILED
            return
        }

        val details: ModelDetails? = _details.value
        if (details == null){
            Log.e("VideoCapture", "Failed to get the details.")
            _modelState.value = ModelState.FAILED
            return
        }

        Log.d("VideoCapture", "Running the model.")
        _modelState.value = ModelState.RUNNING
        viewModelScope.launch(Dispatchers.Default) {
            val outputDetails = fixedModel.run(image, details)
            withContext(Dispatchers.Main) {
                _details.value = outputDetails
                _modelState.value = ModelState.SUCCESS
            }
        }
    }
}
