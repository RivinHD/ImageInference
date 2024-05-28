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
import com.neuralnetwork.imageinference.model.ModelExecutor
import com.neuralnetwork.imageinference.ui.details.DetailsViewModel
import com.neuralnetwork.imageinference.ui.details.ModelDetails
import com.neuralnetwork.imageinference.ui.details.containers.ModelInputType
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.pytorch.executorch.Module

class PhotoCameraViewModel : ViewModel() {

    private val _details = MutableLiveData<ModelDetails>().apply {
        this.value = ModelDetails(ModelInputType.PHOTO)
    }

    private val _modelSuccess = MutableLiveData<Boolean>()

    private val _detailsViewModel = DetailsViewModel(_details, _modelSuccess)

    @ExperimentalZeroShutterLag
    private val _imageCapture = ImageCapture.Builder()
        .setCaptureMode(ImageCapture.CAPTURE_MODE_ZERO_SHUTTER_LAG)
        .build()

    private val _preview: Preview = Preview.Builder()
        .build()

    private val _cameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

    private val _image = MutableLiveData<Bitmap?>()

    val detailsViewModel get() = _detailsViewModel

    val imageCapture
        @ExperimentalZeroShutterLag
        get() = _imageCapture

    val preview get() = _preview

    val cameraSelector get() = _cameraSelector

    var model: Module? = null

    val image: LiveData<Bitmap?> = _image

    val modelSuccess: LiveData<Boolean> = _modelSuccess

    val onImageCaptureCallback = object : ImageCapture.OnImageCapturedCallback() {
        override fun onCaptureSuccess(image: ImageProxy) {
            super.onCaptureSuccess(image)
            val currentImage = image.toBitmap()
            _image.value = currentImage
            runModel(currentImage)
            image.close()
        }

        override fun onError(exception: ImageCaptureException) {
            super.onError(exception)
            _modelSuccess.value = false
        }
    }

    fun runModel(image: Bitmap) {
        val module: Module? = model
        if (module == null) {
            _modelSuccess.value = false
            return
        }

        val details: ModelDetails? = _details.value
        if (details == null) {
            _modelSuccess.value = false
            return
        }

        viewModelScope.launch(Dispatchers.Default) {
            val executor = ModelExecutor(module, image, details)
            executor.run()
            withContext(Dispatchers.Main) {
                _details.value = executor.details
                _modelSuccess.value = true
            }
        }
    }

    fun clearImage() {
        _image.value = null
    }
}
