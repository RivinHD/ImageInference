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
import com.neuralnetwork.imageinference.model.ModelExecutor
import com.neuralnetwork.imageinference.ui.details.DetailsViewModel
import com.neuralnetwork.imageinference.ui.details.ModelDetails
import com.neuralnetwork.imageinference.ui.details.ModelState
import com.neuralnetwork.imageinference.ui.details.containers.ModelInputType
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.pytorch.executorch.Module
import java.util.concurrent.Executor

class VideoCameraViewModel : ViewModel(), ImageAnalysis.Analyzer {
    private val _details = MutableLiveData<ModelDetails>().apply {
        value = ModelDetails(ModelInputType.VIDEO)
    }

    private val _modelSuccess = MutableLiveData<ModelState>()

    private val _isRecording = MutableLiveData<Boolean>().apply {
        value = false
    }

    private val _detailsViewModel = DetailsViewModel(_details, _modelSuccess)

    private val _resolutionSelector = ResolutionSelector.Builder()
        .setResolutionStrategy(
            ResolutionStrategy(
                Size(256, 256),
                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER
            )
        )
        .build()

    private val _imageAnalysis = ImageAnalysis.Builder()
        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .setResolutionSelector(_resolutionSelector)
        .build()

    private val _cameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

    val detailsViewModel get() = _detailsViewModel

    var model: Module? = null

    val imageAnalysis get() = _imageAnalysis
    val cameraSelector get() = _cameraSelector

    val isRecording: LiveData<Boolean> = _isRecording

    override fun analyze(image: ImageProxy) {
        Log.d("TestAnalyzer", "Image size ${image.width} x ${image.height}")
        runModel(image.toBitmap())

        // done, release the ImageProxy object
        image.close()
    }

    private fun runModel(image: Bitmap) {
        val module: Module? = model
        if (module == null) {
            _modelSuccess.value = ModelState.FAILED
            return
        }

        val details: ModelDetails? = _details.value
        if (details == null){
            _modelSuccess.value = ModelState.FAILED
            return
        }

        _modelSuccess.value = ModelState.RUNNING
        viewModelScope.launch(Dispatchers.Default) {
            val executor = ModelExecutor(module, image, details)
            executor.run()
            withContext(Dispatchers.Main) {
                _details.value = executor.details
                _modelSuccess.value = ModelState.SUCCESS
            }
        }
    }

    fun startRecording(executor: Executor) {
        _isRecording.value = true
        imageAnalysis.setAnalyzer(executor, this)
    }

    fun stopRecording() {
        _isRecording.value = false
        imageAnalysis.clearAnalyzer()
    }
}
