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
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import com.neuralnetwork.imageinference.model.ModelExecutor
import com.neuralnetwork.imageinference.ui.details.DetailsViewModel
import com.neuralnetwork.imageinference.ui.details.ModelDetails
import com.neuralnetwork.imageinference.ui.details.containers.ModelInputType
import org.pytorch.executorch.Module

class VideoCameraViewModel : ViewModel(), ImageAnalysis.Analyzer {
    private val _details = MutableLiveData<ModelDetails>().apply {
        this.value = ModelDetails(ModelInputType.VIDEO)
    }
    private val _detailsViewModel = DetailsViewModel(_details)

    val detailsViewModel get() = _detailsViewModel

    var model: Module? = null

    override fun analyze(image: ImageProxy) {
        Log.d("TestAnalyzer", "Image size ${image.height * image.width}")
        runModel(image.toBitmap())
    }

    private fun runModel(image: Bitmap) {
        val module: Module = model ?: return
        val details: ModelDetails = _details.value ?: return
        val executor = ModelExecutor(module, image, details)
    }
}
