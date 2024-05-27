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

package com.neuralnetwork.imageinference.model

import android.graphics.Bitmap
import androidx.core.graphics.scale
import com.neuralnetwork.imageinference.ui.details.ModelDetails
import com.neuralnetwork.imageinference.ui.details.containers.ModelResult
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

class ModelExecutor(
    private val module: Module,
    private val image: Bitmap,
    inDetails: ModelDetails
) : Runnable {

    private var _details: ModelDetails = inDetails.copy()

    val details get() = _details

    override fun run() {

        // Based on https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
        // ResNet50_Weights.IMAGENET1K_V2
        val input: Tensor = TensorImageUtils.bitmapToFloat32Tensor(
            image.scale(256, 256, true),
            16, // 256 - 224 = 32
            16, // To center we have an offset of 16 pixels
            224,
            224,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )

        val startTime: Long = System.nanoTime()
        val output: Tensor = module.forward(EValue.from(input))[0].toTensor()
        _details.evaluationTimeNano = System.nanoTime() - startTime

        var scores = output.dataAsFloatArray
        if (scores.size > 1000) {
            scores = scores.slice(0..<ImageNet.size).toFloatArray()
        }
        _details.results =
            scores.mapIndexed { index, value -> ModelResult(ImageNet.getClass(index), value) }
                .toTypedArray()

    }
}
