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

package com.neuralnetwork.imageinference.model.output

import com.neuralnetwork.imageinference.model.ImageNet
import com.neuralnetwork.imageinference.ui.details.containers.ModelResult
import org.pytorch.executorch.Tensor

/**
 * Transformer for the output of the inference of the ImageNet 1000 classes and applies
 * softmax to the output.
 *
 * @constructor Do not initialize this class.
 */
internal class ImageNet1000AppliedSoftmax {
    companion object {
        /**
         * Transforms the output interpreted as ImageNet 1000 classes to a list of ModelResults
         * and applies softmax to the output.
         *
         * @param output The output to transform.
         * @return The transformed output.
         */
        fun transform(output: Tensor): Array<ModelResult> {
            var scores = output.dataAsFloatArray
            if (scores.size > 1000) {
                scores = scores.slice(0..<ImageNet.size).toFloatArray()
            }

            scores = softmax(scores)

            return scores.mapIndexed { index, value -> ModelResult(ImageNet.getClass(index), value) }
                    .toTypedArray()
        }

        /**
         * Applies softmax to the given data.
         *
         * @param data The data to apply softmax to.
         * @return The data with softmax applied.
         */
        private fun softmax(data: FloatArray): FloatArray{
            val max = data.max()
            val exponents = data.map { kotlin.math.exp(it - max) }
            val exponentsSum = exponents.sum()
            return exponents.map { it / exponentsSum }.toFloatArray()
        }
    }
}
