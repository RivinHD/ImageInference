/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
 */

package com.neuralnetwork.visioninference.model.output

import com.neuralnetwork.visioninference.model.ImageNet
import com.neuralnetwork.visioninference.ui.details.containers.ModelResult
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
