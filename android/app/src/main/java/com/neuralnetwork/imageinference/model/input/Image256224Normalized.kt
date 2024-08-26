/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
 */

package com.neuralnetwork.imageinference.model.input

import android.graphics.Bitmap
import androidx.core.graphics.scale
import com.neuralnetwork.imageinference.model.TensorImageUtils
import org.pytorch.executorch.Tensor

/**
 * Transformer for the inference input of a image where it process in the steps:
 * 1. Resize the image to 256x256.
 * 2. Center crop the image to 224x224.
 * 3. Normalize the image with the predefined mean and standard deviation.
 *
 * @constructor Do not initialize this class.
 */
internal class Image256224Normalized {
    companion object {

        /**
         * Transforms the input image to a tensor that can be used for inference.
         * The following steps are applied:
         * 1. Resize the image to 256x256.
         * 2. Center crop the image to 224x224.
         * 3. Normalize the image with the predefined mean and standard deviation.
         *
         * @param image The image to transform.
         * @return The transformed image as tensor.
         */
        fun transform(image: Bitmap): Tensor {
            // Based on https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch
            // and https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
            val transformed: Tensor = TensorImageUtils.bitmapToFloat32Tensor(
                image.scale(256, 256, true),
                16, // 256 - 224 = 32
                16, // To center we have an offset of 16 pixels
                224,
                224,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
            )

            return transformed
        }
    }
}
