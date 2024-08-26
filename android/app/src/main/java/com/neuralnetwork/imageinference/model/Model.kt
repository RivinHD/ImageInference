/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
 */

package com.neuralnetwork.imageinference.model

import android.content.Context
import android.graphics.Bitmap
import com.neuralnetwork.imageinference.MainActivity
import com.neuralnetwork.imageinference.model.input.Image256244Normalized
import com.neuralnetwork.imageinference.model.output.ImageNet1000AppliedSoftmax
import com.neuralnetwork.imageinference.ui.details.containers.ModelResult
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

/**
 * Container that holds the model related information.
 *
 * @property _file The filename of the model.
 * @property _name The display name of the model.
 * @property _type The underlying type of model.
 * @constructor Creates a model with the given filename, name and type.
 */
data class Model(
    private val _file: String,
    private val _name: String,
    private val _type: ModelType
) {
    /**
     * The executorch model that is loaded from the model file.
     */
    private var module: Module? = null

    /**
     * Gets the name of the model.
     */
    val name get() = _name

    /**
     * Gets the filename of the model.
     */
    val file get() = _file

    /**
     * Runs inference on the model with the given input.
     * The [inDetails] are used as an base information of details and an update version is returned.
     *
     * @param input The input to run the model with.
     * @param inDetails The base details to initialize the new details with.
     * @return The updated details after the model ran.
     */
    fun run(
        input: Any,
        inDetails: ModelDetails
    ): ModelDetails {
        val fixedModule = module ?: return inDetails

        val details = inDetails.copy()
        val moduleInput = processInput(input)

        val startTime: Long = System.nanoTime()
        val output: Tensor = fixedModule.forward(EValue.from(moduleInput))[0].toTensor()
        details.evaluationTimeNano = System.nanoTime() - startTime

        details.results = processOutput(output)
        return details
    }

    /**
     * Loads the model from the assets.
     *
     * @param context The context to load the model from.
     */
    fun load(context: Context){
        module = Module.load(MainActivity.getAsset(context, _file))
    }

    /**
     * Destroys the model and frees the resources.
     */
    fun destroy(){
        module?.destroy()
        module = null
    }

    /**
     * Processes the input of the model and checks the type before it gets passed to the inference.
     *
     * @param input The input to process and transform.
     * @return The transformed input for the inference.
     */
    private fun processInput(input: Any): Tensor {
        return when (_type) {
            ModelType.RESNET50v15 -> {
                if (input !is Bitmap) {
                    throw IllegalArgumentException(
                        "Input is not a Bitmap to transform with Image_256_244_Normalized"
                    )
                }
                Image256244Normalized.transform(input)
            }
        }
    }

    /**
     * Process the output of the models after inference.
     *
     * @param output The output to process and transform.
     * @return The transformed output of the inference.
     */
    private fun processOutput(output: Tensor): Array<ModelResult> {
        return when (_type) {
            ModelType.RESNET50v15 -> {
                ImageNet1000AppliedSoftmax.transform(output)
            }
        }
    }

}
