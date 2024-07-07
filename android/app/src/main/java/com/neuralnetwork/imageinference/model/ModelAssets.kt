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

import android.content.pm.ApplicationInfo
import android.content.res.AssetManager
import android.os.Build
import android.system.ErrnoException
import android.system.Os
import android.util.Log

/**
 * Loads the implemented models from the assets.
 *
 * @constructor Creates a new model assets object.
 *
 * @param assets The asset manager to load the models filepath with.
 * @param applicationInfo The application information to get the native library directory.
 */
class ModelAssets(assets: AssetManager, applicationInfo: ApplicationInfo) {
    private val soc = if (Build.VERSION.SDK_INT > Build.VERSION_CODES.S) {
        Build.SOC_MODEL
    } else {
        ""
    }

    init {
        // Sets the library path for the Snapdragon Neural Processing Engine SDK.
        // See https://developer.qualcomm.com/sites/default/files/docs/snpe/dsp_runtime.html for
        // more information.
        try {
            Os.setenv(
                "ADSP_LIBRARY_PATH",
                applicationInfo.nativeLibraryDir,
                true
            )
        } catch (e: ErrnoException) {
            Log.e(
                "Snapdragon Neural Processing Engine SDK",
                "Cannot set ADSP_LIBRARY_PATH",
                e
            )
        }
    }

    /**
     * The implemented models that are available in the app.
     */
    private val _models = listOf(
        // The model is not activated as it does not finish the inference run.
        /*Model("resnet50v15_htp_fp32_$soc.pte",
            "Resnet50v1.5 (HTP, FP32)",
            ModelType.RESNET50v15),*/
        Model("resnet50v15_htp_int8_$soc.pte",
            "Resnet50v1.5 (HTP, Int8)",
            ModelType.RESNET50v15),
        Model("resnet50v15_htp_int16_$soc.pte",
            "Resnet50v1.5 (HTP, Int16)",
            ModelType.RESNET50v15),
        Model("resnet50v15_htp_int16int4_$soc.pte",
            "Resnet50v1.5 (HTP, Int16Int4)",
            ModelType.RESNET50v15),
        Model("resnet50v15_htp_int8int4_$soc.pte",
            "Resnet50v1.5 (HTP, Int8Int4)",
            ModelType.RESNET50v15),
        Model("resnet50v15_xnnpack_fp32.pte",
            "Resnet50v1.5 (CPU, FP32)",
            ModelType.RESNET50v15),
        Model("resnet50v15_xnnpack_int8.pte",
            "Resnet50v1.5 (CPU, Int8)",
            ModelType.RESNET50v15),
    )

    /**
     * Maps the model filename to the display model name.
     */
    private val _modelsNames: Map<String, String> = _models.associate { it.file to it.name }

    /**
     * Maps the model display name to the model filename.
     */
    private val _modelsValues: Map<String, Model> = _models.associateBy { it.name }

    /**
     * Gets the list of models that are implemented an available through the assets.
     */
    val models: List<String> = setModelsFromAsset(assets)

    companion object {
        /**
         * The default value for the model.
         */
        const val DEFAULT: String = "None"
    }

    /**
     * Gets a model by its name.
     *
     * @param name The name of the model to load.
     * @return The loaded model or null if not found.
     */
    fun getModel(name: String): Model? {
        return _modelsValues.getOrDefault(name, null)
    }

    /**
     * List available models from the assets and loads the filepath.
     *
     * @param assets The asset manager to load the models filepath with.
     * @return A list of available models.
     */
    private fun setModelsFromAsset(assets: AssetManager): List<String> {
        val modelList: List<String>? = assets.list("")
            ?.filter { _modelsNames.containsKey(it) }
            ?.map { _modelsNames.getOrDefault(it, it) }

        return if (modelList.isNullOrEmpty()) {
            listOf(DEFAULT)
        } else {
            modelList
        }
    }
}
