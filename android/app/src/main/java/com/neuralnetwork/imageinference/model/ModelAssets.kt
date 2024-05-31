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

import android.content.res.AssetManager
import kotlin.io.path.Path
import kotlin.io.path.name

class ModelAssets(assets: AssetManager) {

    val models: List<String> = setModelsFromAsset(assets)

    private fun setModelsFromAsset(assets: AssetManager): List<String> {
        val modelList: List<String>? = assets.list("")
            ?.filter { _modelsNames.containsKey(it) }
            ?.map { _modelsNames.getOrDefault(it, it) }

        return if (modelList.isNullOrEmpty()) {
            listOf("None")
        } else {
            modelList
        }
    }

    companion object {
        private val _modelsNames: Map<String, String> = mapOf(
            // "filename" to "display name"
            "resnet50v15_htp_fp32.pte" to "Resnet50v1.5 (HTP, FP32)",
            "resnet50v15_htp_q8.pte" to "Resnet50v1.5 (HTP, Int8)",
            "resnet50v15_xnnpack_fp32.pte" to "Resnet50v1.5 (CPU, FP32)",
            "resnet50v15_xnnpack_q8.pte" to "Resnet50v1.5 (CPU, Int8)"
        )
        private val _modelsValues: Map<String, String> =
            _modelsNames.map { it.value to it.key }.toMap()

        const val DEFAULT: String = "None"

        fun getModelName(fileName: String): String {
            if (fileName == DEFAULT) {
                return DEFAULT
            }

            val path = Path(fileName)
            val modelFileName: String = path.name
            return _modelsNames.getOrDefault(modelFileName, modelFileName)
        }

        fun getModelFile(name: String): String? {
            return _modelsValues.getOrDefault(name, null)
        }
    }
}
