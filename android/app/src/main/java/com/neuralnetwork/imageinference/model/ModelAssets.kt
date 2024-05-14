package com.neuralnetwork.imageinference.model

import android.content.res.AssetManager
import kotlin.io.path.Path

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
            "test.pte" to "test"
        )
        private val _modelsValues: Map<String, String> =
            _modelsNames.map { it.value to it.key }.toMap()

        const val Default: String = "None"

        fun getModelName(fileName: String): String {
            if (fileName == Default) {
                return Default
            }

            val path = Path(fileName)
            val modelFileName: String = path.fileName.toString()
            return _modelsNames.getOrDefault(modelFileName, modelFileName)
        }

        fun getModelFile(name: String): String? {
            return _modelsValues.getOrDefault(name, null)
        }
    }
}
