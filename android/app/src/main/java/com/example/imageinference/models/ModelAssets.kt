package com.example.imageinference.models

import android.content.res.AssetManager
import kotlin.io.path.Path

class ModelAssets(assets: AssetManager) {

    val models: List<String?> = setModelsFromAsset(assets)

    private fun setModelsFromAsset(assets: AssetManager): List<String> {
        val modelList = assets.list("")
            ?.filter { _modelsNames.containsKey(it) }
            ?.map { _modelsNames.getOrDefault(it, it) }

        return if (modelList.isNullOrEmpty()) {
            listOf("None")
        } else {
            modelList;
        }
    }

    companion object {
        private val _modelsNames = mapOf(
            "test.pte" to "test"
        )
        private val _modelsValues = _modelsNames.map { it.value to it.key }.toMap()

        const val default = "None"

        fun getModelName(modelPath: String): String {
            if (modelPath == default) {
                return default
            }
            
            val path = Path(modelPath)
            val modelFileName = path.fileName.toString()
            return _modelsNames.getOrDefault(modelFileName, modelFileName)
        }
    }
}
