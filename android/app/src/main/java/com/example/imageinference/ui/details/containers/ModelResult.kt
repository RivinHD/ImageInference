package com.example.imageinference.ui.details.containers

data class ModelResult(val name: String, val accuracy: Float){
    companion object{
        fun default(): ModelResult{
            return ModelResult("None", 0.0f)
        }
    }
}
