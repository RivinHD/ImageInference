package com.neuralnetwork.imageinference.ui.details.containers

data class ModelResult(val name: String, val accuracy: Float){
    companion object{

        val Default = ModelResult("None", 0.0f)
    }
}
