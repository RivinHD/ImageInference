package com.neuralnetwork.imageinference.model

import org.pytorch.executorch.Module

interface ModelConnector {

    fun getModel(): Module?
    fun getModelName(): String
}
