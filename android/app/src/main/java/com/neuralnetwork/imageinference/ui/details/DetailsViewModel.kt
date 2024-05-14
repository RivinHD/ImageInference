package com.neuralnetwork.imageinference.ui.details

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel

class DetailsViewModel(modelDetail : MutableLiveData<ModelDetails>? = null) : ViewModel() {

    private val _detail = modelDetail ?: MutableLiveData<ModelDetails>()

    val detail : LiveData<ModelDetails> = _detail

}
