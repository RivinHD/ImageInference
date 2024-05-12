package com.example.imageinference.ui.details

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel

class DetailsViewModel(modelDetail : ModelDetails? = null) : ViewModel() {

    private val _detail = MutableLiveData<ModelDetails>().apply {
        this.value = modelDetail
    }

    val detail : LiveData<ModelDetails> = _detail

}
