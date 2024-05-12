package com.example.imageinference.ui.photoCamera

import android.graphics.Bitmap
import androidx.lifecycle.ViewModel
import com.example.imageinference.ui.details.DetailsViewModel
import com.example.imageinference.ui.details.ModelDetails
import com.example.imageinference.ui.details.containers.ModelInputType

class PhotoCameraViewModel : ViewModel() {

    private val detailsViewModel = DetailsViewModel(ModelDetails(ModelInputType.PHOTO))

    /**
     * Get the detail view model from this view model.
     */
    fun getDetailViewModel(): DetailsViewModel {
        return detailsViewModel
    }

    fun runModel(image: Bitmap){
        TODO("Not Implemented")
    }
}
