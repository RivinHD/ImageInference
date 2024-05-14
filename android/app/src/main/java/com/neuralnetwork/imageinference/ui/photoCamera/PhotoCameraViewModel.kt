package com.neuralnetwork.imageinference.ui.photoCamera

import android.graphics.Bitmap
import androidx.lifecycle.ViewModel
import com.neuralnetwork.imageinference.ui.details.DetailsViewModel
import com.neuralnetwork.imageinference.ui.details.ModelDetails
import com.neuralnetwork.imageinference.ui.details.containers.ModelInputType

class PhotoCameraViewModel : ViewModel() {

    private val detailsViewModel = DetailsViewModel(ModelDetails(ModelInputType.PHOTO))

    /**
     * Get the detail view model from this view model.
     */
    fun getDetailViewModel(): DetailsViewModel {
        return detailsViewModel
    }

    fun runModel(image: Bitmap){
        // TODO("Not Implemented")
    }
}
