package com.example.imageinference.ui.videoCamera

import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.lifecycle.ViewModel
import com.example.imageinference.ui.details.DetailsViewModel
import com.example.imageinference.ui.details.ModelDetails
import com.example.imageinference.ui.details.containers.ModelInputType

class VideoCameraViewModel : ViewModel(), ImageAnalysis.Analyzer  {

    private val detailsViewModel = DetailsViewModel(ModelDetails(ModelInputType.VIDEO))

    override fun analyze(image: ImageProxy) {
        Log.d("TestAnalyzer", "Image size ${image.height * image.width}")
        runModel(image.toBitmap())
    }

    /**
     * Get the detail view model from this view model.
     */
    fun getDetailViewModel(): DetailsViewModel {
        return detailsViewModel
    }

    private fun runModel(image: Bitmap){
        TODO("Not Implemented")
    }
}
