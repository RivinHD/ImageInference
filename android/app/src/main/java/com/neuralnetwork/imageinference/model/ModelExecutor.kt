package com.neuralnetwork.imageinference.model

import android.graphics.Bitmap
import com.neuralnetwork.imageinference.ui.details.ModelDetails
import com.neuralnetwork.imageinference.ui.details.containers.ModelResult
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

class ModelExecutor(
    private val module: Module,
    private val image: Bitmap,
    inDetails: ModelDetails
) : Runnable {

    private var _details: ModelDetails = inDetails.copy()

    val details get() = _details

    override fun run() {
        val input: Tensor = TensorImageUtils.bitmapToFloat32Tensor(
            image,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )

        val startTime: Long = System.nanoTime()
        val output: Tensor = module.forward(EValue.from(input))[0].toTensor()
        _details.evaluationTimeNano = System.nanoTime() - startTime

        var scores = output.dataAsFloatArray
        if (scores.size > 1000) {
            scores = scores.slice(0..ImageNet.size).toFloatArray()
        }
        _details.results =
            scores.mapIndexed { index, value -> ModelResult(ImageNet.getClass(index), value) }
                .toTypedArray()

    }
}
