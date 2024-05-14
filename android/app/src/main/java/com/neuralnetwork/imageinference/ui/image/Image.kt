package com.neuralnetwork.imageinference.ui.image

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.widget.ImageView
import androidx.core.content.res.ResourcesCompat
import com.neuralnetwork.imageinference.R
import java.nio.file.Path

data class Image(val path: Path, val name: String) {

    companion object {
        /**
         * A predefined default image.
         *
         * @return The created default image.
         */
        fun default(): Image = Image(kotlin.io.path.Path(""), "DefaultImage")
    }

    /**
     * Loads this image into the give ImageView.
     *
     * @param view The ImageView to load the image into.
     */
    fun loadImageInto(view: ImageView): Bitmap? {
        return if (this == default()) {
            view.setImageDrawable(
                ResourcesCompat.getDrawable(
                    view.resources,
                    R.drawable.ic_image_google,
                    null
                )
            )
            null
        } else {
            val imageFile: Bitmap = BitmapFactory.decodeFile(path.toAbsolutePath().toString())
            view.setImageBitmap(imageFile)
            imageFile
        }
    }
}
