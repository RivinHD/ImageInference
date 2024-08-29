/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
 */

package com.neuralnetwork.visioninference.ui.image

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.widget.ImageView
import androidx.core.content.res.ResourcesCompat
import com.neuralnetwork.visioninference.ImageCollections
import com.neuralnetwork.visioninference.R

/**
 * Container that represents an image.
 *
 * @property uri The uri to the image.
 * @property name The unique name of the image.
 * @constructor Create an image with the given path and name.
 */
data class Image(val uri: Uri, val name: String) {

    /**
     * Creates an image collection from the corresponding data store object.
     */
    constructor(dataStore: ImageCollections.ImageCollection.Image)
            : this(Uri.parse(dataStore.uri), dataStore.name)

    companion object {
        /**
         * A predefined default image.
         *
         * @return The created default image.
         */
        val DEFAULT: Image = Image(Uri.EMPTY, "DefaultImage")
    }

    /**
     * Gets the bitmap of this image.
     *
     * @return The loaded bitmap.
     */
    fun getBitmap(resolver: ContentResolver): Bitmap {
        return ImageDecoder.createSource(resolver, uri).let {
            ImageDecoder.decodeBitmap(it).copy(Bitmap.Config.ARGB_8888, true)
        }
    }

    /**
     * Loads this image into the give ImageView.
     *
     * @param view The ImageView to load the image into.
     */
    fun loadImageInto(view: ImageView): Bitmap? {
        return if (this == DEFAULT) {
            setDefault(view)
            null
        } else {
            val image = getBitmap(view.context.contentResolver)
            view.setImageBitmap(image)
            image
        }
    }

    fun toDataStore(): ImageCollections.ImageCollection.Image {
        return ImageCollections.ImageCollection.Image.newBuilder()
            .setName(this.name)
            .setUri(this.uri.toString())
            .build()
    }

    /**
     * Sets the default image into the given ImageView.
     *
     * @param view The ImageView to load the default image into.
     */
    private fun setDefault(view: ImageView) {
        view.setImageDrawable(
            ResourcesCompat.getDrawable(
                view.resources,
                R.drawable.ic_image_google,
                null
            )
        )
    }
}
