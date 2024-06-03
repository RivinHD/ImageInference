/*
 *  Copyright (c) 2024 by Vincent Gerlach. All rights reserved.
 *
 *  SPDX-License-Identifier: GPL-3.0-or-later
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  in the root folder of this project with the name LICENSE. If not, see <http://www.gnu.org/licenses/>.
 */

package com.neuralnetwork.imageinference.ui.image

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.widget.ImageView
import androidx.core.content.res.ResourcesCompat
import com.neuralnetwork.imageinference.R

/**
 * Container that represents an image.
 *
 * @property uri The uri to the image.
 * @property name The unique name of the image.
 * @constructor Create an image with the given path and name.
 */
data class Image(val uri: Uri, val name: String) {
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
