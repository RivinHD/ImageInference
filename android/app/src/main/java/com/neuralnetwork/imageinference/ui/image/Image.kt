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
