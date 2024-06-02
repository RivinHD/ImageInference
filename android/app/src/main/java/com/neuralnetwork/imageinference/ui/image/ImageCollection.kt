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

/**
 * Container that holds a collection of images.
 *
 * @property imageList The list of images used to fill the collection.
 * @constructor Creates an image collection filled with the given images.
 */
data class ImageCollection(private val _name: String, private val imageList: List<Image>) {

    /**
     * The images of this collection.
     */
    private val _images: MutableList<Image> = imageList.toMutableList()

    /**
     * Gets the images of this collection.
     */
    val images: List<Image> get() = _images

    /**
     * Gets the name of the image collection.
     */
    val name get() = _name

        companion object {
            /**
            * A predefined default image collection.
            *
            * @return The created default image collection.
            */
            val DEFAULT: ImageCollection = ImageCollection("Default", listOf())
        }

        /**
        * Adds an image to the collection.
        *
        * @param image The image to add.
        */
        fun addImage(image: Image) {
            _images.add(image)
        }

        /**
        * Removes an image from the collection.
        *
        * @param image The image to remove.
        */
        fun removeImage(image: Image) {
            if (_images.contains(image)){
                _images.remove(image)
            }
        }
}
