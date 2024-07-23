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

package com.neuralnetwork.imageinference.ui.menu.options.benchmark

import com.neuralnetwork.imageinference.ImageCollections
import com.neuralnetwork.imageinference.ui.image.Image

/**
 * Container that holds a collection of benchmark images.
 *
 * @property name The name of the benchmark collection.
 * @property imageList The list of images used to fill the collection.
 * @property isLabeled Whether the images are labeled or not.
 * @constructor Creates an image collection filled with the given images.
 */
data class BenchmarkCollection(
    public val name: String,
    public val imageList: List<Image>,
    public val isLabeled: Boolean = false
) {
    /**
     * Creates an image collection from the corresponding data store object.
     *
     * @param dataStore The data store object to create the image collection from.
     */
    constructor(dataStore: ImageCollections.ImageCollection)
            : this(dataStore.name, dataStore.imageList.map { Image(it) }, false)
}
