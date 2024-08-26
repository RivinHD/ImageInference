/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
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
