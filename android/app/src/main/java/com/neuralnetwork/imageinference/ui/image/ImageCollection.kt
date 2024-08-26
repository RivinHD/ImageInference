/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
 */

package com.neuralnetwork.imageinference.ui.image

import com.neuralnetwork.imageinference.ImageCollections

/**
 * Container that holds a collection of images.
 *
 * @property imageList The list of images used to fill the collection.
 * @constructor Creates an image collection filled with the given images.
 */
data class ImageCollection(val name: String, private val imageList: List<Image>) {

    /**
     * Creates an image collection from the corresponding data store object.
     *
     * @param dataStore The data store object to create the image collection from.
     */
    constructor(dataStore: ImageCollections.ImageCollection)
            : this(dataStore.name, dataStore.imageList.map { Image(it) })

    /**
     * The images of this collection.
     */
    private val _images: MutableList<Image> = imageList.toMutableList()

    /**
     * Gets the images of this collection.
     */
    val images: List<Image> get() = _images

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
        if (_images.contains(image)) {
            _images.remove(image)
        }
    }

    fun toDataStore(): ImageCollections.ImageCollection {
        return ImageCollections.ImageCollection.newBuilder()
            .setName(this.name)
            .addAllImage(this.images.map { it.toDataStore() })
            .build()
    }
}
