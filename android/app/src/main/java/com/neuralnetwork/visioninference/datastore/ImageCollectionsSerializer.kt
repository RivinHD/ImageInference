/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
 */

package com.neuralnetwork.visioninference.datastore

import android.content.Context
import androidx.datastore.core.CorruptionException
import androidx.datastore.core.DataStore
import androidx.datastore.core.Serializer
import androidx.datastore.dataStore
import com.google.protobuf.InvalidProtocolBufferException
import com.neuralnetwork.visioninference.ImageCollections
import java.io.InputStream
import java.io.OutputStream

/**
 * Serializer for the image collections data store.
 *
 * @constructor Create an empty image collections serializer.
 */
object ImageCollectionsSerializer : Serializer<ImageCollections> {
    override val defaultValue: ImageCollections = ImageCollections.getDefaultInstance()

    override suspend fun readFrom(input: InputStream): ImageCollections {
        try {
            return ImageCollections.parseFrom(input)
        } catch (exception: InvalidProtocolBufferException) {
            throw CorruptionException("Cannot read proto.", exception)
        }
    }

    override suspend fun writeTo(t: ImageCollections, output: OutputStream) {
        t.writeTo(output)
    }
}

/**
 * Initialize the image collections data store.
 */
val Context.imageCollectionsDataStore: DataStore<ImageCollections> by dataStore(
    fileName = "image_collections.pb",
    serializer = ImageCollectionsSerializer
)
