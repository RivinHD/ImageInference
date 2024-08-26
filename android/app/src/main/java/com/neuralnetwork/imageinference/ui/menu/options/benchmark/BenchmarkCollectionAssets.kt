/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
 */

package com.neuralnetwork.imageinference.ui.menu.options.benchmark

import android.content.Context
import android.content.res.AssetManager
import android.net.Uri
import android.provider.ContactsContract.Directory
import androidx.core.net.toUri
import com.neuralnetwork.imageinference.MainActivity
import com.neuralnetwork.imageinference.model.ImageNet
import com.neuralnetwork.imageinference.ui.image.Image
import java.io.File

/**
 * Loads the existing collections from the assets.
 *
 * @constructor Creates a new model assets object.
 *
 * @param context The current context to use.
 */
class BenchmarkCollectionAssets(context: Context) {

    private val _collections = mutableListOf<BenchmarkCollection>()

    private val _collectionsPaths = listOf(
        "labeled_collections/imagenet_2012"
    )

    val collections get() = _collections

    init {
        _collectionsPaths.forEach {
            // Load the collection from the assets.
            val images = mutableListOf<Image>()

            context.assets.list(it)?.forEach { fileName ->
                val file = File(MainActivity.getAsset(context, File(it, fileName).path))
                if (file.isFile) {
                    val classIndex = file.name.split("_")[0].toInt()
                    val className = ImageNet.getClass(classIndex)
                    val image = Image(file.toUri(), className)
                    images.add(image)
                }
            }

            val collection = BenchmarkCollection(it, images, true)
            _collections.add(collection)
        }
    }
}
