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
