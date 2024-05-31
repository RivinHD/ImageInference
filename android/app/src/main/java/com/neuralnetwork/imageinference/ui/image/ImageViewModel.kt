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
import android.net.Uri
import android.view.View
import android.widget.AdapterView
import androidx.datastore.core.DataStore
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.neuralnetwork.imageinference.ImageCollections
import com.neuralnetwork.imageinference.datastore.DataStoreViewModel
import com.neuralnetwork.imageinference.model.ModelExecutor
import com.neuralnetwork.imageinference.ui.details.DetailsViewModel
import com.neuralnetwork.imageinference.ui.details.ModelDetails
import com.neuralnetwork.imageinference.ui.details.ModelState
import com.neuralnetwork.imageinference.ui.details.containers.ModelInputType
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.pytorch.executorch.Module

class ImageViewModel(dataStore: DataStore<ImageCollections>) :
    DataStoreViewModel<ImageCollections>(dataStore) {

    private var inferencedImage: Image = Image.default()

    private val _collections = MutableLiveData<MutableList<ImageCollection>>().apply {
        value = mutableListOf(ImageCollection.default())
        viewModelScope.launch {
            dataStore.data.collect {
                var data = MutableList(it.imageCollectionCount) { collectionIndex ->
                    val collection = it.imageCollectionList[collectionIndex]
                    ImageCollection(collection.name, List(collection.imageCount) { imageIndex ->
                        val image = collection.imageList[imageIndex]
                        Image(Uri.parse(image.uri), image.name)
                    })
                }
                val selectedCollectionIndex: Int
                val selectedImageIndex: Int
                if (data.isEmpty()) {
                    data = mutableListOf(ImageCollection.default())
                    selectedCollectionIndex = 0
                    selectedImageIndex = -1
                } else {
                    selectedCollectionIndex = it.selectedImageCollectionIndex
                    selectedImageIndex =
                        it.imageCollectionList[selectedCollectionIndex].selectedImageIndex
                }
                value = data
                _selectedCollection.value = data[selectedCollectionIndex]
                _images.value = data[selectedCollectionIndex].images
                if (selectedImageIndex != -1) {
                    _selectedImage.value = data[selectedCollectionIndex].images[selectedImageIndex]
                }

            }
        }
    }

    private val _selectedCollection = MutableLiveData<ImageCollection>().apply {
        value = ImageCollection.default()
    }

    private val _images = MutableLiveData<List<Image>>()

    private var _selectedImage = MutableLiveData<Image>().apply {
        value = Image.default()
    }

    private val _details = MutableLiveData<ModelDetails>().apply {
        this.value = ModelDetails(ModelInputType.IMAGE)
    }
    private val _modelSuccess = MutableLiveData<ModelState>().apply {
        value = ModelState.INITIAL
    }

    private val _detailsViewModel = DetailsViewModel(_details, _modelSuccess)

    private val _hasNext = MutableLiveData<Boolean>().apply {
        value = false
    }
    private val _hasBefore = MutableLiveData<Boolean>().apply {
        value = false
    }

    val detailsViewModel get() = _detailsViewModel

    var model: Module? = null

    /**
     * The images that are used for inference.
     */
    val images: LiveData<List<Image>> = _images

    /**
     * The current selected image for inference.
     */
    val selectedImage: LiveData<Image> = _selectedImage

    val hasNext: LiveData<Boolean> = _hasNext

    val hasBefore: LiveData<Boolean> = _hasBefore

    val onImageSelectedListener = object : AdapterView.OnItemSelectedListener {
        override fun onItemSelected(
            parent: AdapterView<*>?,
            view: View?,
            position: Int,
            id: Long
        ) {
            val image = parent?.getItemAtPosition(position) as Image
            selectImage(image.name)
        }

        override fun onNothingSelected(parent: AdapterView<*>?) {
            selectNothing()
        }

    }

    /**
     * OnClickListener that selects the before image.
     */
    val onBeforeClickListener = View.OnClickListener {
        selectBefore()
    }

    /**
     * OnClickListener that selects the next image.
     */
    val onNextClickListener = View.OnClickListener {
        selectNext()
    }

    /**
     * Adds an image to the current set for inference.
     * If the image already exists nothing happens.
     * If the name of the image already exists a count is added with syntax (<count>).
     *
     * @param uris The uri of the images to add.
     * @return true on success.
     */
    fun addImages(uris: List<Uri>): Boolean {
        val selectedCollection = getSelectedCollection()
        var image: Image = Image.default()
        for (uri in uris) {
            val images = _images.value ?: continue
            if (images.any { it.uri == uri }) {
                continue
            }


            var name: String = uri.lastPathSegment ?: "Image"
            val imageNames = images.map { it.name }
            var count = 1
            while (imageNames.contains(name)) {
                name = name.split('(', limit = 2)[0].removeSuffix(" ")
                name += " (${count})"
                count++
            }

            image = Image(uri, name)
            selectedCollection.addImage(image)
        }

        if (image == Image.default()) {
            return false
        }

        _images.value = selectedCollection.images
        _selectedImage.value = image
        return true
    }

    /**
     * Removes an image for the current set for inference.
     * If the image doesn't exists in the set nothing happens.
     *
     * @param image The image to remove from the list.
     */
    fun removeImage(image: Image) {
        val selectedCollection = getSelectedCollection()
        if (!selectedCollection.images.contains(image)) {
            return
        }

        val current: Image? = _selectedImage.value
        if (current != null && image.name == current.name) {
            selectNext()
        }

        selectedCollection.removeImage(image)
        _images.value = selectedCollection.images
    }


    /**
     * Changes the selected image and do inference on the current model.
     *
     * @param name The name of the current image.
     */
    fun selectImage(name: String) {
        _selectedImage.value = _images.value?.first { it.name == name }
    }

    fun getCollectionsNames(): List<String> {
        return _collections.value?.map { it.name } ?: emptyList()
    }

    fun getSelectedCollectionIndex(): Int {
        return _collections.value?.indexOf(getSelectedCollection()) ?: -1
    }

    fun changeCollection(name: String) {
        val collection = _collections.value?.first { it.name == name }
        if (collection != null) {
            _selectedCollection.value = collection
            _images.value = collection.images
            _selectedImage.value = collection.images.firstOrNull()
        }
    }

    fun addCollection(name: String) {
        val collections = _collections.value ?: return
        val new = ImageCollection(name, listOf())
        collections.add(new)
        _collections.value = collections
        _selectedCollection.value = new
        _images.value = new.images
        _selectedImage.value = Image.default()

    }

    fun removeCollection() {
        val collection = getSelectedCollection()
        val collections = _collections.value ?: return

        if (collections.size == 1) {
            return
        }

        collections.remove(collection)
        _collections.value = collections
        _selectedCollection.value = collections.first()
        _images.value = collections.first().images
        _selectedImage.value = Image.default()
    }

    private fun getSelectedCollection(): ImageCollection {
        val selectedCollection = _selectedCollection.value ?: ImageCollection.default()
        if (selectedCollection == ImageCollection.default()) {
            _selectedCollection.value = selectedCollection
        }
        return selectedCollection
    }

    /**
     * Selects the next image in the list.
     *
     * @return true if a next image was found.
     */
    private fun selectNext() {
        val imagesValue = _images.value
        if (imagesValue == null) {
            _hasNext.value = false
            return
        }

        _hasNext.value = true
        val currentIndex = imagesValue.indexOf(_selectedImage.value)
        val nextIndex = (currentIndex + 1) % imagesValue.size
        _selectedImage.value = _images.value?.get(nextIndex)
    }


    /**
     * Selects the next image in the list.
     *
     * @return true if a next image was found.
     */
    private fun selectBefore() {
        val imagesValue = _images.value
        if (imagesValue == null) {
            _hasBefore.value = false
            return
        }

        _hasBefore.value = true
        var currentIndex = imagesValue.indexOf(_selectedImage.value)
        if (currentIndex == -1) {
            currentIndex = 0
        }
        val beforeIndex = (currentIndex + imagesValue.size - 1) % imagesValue.size
        _selectedImage.value = _images.value?.get(beforeIndex)
    }

    /**
     * Selects the default image.
     */
    private fun selectNothing() {
        _selectedImage.value = Image.default()
    }

    fun runModel(resolver: ContentResolver) {
        val module: Module? = model
        if (module == null) {
            _modelSuccess.value = ModelState.FAILED
            return
        }

        val details: ModelDetails? = _details.value
        if (details == null) {
            _modelSuccess.value = ModelState.FAILED
            return
        }

        val image = selectedImage.value
        if (image == null || image == Image.default()) {
            _modelSuccess.value = ModelState.FAILED
            return
        }

        if (image == inferencedImage) {
            _modelSuccess.value = ModelState.FAILED
            return
        }

        inferencedImage = image

        val bitmap: Bitmap = image.getBitmap(resolver)
        _modelSuccess.value = ModelState.RUNNING
        viewModelScope.launch(Dispatchers.Default) {
            val executor = ModelExecutor(module, bitmap, details)
            executor.run()
            withContext(Dispatchers.Main) {
                _details.value = executor.details
                _modelSuccess.value = ModelState.SUCCESS
            }
        }
    }
}
