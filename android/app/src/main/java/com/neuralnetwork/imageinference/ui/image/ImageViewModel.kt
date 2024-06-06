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
import android.util.Log
import androidx.datastore.core.DataStore
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.neuralnetwork.imageinference.ImageCollections
import com.neuralnetwork.imageinference.datastore.DataStoreViewModel
import com.neuralnetwork.imageinference.model.Model
import com.neuralnetwork.imageinference.model.ModelDetails
import com.neuralnetwork.imageinference.model.ModelState
import com.neuralnetwork.imageinference.ui.details.DetailsViewModel
import com.neuralnetwork.imageinference.ui.details.containers.ModelInputType
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * The view model for the image fragment.
 *
 * @constructor Creates an empty image view model.
 *
 * @param dataStore The stored data that is used for the image collections.
 */
class ImageViewModel(dataStore: DataStore<ImageCollections>) :
    DataStoreViewModel<ImageCollections>(dataStore) {

    /**
     * Holds the image collections that store all images for inference.
     */
    private val _collections = MutableLiveData<MutableList<ImageCollection>>().apply {
        value = mutableListOf(ImageCollection.DEFAULT)
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
                    data = mutableListOf(ImageCollection.DEFAULT)
                    selectedCollectionIndex = 0
                    selectedImageIndex = -1
                } else {
                    selectedCollectionIndex = it.selectedImageCollectionIndex
                    selectedImageIndex =
                        it.imageCollectionList[selectedCollectionIndex].selectedImageIndex
                }
                value = data
                _selectedCollection.value = data[selectedCollectionIndex]
                val images = data[selectedCollectionIndex].images
                _images.value = images
                if (images.isNotEmpty() && selectedImageIndex != -1) {
                    _selectedImage.value = images[selectedImageIndex]
                    _hasNext.value = images.size > 1
                    _hasBefore.value = images.size > 1
                }

            }
        }
    }

    /**
     * Holds the selected collection.
     */
    private val _selectedCollection = MutableLiveData<ImageCollection>().apply {
        value = ImageCollection.DEFAULT
    }

    /**
     * Holds the images of the selected collection.
     */
    private val _images = MutableLiveData<List<Image>>()

    /**
     * Holds the selected image for inference.
     */
    private var _selectedImage = MutableLiveData<Image>().apply {
        value = Image.DEFAULT
    }

    /**
     * Holds the details that are update by the model and displayed by the details fragment.
     */
    private val _details = MutableLiveData<ModelDetails>().apply {
        this.value = ModelDetails(ModelInputType.IMAGE)
    }

    /**
     * Holds the state of the model.
     */
    private val _modelState = MutableLiveData<ModelState>().apply {
        value = ModelState.INITIAL
    }

    /**
     * Holds the details view model that is provided to the details fragment.
     */
    private val _detailsViewModel = DetailsViewModel(_details, _modelState)

    /**
     * Holds the state if a next image is available.
     */
    private val _hasNext = MutableLiveData<Boolean>().apply {
        value = false
    }

    /**
     * Holds the state if a before image is available.
     */
    private val _hasBefore = MutableLiveData<Boolean>().apply {
        value = false
    }

    /**
     * Gets the detail view model for the details fragment.
     */
    val detailsViewModel get() = _detailsViewModel

    /**
     * Gets or sets the model for the inference.
     */
    var model: Model? = null

    /**
     * Gets the images that are used for inference.
     */
    val images: LiveData<List<Image>> = _images

    /**
     * Gets the current selected image for inference.
     */
    val selectedImage: LiveData<Image> = _selectedImage

    /**
     * Gets the has next state.
     */
    val hasNext: LiveData<Boolean> = _hasNext

    /**
     * Gets the has before state.
     */
    val hasBefore: LiveData<Boolean> = _hasBefore

    /**
     * Gets the state of the model.
     */
    val modelState: LiveData<ModelState> = _modelState

    /**
     * Callback how the model change is handled.
     */
    val onModelChangedCallback : ((Model?) -> Unit) = {
        model = it
        _modelState.value = ModelState.INITIAL
        _selectedImage.value = _selectedImage.value
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
        var image: Image = Image.DEFAULT
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

        if (image == Image.DEFAULT) {
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

    /**
     * Get the name of the collections.
     *
     * @return The names of the collections.
     */
    fun getCollectionsNames(): List<String> {
        return _collections.value?.map { it.name } ?: emptyList()
    }

    /**
     * Get the selected collection index.
     *
     * @return The index of the selected collection.
     */
    fun getSelectedCollectionIndex(): Int {
        return _collections.value?.indexOf(_selectedCollection.value) ?: 0
    }

    /**
     * Changes the selected collection.
     * This also updates the available images and the selected image.
     * Nothing happens if the collection doesn't exists.
     *
     * @param name The name of the collection to change to.
     */
    fun changeCollection(name: String) {
        val collection = _collections.value?.first { it.name == name }
        if (collection == null) {
            return
        }

        _selectedCollection.value = collection
        _images.value = collection.images
        _selectedImage.value = collection.images.firstOrNull() ?: Image.DEFAULT
    }

    /**
     * Add a collection to the list of collections and select this collection.
     * If the name of the collection already exists a count is added with syntax (<count>).
     *
     * @param name The name of the new collection.
     */
    fun addCollection(name: String) {
        val collections = _collections.value ?: return

        val collectionNames = collections.map { it.name }
        var newName = name
        var count = 1
        while (collectionNames.contains(newName)) {
            newName = newName.split('(', limit = 2)[0].removeSuffix(" ")
            newName += " (${count})"
            count++
        }

        val new = ImageCollection(newName, listOf())
        collections.add(new)
        _collections.value = collections
        _selectedCollection.value = new
        _images.value = new.images
        _selectedImage.value = Image.DEFAULT

    }

    /**
     * Removes the selected collection and selects the first collection.
     */
    fun removeCollection() {
        val collection = getSelectedCollection()
        val collections = _collections.value ?: return

        if (collections.size == 1 || collection == ImageCollection.DEFAULT) {
            return
        }

        collections.remove(collection)
        _collections.value = collections
        _selectedCollection.value = collections.first()
        _images.value = collections.first().images
        _selectedImage.value = Image.DEFAULT
    }

    /**
     * Selects the next image in the list.
     *
     * @return true if a next image was found.
     */
    fun selectNext() {
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
    fun selectBefore() {
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
     * Run the current model on the selected image.
     *
     * @param resolver The content resolver to get the bitmap from the uri.
     */
    fun runModel(resolver: ContentResolver) {
        if (_modelState.value == ModelState.RUNNING){
            Log.d("Image", "The model is already running.")
            return
        }

        Log.d("Image", "Try running the model.")
        val fixedModel: Model? = model
        if (fixedModel == null) {
            Log.d("Image", "Failed to get the model.")
            _modelState.value = ModelState.FAILED
            return
        }

        val details: ModelDetails? = _details.value
        if (details == null) {
            Log.d("Image", "Failed to get the details.")
            _modelState.value = ModelState.FAILED
            return
        }

        val image = selectedImage.value
        if (image == null || image == Image.DEFAULT) {
            Log.d("Image", "Failed to get the image.")
            _modelState.value = ModelState.FAILED
            return
        }


        Log.d("VideoCapture", "Running the model.")
        val bitmap: Bitmap = image.getBitmap(resolver)
        _modelState.value = ModelState.RUNNING
        viewModelScope.launch(Dispatchers.Default) {
            val outputDetails = fixedModel.run(bitmap, details)
            withContext(Dispatchers.Main) {
                _details.value = outputDetails
                _modelState.value = ModelState.SUCCESS
            }
        }
    }

    /**
     * Saved the image collections to the system using data store.
     *
     * @param dataStore The data store object to write to.
     */
    fun saveDatastore(dataStore: DataStore<ImageCollections>) {
        viewModelScope.launch(Dispatchers.IO) {
            dataStore.updateData { data ->
                data.toBuilder()
                    .clearImageCollection()
                    .addAllImageCollection(_collections.value?.map { collection ->
                        ImageCollections.ImageCollection.newBuilder()
                            .setName(collection.name)
                            .addAllImage(collection.images.map { image ->
                                ImageCollections.ImageCollection.Image.newBuilder()
                                    .setName(image.name)
                                    .setUri(image.uri.toString())
                                    .build()
                            })
                            .build()
                    } ?: listOf(
                        ImageCollections.ImageCollection.newBuilder()
                            .setName(ImageCollection.DEFAULT.name)
                            .build()
                    ))
                    .setSelectedImageCollectionIndex(
                        getSelectedCollectionIndex()
                    )
                    .build()
            }
        }
    }

    /**
     * Gets the selected collection.
     * If no collection is selected the first or default collection is selected.
     */
    private fun getSelectedCollection(): ImageCollection {
        val selectedCollection = _selectedCollection.value
        if (selectedCollection == null) {
            val collections = _collections.value
            val selection = collections?.first() ?: ImageCollection.DEFAULT
            _selectedCollection.value = selection
            return selection
        }
        return selectedCollection
    }
}
