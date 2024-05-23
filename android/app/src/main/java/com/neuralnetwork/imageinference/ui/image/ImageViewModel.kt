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
import android.view.View
import android.widget.AdapterView
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.neuralnetwork.imageinference.model.ModelExecutor
import com.neuralnetwork.imageinference.ui.details.DetailsViewModel
import com.neuralnetwork.imageinference.ui.details.ModelDetails
import com.neuralnetwork.imageinference.ui.details.containers.ModelInputType
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.pytorch.executorch.Module
import kotlin.io.path.Path
import kotlin.io.path.exists
import kotlin.io.path.isRegularFile

class ImageViewModel : ViewModel() {

    private val _images = MutableLiveData<List<Image>>()

    private var _selectedImage = MutableLiveData<Image>().apply {
        value = Image.default()
    }

    private val _details = MutableLiveData<ModelDetails>().apply {
        this.value = ModelDetails(ModelInputType.IMAGE)
    }
    private val _modelSuccess = MutableLiveData<Boolean>()

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
            val name = parent?.getItemAtPosition(position) as String
            selectImage(name)
        }

        override fun onNothingSelected(parent: AdapterView<*>?) {
            selectNothing()
        }

    }

    val onBeforeClickListener = View.OnClickListener {
        selectBefore()
    }

    val onNextClickListener = View.OnClickListener {
        selectNext()
    }

    /**
     * Adds an image to the current set for inference.
     * If the image already exists nothing happens.
     * If the name of the image already exists a count is added with syntax (<count>).
     *
     * @param path The path to the image to add.
     * @return true on success.
     */
    fun addImage(path: String): Boolean {
        val parsedPath = Path(path)
        if (!parsedPath.isRegularFile() || !parsedPath.exists()) {
            return false
        }

        if (_images.value?.first { it.path == parsedPath } != null) {
            return false
        }

        var name: String = parsedPath.fileName.toString()
        val imageNames = _images.value?.map { it.name }
        if (imageNames != null) {
            var count = 1
            while (imageNames.contains(name)) {
                name = name.split('(', limit = 2)[0].removeSuffix(" ")
                name += " (${count})"
                count++
            }
        }

        val image = Image(parsedPath, name)
        _images.value = _images.value?.plus(image) ?: listOf(image)
        return true
    }

    /**
     * Removes an image for the current set for inference.
     * If the image doesn't exists in the set nothing happens.
     *
     * @param image The image to remove from the list.
     */
    fun removeImage(image: Image) {
        if (!containsImage(image)) {
            return
        }

        val current: Image? = _selectedImage.value
        if (current != null && image.name == current.name) {
            selectNext()
        }

        _images.value = _images.value?.minus(image)
    }

    /**
     * Checks if the image exists in the current set for inference.
     *
     * @param image The image to check.
     * @return true if the image is contained.
     */
    fun containsImage(image: Image): Boolean {
        return _images.value?.contains(image) ?: false
    }

    /**
     * Changes the selected image and do inference on the current model.
     *
     * @param name The name of the current image.
     */
    fun selectImage(name: String) {
        _selectedImage.value = _images.value?.first { it.name == name }
        runModel()
    }

    /**
     * Selects the next image in the list.
     *
     * @return true if a next image was found.
     */
    private fun selectNext() {
        val imagesValue = _images.value
        if (imagesValue == null)
        {
            _hasNext.value = false
            return
        }

        _hasNext.value = true
        val currentIndex = imagesValue.indexOf(_selectedImage.value)
        val nextIndex = (currentIndex + 1) % imagesValue.count()
        _selectedImage.value = _images.value?.get(nextIndex)
        runModel()
    }


    /**
     * Selects the next image in the list.
     *
     * @return true if a next image was found.
     */
    private fun selectBefore() {
        val imagesValue = _images.value
        if (imagesValue == null){
            _hasBefore.value = false
            return
        }

        _hasBefore.value = true
        val currentIndex = imagesValue.indexOf(_selectedImage.value)
        val beforeIndex = (currentIndex + imagesValue.count() - 1) % imagesValue.count()
        _selectedImage.value = _images.value?.get(beforeIndex)
        runModel()
    }

    /**
     * Selects the default image.
     */
    private fun selectNothing() {
        _selectedImage.value = Image.default()
    }

    private fun runModel() {
        val module: Module? = model
        if (module == null) {
            _modelSuccess.value = false
            return
        }

        val details: ModelDetails? = _details.value
        if (details == null){
            _modelSuccess.value = false
            return
        }

        val path = selectedImage.value?.path?.toAbsolutePath().toString()
        val image: Bitmap = BitmapFactory.decodeFile(path)
        viewModelScope.launch(Dispatchers.Default) {
            val executor = ModelExecutor(module, image, details)
            executor.run()
            withContext(Dispatchers.Main) {
                _details.value = executor.details
                _modelSuccess.value = true
            }
        }
    }
}
