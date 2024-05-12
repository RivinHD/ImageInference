package com.example.imageinference.ui.image

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import kotlin.io.path.Path
import kotlin.io.path.exists
import kotlin.io.path.isRegularFile
import com.example.imageinference.ui.details.DetailsViewModel
import com.example.imageinference.ui.details.ModelDetails
import com.example.imageinference.ui.details.containers.ModelInputType

class ImageViewModel : ViewModel() {

    private val _images = MutableLiveData<List<Image>>()
    private var _selectedImage = MutableLiveData<Image>().apply {
        value = Image.default()
    }
    private val detailsViewModel = DetailsViewModel(ModelDetails(ModelInputType.IMAGE))

    /**
     * The images that are used for inference.
     */
    val images: LiveData<List<Image>> = _images

    /**
     * The current selected image for inference.
     */
    val selectedImage: LiveData<Image> = _selectedImage

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

        val current = _selectedImage.value
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
    fun selectNext(): Boolean {
        val imagesValue = _images.value ?: return false
        val currentIndex = imagesValue.indexOf(_selectedImage.value)
        val nextIndex = (currentIndex + 1) % imagesValue.count()
        _selectedImage.value = _images.value?.get(nextIndex)
        runModel()
        return true
    }


    /**
     * Selects the next image in the list.
     *
     * @return true if a next image was found.
     */
    fun selectBefore(): Boolean {
        val imagesValue = _images.value ?: return false
        val currentIndex = imagesValue.indexOf(_selectedImage.value)
        val beforeIndex = (currentIndex + imagesValue.count() - 1) % imagesValue.count()
        _selectedImage.value = _images.value?.get(beforeIndex)
        runModel()
        return true
    }

    /**
     * Selects the default image.
     */
    fun selectNothing() {
        _selectedImage.value = Image.default()
    }

    /**
     * Get the detail view model from this view model.
     */
    fun getDetailViewModel(): DetailsViewModel {
        return detailsViewModel
    }

    private fun runModel() {
        TODO("Not Implemented")
    }
}
