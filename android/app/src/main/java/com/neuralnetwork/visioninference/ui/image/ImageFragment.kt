/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
 */

package com.neuralnetwork.visioninference.ui.image

import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.view.LayoutInflater
import android.view.MenuInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.EditText
import android.widget.PopupMenu
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.widget.ListPopupWindow
import androidx.core.content.ContextCompat
import androidx.core.graphics.drawable.DrawableCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.neuralnetwork.visioninference.MainActivity
import com.neuralnetwork.visioninference.R
import com.neuralnetwork.visioninference.databinding.FragmentImageBinding
import com.neuralnetwork.visioninference.datastore.DataStoreViewModelFactory
import com.neuralnetwork.visioninference.datastore.imageCollectionsDataStore
import com.neuralnetwork.visioninference.model.ModelConnector
import com.neuralnetwork.visioninference.model.ModelState
import com.neuralnetwork.visioninference.ui.details.DetailsConnector
import com.neuralnetwork.visioninference.ui.details.DetailsViewModel

/**
 * Fragment that uses an image for the model input.
 *
 * @constructor Create empty image fragment.
 */
class ImageFragment : Fragment(), DetailsConnector {
    /**
     * The binding that gets updated from the fragment.
     */
    private var _binding: FragmentImageBinding? = null

    /**
     * The binding that holds the view of this fragment.
     * This property is only valid between onCreateView and onDestroyView.
     */
    private val binding get() = _binding!!

    /**
     * The binding that holds the view of this fragment.
     * This property is only valid between onCreateView and onDestroyView.
     */
    private lateinit var _context: Context

    override fun onAttach(context: Context) {
        super.onAttach(context)
        _context = context
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val vm: ImageViewModel by viewModels {
            DataStoreViewModelFactory(_context.imageCollectionsDataStore)
        }
        _binding = FragmentImageBinding.inflate(inflater, container, false)
        val root: View = binding.root
        val modelConnector = (activity as ModelConnector)
        vm.model = modelConnector.getModel()
        modelConnector.setOnModelChangedListener(vm.onModelChangedCallback)

        val imageSelectionPopup = setupImageSelection(vm)
        setupImageSelectionBefore(vm)
        setupImageSelectionNext(vm)

        setupImageSelectionMenu(vm)

        observeImages(vm, imageSelectionPopup)
        observeSelectedImage(vm)
        observeHasNext(vm)
        observeHasBefore(vm)
        observeModelState(vm)

        return root
    }

    /**
     * Setup the observe on the view model property hasBefore LiveData.
     *
     * @param vm The view model of this fragment.
     */
    private fun observeHasBefore(
        vm: ImageViewModel
    ) {
        val imageSelectionBefore = binding.imageSelectionBefore

        vm.hasBefore.observe(viewLifecycleOwner) {
            imageSelectionBefore.isEnabled = it
        }
    }

    /**
     * Setup the observe on the view model property hasNext LiveData.
     *
     * @param vm The view model of this fragment.
     */
    private fun observeHasNext(
        vm: ImageViewModel
    ) {
        val imageSelectionNext = binding.imageSelectionNext

        vm.hasNext.observe(viewLifecycleOwner) {
            imageSelectionNext.isEnabled = it
        }
    }

    /**
     * Setup the observe on the view model property selectedImage LiveData.
     *
     * @param vm The view model of this fragment.
     */
    private fun observeSelectedImage(
        vm: ImageViewModel
    ) {
        val inferenceImage = binding.visionInferenceImage

        vm.selectedImage.observe(viewLifecycleOwner) {
            if (it == Image.DEFAULT) {
                inferenceImage.setImageResource(R.drawable.ic_image_google)
                return@observe
            }
            it.loadImageInto(inferenceImage)
            vm.runModel(_context.contentResolver)
        }
    }

    /**
     * Setup the observe on the view model property Image LiveData.
     *
     * @param vm The view model of this fragment.
     * @param imageSelectionPopup The popup window for the image selection.
     */
    private fun observeImages(
        vm: ImageViewModel,
        imageSelectionPopup: ListPopupWindow
    ) {
        vm.images.observe(viewLifecycleOwner) { images ->
            val imageAdapter = ImageArrayAdapter(
                _context,
                images
            )
            imageSelectionPopup.setAdapter(imageAdapter)
            checkSelection(vm)
        }
    }

    /**
     * Setup the observe on the view model property models state LiveData.
     *
     * @param vm The view model of this fragment.
     */
    private fun observeModelState(
        vm: ImageViewModel
    ) {
        val progressBar = binding.imageProgressbar

        vm.modelState.observe(viewLifecycleOwner) {
            setButtonDrawable(it)
            checkSelection(vm)

            progressBar.visibility = when (it) {
                ModelState.RUNNING -> View.VISIBLE
                else -> View.GONE
            }
        }
    }

    private fun checkSelection(vm: ImageViewModel){
        val selectionNext = binding.imageSelectionNext
        val selectionBefore = binding.imageSelectionBefore
        val selectionMenu = binding.imageSelectionMenu
        val selection = binding.imageSelection

        val isNotRunning = vm.modelState.value != ModelState.RUNNING
        val isImagesNotEmpty = vm.images.value?.isNotEmpty() ?: false
        val isImagesLarger1 = (vm.images.value?.size ?: 0) > 1

        selectionNext.isEnabled = isNotRunning && isImagesLarger1
        selectionBefore.isEnabled = isNotRunning && isImagesLarger1
        selection.isEnabled = isNotRunning && isImagesNotEmpty
        selectionMenu.isEnabled = isNotRunning
    }

    private fun setButtonDrawable(it: ModelState) {
        val imageSelection = binding.imageSelection
        val colorID = when (it) {
            ModelState.RUNNING -> R.color.loading
            ModelState.SUCCESS -> R.color.success_green
            ModelState.FAILED -> R.color.fail_red
            else -> return
        }

        val drawableID = when (it) {
            ModelState.RUNNING -> R.drawable.inference
            ModelState.SUCCESS -> R.drawable.ic_check_google
            ModelState.FAILED -> R.drawable.ic_close_google
            else -> return
        }

        val drawable = ContextCompat.getDrawable(_context, drawableID)
        val color = ContextCompat.getColor(_context, colorID)
        if (drawable != null) {
            DrawableCompat.setTint(drawable, color)
        }

        imageSelection.setCompoundDrawablesRelativeWithIntrinsicBounds(
            drawable,
            null,
            null,
            null
        )
    }

    /**
     * Setup the image selection menu button with the OnClickListener that shows a popup menu.
     *
     * @param vm The view model of this fragment.
     */
    private fun setupImageSelectionMenu(
        vm: ImageViewModel
    ) {
        val imageSelectionMenu = binding.imageSelectionMenu


        // Registers a photo picker activity launcher in multi-select mode.
        val imageSelectionMenuPickMultipleMedia = registerForActivityResult(
            ActivityResultContracts.PickMultipleVisualMedia()
        ) { uris ->
            for (uri in uris) {
                if (Build.VERSION.SDK_INT > Build.VERSION_CODES.Q) {
                    _context.contentResolver.takePersistableUriPermission(
                        uri,
                        Intent.FLAG_GRANT_READ_URI_PERMISSION
                    )
                }
            }
            vm.addImages(uris)
        }

        imageSelectionMenu.setOnClickListener {
            val popup = PopupMenu(_context, it)

            popup.setOnMenuItemClickListener { it1 ->
                when (it1.itemId) {
                    R.id.image_selection_menu_add -> {
                        if (MainActivity.checkImagePickerPermission(_context)) {
                            imageSelectionMenuPickMultipleMedia.launch(
                                PickVisualMediaRequest(
                                    ActivityResultContracts.PickVisualMedia.ImageOnly
                                )
                            )
                        }
                        true
                    }

                    R.id.image_selection_menu_remove -> {
                        val image = vm.selectedImage.value
                        if (image != null) {
                            val uri = vm.removeImage(image)
                            if (uri != null && Build.VERSION.SDK_INT > Build.VERSION_CODES.Q) {
                                _context.contentResolver.releasePersistableUriPermission(
                                    uri,
                                    Intent.FLAG_GRANT_READ_URI_PERMISSION
                                )
                            }
                        }
                        true
                    }

                    R.id.image_selection_menu_change_collection -> {
                        MaterialAlertDialogBuilder(_context)
                            .setTitle(getString(R.string.change_collection))
                            .setSingleChoiceItems(
                                ArrayAdapter(
                                    _context,
                                    android.R.layout.simple_list_item_single_choice,
                                    vm.getCollectionsNames()
                                ), vm.getSelectedCollectionIndex()
                            ) { dialog, _ ->
                                val selection = (dialog as AlertDialog).listView
                                val position = selection.checkedItemPosition
                                val item = selection.adapter.getItem(position) as String?
                                item?.let { it1 -> vm.changeCollection(it1) }
                                dialog.dismiss()
                            }
                            .show()
                        true
                    }

                    R.id.image_selection_menu_add_collection -> {
                        MaterialAlertDialogBuilder(_context)
                            .setTitle(getString(R.string.add_collection))
                            .setView(R.layout.dialog_add_collection)
                            .setPositiveButton(getString(R.string.add)) { dialog, _ ->
                                val textView =
                                    (dialog as AlertDialog).findViewById<View>(R.id.collection_name) as EditText
                                vm.addCollection(textView.text.toString())
                                dialog.dismiss()
                            }
                            .setNegativeButton(getString(R.string.cancel)) { dialog, _ -> dialog.dismiss() }
                            .show()
                        true
                    }

                    R.id.image_selection_menu_remove_collection -> {
                        val currentCollectionName =
                            vm.getCollectionsNames()[vm.getSelectedCollectionIndex()]
                        MaterialAlertDialogBuilder(_context)
                            .setTitle(getString(R.string.remove_collection))
                            .setMessage(
                                getString(
                                    R.string.are_you_sure_you_want_to_remove_the_collection,
                                    currentCollectionName
                                )
                            )
                            .setPositiveButton(getString(R.string.yes)) { dialog, _ ->
                                val uris = vm.removeCollection()
                                if (uris != null && Build.VERSION.SDK_INT > Build.VERSION_CODES.Q) {
                                    for (uri in uris) {
                                        _context.contentResolver.releasePersistableUriPermission(
                                            uri,
                                            Intent.FLAG_GRANT_READ_URI_PERMISSION
                                        )
                                    }
                                }
                                dialog.dismiss()
                            }
                            .setNegativeButton(getString(R.string.no)) { dialog, _ ->
                                dialog.dismiss()
                            }
                            .show()
                        true
                    }

                    else -> false
                }
            }

            val popupInflater: MenuInflater = popup.menuInflater
            popupInflater.inflate(R.menu.image_selection_menu, popup.menu)
            popup.show()
        }
    }

    /**
     * Setup image selection next button with the OnClickListener.
     *
     * @param vm The view model of this fragment.
     */
    private fun setupImageSelectionNext(
        vm: ImageViewModel
    ) {
        val imageSelectionNext = binding.imageSelectionNext
        imageSelectionNext.setOnClickListener {
            vm.selectNext()
        }
    }

    /**
     * Setup the observe on the view model property hasBefore LiveData.
     *
     * @param vm The view model of this fragment.
     */
    private fun setupImageSelectionBefore(
        vm: ImageViewModel
    ) {
        val imageSelectionBefore = binding.imageSelectionBefore
        imageSelectionBefore.setOnClickListener {
            vm.selectBefore()
        }
    }

    /**
     * Setup image selection button with the OnClickListener.
     * That also provides a popup window for the image selection.
     *
     * @param vm The view model of this fragment.
     * @return The popup window for the image selection.
     */
    private fun setupImageSelection(
        vm: ImageViewModel
    ): ListPopupWindow {
        val imageSelection = binding.imageSelection

        val imageSelectionPopup = ListPopupWindow(
            _context,
            null,
            androidx.appcompat.R.attr.listPopupWindowStyle
        )

        // TODO change the popup to a dialog that uses GridView to show all images
        imageSelectionPopup.anchorView = imageSelection
        imageSelectionPopup.setOnItemClickListener { parent: AdapterView<*>?,
                                                     _: View?,
                                                     position:
                                                     Int, _: Long ->
            val image = parent?.getItemAtPosition(position) as Image
            vm.selectImage(image.name)
            imageSelectionPopup.dismiss()
        }

        imageSelection.setOnClickListener {
            imageSelectionPopup.show()
        }

        return imageSelectionPopup
    }

    override fun onDestroyView() {
        val vm: ImageViewModel by viewModels {
            DataStoreViewModelFactory(_context.imageCollectionsDataStore)
        }
        val modelConnector = (activity as ModelConnector)
        modelConnector.removeOnModelChangeListener(vm.onModelChangedCallback)
        _binding = null
        super.onDestroyView()
    }

    override fun getDetailViewModel(): DetailsViewModel {
        val vm: ImageViewModel by viewModels {
            DataStoreViewModelFactory(_context.imageCollectionsDataStore)
        }
        return vm.detailsViewModel
    }

}
