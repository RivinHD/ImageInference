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

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.MenuInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.PopupMenu
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.neuralnetwork.imageinference.R
import com.neuralnetwork.imageinference.databinding.FragmentImageBinding
import com.neuralnetwork.imageinference.model.ModelConnector
import com.neuralnetwork.imageinference.ui.details.DetailsConnector
import com.neuralnetwork.imageinference.ui.details.DetailsViewModel

class ImageFragment : Fragment(), DetailsConnector {

    private var _binding: FragmentImageBinding? = null

    // This property is only valid between onCreateView and
    // onDestroyView.
    private val binding get() = _binding!!
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
        // region Initialization
        val vm = ViewModelProvider(this)[ImageViewModel::class.java]
        _binding = FragmentImageBinding.inflate(inflater, container, false)
        val root: View = binding.root
        vm.model = (activity as ModelConnector).getModel()

        val inferenceImage = binding.imageInferenceImage
        val imageSelection = binding.imageSelection
        val imageSelectionBefore = binding.imageSelectionBefore
        val imageSelectionNext = binding.imageSelectionNext
        val imageSelectionMenu = binding.imageSelectionMenu
        // endregion

        // region Listeners
        imageSelection.onItemSelectedListener = vm.onImageSelectedListener
        imageSelectionBefore.setOnClickListener(vm.onBeforeClickListener)
        imageSelectionNext.setOnClickListener(vm.onNextClickListener)

        // Registers a photo picker activity launcher in multi-select mode.
        val imageSelectionMenuPickMultipleMedia =
            registerForActivityResult(ActivityResultContracts.PickMultipleVisualMedia()) { uris ->
                for (uri in uris) {
                    _context.contentResolver.takePersistableUriPermission(
                        uri,
                        Intent.FLAG_GRANT_READ_URI_PERMISSION
                    )
                    uri.path?.let { vm.addImage(it) }
                }
            }
        imageSelectionMenu.setOnClickListener {
            val popup = PopupMenu(_context, it)
            popup.setOnMenuItemClickListener { it1 ->
                when (it1.itemId) {
                    R.id.image_selection_menu_add -> {
                        imageSelectionMenuPickMultipleMedia.launch(
                            PickVisualMediaRequest(
                                ActivityResultContracts.PickVisualMedia.ImageOnly
                            )
                        )
                        true
                    }

                    R.id.image_selection_menu_remove -> {
                        vm.selectedImage.value?.let { it2 -> vm.removeImage(it2) }
                        true
                    }

                    else -> false
                }
            }
            val popupInflater: MenuInflater = popup.menuInflater
            popupInflater.inflate(R.menu.image_selection_menu, popup.menu)
            popup.show()
        }
        // endregion

        // region Observes
        vm.images.observe(viewLifecycleOwner) { images ->
            imageSelection.adapter = ArrayAdapter(
                _context,
                android.R.layout.simple_spinner_dropdown_item,
                images.map { it.name }
            )
            imageSelectionBefore.isEnabled = images.count() > 1
            imageSelectionNext.isEnabled = images.count() > 1
        }
        vm.selectedImage.observe(viewLifecycleOwner) {
            it.loadImageInto(inferenceImage)
        }
        vm.hasNext.observe(viewLifecycleOwner) {
            imageSelectionNext.isEnabled = it
        }
        vm.hasBefore.observe(viewLifecycleOwner) {
            imageSelectionBefore.isEnabled = it
        }
        // endregion

        return root
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    override fun getDetailViewModel(): DetailsViewModel {
        val imageViewModel = ViewModelProvider(this)[ImageViewModel::class.java]
        return imageViewModel.detailsViewModel
    }

}
