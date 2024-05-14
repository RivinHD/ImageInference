package com.neuralnetwork.imageinference.ui.image

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.MenuInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.PopupMenu
import android.widget.Spinner
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.neuralnetwork.imageinference.R
import com.neuralnetwork.imageinference.databinding.FragmentImageBinding
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
        val imageViewModel = ViewModelProvider(this)[ImageViewModel::class.java]
        _binding = FragmentImageBinding.inflate(inflater, container, false)
        val root: View = binding.root

        val inferenceImage = binding.imageInferenceImage
        val imageSelection = binding.imageSelection
        val imageSelectionBefore = binding.imageSelectionBefore
        val imageSelectionNext = binding.imageSelectionNext
        val imageSelectionMenu = binding.imageSelectionMenu

        imageSelectionBefore.isEnabled = false
        imageSelectionNext.isEnabled = false

        imageSelection.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?,
                view: View?,
                position: Int,
                id: Long
            ) {
                val name = parent?.getItemAtPosition(position) as String
                imageViewModel.selectImage(name)
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                imageViewModel.selectNothing()
            }

        }
        imageSelectionBefore.setOnClickListener {
            it.isEnabled = imageViewModel.selectNext()
        }
        imageSelectionNext.setOnClickListener {
            it.isEnabled = imageViewModel.selectBefore()
        }
        // Registers a photo picker activity launcher in multi-select mode.
        val imageSelectionMenuPickMultipleMedia =
            registerForActivityResult(ActivityResultContracts.PickMultipleVisualMedia()) { uris ->
                for (uri in uris) {
                    _context.contentResolver.takePersistableUriPermission(
                        uri,
                        Intent.FLAG_GRANT_READ_URI_PERMISSION
                    )
                    uri.path?.let { imageViewModel.addImage(it) }
                }
            }
        val imageSelectionMenuOnMenuItemClickListener = PopupMenu.OnMenuItemClickListener {
            when (it.itemId) {
                R.id.image_selection_menu_add -> {
                    imageSelectionMenuPickMultipleMedia.launch(
                        PickVisualMediaRequest(
                            ActivityResultContracts.PickVisualMedia.ImageOnly
                        )
                    )
                    true
                }

                R.id.image_selection_menu_remove -> {
                    imageViewModel.selectedImage.value?.let { it1 -> imageViewModel.removeImage(it1) }
                    true
                }

                else -> false
            }
        }
        imageSelectionMenu.setOnClickListener {
            val popup = PopupMenu(_context, it)
            popup.setOnMenuItemClickListener(imageSelectionMenuOnMenuItemClickListener)
            val popupInflater: MenuInflater = popup.menuInflater
            popupInflater.inflate(R.menu.image_selection_menu, popup.menu)
            popup.show()
        }


        imageViewModel.images.observe(viewLifecycleOwner) { images ->
            imageSelection.adapter = ArrayAdapter(
                _context,
                android.R.layout.simple_spinner_dropdown_item,
                images.map { it.name }
            )
            imageSelectionBefore.isEnabled = images.count() > 1
            imageSelectionNext.isEnabled = images.count() > 1
        }
        imageViewModel.selectedImage.observe(viewLifecycleOwner) {
            it.loadImageInto(inferenceImage)
        }

        return root
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    override fun getDetailViewModel(): DetailsViewModel {
        val imageViewModel = ViewModelProvider(this)[ImageViewModel::class.java]
        return imageViewModel.getDetailViewModel()
    }

}
