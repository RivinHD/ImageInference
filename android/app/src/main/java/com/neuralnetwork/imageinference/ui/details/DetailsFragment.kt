package com.neuralnetwork.imageinference.ui.details

import android.content.Context
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.neuralnetwork.imageinference.R
import com.neuralnetwork.imageinference.databinding.FragmentDetailsBinding

class DetailsFragment : Fragment() {


    private var _binding: FragmentDetailsBinding? = null

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
        _binding = FragmentDetailsBinding.inflate(inflater, container, false)
        val root: View = binding.root

        val detailsViewModel: DetailsViewModel = when (parentFragment?.id) {
            R.id.navigation_image,
            R.id.navigation_photo_camera,
            R.id.navigation_video_camera -> {
                (requireParentFragment() as DetailsConnector).getDetailViewModel()
            }

            else -> {
                ViewModelProvider(this)[DetailsViewModel::class.java]
            }
        }

        val class1 = binding.option1Class
        val accuracy1 = binding.option1Percentage
        val class2 = binding.option2Class
        val accuracy2 = binding.option2Percentage
        val class3 = binding.option3Class
        val accuracy3 = binding.option3Percentage
        val class4 = binding.option4Class
        val accuracy4 = binding.option4Percentage
        val class5 = binding.option5Class
        val accuracy5 = binding.option5Percentage

        return root
    }
}
