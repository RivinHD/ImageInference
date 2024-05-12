package com.example.imageinference.ui.videoCamera

import android.content.Context
import android.os.Bundle
import android.util.Size
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.example.imageinference.MainActivity
import com.example.imageinference.R
import com.example.imageinference.databinding.FragmentVideoCameraBinding
import com.example.imageinference.ui.details.DetailsConnector
import com.example.imageinference.ui.details.DetailsViewModel
import com.google.common.util.concurrent.ListenableFuture

class VideoCameraFragment : Fragment(), DetailsConnector {

    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>

    private var _binding: FragmentVideoCameraBinding? = null

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
        val videoCameraViewModel = ViewModelProvider(this)[VideoCameraViewModel::class.java]
        _binding = FragmentVideoCameraBinding.inflate(inflater, container, false)
        val root: View = binding.root

        val videoRecord = binding.videoRecord
        val videoPreview = binding.videoPreview

        // Request camera permission
        MainActivity.checkCameraPermission(_context)

        // Setup camera preview and capture
        cameraProviderFuture = ProcessCameraProvider.getInstance(_context)
        val resolutionSelector = ResolutionSelector.Builder()
            .setResolutionStrategy(
                ResolutionStrategy(
                    Size(256, 256),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER
                )
            )
            .build()
        val imageAnalysis = ImageAnalysis.Builder()
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setResolutionSelector(resolutionSelector)
            .build()
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview: Preview = Preview.Builder()
                .build()

            val cameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            preview.setSurfaceProvider(videoPreview.getSurfaceProvider())

            cameraProvider.bindToLifecycle(
                viewLifecycleOwner,
                cameraSelector,
                imageAnalysis,
                preview
            )
        }, ContextCompat.getMainExecutor(_context))

        videoRecord.setOnClickListener {
            if (videoRecord.text == getString(R.string.video_record)) {
                videoRecord.text = getString(R.string.stop)
                imageAnalysis.setAnalyzer(
                    ContextCompat.getMainExecutor(_context),
                    videoCameraViewModel
                )
            } else {
                videoRecord.text = getString(R.string.video_record)
                imageAnalysis.clearAnalyzer()
            }
        }

        return root
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    override fun getDetailViewModel(): DetailsViewModel {
        val videoCameraViewModel = ViewModelProvider(this)[VideoCameraViewModel::class.java]
        return videoCameraViewModel.getDetailViewModel()
    }
}
