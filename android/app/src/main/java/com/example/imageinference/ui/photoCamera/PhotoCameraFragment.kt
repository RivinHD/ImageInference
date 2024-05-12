package com.example.imageinference.ui.photoCamera


import android.content.Context
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalZeroShutterLag
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.example.imageinference.MainActivity
import com.example.imageinference.R
import com.example.imageinference.databinding.FragmentPhotoCameraBinding
import com.example.imageinference.ui.details.DetailsConnector
import com.example.imageinference.ui.details.DetailsViewModel
import com.google.common.util.concurrent.ListenableFuture

class PhotoCameraFragment : Fragment(), DetailsConnector {

    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>

    private var _binding: FragmentPhotoCameraBinding? = null

    // This property is only valid between onCreateView and
    // onDestroyView.
    private val binding get() = _binding!!

    private lateinit var _context: Context

    override fun onAttach(context: Context) {
        super.onAttach(context)
        _context = context
    }

    @ExperimentalZeroShutterLag
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val photoCameraViewModel = ViewModelProvider(this)[PhotoCameraViewModel::class.java]
        _binding = FragmentPhotoCameraBinding.inflate(inflater, container, false)
        val root: View = binding.root

        val photoCapture = binding.photoCapture
        val photoView = binding.photoInferenceImage
        val photoPreview = binding.photoPreview

        // Request camera permission
        MainActivity.checkCameraPermission(_context)

        // Setup camera preview and capture
        cameraProviderFuture = ProcessCameraProvider.getInstance(_context)
        val imageCapture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_ZERO_SHUTTER_LAG)
            .build()
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview: Preview = Preview.Builder()
                .build()

            val cameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            preview.setSurfaceProvider(photoPreview.getSurfaceProvider())

            cameraProvider.bindToLifecycle(
                viewLifecycleOwner,
                cameraSelector,
                imageCapture,
                preview
            )
        }, ContextCompat.getMainExecutor(_context))


        photoCapture.setOnClickListener {
            if (photoPreview.visibility == View.VISIBLE) {
                // Take the current photo from the camera preview
                var photoSuccess = true
                imageCapture.takePicture(
                    ContextCompat.getMainExecutor(_context),
                    object : ImageCapture.OnImageCapturedCallback() {
                        override fun onCaptureSuccess(image: ImageProxy) {
                            super.onCaptureSuccess(image)
                            photoView.setImageBitmap(image.toBitmap())
                            photoCameraViewModel.runModel(image.toBitmap())
                        }

                        override fun onError(exception: ImageCaptureException) {
                            super.onError(exception)
                            photoSuccess = false
                        }
                    }
                )

                if (photoSuccess) {
                    photoPreview.visibility = View.INVISIBLE
                    photoCapture.text = getString(R.string.photo_clear)
                }

            } else {
                // Delete the current photo and activate the preview again
                photoPreview.visibility = View.VISIBLE
                photoCapture.text = getString(R.string.photo_capture)
            }
        }

        return root
    }

    override fun onDestroyView() {
        super.onDestroyView()
        cameraProviderFuture.cancel(true)
        _binding = null
    }

    override fun getDetailViewModel(): DetailsViewModel {
        val photoCameraViewModel = ViewModelProvider(this)[PhotoCameraViewModel::class.java]
        return photoCameraViewModel.getDetailViewModel()
    }
}
