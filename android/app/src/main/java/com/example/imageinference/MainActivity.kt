package com.example.imageinference

import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.ArrayAdapter
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.navigation.findNavController
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.setupActionBarWithNavController
import androidx.navigation.ui.setupWithNavController
import com.example.imageinference.databinding.ActivityMainBinding
import com.example.imageinference.models.ModelAssets
import com.google.android.material.bottomnavigation.BottomNavigationView
import java.io.File
import java.io.FileOutputStream


class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.modelToolbar)

        val navView: BottomNavigationView = binding.navView

        val navController = findNavController(R.id.nav_host_fragment_activity_main)
        // Passing each menu ID as a set of Ids because each
        // menu should be considered as top level destinations.
        val appBarConfiguration = AppBarConfiguration(
            setOf(
                R.id.navigation_photo_camera,
                R.id.navigation_video_camera,
                R.id.navigation_image,
                R.id.navigation_menu
            )
        )
        setupActionBarWithNavController(navController, appBarConfiguration)
        navView.setupWithNavController(navController)

        val modelAssets = ModelAssets(assets)
        val modelSelector = binding.modelSelector
        modelSelector.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            modelAssets.models
        )

    }

    companion object {
        /**
         * Check for the camera permission and ask if needed.
         *
         * @param context The context for the permission check.
         */
        fun checkCameraPermission(context: Context) {
            if (ContextCompat.checkSelfPermission(
                    context,
                    android.Manifest.permission.CAMERA
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                ActivityCompat.requestPermissions(
                    context as Activity,
                    arrayOf(android.Manifest.permission.CAMERA),
                    100
                )
            }

        }

        /**
         * Loads the selected asset into the directory that holds the applications files
         * and returns the absolut path.
         *
         * @param context The context for the application file directory.
         * @param assetName The name of the asset.
         * @return The absolut path to the loaded asset.
         */
        fun getAsset(context: Context, assetName: String): String {
            val file = File(context.filesDir, assetName)
            if (file.exists() && file.length() > 0) {
                return file.absolutePath
            }

            context.assets.open(assetName).use { inStream ->
                FileOutputStream(file).use { outStream ->
                    val buffer = ByteArray(4096)
                    var read: Int
                    while (inStream.read(buffer).also { read = it } != -1) {
                        outStream.write(buffer, 0, read)
                    }
                    outStream.flush()
                }
                return file.absolutePath
            }
        }
    }

}
