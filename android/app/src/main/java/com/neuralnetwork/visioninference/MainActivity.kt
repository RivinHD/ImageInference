/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
 */

package com.neuralnetwork.visioninference

import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.AdapterView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.net.toUri
import androidx.navigation.findNavController
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.setupActionBarWithNavController
import androidx.navigation.ui.setupWithNavController
import com.google.android.material.bottomnavigation.BottomNavigationView
import com.google.android.material.textfield.MaterialAutoCompleteTextView
import com.neuralnetwork.visioninference.databinding.ActivityMainBinding
import com.neuralnetwork.visioninference.model.Model
import com.neuralnetwork.visioninference.model.ModelAssets
import com.neuralnetwork.visioninference.model.ModelConnector
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter


class MainActivity : AppCompatActivity(), ModelConnector {
    /**
     * The binding that holds the view of this activity.
     */
    private lateinit var binding: ActivityMainBinding

    /**
     * The current selected model.
     */
    private var model: Model? = null

    /**
     * The name of the current selected model.
     */
    private var modelName: String = ModelAssets.DEFAULT

    /**
     * Callback to use when the model has changed.
     */
    private var _modelChangedCallback: ((m: Model?) -> Unit)? = null

    /**
     * Checks the available models and loads them into the model selector.
     */
    private lateinit var _modelAssets: ModelAssets

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        _modelAssets = ModelAssets(assets, applicationInfo)
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

        setupModelSelector()
    }

    /**
     * Setup the model selector with with listener and items
     */
    private fun setupModelSelector() {
        val modelSelector = binding.modelSelector
        (modelSelector.editText as? MaterialAutoCompleteTextView)?.setSimpleItems(
            _modelAssets.models.toTypedArray()
        )
        modelSelector.isEnabled = _modelAssets.models.isNotEmpty()

        (modelSelector.editText as? MaterialAutoCompleteTextView)?.onItemClickListener =
            AdapterView.OnItemClickListener { parent, view, position, id ->
                modelName = parent?.getItemAtPosition(position) as String
                Log.d("MainActivity", "Selected Model: $modelName.")
                model = _modelAssets.getModel(modelName)
                model?.load(applicationContext)
                _modelChangedCallback?.let { it(model) }
            }
    }

    companion object {
        /**
         * Check for the camera permission and ask if needed.
         *
         * @param context The context for the permission check.
         * @return True if the permission is granted, false otherwise.
         */
        fun checkCameraPermission(context: Context): Boolean {
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

            return ContextCompat.checkSelfPermission(
                context,
                android.Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        }

        /**
         * Check for the image media permission and ask if needed.
         *
         * @param context The context for the permission check.
         * @return True if the permission is granted, false otherwise.
         */
        fun checkImagePickerPermission(context: Context): Boolean {
            val permissions =
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.Q) {
                    arrayOf(android.Manifest.permission.READ_EXTERNAL_STORAGE)
                } else {
                    emptyArray()
                }

            if (permissions.any {
                    ContextCompat.checkSelfPermission(
                        context,
                        it
                    ) != PackageManager.PERMISSION_GRANTED
                }
            ) {

                ActivityCompat.requestPermissions(
                    context as Activity,
                    permissions,
                    100
                )
            }

            return permissions.all {
                ContextCompat.checkSelfPermission(
                    context,
                    it
                ) == PackageManager.PERMISSION_GRANTED
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
            file.parentFile?.mkdirs()
            val updateTime = context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime

            if (file.exists() && file.length() > 0 && file.lastModified() >= updateTime) {
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

        /**
         * Save a file to the cache
         *
         * @param context The context for the application file directory.
         * @param content The content of the file to write.
         * @return The created file.
         */
        fun cacheSave(context: Context, filename: String, content: String): File? {
            val directory = context.cacheDir
            val file = File(directory, filename)
            file.parentFile?.mkdirs()

            val outputStream = context.contentResolver.openOutputStream(file.toUri()) ?: return null
            val outputWriter = OutputStreamWriter(outputStream)
            outputWriter.write(content)
            outputWriter.close()
            outputStream.close()
            return file
        }

        /**
         * Remove a file from the cache.
         * Also checks if the given uri is part of the cache.
         *
         * @param context The context for the application file directory.
         * @param file The file to remove.
         */
        fun cacheRemove(context: Context, file: File) {
            try {
                val directory = context.cacheDir

                if (directory != null && directory.startsWith(file) && file.isFile) {
                    file.delete()
                }
            } catch (e: Exception) {
                when (e) {
                    is IllegalArgumentException, is SecurityException -> return
                    else -> throw e
                }
            }
        }
    }

    override fun getModel(): Model? {
        return model
    }

    override fun getModelName(): String {
        return modelName
    }

    override fun setOnModelChangedListener(callback: ((m: Model?) -> Unit)) {
        _modelChangedCallback = callback
    }

    override fun removeOnModelChangeListener(callback: (m: Model?) -> Unit) {
        if (_modelChangedCallback == callback) {
            _modelChangedCallback = null
        }
    }

}
