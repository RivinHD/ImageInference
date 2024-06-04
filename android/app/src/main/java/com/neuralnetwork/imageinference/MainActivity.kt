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

package com.neuralnetwork.imageinference

import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.system.ErrnoException
import android.system.Os
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.AutoCompleteTextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.navigation.findNavController
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.setupActionBarWithNavController
import androidx.navigation.ui.setupWithNavController
import com.google.android.material.bottomnavigation.BottomNavigationView
import com.google.android.material.textfield.MaterialAutoCompleteTextView
import com.neuralnetwork.imageinference.databinding.ActivityMainBinding
import com.neuralnetwork.imageinference.model.Model
import com.neuralnetwork.imageinference.model.ModelAssets
import com.neuralnetwork.imageinference.model.ModelConnector
import java.io.File
import java.io.FileOutputStream


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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Sets the library path for the Snapdragon Neural Processing Engine SDK.
        // See https://developer.qualcomm.com/sites/default/files/docs/snpe/dsp_runtime.html for
        // more information.
        try {
            Os.setenv(
                "ADSP_LIBRARY_PATH",
                applicationInfo.nativeLibraryDir,
                true)
        } catch (e: ErrnoException) {
            Log.e(
                "Snapdragon Neural Processing Engine SDK",
                "Cannot set ADSP_LIBRARY_PATH",
                e
            )
        }

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
        setupModelSelector(modelAssets)
    }

    /**
     * Setup the model selector with with listener and items
     *
     * @param modelAssets The models available from the assets.
     */
    private fun setupModelSelector(modelAssets: ModelAssets) {
        val modelSelector = binding.modelSelector
        (modelSelector.editText as? MaterialAutoCompleteTextView)?.setSimpleItems(
            modelAssets.models.toTypedArray()
        )
        modelSelector.isEnabled = modelAssets.models.isNotEmpty()

        (modelSelector.editText as? MaterialAutoCompleteTextView)?.onItemClickListener =
            AdapterView.OnItemClickListener { parent, view, position, id ->
                modelName = parent?.getItemAtPosition(position) as String
                Log.d("MainActivity", "Selected Model: $modelName.")
                model = ModelAssets.getModel(modelName)
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
        if (_modelChangedCallback == callback)
        {
            _modelChangedCallback = null
        }
    }

}
