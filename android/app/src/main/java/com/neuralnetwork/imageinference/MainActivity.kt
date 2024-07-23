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
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.AdapterView
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
        if (_modelChangedCallback == callback) {
            _modelChangedCallback = null
        }
    }

}
