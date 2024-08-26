/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
 */

package com.neuralnetwork.imageinference.datastore

import androidx.datastore.core.DataStore
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.neuralnetwork.imageinference.ImageCollections
import com.neuralnetwork.imageinference.ui.image.ImageViewModel

/**
 * Provides a view model factory for the data store view model.
 * WARNING: The view model that is created by this factory must implement the DataStoreViewModel.
 *
 * @param D The DataStore type created from proto.
 * @property dataStore The actual data storage from the context.
 * @constructor Create Data store view model factory
 * that inputs the given data store to the view model.
 */
class DataStoreViewModelFactory<D>(private val dataStore: DataStore<D>) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (DataStoreViewModel::class.java.isAssignableFrom(modelClass)) {
            @Suppress("UNCHECKED_CAST")
            when (modelClass){
                ImageViewModel::class.java -> {
                    return ImageViewModel(dataStore as DataStore<ImageCollections>) as T
                }
                else -> throw NotImplementedError("The given ViewModel is not implemented in the factory.")
            }
        }

        throw IllegalArgumentException("The given ViewModel must be an instance of DataStoreViewModel.")
    }
}
