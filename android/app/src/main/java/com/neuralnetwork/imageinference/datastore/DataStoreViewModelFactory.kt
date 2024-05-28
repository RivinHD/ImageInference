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
