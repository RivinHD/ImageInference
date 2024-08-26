/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
 */

package com.neuralnetwork.imageinference.datastore

import androidx.datastore.core.DataStore
import androidx.lifecycle.ViewModel

/**
 * Provides a view model base that connect a DataStore to the view model.
 *
 * @param T The type of the data store.
 * @property dataStore The data store object to use.
 * @constructor Creates an empty data store view model.
 */
open class DataStoreViewModel<T>(private val dataStore: DataStore<T>) : ViewModel()
