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

package com.neuralnetwork.imageinference.ui.menu.options.benchmark

import android.content.Context
import android.content.Intent
import androidx.activity.result.ActivityResult
import androidx.activity.result.contract.ActivityResultContract

class BenchmarkSaveContract : ActivityResultContract<Intent, ActivityResult>() {
    /**
     * Holds the content that was given by the input intent.
     */
    private var _content: String? = null

    override fun createIntent(context: Context, input: Intent): Intent {
        _content = input.getStringExtra(EXTRA_CONTENT)
        return input
    }

    override fun parseResult(resultCode: Int, intent: Intent?): ActivityResult {
        intent?.putExtra(EXTRA_CONTENT, _content)
        return ActivityResult(resultCode, intent)
    }

    companion object {
        private const val PACKAGE_NAME = "com.neuralnetwork.imageinference"
        const val EXTRA_CONTENT = "$PACKAGE_NAME.Content"
    }
}
