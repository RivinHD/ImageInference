/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
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
