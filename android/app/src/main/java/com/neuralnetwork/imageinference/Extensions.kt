/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
 */

package com.neuralnetwork.imageinference

import android.graphics.Bitmap
import android.graphics.Matrix

/** ===========================================
 *  This file contains various type extensions.
 * ============================================
 */

/**
 * Rotates the bitmap by the given degrees.
 *
 * @param degrees The degrees to rotate the bitmap by.
 * @return The rotated bitmap.
 */
fun Bitmap.rotate(degrees: Float): Bitmap {
    val matrix = Matrix().apply { postRotate(degrees) }
    return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
}
