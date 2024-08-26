/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
 */

package com.neuralnetwork.imageinference.ui.details.containers

/**
 * The type of input the model can be used with.
 */
enum class ModelInputType {
    /**
     * The model is used with video input, which corresponds to a stream of photos.
     */
    VIDEO,

    /**
     * The model is used with a single photo input.
     */
    PHOTO,

    /**
     * The model is used with a image input where the image can be swapped.
     */
    IMAGE
}
