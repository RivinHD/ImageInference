/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
 */

package com.neuralnetwork.imageinference.ui.image

import android.content.Context
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.ImageView
import android.widget.TextView
import com.neuralnetwork.imageinference.R

/**
 * The Adapter that displays images in a list.
 *
 * @constructor Creates an image array adapter filled with the given images.
 *
 * @param context The context the adapter is created in.
 * @param objects The list of images used to fill the adapter.
 */
class ImageArrayAdapter(context: Context, objects: List<Image>)
    : ArrayAdapter<Image>(context, R.layout.item_image_text, R.id.item_text, objects) {

    override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
        val inflater = LayoutInflater.from(parent.context)
        val mainView : View = convertView ?: inflater.inflate(R.layout.item_image_text, parent, false)
        val imageView = mainView.findViewById<ImageView>(R.id.item_image)
        val textView = mainView.findViewById<TextView>(R.id.item_text)
        val image = getItem(position)
        image?.loadImageInto(imageView)
        textView.text = image?.name
        return mainView
    }

    override fun getDropDownView(position: Int, convertView: View?, parent: ViewGroup): View {
        return getView(position, convertView, parent)
    }
}
