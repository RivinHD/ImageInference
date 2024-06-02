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
