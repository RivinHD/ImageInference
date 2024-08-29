/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
 */

package com.neuralnetwork.visioninference.ui.menu

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import com.neuralnetwork.visioninference.databinding.FragmentMenuBinding
import com.neuralnetwork.visioninference.ui.menu.options.SettingsActivity
import com.neuralnetwork.visioninference.ui.menu.options.benchmark.BenchmarkActivity

class MenuFragment : Fragment() {

    /**
     * The binding that gets updated from the fragment.
     */
    private var _binding: FragmentMenuBinding? = null

    /**
     * The binding that holds the view of this fragment.
     * This property is only valid between onCreateView and onDestroyView.
     */
    private val binding get() = _binding!!

    /**
     * The context object of the fragment.
     * This property is only valid between onAttach and onDetach.
     */
    private lateinit var _context: Context

    override fun onAttach(context: Context) {
        super.onAttach(context)
        _context = context
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentMenuBinding.inflate(inflater, container, false)
        val root: View = binding.root

        setupBenchmark()
        setupSettings()

        return root
    }

    /**
     * Setup the benchmark button with the OnClickListener that creates a new activity.
     */
    private fun setupBenchmark() {
        val benchmark = binding.menuBenchmark
        benchmark.setOnClickListener {
            startActivity(Intent(_context, BenchmarkActivity::class.java))
        }
    }

    private fun setupSettings() {
        val settings = binding.menuSettings
        settings.setOnClickListener {
            startActivity(Intent(_context, SettingsActivity::class.java))
        }
    }
}
