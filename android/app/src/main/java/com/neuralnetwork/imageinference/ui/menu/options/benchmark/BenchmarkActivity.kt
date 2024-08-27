/*
 * SPDX-FileCopyrightText: © 2024 Vincent Gerlach
 *
 * SPDX-License-Identifier: MIT
 */

package com.neuralnetwork.imageinference.ui.menu.options.benchmark

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.AdapterView
import androidx.activity.result.ActivityResult
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.lifecycleScope
import androidx.preference.PreferenceManager
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.textfield.MaterialAutoCompleteTextView
import com.neuralnetwork.imageinference.MainActivity
import com.neuralnetwork.imageinference.R
import com.neuralnetwork.imageinference.databinding.ActivityBenchmarkBinding
import com.neuralnetwork.imageinference.datastore.imageCollectionsDataStore
import com.neuralnetwork.imageinference.model.Model
import com.neuralnetwork.imageinference.model.ModelAssets
import com.neuralnetwork.imageinference.model.ModelDetails
import com.neuralnetwork.imageinference.model.ModelState
import com.neuralnetwork.imageinference.ui.details.containers.ModelInputType
import com.neuralnetwork.imageinference.ui.menu.options.benchmark.BenchmarkSaveContract.Companion.EXTRA_CONTENT
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.OutputStreamWriter
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.CancellationException

class BenchmarkActivity : AppCompatActivity() {
    /**
     * The binding that holds the view of this activity.
     */
    private lateinit var binding: ActivityBenchmarkBinding

    /**
     * The current selected model.
     */
    private var _model: Model? = null

    /**
     * The name of the current selected model.
     */
    private var _modelName: String = ModelAssets.DEFAULT

    /**
     * Checks the available models and loads them into the model selector.
     */
    private lateinit var _modelAssets: ModelAssets

    /**
     * Checks the available models and loads them into the model selector.
     */
    private lateinit var _benchmarkAssets: BenchmarkCollectionAssets

    /**
     * Holds the state of the model.
     */
    private val _modelState = MutableLiveData<ModelState>().apply {
        value = ModelState.INITIAL
    }

    /**
     * The current selected collection.
     */
    private var _collection: BenchmarkCollection? = null

    /**
     * The name of the current selected collection.
     */
    private var _collectionName: String? = null

    /**
     * Holds the image collections that are available in the app.
     */
    private var _imageCollections = mutableListOf<BenchmarkCollection>()

    /**
     * The job that runs the benchmark.
     */
    private var _benchmarkJob: Job? = null

    /**
     * Holds the benchmark details for each collection.
     * The outer map holds the collection name and the inner map holds the model name and the details.
     */
    private val _benchmarks =
        MutableLiveData<MutableMap<String, MutableMap<String, BenchmarkDetails>>?>()

    private val loadingCollections = MutableLiveData<Boolean>().apply {
        value = true
    }

    /**
     * Holds the registered save to file launcher.
     */
    private val _saveToFileLauncher = registerForActivityResult(
        BenchmarkSaveContract()
    ) {
        val result = it.data
        if (result == null) {
            Log.d("Benchmark Save", "Failed to get the intent.")
            return@registerForActivityResult
        }

        val content = result.getStringExtra(EXTRA_CONTENT)
        if (content == null) {
            Log.d("Benchmark Save", "Failed to get the content.")
            return@registerForActivityResult
        }
        writeToFile(it, content)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityBenchmarkBinding.inflate(layoutInflater)
        setContentView(binding.root)

        _modelAssets = ModelAssets(assets, applicationInfo)
        val context = this

        lifecycleScope.launch {
            loadingCollections.value = true
            _benchmarkAssets = withContext(Dispatchers.IO) {
                BenchmarkCollectionAssets(context)
            }
            withContext(Dispatchers.Main) {
                _imageCollections.addAll(_benchmarkAssets.collections)
            }
            loadingCollections.value = false
        }

        checkRunBenchmark()
        checkSaveShareBenchmark()

        observeModelState()
        observeLoadingCollections()

        setupTopAppBar()
        setupModelSelector()
        setupRunBenchmark()
        setupBenchmarkDetails()
        setupBenchmarkSave()
        setupBenchmarkShare()
    }

    private fun observeLoadingCollections() {
        val collectionProgressbar = binding.collectionProgressbar
        val collectionSelector = binding.collectionSelector
        loadingCollections.observe(this) {
            if (it == false) {
                checkRunBenchmark()
                checkSaveShareBenchmark()
                setupImageCollections()
            }
            collectionProgressbar.visibility = if (it) View.VISIBLE else View.GONE
        }
    }

    /**
     * Setup the benchmark share button with the on click listener.
     */
    private fun setupBenchmarkShare() {
        val share = binding.shareBenchmark
        share.setOnClickListener {
            val content = getJsonOfBenchmark()
            if (content != null) {
                val filename = getJsonFilename()
                shareFile(filename, content)
            }
        }
    }

    /**
     * Setup the benchmark save button with the on click listener.
     */
    private fun setupBenchmarkSave() {
        val save = binding.saveBenchmark
        save.setOnClickListener {
            val content = getJsonOfBenchmark()
            if (content != null) {
                val filename = getJsonFilename()
                saveFile(filename, content)
            }
        }
    }

    /**
     * Observe the model state and update the UI accordingly.
     */
    private fun observeModelState() {
        val info = binding.modelStateInfo
        val progressbar = binding.benchmarkProgressbar
        val run = binding.runBenchmark
        _modelState.observe(this) {
            info.text = getString(
                when (it) {
                    null, ModelState.INITIAL, ModelState.NO_DATA_SELECTED -> R.string.no_data
                    ModelState.RUNNING -> R.string.inference_running
                    ModelState.SUCCESS -> R.string.inference_successful
                    ModelState.FAILED -> R.string.inference_failed
                    ModelState.NO_MODEL_SELECTED -> R.string.select_a_model_to_start_inference
                    ModelState.CANCELLED -> R.string.inference_was_cancelled
                }
            )
            progressbar.visibility = when (it) {
                ModelState.RUNNING -> View.VISIBLE
                else -> View.GONE
            }
            run.text = getString(
                when (it) {
                    ModelState.RUNNING -> R.string.cancel
                    else -> R.string.run
                }
            )
            checkSaveShareBenchmark()
        }
    }

    /**
     * Setup the benchmark details with the adapter and the recycler view.
     */
    private fun setupBenchmarkDetails() {
        val details = binding.benchmarkDetails
        _benchmarks.observe(this) {
            checkRunBenchmark()

            if (it == null) {
                return@observe
            }

            val benchmarkCollection = it.getOrDefault(_collectionName, null)
            if (benchmarkCollection != null) {
                details.adapter = BenchmarkDetailsAdapter(benchmarkCollection.values.toTypedArray())
            }

            checkSaveShareBenchmark()
        }
    }

    /**
     * Setup the run benchmark button with On Click Listener.
     */
    private fun setupRunBenchmark() {
        val run = binding.runBenchmark
        val save = binding.saveBenchmark
        val share = binding.shareBenchmark
        run.setOnClickListener {
            save.isEnabled = false
            share.isEnabled = false

            val job = _benchmarkJob
            if (job == null || job.isActive.not()) {
                run.text = getString(R.string.cancel)
                runBenchmark()
            } else {
                run.text = getString(R.string.run)
                _benchmarkJob?.cancel(CancellationException("User cancelled the benchmark."))
                _modelState.value = ModelState.CANCELLED
                binding.modelSelector.isEnabled = true
                binding.collectionSelector.isEnabled = true
            }
        }
    }

    /**
     * Setup the top app bar with with navigation listener.
     */
    private fun setupTopAppBar() {
        val toolbar = binding.topAppBar
        toolbar.setNavigationOnClickListener {
            onBackPressedDispatcher.onBackPressed()
        }
        toolbar.setOnMenuItemClickListener {
            when (it.itemId) {
                R.id.help -> {
                    MaterialAlertDialogBuilder(this).setTitle(getString(R.string.benchmark_help))
                        .setMessage(
                            getString(R.string.benchmark_help_text)
                        ).setPositiveButton(getString(R.string.ok)) { dialog, _ ->
                            dialog.dismiss()
                        }.show()
                    true
                }

                else -> false
            }
        }
    }

    /**
     * Setup the model selector with with listener and items.
     */
    private fun setupModelSelector() {
        val modelSelector = binding.modelSelector
        (modelSelector.editText as? MaterialAutoCompleteTextView)?.setSimpleItems(
            _modelAssets.models.toTypedArray()
        )
        modelSelector.isEnabled = _modelAssets.models.isNotEmpty()

        (modelSelector.editText as? MaterialAutoCompleteTextView)?.onItemClickListener =
            AdapterView.OnItemClickListener { parent, view, position, id ->
                _modelName = parent?.getItemAtPosition(position) as String
                _model = _modelAssets.getModel(_modelName)
                _model?.load(applicationContext)
                checkRunBenchmark()
            }
    }

    /**
     * Setup the model selector with with listener and items.
     */
    private fun setupImageCollections() {
        val collectionSelector = binding.collectionSelector
        collectionSelector.isEnabled = false
        lifecycleScope.launch {
            imageCollectionsDataStore.data.collect {
                _imageCollections += it.imageCollectionList.map { it1 -> BenchmarkCollection(it1) }

                (collectionSelector.editText as? MaterialAutoCompleteTextView)?.setSimpleItems(
                    _imageCollections.map { it1 -> it1.name }.toTypedArray()
                )
                collectionSelector.isEnabled = _imageCollections.isNotEmpty()

                (collectionSelector.editText as? MaterialAutoCompleteTextView)?.onItemClickListener =
                    AdapterView.OnItemClickListener { parent, view, position, id ->
                        _collectionName = parent?.getItemAtPosition(position) as String
                        _collection =
                            _imageCollections.first { it1 -> it1.name == _collectionName }
                        _benchmarks.value = _benchmarks.value
                    }
                checkRunBenchmark()
            }
        }
    }

    /**
     * Check if the run benchmark button should be enabled.
     */
    private fun checkRunBenchmark() {
        val run = binding.runBenchmark
        run.isEnabled = (_model != null && _collection != null)
    }

    /**
     * Check if the save and share benchmark buttons should be enabled.
     */
    private fun checkSaveShareBenchmark() {
        val save = binding.saveBenchmark
        val share = binding.shareBenchmark
        val enabled = (_modelState.value != ModelState.RUNNING &&
                _collectionName != null &&
                _benchmarks.value?.get(_collectionName)?.isNotEmpty() == true
                )
        save.isEnabled = enabled
        share.isEnabled = enabled
    }

    /**
     * Run the current model on the selected collection.
     */
    private fun runBenchmark() {
        if (_modelState.value == ModelState.RUNNING) {
            Log.d("Benchmark", "The model is already running.")
            return
        }

        Log.d("Benchmark", "Try running the model.")
        val fixedModel: Model? = _model
        if (fixedModel == null) {
            Log.d("Benchmark", "Failed to get the model.")
            _modelState.value = ModelState.NO_MODEL_SELECTED
            return
        }

        val fixedCollection = _collection
        if (fixedCollection == null || fixedCollection.imageList.isEmpty()) {
            Log.d("Benchmark", "Failed to get collection.")
            _modelState.value = ModelState.NO_DATA_SELECTED
            return
        }

        val progressbar = binding.benchmarkProgressbar
        progressbar.visibility = View.VISIBLE
        progressbar.progress = 0
        progressbar.min = 0
        progressbar.max = fixedCollection.imageList.size

        val benchmarkCollections = _benchmarks.value ?: mutableMapOf()
        val benchmarkModels = benchmarkCollections.getOrPut(fixedCollection.name) { mutableMapOf() }
        val benchmark = benchmarkModels.getOrPut(fixedModel.name) {
            BenchmarkDetails(
                fixedCollection.name,
                fixedModel.name
            )
        }
        val addDetails = if (fixedCollection.isLabeled) {
            { bench: BenchmarkDetails, details: ModelDetails, label: String ->
                bench.addDetails(details, label)
            }
        } else {
            { bench: BenchmarkDetails, details: ModelDetails, _: String ->
                bench.addDetails(details)
            }
        }

        val preferences = PreferenceManager.getDefaultSharedPreferences(this)
        val warmupCount = preferences.getInt(getString(R.string.warmup_samples_count), 25)

        _modelState.value = ModelState.RUNNING
        binding.modelSelector.isEnabled = false
        binding.collectionSelector.isEnabled = false
        Log.d("Benchmark", "Running the model.")
        _benchmarkJob = lifecycleScope.launch {
            withContext(Dispatchers.Default) {
                for (i in 0..warmupCount) {
                    val warmupImage = fixedCollection.imageList.first()
                    val warmupDetails = ModelDetails(ModelInputType.IMAGE)
                    val warmupBitmap = warmupImage.getBitmap(contentResolver)
                    fixedModel.run(warmupBitmap, warmupDetails)
                }
            }

            for (image in fixedCollection.imageList) {
                val outputDetails = withContext(Dispatchers.Default) {
                    val details = ModelDetails(ModelInputType.IMAGE)
                    val bitmap = image.getBitmap(contentResolver)
                    fixedModel.run(bitmap, details)
                }
                progressbar.setProgress(progressbar.progress + 1, false)
                addDetails(benchmark, outputDetails, image.name)
                _benchmarks.value = benchmarkCollections
            }

            _modelState.value = ModelState.SUCCESS
            progressbar.visibility = View.GONE
            binding.modelSelector.isEnabled = true
            binding.collectionSelector.isEnabled = true
        }
    }

    /**
     * Saves a file with the given filename to the filesystem.
     * The directory is decided by the user.
     *
     * @param filename The name of the file to save.
     * @param content The content of the file to write.
     */
    private fun saveFile(filename: String, content: String) {
        val intent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = MIMETYPE_JSON
            putExtra(Intent.EXTRA_TITLE, filename)
            putExtra(EXTRA_CONTENT, content)
        }
        _saveToFileLauncher.launch(intent)
    }

    /**
     * Writes the given content to a file based on the given activity result.
     *
     * @param result The activity result that holds the file path.
     * @param content The content to write to the file.
     */
    private fun writeToFile(result: ActivityResult, content: String) {
        if (result.resultCode != Activity.RESULT_OK) {
            return
        }

        val intent = result.data ?: return
        val uri = intent.data ?: return
        val outputStream = contentResolver.openOutputStream(uri) ?: return
        val outputWriter = OutputStreamWriter(outputStream)
        outputWriter.write(content)
        outputWriter.close()
        outputStream.close()
    }

    /**
     * Shares the content of file with the corresponding filename.
     *
     * @param filename The name of the file to share.
     * @param content The content of the file to share.
     */
    private fun shareFile(filename: String, content: String) {
        val file = MainActivity.cacheSave(this, FILE_PROVIDER_NAME, content)

        val sendIntent = if (file == null) {
            Intent().apply {
                action = Intent.ACTION_SEND
                type = "text/plain"
                putExtra(Intent.EXTRA_TITLE, filename)
                putExtra(Intent.EXTRA_TEXT, content)
            }
        } else {
            val shareUri = FileProvider.getUriForFile(this, FILE_PROVIDER_AUTHORITY, file, filename)
            Intent().apply {
                action = Intent.ACTION_SEND
                type = MIMETYPE_JSON
                putExtra(Intent.EXTRA_TITLE, filename)
                putExtra(Intent.EXTRA_STREAM, shareUri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
        }

        val shareIntent = Intent.createChooser(sendIntent, getString(R.string.save_benchmark))
        startActivity(shareIntent)

        if (file != null) {
            MainActivity.cacheRemove(this, file)
        }
    }

    /**
     * Get the filename for the benchmark json with the current date.
     *
     * @return The filename for the benchmark.
     */
    private fun getJsonFilename(): String {
        val timeFormat = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm")
        val time = LocalDateTime.now().format(timeFormat)
        val filename = "ImageInference_Benchmark_$time"
        return filename
    }

    /**
     * Get the benchmark data as a json string.
     *
     * @return The json string or null on failure.
     */
    private fun getJsonOfBenchmark(): String? {
        val benchmarks = _benchmarks.value
        if (benchmarks?.isNotEmpty() == true) {
            val benchmark = benchmarks.getOrDefault(_collectionName, null)
            if (benchmark == null) {
                Log.d("Benchmark", "Failed to get the benchmark.")
                return null
            }
            return Json.encodeToString(benchmark)
        }
        return null
    }

    companion object {
        private const val MIMETYPE_JSON = "application/json"
        private const val FILE_PROVIDER_AUTHORITY = "com.neuralnetwork.imageinference.fileprovider"
        private const val FILE_PROVIDER_NAME = "shared_benchmark.json"
    }
}
