package com.example.androidobjecttracker

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.camera.video.VideoCapture
import androidx.camera.view.PreviewView
import androidx.compose.runtime.*
import androidx.core.content.ContextCompat
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import com.example.modelengine.AppLog
import com.example.modelengine.StaticTracker
import com.example.modelengine.SOTExecutor
import com.example.androidobjecttracker.ui.AppState
import com.example.androidobjecttracker.ui.TrackingScreen
import androidx.compose.ui.geometry.Offset
import com.example.modelengine.ModelExecutor
import com.example.modelengine.SortTracker
import java.text.SimpleDateFormat
import java.util.*
import android.net.Uri
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    // ---- Camera infrastructure ----
    private lateinit var cameraExecutor: ExecutorService
    private var previewView: PreviewView? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null

    // ---- ML components ----
    private lateinit var modelExecutor: ModelExecutor
    private lateinit var sotExecutor: SOTExecutor
    private val staticTracker = StaticTracker()

    // ---- Pipeline components ----
    private lateinit var videoEncoder: VideoEncoder
    private lateinit var videoProcessor: FastVideoProcessor

    // ---- Compose-observable state ----
    private var appState          by mutableStateOf(AppState.IDLE)
    private var processingProgress by mutableFloatStateOf(0f)
    private var processingEtaMs   by mutableLongStateOf(-1L)
    private var currentFrame      by mutableStateOf<Bitmap?>(null)
    private var resultVideoUri    by mutableStateOf<Uri?>(null)
    private var useSot            by mutableStateOf(false)
    private var finalTimeMillis   = 0L

    // ---- Result state ----
    private var resultVideoFile: File? = null

    // ---- Gallery picker ----
    private val pickVideoLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let { processVideo(it) }
        }

    // =========================================================================
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialise ML
        modelExecutor = ModelExecutor(this).also { it.loadModel("yolo26n.ptl") }
        sotExecutor   = SOTExecutor().also   { it.loadModel("AbaViTrack_lite.ptl", this) }

        // Initialise pipeline
        videoEncoder   = VideoEncoder(this)
        videoProcessor = FastVideoProcessor(this, modelExecutor, sotExecutor, staticTracker, videoEncoder)

        // Camera thread
        cameraExecutor = Executors.newSingleThreadExecutor()
        previewView    = PreviewView(this).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }

        if (!allPermissionsGranted()) requestPermissions()

        setContent {
            // These are read on every recomposition.
            // currentFrame is a MutableState, so when FastVideoProcessor delivers
            // a new annotated frame, recomposition fires and both the frame AND
            // the latest pin/path state are refreshed together.
            val pins    = staticTracker.getPins()
            val carPath = sotExecutor.carPath.toList()   // snapshot to avoid concurrent modification

            TrackingScreen(
                appState           = appState,
                previewView        = previewView,
                currentFrame       = currentFrame,
                resultVideoUri     = resultVideoUri,
                processingProgress = processingProgress,
                processingEtaMs    = processingEtaMs,
                pins               = pins,
                carPath            = carPath,
                useSot             = useSot,
                finalStats         = if (appState == AppState.RESULTS)
                                        Pair(finalTimeMillis, pins.count { it.isFallen })
                                     else null,
                onRecordToggle     = { if (appState == AppState.RECORDING) stopRecording() else startRecording() },
                onPickVideo        = { pickVideoLauncher.launch("video/*") },
                onToggleSot        = { useSot = !useSot },
                onSaveResult       = { saveProcessedVideo() },
                onReset            = { resetAll() }
            )
        }
    }

    // =========================================================================
    override fun onResume() {
        super.onResume()
        if (allPermissionsGranted() && appState in listOf(AppState.IDLE, AppState.RECORDING)) {
            startCamera()
        }
    }

    override fun onPause() {
        super.onPause()
        cameraProvider?.unbindAll()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    // =========================================================================
    // Video processing
    // =========================================================================

    private fun processVideo(uri: Uri) {
        // Release camera hardware so it does not compete with MediaCodec
        try { cameraProvider?.unbindAll() } catch (e: Exception) { AppLog.e("Camera unbind", e) }

        appState = AppState.PROCESSING
        AppLog.i("MainActivity: Processing $uri")

        // Reset all tracker state before a new run
        modelExecutor.reset()
        sotExecutor.reset()
        staticTracker.reset()
        resultVideoFile = null
        resultVideoUri  = null

        lifecycleScope.launch(Dispatchers.Default) {
            val startMs = System.currentTimeMillis()

            videoProcessor.processVideo(
                uri               = uri,
                onProgress        = { progress, etaMs ->
                    processingProgress = progress
                    processingEtaMs    = etaMs
                },
                onFrameProcessed  = { annotatedFrame ->
                    // Swap in the new annotated frame; recycle the old one
                    val old = currentFrame
                    currentFrame = annotatedFrame
                    if (old != null && old !== annotatedFrame) old.recycle()
                }
            )

            resultVideoFile = videoEncoder.finish()
            resultVideoUri  = resultVideoFile?.let { Uri.fromFile(it) }

            withContext(Dispatchers.Main) {
                finalTimeMillis = System.currentTimeMillis() - startMs
                AppLog.i("MainActivity: Processing done in ${finalTimeMillis}ms")
                appState = AppState.RESULTS
            }
        }
    }

    private fun saveProcessedVideo() {
        val file = resultVideoFile ?: return
        appState           = AppState.PROCESSING
        processingProgress = 0f

        lifecycleScope.launch(Dispatchers.Default) {
            val savedUri = videoEncoder.saveToGallery(file)
            withContext(Dispatchers.Main) {
                appState = AppState.RESULTS
                val msg = if (savedUri != null) "Video saved to gallery!" else "Save failed."
                Toast.makeText(this@MainActivity, msg, Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun resetAll() {
        appState = AppState.IDLE
        modelExecutor.reset()
        sotExecutor.reset()
        staticTracker.reset()
        val old = currentFrame
        currentFrame = null
        old?.recycle()
        processingProgress = 0f
        processingEtaMs    = -1L
        resultVideoFile    = null
        resultVideoUri     = null
        startCamera()   // re-bind camera after reset
    }

    // =========================================================================
    // Camera
    // =========================================================================

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            cameraProvider = future.get()
            val provider   = cameraProvider ?: return@addListener

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView?.surfaceProvider)
            }

            val recorder = Recorder.Builder()
                .setQualitySelector(
                    QualitySelector.from(Quality.HD, FallbackStrategy.lowerQualityOrHigherThan(Quality.HD))
                )
                .build()
            videoCapture = VideoCapture.withOutput(recorder)

            val resolutionSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(Size(640, 640), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                )
                .build()

            val imageAnalysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            // Live viewfinder analysis — YOLO+SORT on each preview frame.
            // Not used in the output video (which uses the bootstrap architecture),
            // but drives any live overlay the UI might want to show.
            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                val bitmap = imageProxyToBitmap(imageProxy)
                imageProxy.close()
                // We run detectAndTrack here just to keep inference warm and
                // demonstrate live detection in the viewfinder if the UI adds it.
                modelExecutor.detectAndTrack(bitmap)
                bitmap.recycle()
            }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA,
                    preview, imageAnalysis, videoCapture)
            } catch (exc: Exception) {
                Log.e(TAG, "Camera binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // =========================================================================
    // Recording
    // =========================================================================

    private fun startRecording() {
        val vc = videoCapture ?: return
        appState = AppState.RECORDING

        val name = SimpleDateFormat(FILENAME_FORMAT, Locale.US).format(System.currentTimeMillis())
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
            put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/CameraX-Video")
        }
        val options = MediaStoreOutputOptions
            .Builder(contentResolver, MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
            .setContentValues(contentValues)
            .build()

        recording = vc.output.prepareRecording(this, options)
            .start(ContextCompat.getMainExecutor(this)) { event ->
                when (event) {
                    is VideoRecordEvent.Start    -> { /* recording begun */ }
                    is VideoRecordEvent.Finalize -> {
                        if (!event.hasError()) {
                            processVideo(event.outputResults.outputUri)
                        } else {
                            appState = AppState.IDLE
                            recording?.close()
                            recording = null
                            Log.e(TAG, "Recording error: ${event.error}")
                        }
                    }
                }
            }
    }

    private fun stopRecording() {
        recording?.stop()
        recording = null
    }

    // =========================================================================
    // Utilities
    // =========================================================================

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val bitmap = imageProxy.toBitmap()
        val rotation = imageProxy.imageInfo.rotationDegrees
        if (rotation == 0) return bitmap

        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        bitmap.recycle()
        return rotated
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestPermissions() {
        activityResultLauncher.launch(REQUIRED_PERMISSIONS)
    }

    private val activityResultLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
            if (allPermissionsGranted()) startCamera()
        }

    companion object {
        private const val TAG             = "TrackerApp"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private val REQUIRED_PERMISSIONS  = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO
        )
    }
}
