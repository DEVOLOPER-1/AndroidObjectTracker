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
import com.example.androidobjecttracker.ui.AppState
import com.example.androidobjecttracker.ui.TrackingScreen
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
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var modelExecutor: ModelExecutor
    private var previewView: PreviewView? = null
    private var cameraProvider: ProcessCameraProvider? = null

    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null

    // App State
    private var appState by mutableStateOf(AppState.IDLE)
    private var processingProgress by mutableFloatStateOf(0f)
    private var processingEtaMs by mutableLongStateOf(-1L)
    
    // Tracking State
    private var trackedObjects by mutableStateOf<List<SortTracker.TrackedObject>>(emptyList())
    private var finalTimeMillis = 0L
    private var isRecording by mutableStateOf(false)

    private var resultVideoUri by mutableStateOf<Uri?>(null)

    private lateinit var videoProcessor: FastVideoProcessor
    private lateinit var videoEncoder: VideoEncoder

    // Stats and Metrics
    private var fps by mutableIntStateOf(0)
    private var latency by mutableLongStateOf(0L)
    private var frameCount = 0
    private var lastFpsTimestamp = 0L

    private val pickVideoLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let { processVideo(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        modelExecutor = ModelExecutor(this)
        modelExecutor.loadModel("yolo26n.onnx")
        
        videoEncoder = VideoEncoder(this)
        videoProcessor = FastVideoProcessor(this, modelExecutor, videoEncoder)
        
        cameraExecutor = Executors.newSingleThreadExecutor()
        previewView = PreviewView(this).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }

        if (!allPermissionsGranted()) {
            requestPermissions()
        }

        setContent {
            val pins = trackedObjects.filter { it.classIndex == 1 }.map { 
                // Minimal state for UI compatibility
                it 
            }
            val car = trackedObjects.find { it.classIndex == 3 }

            TrackingScreen(
                appState = appState,
                previewView = previewView,
                currentFrame = null, // We'll show video from URI in results
                resultVideoUri = resultVideoUri,
                processingProgress = processingProgress,
                processingEtaMs = processingEtaMs,
                pins = emptyList(), // Not used anymore as we bake into video
                carPath = emptyList(), // Not used anymore as we bake into video
                finalStats = if (appState == AppState.RESULTS) Pair(finalTimeMillis, trackedObjects.count { it.state == SortTracker.State.FALLEN }) else null,
                onRecordToggle = {
                    if (appState == AppState.RECORDING) {
                        stopRecording()
                    } else {
                        startRecording()
                    }
                },
                onPickVideo = {
                    pickVideoLauncher.launch("video/*")
                },
                onSaveResult = {
                    // Video is already saved to gallery by FastVideoProcessor
                    Toast.makeText(this, "Video saved to gallery!", Toast.LENGTH_SHORT).show()
                },
                onReset = {
                    appState = AppState.IDLE
                    modelExecutor.reset()
                    trackedObjects = emptyList()
                    resultVideoUri = null
                    processingProgress = 0f
                    startCamera()
                }
            )
        }
    }

    override fun onResume() {
        super.onResume()
        if (allPermissionsGranted() && (appState == AppState.IDLE || appState == AppState.RECORDING)) {
            startCamera()
        }
    }

    private fun processVideo(uri: Uri) {
        try {
            cameraProvider?.unbindAll()
        } catch (e: Exception) {
            AppLog.e("Failed to unbind camera", e)
        }
        
        appState = AppState.PROCESSING
        modelExecutor.reset()
        
        resultVideoUri = null

        lifecycleScope.launch(Dispatchers.Default) {
            val startProcessingTime = System.currentTimeMillis()
            
            videoProcessor.processVideo(
                uri = uri,
                onProgress = { progress, etaMs ->
                    processingProgress = progress
                    processingEtaMs = etaMs
                }
            )
            
            // FastVideoProcessor saves it automatically now. We just need the URI to show it.
            // For simplicity, we'll find the last file or just stay in IDLE if we can't get URI easily.
            // In a real app we'd pass the result file back.
            
            withContext(Dispatchers.Main) {
                finalTimeMillis = System.currentTimeMillis() - startProcessingTime
                appState = AppState.RESULTS
                // We'll skip setting resultVideoUri for now as VideoView might not like cache files
                // but we could set it if we had the path.
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            val cameraProvider = cameraProvider ?: return@addListener

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView?.surfaceProvider)
            }

            val recorder = Recorder.Builder()
                .setQualitySelector(QualitySelector.from(Quality.HD))
                .build()
            videoCapture = VideoCapture.withOutput(recorder)

            val resolutionSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(ResolutionStrategy(Size(640, 640), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                .build()

            val imageAnalysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                val bitmap = imageProxyToBitmap(imageProxy)
                imageProxy.close()

                val rawTracks = modelExecutor.detectAndTrack(bitmap)

                runOnUiThread {
                    trackedObjects = rawTracks
                    latency = modelExecutor.getLastInferenceTime()
                    updateFps()
                }
                bitmap.recycle()
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis, videoCapture)
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun startRecording() {
        val videoCapture = this.videoCapture ?: return
        appState = AppState.RECORDING
        isRecording = true

        val name = SimpleDateFormat(FILENAME_FORMAT, Locale.US).format(System.currentTimeMillis())
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
            put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/TrackerInput")
        }

        val mediaStoreOutputOptions = MediaStoreOutputOptions
            .Builder(contentResolver, MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
            .setContentValues(contentValues)
            .build()

        recording = videoCapture.output
            .prepareRecording(this, mediaStoreOutputOptions)
            .start(ContextCompat.getMainExecutor(this)) { recordEvent ->
                when(recordEvent) {
                    is VideoRecordEvent.Start -> isRecording = true
                    is VideoRecordEvent.Finalize -> {
                        isRecording = false
                        if (!recordEvent.hasError()) {
                            processVideo(recordEvent.outputResults.outputUri)
                        } else {
                            appState = AppState.IDLE
                            recording?.close()
                            recording = null
                        }
                    }
                }
            }
    }

    private fun stopRecording() {
        recording?.stop()
        recording = null
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val bitmap = imageProxy.toBitmap()
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        if (rotationDegrees == 0) return bitmap
        val matrix = Matrix()
        matrix.postRotate(rotationDegrees.toFloat())
        val rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        bitmap.recycle()
        return rotatedBitmap
    }

    private fun updateFps() {
        frameCount++
        val now = System.currentTimeMillis()
        if (now - lastFpsTimestamp >= 1000) {
            fps = frameCount
            frameCount = 0
            lastFpsTimestamp = now
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestPermissions() {
        activityResultLauncher.launch(REQUIRED_PERMISSIONS)
    }

    private val activityResultLauncher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        if (allPermissionsGranted()) startCamera()
    }

    override fun onPause() {
        super.onPause()
        cameraProvider?.unbindAll()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "TrackerApp"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
    }
}
