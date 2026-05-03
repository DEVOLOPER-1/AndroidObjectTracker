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
    private var trackedObjects by mutableStateOf<List<SortTracker.Track>>(emptyList())
    private var currentFrame by mutableStateOf<Bitmap?>(null)
    private var finalTimeMillis = 0L
    private var isRecording by mutableStateOf(false)

    private var resultVideoFile: File? = null
    private var resultVideoUri by mutableStateOf<Uri?>(null)

    private lateinit var videoProcessor: FastVideoProcessor
    private lateinit var videoEncoder: VideoEncoder
    private val staticTracker = StaticTracker()
    private lateinit var sotExecutor: SOTExecutor
    private var useSot by mutableStateOf(false)

    // Stats and Metrics
    private var fps by mutableIntStateOf(0)
    private var latency by mutableLongStateOf(0L)
    private var frameCount = 0
    private var lastFpsTimestamp = 0L

    private val pickVideoLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let { processVideo(it) }
    }

    private val UNDERSAMPLING_FACTOR = 10 // Process 1 out of 10 frames (~3 fps)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        modelExecutor = ModelExecutor(this)
        modelExecutor.loadModel("yolo26n.ptl")
        
        sotExecutor = SOTExecutor(this)
        sotExecutor.loadModel("AbaViTrack_lite.ptl")
        
        videoProcessor = FastVideoProcessor(this, modelExecutor)
        videoEncoder = VideoEncoder(this)
        
        cameraExecutor = Executors.newSingleThreadExecutor()
        previewView = PreviewView(this).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }

        if (!allPermissionsGranted()) {
            requestPermissions()
        }

        setContent {
            val pins = staticTracker.getPins()
            val carPath = staticTracker.getCarPath()

            TrackingScreen(
                appState = appState,
                previewView = previewView,
                currentFrame = currentFrame,
                resultVideoUri = resultVideoUri,
                processingProgress = processingProgress,
                processingEtaMs = processingEtaMs,
                pins = pins,
                carPath = carPath,
                useSot = useSot,
                finalStats = if (appState == AppState.RESULTS) Pair(finalTimeMillis, pins.count { it.isFallen }) else null,
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
                onToggleSot = { useSot = !useSot },
                onSaveResult = {
                    saveProcessedVideo()
                },
                onReset = {
                    appState = AppState.IDLE
                    modelExecutor.reset()
                    sotExecutor.reset()
                    staticTracker.reset()
                    trackedObjects = emptyList()
                    currentFrame = null
                    processingProgress = 0f
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
        // Completely stop and unbind camera to free hardware resources for decoding
        try {
            cameraProvider?.unbindAll()
        } catch (e: Exception) {
            AppLog.e("Failed to unbind camera", e)
        }
        
        appState = AppState.PROCESSING
        AppLog.i("Video processing requested for: $uri (SOT: $useSot)")
        modelExecutor.reset()
        staticTracker.reset()
        if (useSot) sotExecutor.reset()
        
        resultVideoFile = null
        resultVideoUri = null

        lifecycleScope.launch(Dispatchers.Default) {
            val startProcessingTime = System.currentTimeMillis()
            AppLog.d("Processing started at: $startProcessingTime")
            
            var frameIndex = 0

            videoProcessor.processVideo(
                uri = uri,
                undersamplingFactor = UNDERSAMPLING_FACTOR,
                onProgress = { progress, etaMs ->
                    processingProgress = progress
                    processingEtaMs = etaMs
                },
                onFrameProcessed = { frame, tracks ->
                    if (frameIndex == 0) {
                        // tracks are now in frame coordinates (1080p)
                        staticTracker.initialize(tracks.map { SortTracker.Detection(it.bbox, it.classIndex, 1.0f) })
                        
                        if (useSot) {
                            val car = tracks.find { it.classIndex == 0 || it.classIndex == 13 || it.classIndex == 2 }
                            car?.let { 
                                AppLog.i("Initializing SOT for car at ${it.bbox}")
                                sotExecutor.init(frame, it.bbox) 
                            }
                        }
                        
                        videoEncoder.start(frame.width, frame.height, 30, UNDERSAMPLING_FACTOR)
                    } else {
                        // Hybrid Logic
                        val detections = tracks.map { SortTracker.Detection(it.bbox, it.classIndex, 1.0f) }
                        if (useSot) {
                            val carBbox = sotExecutor.update(frame)
                            staticTracker.update(detections, carBbox)
                        } else {
                            staticTracker.update(detections)
                        }
                    }

                    // Map static objects for UI
                    val pins = staticTracker.getPins()
                    val carPath = staticTracker.getCarPath()
                    
                    val oldFrame = currentFrame
                    currentFrame = frame
                    trackedObjects = emptyList() // Clear old tracks to prioritize static/SOT visuals
                    
                    // Incremental Encoding with Baked Annotations
                    videoEncoder.addFrame(frame, pins, carPath)

                    oldFrame?.recycle()
                    frameIndex++
                }
            )
            
            resultVideoFile = videoEncoder.finish()
            resultVideoUri = resultVideoFile?.let { f -> Uri.fromFile(f) }

            withContext(Dispatchers.Main) {
                finalTimeMillis = System.currentTimeMillis() - startProcessingTime
                AppLog.i("Processing finished in ${finalTimeMillis}ms")
                appState = AppState.RESULTS
            }
        }
    }

    private fun saveProcessedVideo() {
        val file = resultVideoFile ?: return
        
        appState = AppState.PROCESSING
        processingProgress = 0f
        
        lifecycleScope.launch(Dispatchers.Default) {
            AppLog.i("Saving video to gallery...")
            val finalUri = videoEncoder.saveToGallery(file)
            
            withContext(Dispatchers.Main) {
                appState = AppState.RESULTS
                if (finalUri != null) {
                    Toast.makeText(this@MainActivity, "Video saved to gallery!", Toast.LENGTH_LONG).show()
                } else {
                    Toast.makeText(this@MainActivity, "Failed to save video.", Toast.LENGTH_LONG).show()
                }
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
                .setQualitySelector(
                    QualitySelector.from(
                        Quality.HD,
                        FallbackStrategy.lowerQualityOrHigherThan(Quality.HD)
                    )
                )
                .build()
            videoCapture = VideoCapture.withOutput(recorder)

            val resolutionSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(
                        Size(640, 640),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                    )
                )
                .build()

            val imageAnalysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                val bitmap = imageProxyToBitmap(imageProxy)
                imageProxy.close()

                val rawTracks = modelExecutor.detectAndTrack(bitmap)
                val viewTracks = rawTracks.map { track ->
                    val mappedBbox = scaleYoloToView(track.bbox, bitmap.width, bitmap.height)
                    val mappedPath = track.path.map { scaleYoloToView(it, bitmap.width, bitmap.height) }
                    track.copy(bbox = mappedBbox).apply {
                        path.clear()
                        path.addAll(mappedPath)
                    }
                }

                runOnUiThread {
                    trackedObjects = viewTracks
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

    private fun scaleYoloToView(bbox: RectF, imgW: Int, imgH: Int): RectF {
        // YOLO output is relative to 640x640 input
        val scaleX = imgW.toFloat() / 640f
        val scaleY = imgH.toFloat() / 640f
        
        val imgBbox = RectF(
            bbox.left * scaleX,
            bbox.top * scaleY,
            bbox.right * scaleX,
            bbox.bottom * scaleY
        )
        
        // Now scale from image to view
        return scaleBboxToView(imgBbox, imgW, imgH)
    }

    private fun startRecording() {
        val videoCapture = this.videoCapture ?: return
        appState = AppState.RECORDING
        isRecording = true

        val name = SimpleDateFormat(FILENAME_FORMAT, Locale.US)
            .format(System.currentTimeMillis())
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
            put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/CameraX-Video")
        }

        val mediaStoreOutputOptions = MediaStoreOutputOptions
            .Builder(contentResolver, MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
            .setContentValues(contentValues)
            .build()

        recording = videoCapture.output
            .prepareRecording(this, mediaStoreOutputOptions)
            .start(ContextCompat.getMainExecutor(this)) { recordEvent ->
                when(recordEvent) {
                    is VideoRecordEvent.Start -> {
                        isRecording = true
                    }
                    is VideoRecordEvent.Finalize -> {
                        isRecording = false
                        if (!recordEvent.hasError()) {
                            val uri = recordEvent.outputResults.outputUri
                            val msg = "Video capture succeeded: $uri"
                            Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                            Log.d(TAG, msg)
                            processVideo(uri)
                        } else {
                            appState = AppState.IDLE
                            recording?.close()
                            recording = null
                            Log.e(TAG, "Video capture ends with error: ${recordEvent.error}")
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
        // Original bitmap is no longer needed
        bitmap.recycle()
        return rotatedBitmap
    }

    private fun scaleBboxToView(bbox: RectF, imgW: Int, imgH: Int): RectF {
        val view = previewView ?: return bbox
        val viewW = view.width.toFloat()
        val viewH = view.height.toFloat()
        
        // Calculate the scale factors while preserving aspect ratio (FIT_CENTER/FILL_CENTER)
        // ImageAnalysis usually outputs images in a landscape-like orientation relative to sensor,
        // but our bitmap is already rotated to match the portrait display.
        
        // If image is 1088x1088 (square) and screen is 720x1560 (tall)
        // We need to find how that square is mapped into the tall view.
        val imgAspectRatio = imgW.toFloat() / imgH
        val viewAspectRatio = viewW / viewH

        var finalScale: Float
        var offsetX = 0f
        var offsetY = 0f

        if (viewAspectRatio > imgAspectRatio) {
            // View is wider than image (relative to aspect ratio)
            finalScale = viewH / imgH
            offsetX = (viewW - imgW * finalScale) / 2f
        } else {
            // View is taller than image
            finalScale = viewW / imgW
            offsetY = (viewH - imgH * finalScale) / 2f
        }
        
        return RectF(
            bbox.left * finalScale + offsetX,
            bbox.top * finalScale + offsetY,
            bbox.right * finalScale + offsetX,
            bbox.bottom * finalScale + offsetY
        )
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
        private val REQUIRED_PERMISSIONS = mutableListOf (
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO
        ).toTypedArray()
    }
}
