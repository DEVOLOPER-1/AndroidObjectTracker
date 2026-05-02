package com.example.androidobjecttracker

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.os.Build
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
import com.example.androidobjecttracker.ui.TrackingScreen
import com.example.modelengine.ModelExecutor
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var modelExecutor: ModelExecutor
    private var previewView: PreviewView? = null

    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null

    // Compose States
    private var trackedBboxes by mutableStateOf<Map<Int, RectF>>(emptyMap())
    private var fps by mutableIntStateOf(0)
    private var latency by mutableLongStateOf(0L)
    private var isRecording by mutableStateOf(false)

    // FPS Calculation
    private var frameCount = 0
    private var lastFpsTimestamp = 0L

    // Initialization state
    private val pendingRois = mutableListOf<RectF>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        modelExecutor = ModelExecutor(this)
        modelExecutor.loadModel("AbaViTrack_lite.ptl")
        
        cameraExecutor = Executors.newSingleThreadExecutor()
        previewView = PreviewView(this)

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissions()
        }

        setContent {
            TrackingScreen(
                previewView = previewView!!,
                trackedBboxes = trackedBboxes,
                fps = fps,
                latency = latency,
                isRecording = isRecording,
                onRoiSelected = { roi ->
                    synchronized(pendingRois) {
                        pendingRois.add(roi)
                    }
                },
                onReset = {
                    modelExecutor.reset()
                    trackedBboxes = emptyMap()
                    synchronized(pendingRois) {
                        pendingRois.clear()
                    }
                },
                onToggleRecording = {
                    if (isRecording) stopRecording() else startRecording()
                }
            )
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView?.surfaceProvider)
            }

            val recorder = Recorder.Builder()
                .setQualitySelector(QualitySelector.from(Quality.HIGHEST))
                .build()
            videoCapture = VideoCapture.withOutput(recorder)

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(1280, 720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                processImageProxy(imageProxy)
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

    private fun processImageProxy(imageProxy: ImageProxy) {
        val bitmap = imageProxyToBitmap(imageProxy)
        imageProxy.close()

        // 1. Process new ROIs
        synchronized(pendingRois) {
            if (pendingRois.isNotEmpty()) {
                Log.d(TAG, "Processing ${pendingRois.size} pending ROIs")
                for (roi in pendingRois) {
                    val imageRoi = scaleBboxToImage(roi, bitmap.width, bitmap.height)
                    Log.d(TAG, "Adding tracker for ROI: $roi -> Image ROI: $imageRoi")
                    modelExecutor.addTracker(bitmap, imageRoi)
                }
                pendingRois.clear()
            }
        }

        // 2. Run Tracking
        val results = modelExecutor.trackAll(bitmap)
        
        // Update UI state using a temporary map to trigger recomposition properly
        val updatedBboxes = results.mapValues { (id, bbox) ->
            val viewBbox = scaleBboxToView(bbox, bitmap.width, bitmap.height)
            if (results.size < 5) { // Avoid spamming if many trackers
                Log.d(TAG, "Object $id: img=$bbox -> view=$viewBbox")
            }
            viewBbox
        }
        
        // Use the main thread to update Compose state if not already there
        runOnUiThread {
            if (updatedBboxes.isNotEmpty() || trackedBboxes.isNotEmpty()) {
                trackedBboxes = updatedBboxes
            }
            latency = modelExecutor.getLastInferenceTime()
            updateFps()
        }
    }

    private fun startRecording() {
        val videoCapture = this.videoCapture ?: return

        isRecording = true

        val name = SimpleDateFormat(FILENAME_FORMAT, Locale.US)
            .format(System.currentTimeMillis())
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
            if (Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
                put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/CameraX-Video")
            }
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
                            val msg = "Video capture succeeded: ${recordEvent.outputResults.outputUri}"
                            Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                            Log.d(TAG, msg)
                        } else {
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
        val bitmap = Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
        imageProxy.planes[0].buffer.rewind()
        bitmap.copyPixelsFromBuffer(imageProxy.planes[0].buffer)
        
        val matrix = Matrix()
        matrix.postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
        
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun scaleBboxToView(bbox: RectF, imgW: Int, imgH: Int): RectF {
        val viewW = previewView?.width ?: 1
        val viewH = previewView?.height ?: 1
        
        // CameraX ImageAnalysis usually provides images in a specific orientation.
        // The bitmap we use is already rotated to match the display.
        // We need to handle the aspect ratio: CenterCrop or Fit? 
        // PreviewView usually defaults to FILL_CENTER.
        
        val scaleX = viewW.toFloat() / imgW
        val scaleY = viewH.toFloat() / imgH
        
        // Use the same scale for both if maintaining aspect ratio, 
        // but here we map directly to the view's pixel space.
        return RectF(
            bbox.left * scaleX,
            bbox.top * scaleY,
            bbox.right * scaleX,
            bbox.bottom * scaleY
        )
    }

    private fun scaleBboxToImage(roi: RectF, imgW: Int, imgH: Int): RectF {
        val viewW = previewView?.width ?: 1
        val viewH = previewView?.height ?: 1
        
        val scaleX = imgW.toFloat() / viewW
        val scaleY = imgH.toFloat() / viewH
        
        return RectF(
            roi.left * scaleX,
            roi.top * scaleY,
            roi.right * scaleX,
            roi.bottom * scaleY
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
        ).apply {
            if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
            }
        }.toTypedArray()
    }
}
