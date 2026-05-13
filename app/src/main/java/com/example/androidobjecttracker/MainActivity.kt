package com.example.androidobjecttracker

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.camera.view.PreviewView
import androidx.compose.runtime.*
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.androidobjecttracker.pipeline.PipelineOrchestrator
import com.example.androidobjecttracker.pipeline.PipelineParams
import com.example.androidobjecttracker.processor.FastVideoProcessor
import com.example.androidobjecttracker.processor.VideoEncoder
import com.example.androidobjecttracker.ui.AppState
import com.example.androidobjecttracker.ui.TrackingScreen
import com.example.androidobjecttracker.utils.AppLog
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private lateinit var cameraExecutor: ExecutorService
    private var previewView: PreviewView? = null
    private var cameraProvider: ProcessCameraProvider? = null

    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null

    // App State
    private var appState by mutableStateOf(AppState.IDLE)
    private var resultVideoUri by mutableStateOf<Uri?>(null)
    private var pipelineParams by mutableStateOf(PipelineParams())
    private var processingProgress by mutableFloatStateOf(0f)

    private lateinit var orchestrator: PipelineOrchestrator
    private lateinit var videoProcessor: FastVideoProcessor

    private val pickVideoLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let { startInAppProcessing(it) }
    }

    private var lastClickTime = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        orchestrator = PipelineOrchestrator(this)
        videoProcessor = FastVideoProcessor(this, orchestrator, VideoEncoder(this))
        
        cameraExecutor = Executors.newSingleThreadExecutor()
        previewView = PreviewView(this).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }

        if (!allPermissionsGranted()) {
            requestPermissions()
        }

        setContent {
            TrackingScreen(
                appState = appState,
                previewView = previewView,
                resultVideoUri = resultVideoUri,
                pipelineParams = pipelineParams,
                processingProgress = processingProgress,
                onParamsChange = { pipelineParams = it },
                onRecordToggle = {
                    val now = System.currentTimeMillis()
                    if (now - lastClickTime < 1000) return@TrackingScreen
                    lastClickTime = now

                    AppLog.i("Record Toggle Clicked. State: $appState")
                    if (appState == AppState.RECORDING) {
                        stopRecording()
                    } else {
                        startRecording()
                    }
                },
                onReset = {
                    appState = AppState.IDLE
                    resultVideoUri = null
                    processingProgress = 0f
                    startCamera()
                },
                onPickVideo = {
                    pickVideoLauncher.launch("video/*")
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

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            val cameraProvider = cameraProvider ?: return@addListener

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView?.surfaceProvider)
            }

            val qualitySelector = QualitySelector.from(
                Quality.HD,
                FallbackStrategy.higherQualityOrLowerThan(Quality.HD)
            )
            val recorder = Recorder.Builder()
                .setQualitySelector(qualitySelector)
                .build()
            videoCapture = VideoCapture.withOutput(recorder)

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, videoCapture)
                AppLog.i("Camera bound successfully")
            } catch (exc: Exception) {
                AppLog.e("Use case binding failed", exc)
                Toast.makeText(this, "Camera binding failed: ${exc.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun startRecording() {
        AppLog.i("startRecording() called")
        val videoCapture = this.videoCapture
        if (videoCapture == null) {
            AppLog.e("videoCapture is null, cannot start recording")
            Toast.makeText(this, "Camera not ready", Toast.LENGTH_SHORT).show()
            return
        }
        
        appState = AppState.RECORDING

        val name = SimpleDateFormat(FILENAME_FORMAT, Locale.US).format(System.currentTimeMillis())
        val movieDir = File(externalMediaDirs.firstOrNull() ?: filesDir, "TrackerInput")
        if (!movieDir.exists()) movieDir.mkdirs()
        val movieFile = File(movieDir, "$name.mp4")

        val fileOutputOptions = FileOutputOptions.Builder(movieFile).build()

        recording = videoCapture.output
            .prepareRecording(this, fileOutputOptions)
            .start(ContextCompat.getMainExecutor(this)) { recordEvent ->
                when(recordEvent) {
                    is VideoRecordEvent.Start -> {
                        AppLog.i("Recording started")
                    }
                    is VideoRecordEvent.Finalize -> {
                        if (!recordEvent.hasError()) {
                            val uri = Uri.fromFile(movieFile)
                            AppLog.i("Recording finalized: ${movieFile.absolutePath}")
                            startInAppProcessing(uri)
                        } else {
                            AppLog.e("Recording failed: ${recordEvent.error}")
                            appState = AppState.IDLE
                        }
                    }
                }
            }
    }

    private fun stopRecording() {
        recording?.stop()
        recording = null
    }

    private fun startInAppProcessing(videoUri: Uri) {
        appState = AppState.PROCESSING
        processingProgress = 0f
        
        lifecycleScope.launch(Dispatchers.Main) {
            try {
                cameraProvider?.unbindAll()
                
                val resultUri = videoProcessor.processVideo(
                    uri = videoUri,
                    params = pipelineParams,
                    onProgress = { progress ->
                        processingProgress = progress
                    }
                )
                
                if (resultUri != null) {
                    resultVideoUri = resultUri
                    appState = AppState.RESULTS
                    Toast.makeText(this@MainActivity, "Processing complete!", Toast.LENGTH_SHORT).show()
                } else {
                    appState = AppState.IDLE
                    Toast.makeText(this@MainActivity, "Processing failed!", Toast.LENGTH_LONG).show()
                }
            } catch (e: Exception) {
                AppLog.e("Processing error", e)
                appState = AppState.IDLE
                Toast.makeText(this@MainActivity, "Error: ${e.message}", Toast.LENGTH_LONG).show()
            }
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
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO,
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) 
                Manifest.permission.READ_MEDIA_VIDEO 
            else 
                Manifest.permission.READ_EXTERNAL_STORAGE
        )
    }
}
