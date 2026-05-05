package com.example.androidobjecttracker.ui

import android.graphics.Bitmap
import android.graphics.PointF
import android.net.Uri
import android.widget.VideoView
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.example.modelengine.SortTracker

enum class AppState {
    IDLE, RECORDING, PROCESSING, RESULTS
}

@Composable
fun TrackingScreen(
    appState: AppState,
    previewView: PreviewView?,
    currentFrame: Bitmap?,
    resultVideoUri: Uri?,
    processingProgress: Float,
    processingEtaMs: Long,
    pins: List<SortTracker.TrackedObject>,
    carPath: List<PointF>,
    finalStats: Pair<Long, Int>?,
    onRecordToggle: () -> Unit,
    onPickVideo: () -> Unit,
    onSaveResult: () -> Unit,
    onReset: () -> Unit
) {
    Box(modifier = Modifier.fillMaxSize()) {
        when (appState) {
            AppState.IDLE, AppState.RECORDING -> {
                CameraCaptureView(previewView, appState == AppState.RECORDING, onRecordToggle, onPickVideo)
            }
            AppState.PROCESSING -> {
                ProcessingView(processingProgress, processingEtaMs)
            }
            AppState.RESULTS -> {
                ResultsView(currentFrame, resultVideoUri, pins, carPath, finalStats, onSaveResult, onReset)
            }
        }
    }
}

@Composable
fun Label(text: String, x: Float, y: Float, color: Color) {
    val density = androidx.compose.ui.platform.LocalDensity.current.density
    Box(
        modifier = Modifier
            .offset(x = (x / density).dp, y = (y / density).dp)
            .background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(4.dp))
            .padding(horizontal = 4.dp, vertical = 2.dp)
    ) {
        Text(text = text, color = color, fontSize = 10.sp, fontWeight = androidx.compose.ui.text.font.FontWeight.Bold)
    }
}

@Composable
fun CameraCaptureView(
    previewView: PreviewView?,
    isRecording: Boolean,
    onRecordToggle: () -> Unit,
    onPickVideo: () -> Unit
) {
    Box(modifier = Modifier.fillMaxSize()) {
        if (previewView != null) {
            AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
        }
        
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 48.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (!isRecording) {
                Button(onClick = onPickVideo, colors = ButtonDefaults.buttonColors(containerColor = Color.White, contentColor = Color.Black)) {
                    Text("PICK FROM GALLERY")
                }
                Spacer(modifier = Modifier.height(16.dp))
            }
            
            IconButton(
                onClick = onRecordToggle,
                modifier = Modifier
                    .size(80.dp)
                    .background(Color.White, CircleShape)
                    .padding(4.dp)
                    .background(Color.Black, CircleShape)
                    .padding(2.dp)
                    .background(if (isRecording) Color.Red else Color.White, CircleShape)
            ) {}
            
            if (isRecording) {
                Text("RECORDING...", color = Color.Red, modifier = Modifier.padding(top = 8.dp), style = MaterialTheme.typography.labelLarge)
            }
        }
    }
}

@Composable
fun ProcessingView(progress: Float, etaMs: Long) {
    Box(modifier = Modifier.fillMaxSize().background(Color.Black), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            CircularProgressIndicator(progress = { progress }, color = Color.Cyan, strokeWidth = 6.dp, modifier = Modifier.size(100.dp))
            Spacer(modifier = Modifier.height(24.dp))
            Text("PROCESSING VIDEO...", color = Color.White, style = MaterialTheme.typography.headlineSmall)
            Text("${(progress * 100).toInt()}%", color = Color.Cyan, modifier = Modifier.padding(top = 8.dp))
            
            if (etaMs > 0) {
                Text(
                    text = "Estimated time remaining: ${etaMs / 1000}s",
                    color = Color.White.copy(alpha = 0.7f),
                    modifier = Modifier.padding(top = 16.dp),
                    style = MaterialTheme.typography.bodyMedium
                )
            }
        }
    }
}

@Composable
fun ResultsView(
    frame: Bitmap?,
    videoUri: Uri?,
    pins: List<SortTracker.TrackedObject>,
    carPath: List<PointF>,
    stats: Pair<Long, Int>?,
    onSave: () -> Unit,
    onReset: () -> Unit
) {
    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        if (videoUri != null) {
            AndroidView(
                factory = { context ->
                    VideoView(context).apply {
                        setVideoURI(videoUri)
                        setOnPreparedListener { it.isLooping = true }
                        start()
                    }
                },
                modifier = Modifier.fillMaxSize()
            )
        } else if (frame != null) {
            Image(
                bitmap = frame.asImageBitmap(),
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Fit
            )
        }

        // Top Stats Overlay
        stats?.let { (time, count) ->
            Column(
                modifier = Modifier
                    .padding(16.dp)
                    .background(Color.Black.copy(0.7f), RoundedCornerShape(8.dp))
                    .padding(12.dp)
            ) {
                Text("Run Time: ${time / 1000}s", color = Color.White)
                Text("Pins Hit: $count", color = Color.Yellow, style = MaterialTheme.typography.titleLarge)
            }
        }

        // Bottom Actions
        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(32.dp)
                .fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            Button(
                onClick = onReset,
                colors = ButtonDefaults.buttonColors(containerColor = Color.Gray)
            ) {
                Text("RESTART")
            }
            Button(
                onClick = onSave,
                colors = ButtonDefaults.buttonColors(containerColor = Color.Cyan, contentColor = Color.Black)
            ) {
                Text("SAVE VIDEO")
            }
        }
    }
}
