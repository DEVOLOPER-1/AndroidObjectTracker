package com.example.androidobjecttracker.ui

import android.graphics.Bitmap
import android.graphics.PointF
import android.net.Uri
import android.widget.VideoView
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.foundation.Image
import com.example.modelengine.StaticTracker

enum class AppState {
    IDLE, RECORDING, PROCESSING, RESULTS
}

// =============================================================================
// Root screen router
// =============================================================================

@Composable
fun TrackingScreen(
    appState: AppState,
    previewView: PreviewView?,
    currentFrame: Bitmap?,
    resultVideoUri: Uri?,
    processingProgress: Float,
    processingEtaMs: Long,
    pins: List<StaticTracker.PinState>,
    carPath: List<PointF>,
    useSot: Boolean,
    finalStats: Pair<Long, Int>?,
    onRecordToggle: () -> Unit,
    onPickVideo: () -> Unit,
    onToggleSot: () -> Unit,
    onSaveResult: () -> Unit,
    onReset: () -> Unit
) {
    Box(modifier = Modifier.fillMaxSize()) {
        when (appState) {
            AppState.IDLE, AppState.RECORDING -> {
                CameraCaptureView(
                    previewView    = previewView,
                    isRecording    = appState == AppState.RECORDING,
                    useSot         = useSot,
                    onRecordToggle = onRecordToggle,
                    onPickVideo    = onPickVideo,
                    onToggleSot    = onToggleSot
                )
            }
            AppState.PROCESSING -> {
                ProcessingView(
                    currentFrame = currentFrame,
                    progress     = processingProgress,
                    etaMs        = processingEtaMs
                )
            }
            AppState.RESULTS -> {
                ResultsView(
                    videoUri  = resultVideoUri,
                    lastFrame = currentFrame,
                    pins      = pins,
                    stats     = finalStats,
                    onSave    = onSaveResult,
                    onReset   = onReset
                )
            }
        }
    }
}

// =============================================================================
// IDLE / RECORDING — Camera preview + controls
// =============================================================================

@Composable
fun CameraCaptureView(
    previewView: PreviewView?,
    isRecording: Boolean,
    useSot: Boolean,
    onRecordToggle: () -> Unit,
    onPickVideo: () -> Unit,
    onToggleSot: () -> Unit
) {
    Box(modifier = Modifier.fillMaxSize()) {

        // Live camera feed
        if (previewView != null) {
            AndroidView(
                factory  = { previewView },
                modifier = Modifier.fillMaxSize()
            )
        }

        // ---- Advanced-tracking toggle (top-right) ----
        if (!isRecording) {
            Row(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(16.dp)
                    .background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(8.dp))
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text  = "SOT TRACKING",
                    color = Color.White,
                    style = MaterialTheme.typography.labelSmall
                )
                Spacer(modifier = Modifier.width(8.dp))
                Switch(checked = useSot, onCheckedChange = { onToggleSot() })
            }
        }

        // ---- Bottom controls ----
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 48.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (!isRecording) {
                Button(
                    onClick = onPickVideo,
                    colors  = ButtonDefaults.buttonColors(
                        containerColor = Color.White,
                        contentColor   = Color.Black
                    )
                ) {
                    Text("PICK FROM GALLERY")
                }
                Spacer(modifier = Modifier.height(16.dp))
            }

            // Record button
            Box(
                modifier = Modifier
                    .size(80.dp)
                    .background(Color.White, CircleShape)
                    .padding(4.dp)
                    .background(Color.Black, CircleShape)
                    .padding(2.dp)
                    .background(if (isRecording) Color.Red else Color.White, CircleShape),
                contentAlignment = Alignment.Center
            ) {
                IconButton(onClick = onRecordToggle, modifier = Modifier.fillMaxSize()) {}
            }

            if (isRecording) {
                Text(
                    text     = "● RECORDING",
                    color    = Color.Red,
                    modifier = Modifier.padding(top = 8.dp),
                    style    = MaterialTheme.typography.labelLarge
                )
            }
        }
    }
}

// =============================================================================
// PROCESSING — Live annotated-frame preview + progress overlay
//
// BUG 9 FIX: The previous version drew a second Canvas layer (carPath + pins)
// on top of the current frame in SCREEN coordinates, while the path/pin
// coordinates are in FRAME pixel space.  These two spaces differ whenever the
// frame is not the same size as the screen (always true on real devices), so
// the overlay annotations appeared at completely wrong positions AND were
// rendered twice (once baked into the bitmap by FastVideoProcessor, once by
// the Compose Canvas).
//
// Fix: display the annotated frame as-is (annotations already baked in by
// FastVideoProcessor) and show only the non-spatial UI elements (progress
// bar, ETA text) on top.
// =============================================================================

@Composable
fun ProcessingView(
    currentFrame: Bitmap?,
    progress: Float,
    etaMs: Long
) {
    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {

        // Annotated frame (annotations already baked by FastVideoProcessor)
        if (currentFrame != null) {
            Image(
                bitmap             = currentFrame.asImageBitmap(),
                contentDescription = "Processing frame",
                modifier           = Modifier.fillMaxSize(),
                contentScale       = ContentScale.Fit
            )
        }

        // Semi-transparent progress overlay — pure UI elements, no spatial drawing
        Box(
            modifier        = Modifier
                .fillMaxSize()
                .background(Color.Black.copy(alpha = 0.35f)),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                CircularProgressIndicator(
                    progress  = { progress },
                    color     = Color.Cyan,
                    strokeWidth = 6.dp,
                    modifier  = Modifier.size(100.dp)
                )
                Spacer(modifier = Modifier.height(24.dp))
                Text(
                    text  = "PROCESSING VIDEO…",
                    color = Color.White,
                    style = MaterialTheme.typography.headlineSmall
                )
                Text(
                    text     = "${(progress * 100).toInt()}%",
                    color    = Color.Cyan,
                    modifier = Modifier.padding(top = 8.dp),
                    style    = MaterialTheme.typography.titleLarge
                )
                if (etaMs > 0) {
                    Text(
                        text     = "ETA: ${etaMs / 1000}s remaining",
                        color    = Color.White.copy(alpha = 0.7f),
                        modifier = Modifier.padding(top = 16.dp),
                        style    = MaterialTheme.typography.bodyMedium
                    )
                } else {
                    Text(
                        text     = "Calculating ETA…",
                        color    = Color.White.copy(alpha = 0.5f),
                        modifier = Modifier.padding(top = 16.dp),
                        style    = MaterialTheme.typography.bodyMedium
                    )
                }
            }
        }
    }
}

// =============================================================================
// RESULTS — Annotated video playback + score summary + actions
//
// BUG 10 FIX: The previous version drew a second Canvas overlay on the result
// frame/video in wrong screen coordinates (same root cause as BUG 9).
//
// Fix: the VideoView plays the annotated MP4 (annotations baked in by
// FastVideoProcessor/VideoEncoder), so no Compose Canvas overlay is needed.
// Only non-spatial elements are drawn on top: the stats card and the buttons.
// =============================================================================

@Composable
fun ResultsView(
    videoUri:  Uri?,
    lastFrame: Bitmap?,
    pins:      List<StaticTracker.PinState>,
    stats:     Pair<Long, Int>?,
    onSave:    () -> Unit,
    onReset:   () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
    ) {

        // ---- Main content: annotated video or last annotated frame ----
        if (videoUri != null) {
            // Play the encoded output video (annotations already baked in)
            AndroidView(
                factory  = { context ->
                    VideoView(context).apply {
                        setVideoURI(videoUri)
                        setOnPreparedListener { mp ->
                            mp.isLooping = true
                            mp.start()
                        }
                    }
                },
                modifier = Modifier.fillMaxSize()
            )
        } else if (lastFrame != null) {
            // Fallback: show the last processed frame (annotations baked in)
            Image(
                bitmap             = lastFrame.asImageBitmap(),
                contentDescription = "Last processed frame",
                modifier           = Modifier.fillMaxSize(),
                contentScale       = ContentScale.Fit
            )
        }

        // ---- Score summary overlay (top-left) ----
        stats?.let { (elapsedMs, pinsHit) ->
            Column(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(16.dp)
                    .background(Color.Black.copy(alpha = 0.72f), RoundedCornerShape(10.dp))
                    .padding(14.dp)
            ) {
                Text(
                    text  = "⏱  ${elapsedMs / 1000}s",
                    color = Color.White,
                    style = MaterialTheme.typography.titleMedium
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text  = "🎳  $pinsHit pin${if (pinsHit != 1) "s" else ""} knocked down",
                    color = Color.Yellow,
                    style = MaterialTheme.typography.titleLarge
                )

                // Per-pin fall order summary
                val fallenPins = pins.filter { it.isFallen }.sortedBy { it.fallOrder }
                if (fallenPins.isNotEmpty()) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text     = "Fall order: " + fallenPins.joinToString(" → ") { "#${it.fallOrder}" },
                        color    = Color.White.copy(alpha = 0.85f),
                        fontSize = 13.sp
                    )
                }
            }
        }

        // ---- Action buttons (bottom-center) ----
        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(32.dp)
                .fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            Button(
                onClick = onReset,
                colors  = ButtonDefaults.buttonColors(containerColor = Color.DarkGray)
            ) {
                Text("RESTART")
            }
            Button(
                onClick = onSave,
                colors  = ButtonDefaults.buttonColors(
                    containerColor = Color.Cyan,
                    contentColor   = Color.Black
                )
            ) {
                Text("SAVE TO GALLERY")
            }
        }
    }
}
