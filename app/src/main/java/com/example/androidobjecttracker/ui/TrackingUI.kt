package com.example.androidobjecttracker.ui

import android.net.Uri
import android.widget.VideoView
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.example.androidobjecttracker.pipeline.PipelineParams

enum class AppState {
    IDLE, RECORDING, PROCESSING, RESULTS
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TrackingScreen(
    appState: AppState,
    previewView: PreviewView?,
    resultVideoUri: Uri?,
    processingVideoUri: Uri?,
    pipelineParams: PipelineParams,
    processingProgress: Float,
    onParamsChange: (PipelineParams) -> Unit,
    onRecordToggle: () -> Unit,
    onReset: () -> Unit,
    onPickVideo: () -> Unit
) {
    var showSettings by remember { mutableStateOf(false) }

    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = Color(0xFF00E676),
            secondary = Color(0xFF00B0FF),
            background = Color(0xFF121212),
            surface = Color(0xFF1E1E1E)
        )
    ) {
        Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            Box(modifier = Modifier.fillMaxSize()) {
                when (appState) {
                    AppState.IDLE, AppState.RECORDING -> {
                        CameraCaptureView(
                            previewView,
                            appState == AppState.RECORDING,
                            onRecordToggle,
                            onPickVideo,
                            onOpenSettings = { showSettings = true }
                        )
                    }
                    AppState.PROCESSING -> {
                        ProcessingView(processingVideoUri, processingProgress)
                    }
                    AppState.RESULTS -> {
                        ResultsView(resultVideoUri, onReset)
                    }
                }

                if (showSettings) {
                    SettingsSheet(
                        params = pipelineParams,
                        onDismiss = { showSettings = false },
                        onSave = {
                            onParamsChange(it)
                            showSettings = false
                        }
                    )
                }
            }
        }
    }
}

@Composable
fun SettingsSheet(
    params: PipelineParams,
    onDismiss: () -> Unit,
    onSave: (PipelineParams) -> Unit
) {
    var processEvery by remember { mutableStateOf(params.processEvery.toString()) }
    var trailLength by remember { mutableStateOf(params.trailLength.toString()) }
    var trailMinAlpha by remember { mutableStateOf(params.trailMinAlpha.toString()) }
    var preferredClass by remember { mutableStateOf(params.preferredClass) }
    var nmsScore by remember { mutableStateOf(params.nmsScoreThreshold.toString()) }
    var nmsIou by remember { mutableStateOf(params.nmsIouThreshold.toString()) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Pipeline Settings") },
        text = {
            Column(
                modifier = Modifier
                    .verticalScroll(rememberScrollState())
                    .padding(8.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                OutlinedTextField(
                    value = preferredClass,
                    onValueChange = { preferredClass = it },
                    label = { Text("Preferred Class") },
                    modifier = Modifier.fillMaxWidth()
                )

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(
                        value = processEvery,
                        onValueChange = { processEvery = it },
                        label = { Text("Process Every N") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.weight(1f)
                    )
                    OutlinedTextField(
                        value = trailLength,
                        onValueChange = { trailLength = it },
                        label = { Text("Trail Length") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.weight(1f)
                    )
                }

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(
                        value = trailMinAlpha,
                        onValueChange = { trailMinAlpha = it },
                        label = { Text("Min Alpha") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
                        modifier = Modifier.weight(1f)
                    )
                    OutlinedTextField(
                        value = nmsScore,
                        onValueChange = { nmsScore = it },
                        label = { Text("Score Thresh") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
                        modifier = Modifier.weight(1f)
                    )
                }

                OutlinedTextField(
                    value = nmsIou,
                    onValueChange = { nmsIou = it },
                    label = { Text("IoU Thresh") },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
                    modifier = Modifier.fillMaxWidth()
                )
            }
        },
        confirmButton = {
            Button(onClick = {
                onSave(
                    params.copy(
                        processEvery = processEvery.toIntOrNull() ?: params.processEvery,
                        trailLength = trailLength.toIntOrNull() ?: params.trailLength,
                        trailMinAlpha = trailMinAlpha.toFloatOrNull() ?: params.trailMinAlpha,
                        preferredClass = preferredClass,
                        nmsScoreThreshold = nmsScore.toFloatOrNull() ?: params.nmsScoreThreshold,
                        nmsIouThreshold = nmsIou.toFloatOrNull() ?: params.nmsIouThreshold
                    )
                )
            }) {
                Text("SAVE")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("CANCEL") }
        }
    )
}

@Composable
fun CameraCaptureView(
    previewView: PreviewView?,
    isRecording: Boolean,
    onRecordToggle: () -> Unit,
    onPickVideo: () -> Unit,
    onOpenSettings: () -> Unit
) {
    Box(modifier = Modifier.fillMaxSize()) {
        if (previewView != null) {
            AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
        }

        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.verticalGradient(
                        colors = listOf(Color.Transparent, Color.Black.copy(alpha = 0.4f)),
                        startY = 500f
                    )
                )
        )

        if (!isRecording) {
            Column(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                IconButton(
                    onClick = onOpenSettings,
                    modifier = Modifier
                        .background(Color.Black.copy(alpha = 0.5f), CircleShape)
                ) {
                    Icon(Icons.Default.Settings, contentDescription = "Settings", tint = Color.White)
                }

                Button(
                    onClick = onPickVideo,
                    colors = ButtonDefaults.buttonColors(containerColor = Color.White.copy(alpha = 0.7f), contentColor = Color.Black)
                ) {
                    Text("GALLERY")
                }
            }
        }

        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 64.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (!isRecording) {
                Text(
                    "PIN TRACKER PRO",
                    style = MaterialTheme.typography.headlineMedium,
                    color = Color.White,
                    fontWeight = FontWeight.ExtraBold,
                    letterSpacing = 4.sp,
                    modifier = Modifier.padding(bottom = 32.dp)
                )
            }

            if (isRecording) {
                Text(
                    "REC",
                    color = Color.Red,
                    fontWeight = FontWeight.Bold,
                    fontSize = 18.sp,
                    modifier = Modifier
                        .background(Color.Black.copy(alpha = 0.5f), RoundedCornerShape(4.dp))
                        .padding(horizontal = 12.dp, vertical = 4.dp)
                )
                Spacer(modifier = Modifier.height(16.dp))
            }

            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier
                    .size(88.dp)
                    .clip(CircleShape)
                    .background(Color.White.copy(alpha = 0.2f))
                    .padding(8.dp)
                    .clip(CircleShape)
                    .background(if (isRecording) Color.Red else Color.White)
                    .clickable { onRecordToggle() }
            ) {}

            Spacer(modifier = Modifier.height(16.dp))
            Text(
                if (isRecording) "Stop Recording" else "Start Recording",
                color = Color.White,
                fontWeight = FontWeight.Medium
            )
        }
    }
}

@Composable
fun ProcessingView(videoUri: Uri?, progress: Float) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black),
        contentAlignment = Alignment.Center
    ) {
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
            // Dim the background video to make text readable
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black.copy(alpha = 0.7f))
            )
        }

        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(24.dp)
        ) {
            CircularProgressIndicator(
                progress = { progress },
                color = Color(0xFF00E676),
                strokeWidth = 6.dp,
                modifier = Modifier.size(120.dp)
            )
            
            Text(
                "AI ANALYSIS IN PROGRESS",
                style = MaterialTheme.typography.titleLarge,
                color = Color.White,
                fontWeight = FontWeight.Bold,
                letterSpacing = 2.sp
            )
            
            Text(
                "Analyzing Pin Physics: ${(progress * 100).toInt()}%",
                style = MaterialTheme.typography.bodyLarge,
                color = Color(0xFF00E676),
                fontWeight = FontWeight.Medium
            )
            
            Text(
                "Applying Classical CV Engineering Pipeline...",
                style = MaterialTheme.typography.bodySmall,
                color = Color.Gray
            )
        }
    }
}

@Composable
fun ResultsView(
    videoUri: Uri?,
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
        } else {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator()
            }
        }

        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(32.dp)
                .fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Button(
                onClick = onReset,
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier.height(56.dp).fillMaxWidth(0.6f),
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
            ) {
                Icon(Icons.Default.Refresh, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("RECORD AGAIN", fontWeight = FontWeight.Bold)
            }
        }
    }
}
