package com.example.androidobjecttracker.ui

import android.graphics.RectF
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView

@Composable
fun TrackingScreen(
    previewView: PreviewView,
    trackedBboxes: Map<Int, RectF>,
    fps: Int,
    latency: Long,
    isRecording: Boolean,
    onRoiSelected: (RectF) -> Unit,
    onReset: () -> Unit,
    onToggleRecording: () -> Unit
) {
    var selectionStart by remember { mutableStateOf<Offset?>(null) }
    var selectionCurrent by remember { mutableStateOf<Offset?>(null) }
    var isAddingObject by remember { mutableStateOf(false) }

    Box(modifier = Modifier.fillMaxSize()) {
        // 1. Camera Preview
        AndroidView(
            factory = { previewView },
            modifier = Modifier.fillMaxSize()
        )

        // 2. Overlays
        Canvas(
            modifier = Modifier
                .fillMaxSize()
                .then(
                    if (isAddingObject) {
                        Modifier.pointerInput(Unit) {
                            detectDragGestures(
                                onDragStart = { offset ->
                                    selectionStart = offset
                                    selectionCurrent = offset
                                },
                                onDrag = { change, _ ->
                                    selectionCurrent = change.position
                                },
                                onDragEnd = {
                                    if (selectionStart != null && selectionCurrent != null) {
                                        val rect = RectF(
                                            minOf(selectionStart!!.x, selectionCurrent!!.x),
                                            minOf(selectionStart!!.y, selectionCurrent!!.y),
                                            maxOf(selectionStart!!.x, selectionCurrent!!.x),
                                            maxOf(selectionStart!!.y, selectionCurrent!!.y)
                                        )
                                        if (rect.width() > 20 && rect.height() > 20) {
                                            onRoiSelected(rect)
                                            isAddingObject = false // Exit add mode after selection
                                        }
                                    }
                                    selectionStart = null
                                    selectionCurrent = null
                                }
                            )
                        }
                    } else Modifier
                )
        ) {
            // Draw ROI Selector (while dragging)
            val start = selectionStart
            val current = selectionCurrent
            if (isAddingObject && start != null && current != null) {
                drawRect(
                    color = Color.Yellow,
                    topLeft = Offset(
                        minOf(start.x, current.x),
                        minOf(start.y, current.y)
                    ),
                    size = Size(
                        kotlin.math.abs(current.x - start.x),
                        kotlin.math.abs(current.y - start.y)
                    ),
                    style = Stroke(width = 2.dp.toPx())
                )
            }

            // Draw ALL Tracked Bounding Boxes
            trackedBboxes.forEach { (id, bbox) ->
                drawRect(
                    color = getBoxColor(id),
                    topLeft = Offset(bbox.left, bbox.top),
                    size = Size(bbox.width(), bbox.height()),
                    style = Stroke(width = 4.dp.toPx())
                )
            }
        }

        // 3. Status/Instruction text
        if (isAddingObject) {
            Text(
                text = "Draw a box around the object",
                color = Color.Yellow,
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 100.dp)
                    .background(Color.Black.copy(alpha = 0.5f), RoundedCornerShape(4.dp))
                    .padding(8.dp)
            )
        }

        // 4. Stats Overlay
        StatsOverlay(fps, latency)

        // 5. Controls
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 32.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(24.dp)
            ) {
                // Reset Button
                IconButton(
                    onClick = {
                        onReset()
                        isAddingObject = false
                    },
                    modifier = Modifier
                        .background(Color.DarkGray.copy(alpha = 0.7f), CircleShape)
                        .size(56.dp)
                ) {
                    Icon(Icons.Default.Refresh, contentDescription = "Reset", tint = Color.White)
                }

                // Recording Button
                IconButton(
                    onClick = onToggleRecording,
                    modifier = Modifier
                        .size(72.dp)
                        .background(if (isRecording) Color.Red else Color.White, CircleShape)
                        .padding(4.dp)
                        .background(Color.Black, CircleShape)
                        .padding(2.dp)
                        .background(if (isRecording) Color.Red else Color.White, CircleShape)
                ) {
                    // Visual indicator for recording
                }

                // Add Object Button
                IconButton(
                    onClick = { isAddingObject = !isAddingObject },
                    modifier = Modifier
                        .background(if (isAddingObject) Color.Yellow else Color.DarkGray.copy(alpha = 0.7f), CircleShape)
                        .size(56.dp)
                ) {
                    Icon(
                        Icons.Default.Add, 
                        contentDescription = "Add Object", 
                        tint = if (isAddingObject) Color.Black else Color.White
                    )
                }
            }
            
            if (isRecording) {
                Text(
                    text = "RECORDING",
                    color = Color.Red,
                    fontSize = 12.sp,
                    modifier = Modifier.padding(top = 8.dp)
                )
            }
        }
    }
}

fun getBoxColor(id: Int): Color {
    val colors = listOf(Color.Green, Color.Cyan, Color.Magenta, Color.Yellow, Color.Red, Color.Blue)
    return colors[id % colors.size]
}

@Composable
fun StatsOverlay(fps: Int, latency: Long) {
    Column(
        modifier = Modifier
            .padding(16.dp)
            .background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(8.dp))
            .padding(8.dp)
            .width(120.dp),
        horizontalAlignment = Alignment.Start
    ) {
        Text(text = "FPS: $fps", color = Color.White, fontSize = 14.sp)
        Text(text = "Latency: ${latency}ms", color = Color.White, fontSize = 14.sp)
    }
}
