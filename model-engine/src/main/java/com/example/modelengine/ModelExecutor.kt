package com.example.modelengine

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock

/**
 * ModelExecutor handles the initialization of YoloExecutor and SortTracker.
 * Acts as the bridge between the app and the tracking engine.
 */
class ModelExecutor(private val context: Context) {
    private var yoloExecutor: YoloExecutor? = null
    private val sortTracker = SortTracker()
    private var lastInferenceTime: Long = 0

    fun loadModel(assetName: String) {
        try {
            yoloExecutor = YoloExecutor(context, assetName)
            AppLog.i("ONNX Model successfully loaded: $assetName")
        } catch (e: Exception) {
            AppLog.e("Failed to load ONNX model", e)
        }
    }

    fun reset() {
        sortTracker.reset()
        lastInferenceTime = 0
    }

    /**
     * Executes YOLO inference via ONNX and updates object tracks.
     */
    fun detectAndTrack(bitmap: Bitmap): List<SortTracker.TrackedObject> {
        val executor = yoloExecutor ?: return emptyList()
        val startTime = SystemClock.elapsedRealtime()

        // 1. Inference via ONNX
        val detections = executor.detect(bitmap)

        // 2. Update Tracker
        val tracks = sortTracker.update(detections)

        lastInferenceTime = SystemClock.elapsedRealtime() - startTime
        return tracks
    }

    fun getLastInferenceTime(): Long = lastInferenceTime
}
