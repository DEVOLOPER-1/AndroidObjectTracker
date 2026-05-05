package com.example.modelengine

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock

/**
 * Orchestrates the Bootstrap Tracking Architecture:
 *
 *   Frame 0  → YoloExecutor runs ONCE.
 *              Car detections (class 3) → SortTracker (centroid identity).
 *              Pin detections (class 1) → StaticTracker (luminance baseline).
 *
 *   Frame N+ → YoloExecutor runs ONLY for the car.
 *              StaticTracker handles all pin fall detection — no YOLO for pins.
 *
 * This eliminates per-frame jitter on static objects and cuts inference cost
 * significantly after the first frame.
 */
class ModelExecutor(private val context: Context) {

    private var yoloExecutor: YoloExecutor? = null
    private val sortTracker  = SortTracker()
    private val staticTracker = StaticTracker()

    @Volatile private var isBootstrapped = false
    private var lastInferenceTime: Long = 0

    // -------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------

    fun loadModel(assetName: String) {
        try {
            yoloExecutor = YoloExecutor(context, assetName)
            AppLog.i("ModelExecutor: ONNX model loaded → $assetName")
        } catch (e: Exception) {
            AppLog.e("ModelExecutor: Failed to load model", e)
        }
    }

    /**
     * Full reset: clears both trackers and drops the bootstrap flag so the
     * next call to detectAndTrack() is treated as frame 0 again.
     */
    fun reset() {
        sortTracker.reset()
        staticTracker.reset()
        isBootstrapped = false
        lastInferenceTime = 0
        AppLog.i("ModelExecutor: reset — bootstrap gate cleared.")
    }

    fun getLastInferenceTime(): Long = lastInferenceTime

    // -------------------------------------------------------------------
    // Core pipeline
    // -------------------------------------------------------------------

    fun detectAndTrack(bitmap: Bitmap): List<SortTracker.TrackedObject> {
        val executor = yoloExecutor ?: return emptyList()
        val startTime = SystemClock.elapsedRealtime()

        val tracks = if (!isBootstrapped) {
            runBootstrap(executor, bitmap)
        } else {
            runTracking(executor, bitmap)
        }

        lastInferenceTime = SystemClock.elapsedRealtime() - startTime
        return tracks
    }

    // -------------------------------------------------------------------
    // Frame 0 — Bootstrap
    // -------------------------------------------------------------------

    private fun runBootstrap(
        executor: YoloExecutor,
        bitmap: Bitmap
    ): List<SortTracker.TrackedObject> {

        AppLog.i("ModelExecutor: *** BOOTSTRAP FRAME — running YOLO ***")
        val allDetections = executor.detect(bitmap)

        // ── DIAGNOSTIC: print every raw detection so we can see actual class indices ──
        if (allDetections.isEmpty()) {
            AppLog.i("ModelExecutor: YOLO returned 0 detections (confidence threshold too high?)")
        } else {
            allDetections.forEachIndexed { i, det ->
                AppLog.i(
                    "ModelExecutor: Detection[$i] " +
                            "class=${det.classIndex} conf=${"%.2f".format(det.confidence)} " +
                            "bbox=(${det.bbox.left.toInt()},${det.bbox.top.toInt()}," +
                            "${det.bbox.right.toInt()},${det.bbox.bottom.toInt()})"
                )
            }
        }
        // ── END DIAGNOSTIC ──

        val pinDetections = allDetections.filter { it.classIndex == 1 }
        val carDetections = allDetections.filter { it.classIndex == 3 }
        // ... rest unchanged
        AppLog.i(
            "ModelExecutor: Bootstrap found ${carDetections.size} car(s) " +
                    "and ${pinDetections.size} pin(s)."
        )

        // Hand pins off to StaticTracker
        if (pinDetections.isNotEmpty()) {
            staticTracker.initialize(bitmap, pinDetections.map { it.bbox })
        } else {
            AppLog.i("ModelExecutor: No pins detected on frame 0 — StaticTracker idle.")
        }

        // Hand car(s) off to SortTracker for identity assignment
        val carTracks = sortTracker.update(carDetections)

        isBootstrapped = true

        // On frame 0 the pin luminance equals baseline → all STANDING, correct.
        val pinTracks = staticTracker.update(bitmap)

        return carTracks + pinTracks
    }

    // -------------------------------------------------------------------
    // Frame N+ — Continuous tracking
    // -------------------------------------------------------------------

    private fun runTracking(
        executor: YoloExecutor,
        bitmap: Bitmap
    ): List<SortTracker.TrackedObject> {

        // YOLO only for the car — skip pin class entirely
        val allDetections = executor.detect(bitmap)
        val carDetections = allDetections.filter { it.classIndex == 3 }

        val carTracks = sortTracker.update(carDetections)

        // Pins handled by cheap luminance comparison — no neural network
        val pinTracks = staticTracker.update(bitmap)

        return carTracks + pinTracks
    }
}