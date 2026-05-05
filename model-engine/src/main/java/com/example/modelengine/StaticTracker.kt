package com.example.modelengine

import android.graphics.Bitmap
import android.graphics.RectF
import kotlin.math.abs

/**
 * Non-ML, pixel-comparison tracker for bowling pins.
 * Initialized ONCE by YOLO on frame 0 with the baseline average luminance
 * of each detected pin bounding box.
 *
 * On frame N, it recomputes the average luminance within each stored bbox.
 * If the absolute difference exceeds FALL_THRESHOLD, the pin is marked FALLEN
 * and assigned a sequential fall order number.
 */
class StaticTracker {

    companion object {
        private const val FALL_THRESHOLD = 25.0f
    }

    data class PinState(
        val id: Int,
        val bbox: RectF,
        val baselineLuminance: Float,
        var state: SortTracker.State = SortTracker.State.STANDING,
        var fallOrder: Int? = null
    )

    private val pins = mutableListOf<PinState>()
    private var fallCount = 0

    // --- Public API ---

    fun isInitialized(): Boolean = pins.isNotEmpty()

    /**
     * Called ONCE on frame 0.
     * Stores each pin's bounding box and its baseline average luminance.
     */
    fun initialize(bitmap: Bitmap, pinBboxes: List<RectF>) {
        pins.clear()
        fallCount = 0
        pinBboxes.forEachIndexed { index, bbox ->
            val baseline = computeAverageLuminance(bitmap, bbox)
            pins.add(
                PinState(
                    id = index + 1,
                    bbox = bbox,
                    baselineLuminance = baseline
                )
            )
            AppLog.i(
                "StaticTracker: Pin #${index + 1} initialized. " +
                        "Baseline=${baseline}, BBox=(${bbox.left.toInt()},${bbox.top.toInt()}," +
                        "${bbox.right.toInt()},${bbox.bottom.toInt()})"
            )
        }
    }

    /**
     * Called on every frame after frame 0.
     * Checks each standing pin for a luminance change. Returns the full pin
     * state list as TrackedObjects so the VideoEncoder can render them
     * without any extra conversion.
     */
    fun update(bitmap: Bitmap): List<SortTracker.TrackedObject> {
        val result = mutableListOf<SortTracker.TrackedObject>()

        for (pin in pins) {
            if (pin.state == SortTracker.State.STANDING) {
                val current = computeAverageLuminance(bitmap, pin.bbox)
                val diff = abs(current - pin.baselineLuminance)

                if (diff > FALL_THRESHOLD) {
                    fallCount++
                    pin.state = SortTracker.State.FALLEN
                    pin.fallOrder = fallCount
                    AppLog.i(
                        "StaticTracker: PIN FELL! " +
                                "ID=${pin.id}, FallOrder=$fallCount, LumDiff=${"%.1f".format(diff)}"
                    )
                }
            }

            result.add(
                SortTracker.TrackedObject(
                    id = pin.id,
                    classIndex = 1,
                    bbox = RectF(pin.bbox), // defensive copy
                    lastCx = pin.bbox.centerX(),
                    lastCy = pin.bbox.centerY(),
                    state = pin.state,
                    fallOrder = pin.fallOrder
                )
            )
        }

        return result
    }

    fun getFallCount(): Int = fallCount

    fun reset() {
        pins.clear()
        fallCount = 0
        AppLog.i("StaticTracker: reset.")
    }

    // --- Math ---

    /**
     * Computes average pixel luminance using the standard Rec. 601 formula:
     * Y = 0.299R + 0.587G + 0.114B
     *
     * Clamps the bbox to the bitmap boundaries before sampling to avoid
     * ArrayIndexOutOfBoundsException on edge detections.
     */
    private fun computeAverageLuminance(bitmap: Bitmap, bbox: RectF): Float {
        val left   = bbox.left.toInt().coerceIn(0, bitmap.width - 1)
        val top    = bbox.top.toInt().coerceIn(0, bitmap.height - 1)
        val right  = bbox.right.toInt().coerceIn(left + 1, bitmap.width)
        val bottom = bbox.bottom.toInt().coerceIn(top + 1, bitmap.height)

        val w = right - left
        val h = bottom - top
        if (w <= 0 || h <= 0) return 0f

        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, left, top, w, h)

        var total = 0.0
        for (pixel in pixels) {
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8)  and 0xFF
            val b =  pixel         and 0xFF
            total += 0.299 * r + 0.587 * g + 0.114 * b
        }
        return (total / pixels.size).toFloat()
    }
}