package com.example.modelengine

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import kotlin.math.abs

/**
 * Tracks pin falls by comparing current-frame luminance against a per-pin baseline.
 *
 * Why pixel comparison works here:
 *   Bowling pins do not move until struck by the car. While they are upright
 *   the average intensity inside their bounding box is stable. When a pin falls,
 *   it is displaced (or knocked out of the box) and the background pixel
 *   values replace it — causing a large, reliable luminance change.
 *
 * Algorithm (per directive):
 *   Frame 0  → initialize(): record baseline average intensity for each pin bbox.
 *   Frame N  → update():     compute new average; if |new − baseline| > 25.0,
 *                            mark pin as fallen and assign an incremental order number.
 */
class StaticTracker {

    data class PinState(
        val id: Int,
        val rect: RectF,
        var isFallen: Boolean = false,
        var fallOrder: Int    = 0,         // 0 means "not fallen yet"
        var baselineIntensity: Float = 0f
    )

    private val pins          = mutableListOf<PinState>()
    private var nextFallOrder = 1

    // Per directive: fall is detected when |current − baseline| > 25.0
    private val fallThreshold = 25.0f   // (BUG 8 FIX: was 20.0f)

    // -------------------------------------------------------------------------
    /**
     * Called ONCE on Frame 0 after YOLO detection.
     * Stores the initial pixel intensity inside each pin's bounding box.
     */
    fun initialize(frame: Bitmap, pinRects: List<RectF>) {
        pins.clear()
        nextFallOrder = 1

        pinRects.forEachIndexed { index, rect ->
            val intensity = getAverageIntensity(frame, rect)
            pins.add(
                PinState(
                    id                = index,
                    rect              = RectF(rect),   // defensive copy
                    baselineIntensity = intensity
                )
            )
            AppLog.d("StaticTracker: Pin $index baseline = ${"%.1f".format(intensity)}")
        }
        AppLog.i("StaticTracker: ${pins.size} pins initialised")
    }

    // -------------------------------------------------------------------------
    /**
     * Called on every frame (including Frame 0 — on Frame 0 the diff ≈ 0).
     * Returns the current state list (a copy; modifications by the caller
     * do not affect internal state).
     */
    fun update(frame: Bitmap): List<PinState> {
        pins.forEach { pin ->
            if (!pin.isFallen) {
                val current = getAverageIntensity(frame, pin.rect)
                val diff    = abs(current - pin.baselineIntensity)

                if (diff > fallThreshold) {
                    pin.isFallen   = true
                    pin.fallOrder  = nextFallOrder++
                    AppLog.i(
                        "StaticTracker: Pin ${pin.id} FELL " +
                        "(baseline=${pin.baselineIntensity}, now=$current, Δ=${"%.1f".format(diff)}, " +
                        "order=${pin.fallOrder})"
                    )
                }
            }
        }
        return pins.toList()   // return a snapshot so callers cannot mutate state
    }

    // -------------------------------------------------------------------------
    /** Returns a snapshot of all pin states (safe to read from any thread). */
    fun getPins(): List<PinState> = pins.toList()

    fun reset() {
        pins.clear()
        nextFallOrder = 1
    }

    // -------------------------------------------------------------------------
    /**
     * Computes the average luminance (grayscale intensity) of all pixels within
     * [rect] in [bitmap].  The rect is clamped to the bitmap bounds.
     *
     * Grayscale formula:  I = (R + G + B) / 3
     * (Simple average, not perceptual Luma, but consistent across frames.)
     */
    private fun getAverageIntensity(bitmap: Bitmap, rect: RectF): Float {
        val left   = rect.left.toInt().coerceIn(0, bitmap.width  - 1)
        val top    = rect.top.toInt().coerceIn(0, bitmap.height - 1)
        val right  = rect.right.toInt().coerceIn(0, bitmap.width  - 1)
        val bottom = rect.bottom.toInt().coerceIn(0, bitmap.height - 1)

        val width  = right  - left
        val height = bottom - top
        if (width <= 0 || height <= 0) return 0f

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, left, top, width, height)

        var totalLuminance = 0L
        for (p in pixels) {
            // Integer average of R, G, B channels (0–255 each)
            totalLuminance += (Color.red(p) + Color.green(p) + Color.blue(p)) / 3
        }
        return totalLuminance.toFloat() / (width * height)
    }
}
