package com.example.modelengine

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import kotlin.math.abs

/**
 * Tracks pin falls by comparing pixel data within static regions against a baseline intensity.
 * This pivot avoids YOLO detection failures in static environments.
 */
class StaticTracker {

    data class PinState(
        val id: Int,
        val rect: RectF,
        var isFallen: Boolean = false,
        var fallOrder: Int? = null,
        var baselineIntensity: Float = 0f,
    )

    private val pins = mutableListOf<PinState>()
    private var nextFallOrder = 1
    private val thresholdPinFall = 20.0f // Intensity change threshold for fall detection

    /**
     * Captures initial pixel intensity baselines for all pins on Frame 0.
     */
    fun initialize(frame: Bitmap, pinRects: List<RectF>) {
        pins.clear()
        nextFallOrder = 1
        
        pinRects.forEachIndexed { index, rect ->
            val intensity = getAverageIntensity(frame, rect)
            pins.add(PinState(id = index, rect = rect, baselineIntensity = intensity))
        }
        AppLog.i("StaticTracker: Pins initialized with pixel baselines")
    }

    /**
     * Compares current frame pixels against baselines to detect changes (falls).
     */
    fun update(frame: Bitmap): List<PinState> {
        pins.forEach { pin ->
            if (!pin.isFallen) {
                val currentIntensity = getAverageIntensity(frame, pin.rect)
                val diff = abs(currentIntensity - pin.baselineIntensity)
                
                if (diff > thresholdPinFall) {
                    pin.isFallen = true
                    pin.fallOrder = nextFallOrder++
                    AppLog.i("StaticTracker: Pin ${pin.id} fell (Baseline: ${pin.baselineIntensity}, Current: $currentIntensity, Diff: $diff)")
                }
            }
        }
        return pins
    }

    private fun getAverageIntensity(bitmap: Bitmap, rect: RectF): Float {
        val left = rect.left.toInt().coerceIn(0, bitmap.width - 1)
        val top = rect.top.toInt().coerceIn(0, bitmap.height - 1)
        val right = rect.right.toInt().coerceIn(0, bitmap.width - 1)
        val bottom = rect.bottom.toInt().coerceIn(0, bitmap.height - 1)
        
        val width = right - left
        val height = bottom - top
        if (width <= 0 || height <= 0) return 0f
        
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, left, top, width, height)
        
        var total = 0L
        for (p in pixels) {
            // Simple grayscale conversion for intensity comparison
            total += (Color.red(p) + Color.green(p) + Color.blue(p)) / 3
        }
        return total.toFloat() / (width * height)
    }

    fun getPins() = pins.toList()

    fun reset() {
        pins.clear()
        nextFallOrder = 1
    }
}
