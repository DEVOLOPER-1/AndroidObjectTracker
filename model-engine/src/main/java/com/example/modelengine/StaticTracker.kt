package com.example.modelengine

import android.graphics.PointF
import android.graphics.RectF
import kotlin.math.sqrt

/**
 * A tracker that initializes object positions once and tracks state changes
 * based on Euclidean distance from the starting point.
 */
class StaticTracker {
    
    data class PinState(
        val initialCentroid: PointF,
        var currentCentroid: PointF,
        var isFallen: Boolean = false,
        var fallOrder: Int? = null
    )

    private val pins = mutableListOf<PinState>()
    private val carPath = mutableListOf<PointF>()
    private var nextFallOrder = 1
    private val FALL_THRESHOLD = 50f // Pixels
    private val CAR_MOVE_THRESHOLD = 5f

    // Class Mappings based on logs
    private val CAR_CLASSES = listOf(0, 13, 2)
    private val PIN_CLASSES = listOf(1, 4, 7)

    /**
     * Call this on Frame 0 to lock the initial positions of the 5 pins and the car.
     */
    fun initialize(initialDetections: List<SortTracker.Detection>) {
        pins.clear()
        carPath.clear()
        nextFallOrder = 1

        val detectedPins = initialDetections.filter { it.classIndex in PIN_CLASSES }
        val detectedCar = initialDetections.find { it.classIndex in CAR_CLASSES }

        detectedPins.take(5).forEach { pin ->
            val center = PointF(pin.bbox.centerX(), pin.bbox.centerY())
            pins.add(PinState(center, center))
        }

        detectedCar?.let { car ->
            carPath.add(PointF(car.bbox.centerX(), car.bbox.centerY()))
        }
        
        AppLog.i("StaticTracker Initialized: ${pins.size} pins, Car found: ${detectedCar != null}")
    }

    /**
     * Process a frame and update states based on distance.
     */
    fun update(currentDetections: List<SortTracker.Detection>, externalCarBbox: RectF? = null) {
        // 1. Update Pins
        pins.forEach { pin ->
            if (!pin.isFallen) {
                // Find the detection closest to the INITIAL centroid
                val closest = currentDetections
                    .filter { it.classIndex in PIN_CLASSES || it.classIndex == 2 } // 2 might be fallen pin in some models
                    .minByOrNull { dist(it.bbox.centerX(), it.bbox.centerY(), pin.initialCentroid.x, pin.initialCentroid.y) }

                closest?.let { det ->
                    val curX = det.bbox.centerX()
                    val curY = det.bbox.centerY()
                    val distanceFromStart = dist(curX, curY, pin.initialCentroid.x, pin.initialCentroid.y)
                    
                    pin.currentCentroid = PointF(curX, curY)
                    
                    if (distanceFromStart > FALL_THRESHOLD) {
                        pin.isFallen = true
                        pin.fallOrder = nextFallOrder++
                        AppLog.i("PIN FELL! Distance: $distanceFromStart, Order: ${pin.fallOrder}")
                    }
                }
            }
        }

        // 2. Update Car Path
        if (externalCarBbox != null) {
            val newPoint = PointF(externalCarBbox.centerX(), externalCarBbox.centerY())
            carPath.add(newPoint)
            if (carPath.size > 200) carPath.removeAt(0)
        } else {
            val carDet = currentDetections.find { it.classIndex in CAR_CLASSES }
            carDet?.let { det ->
                val lastPoint = carPath.lastOrNull()
                val newPoint = PointF(det.bbox.centerX(), det.bbox.centerY())
                
                if (lastPoint == null || dist(newPoint.x, newPoint.y, lastPoint.x, lastPoint.y) > CAR_MOVE_THRESHOLD) {
                    carPath.add(newPoint)
                    if (carPath.size > 200) carPath.removeAt(0)
                }
            }
        }
    }

    fun getPins(): List<PinState> = pins.toList()
    fun getCarPath(): List<PointF> = carPath.toList()

    private fun dist(x1: Float, y1: Float, x2: Float, y2: Float): Float {
        return sqrt(((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)).toDouble()).toFloat()
    }

    fun reset() {
        pins.clear()
        carPath.clear()
        nextFallOrder = 1
    }
}
