package com.example.modelengine

import android.graphics.PointF
import android.graphics.RectF
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.sqrt

/**
 * Centroid-based tracker for maintaining object identities across frames.
 * Implements specific logic for Pin Fall detection and Car Trajectory tracking.
 */
class SortTracker {
    private val nextId = AtomicInteger(1)
    private var currentFallCount = 0
    private val tracks = mutableListOf<TrackedObject>()
    
    // Configuration
    private val maxAge = 10
    private val distanceThreshold = 150f // Max pixels for centroid matching
    private val fallMovementThreshold = 50f // Pixels moved to confirm fall

    data class TrackedObject(
        val id: Int,
        val classIndex: Int,
        var bbox: RectF,
        var lastCx: Float,
        var lastCy: Float,
        var age: Int = 0,
        var state: State = State.STANDING,
        var fallOrder: Int? = null,
        val trajectory: MutableList<PointF> = mutableListOf()
    )

    enum class State {
        STANDING,
        FALLEN
    }

    /**
     * Updates tracks with new detections from YOLO.
     */
    fun update(detections: List<YoloExecutor.Detection>): List<TrackedObject> {
        // 1. Predict/Age existing tracks
        tracks.forEach { it.age++ }

        val matchedDetections = mutableSetOf<Int>()
        val matchedTracks = mutableSetOf<Int>()

        // 2. Greedy Centroid Matching
        // In a real SORT we'd use Hungarian, but requirements specify Centroid matching.
        val assignments = mutableListOf<Pair<Int, Int>>()
        
        // Calculate all distances
        val distanceMatrix = Array(tracks.size) { tIdx ->
            FloatArray(detections.size) { dIdx ->
                calculateDistance(tracks[tIdx].lastCx, tracks[tIdx].lastCy, 
                                 detections[dIdx].bbox.centerX(), detections[dIdx].bbox.centerY())
            }
        }

        while (true) {
            var minDistance = distanceThreshold
            var bestT = -1
            var bestD = -1

            for (t in tracks.indices) {
                if (t in matchedTracks) continue
                for (d in detections.indices) {
                    if (d in matchedDetections) continue
                    if (distanceMatrix[t][d] < minDistance) {
                        minDistance = distanceMatrix[t][d]
                        bestT = t
                        bestD = d
                    }
                }
            }

            if (bestT != -1) {
                matchedTracks.add(bestT)
                matchedDetections.add(bestD)
                assignments.add(bestT to bestD)
            } else {
                break
            }
        }

        // 3. Update Matched Tracks
        for ((tIdx, dIdx) in assignments) {
            val track = tracks[tIdx]
            val detection = detections[dIdx]
            
            val newCx = detection.bbox.centerX()
            val newCy = detection.bbox.centerY()
            
            // Pin Fall Logic (classIndex 1)
            if (track.classIndex == 1 && track.state == State.STANDING) {
                val ratio = detection.bbox.width() / detection.bbox.height()
                val movement = calculateDistance(track.lastCx, track.lastCy, newCx, newCy)
                
                if (ratio > 1.0f || movement > fallMovementThreshold) {
                    track.state = State.FALLEN
                    currentFallCount++
                    track.fallOrder = currentFallCount
                    AppLog.i("PIN FELL! ID: ${track.id}, Order: $currentFallCount, Ratio: $ratio")
                }
            }

            // Car Trajectory Logic (classIndex 3)
            if (track.classIndex == 3) {
                track.trajectory.add(PointF(newCx, newCy))
                if (track.trajectory.size > 200) track.trajectory.removeAt(0)
            }

            track.bbox = detection.bbox
            track.lastCx = newCx
            track.lastCy = newCy
            track.age = 0
        }

        // 4. Create New Tracks
        for (dIdx in detections.indices) {
            if (dIdx !in matchedDetections) {
                val det = detections[dIdx]
                val newTrack = TrackedObject(
                    id = nextId.getAndIncrement(),
                    classIndex = det.classIndex,
                    bbox = det.bbox,
                    lastCx = det.bbox.centerX(),
                    lastCy = det.bbox.centerY()
                )
                
                // If it's a car, start trajectory
                if (newTrack.classIndex == 3) {
                    newTrack.trajectory.add(PointF(newTrack.lastCx, newTrack.lastCy))
                }
                
                // Initial state check for pins
                if (newTrack.classIndex == 1) {
                    val ratio = det.bbox.width() / det.bbox.height()
                    if (ratio > 1.0f) {
                        newTrack.state = State.FALLEN
                        // Not incrementing fallOrder here because we didn't see it fall? 
                        // Or maybe we should if we want to count all fallen pins.
                        // "Assign it a fallOrder... and lock its state" implies seeing transition.
                    }
                }

                tracks.add(newTrack)
                AppLog.d("New track created: ID=${newTrack.id}, Class=${newTrack.classIndex}")
            }
        }

        // 5. Cleanup Dead Tracks
        tracks.removeAll { it.age > maxAge }

        return tracks.toList()
    }

    private fun calculateDistance(x1: Float, y1: Float, x2: Float, y2: Float): Float {
        val dx = x1 - x2
        val dy = y1 - y2
        return sqrt(dx * dx + dy * dy)
    }

    fun reset() {
        tracks.clear()
        nextId.set(1)
        currentFallCount = 0
    }
}
