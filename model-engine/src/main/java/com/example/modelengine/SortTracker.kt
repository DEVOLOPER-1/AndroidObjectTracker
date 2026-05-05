package com.example.modelengine

import android.graphics.PointF
import android.graphics.RectF
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.sqrt

/**
 * Centroid-based identity tracker.
 *
 * After the bootstrap migration, this class is responsible ONLY for the car
 * (classIndex 3). Pins are tracked by StaticTracker.
 *
 * On each frame it:
 *   1. Ages every existing track by +1.
 *   2. Greedily matches new detections to existing tracks by minimum
 *      Euclidean centroid distance.
 *   3. Updates matched tracks (resets age, appends trajectory point).
 *   4. Creates new tracks for unmatched detections.
 *   5. Purges tracks that have not been matched for > maxAge frames.
 */
class SortTracker {

    // --- Configuration ---
    private val maxAge             = 10
    private val distanceThreshold  = 150f  // px — max centroid distance for a valid match

    // --- State ---
    private val nextId = AtomicInteger(1)
    private val tracks = mutableListOf<TrackedObject>()

    // --- Data model (shared with StaticTracker output) ---

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

    enum class State { STANDING, FALLEN }

    // --- Public API ---

    /**
     * Feed new detections from YoloExecutor. Returns the current live track list.
     * Should be called with CAR detections only after bootstrap.
     */
    fun update(detections: List<YoloExecutor.Detection>): List<TrackedObject> {

        // 1. Age all tracks
        tracks.forEach { it.age++ }

        val matchedDetections = mutableSetOf<Int>()
        val matchedTracks     = mutableSetOf<Int>()
        val assignments       = mutableListOf<Pair<Int, Int>>() // trackIdx → detectionIdx

        // 2. Build distance matrix
        val distMatrix = Array(tracks.size) { tIdx ->
            FloatArray(detections.size) { dIdx ->
                euclidean(
                    tracks[tIdx].lastCx, tracks[tIdx].lastCy,
                    detections[dIdx].bbox.centerX(), detections[dIdx].bbox.centerY()
                )
            }
        }

        // 3. Greedy minimum-distance matching
        while (true) {
            var minDist = distanceThreshold
            var bestT = -1
            var bestD = -1

            for (t in tracks.indices) {
                if (t in matchedTracks) continue
                for (d in detections.indices) {
                    if (d in matchedDetections) continue
                    if (distMatrix[t][d] < minDist) {
                        minDist = distMatrix[t][d]
                        bestT = t
                        bestD = d
                    }
                }
            }

            if (bestT == -1) break
            matchedTracks.add(bestT)
            matchedDetections.add(bestD)
            assignments.add(bestT to bestD)
        }

        // 4. Update matched tracks
        for ((tIdx, dIdx) in assignments) {
            val track = tracks[tIdx]
            val det   = detections[dIdx]
            val newCx = det.bbox.centerX()
            val newCy = det.bbox.centerY()

            // Append trajectory for the car
            if (track.classIndex == 3) {
                track.trajectory.add(PointF(newCx, newCy))
                if (track.trajectory.size > 300) track.trajectory.removeAt(0)
            }

            track.bbox  = det.bbox
            track.lastCx = newCx
            track.lastCy = newCy
            track.age   = 0
        }

        // 5. Create new tracks for unmatched detections
        for (dIdx in detections.indices) {
            if (dIdx in matchedDetections) continue
            val det = detections[dIdx]
            val cx  = det.bbox.centerX()
            val cy  = det.bbox.centerY()
            val newTrack = TrackedObject(
                id         = nextId.getAndIncrement(),
                classIndex = det.classIndex,
                bbox       = det.bbox,
                lastCx     = cx,
                lastCy     = cy
            )
            if (newTrack.classIndex == 3) {
                newTrack.trajectory.add(PointF(cx, cy))
            }
            tracks.add(newTrack)
            AppLog.d("SortTracker: new track ID=${newTrack.id} class=${newTrack.classIndex}")
        }

        // 6. Purge dead tracks
        tracks.removeAll { it.age > maxAge }

        return tracks.toList()
    }

    fun reset() {
        tracks.clear()
        nextId.set(1)
        AppLog.i("SortTracker: reset.")
    }

    // --- Math ---

    private fun euclidean(x1: Float, y1: Float, x2: Float, y2: Float): Float {
        val dx = x1 - x2
        val dy = y1 - y2
        return sqrt(dx * dx + dy * dy)
    }
}