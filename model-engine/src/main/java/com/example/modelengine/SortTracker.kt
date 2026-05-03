package com.example.modelengine

import android.graphics.RectF
import java.util.concurrent.atomic.AtomicInteger

/**
 * A lightweight tracking algorithm similar to SORT.
 * Maintains object identities across frames based on IoU.
 */
class SortTracker {
    private val nextId = AtomicInteger(0)
    private val tracks = mutableListOf<Track>()
    private val maxAge = 5 // Number of frames to keep a track without detection

    data class Track(
        val id: Int,
        var bbox: RectF,
        var classIndex: Int,
        var age: Int = 0,
        var hits: Int = 1,
        var fallOrder: Int? = null,
        val path: MutableList<RectF> = mutableListOf() // Store historical centers as RectFs for simplicity in coordinate mapping
    )

    data class Detection(
        val bbox: RectF,
        val classIndex: Int,
        val confidence: Float
    )

    private var currentFallCount = 0

    fun update(detections: List<Detection>): List<Track> {
        // ... (Prediction logic remains same)
        for (track in tracks) {
            track.age++
        }

        // Association logic...
        val matchedDetections = mutableSetOf<Int>()
        val matchedTracks = mutableSetOf<Int>()
        val assignments = mutableListOf<Pair<Int, Int>>()
        
        val iouMatrix = Array(tracks.size) { t ->
            FloatArray(detections.size) { d ->
                calculateIoU(tracks[t].bbox, detections[d].bbox)
            }
        }

        while (true) {
            var maxIoU = 0.3f
            var bestT = -1
            var bestD = -1

            for (t in tracks.indices) {
                if (t in matchedTracks) continue
                for (d in detections.indices) {
                    if (d in matchedDetections) continue
                    if (iouMatrix[t][d] > maxIoU) {
                        maxIoU = iouMatrix[t][d]
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

        // 3. Update matched tracks
        for ((tIdx, dIdx) in assignments) {
            val track = tracks[tIdx]
            val detection = detections[dIdx]
            
            // State Machine for Pin Fall
            if (track.classIndex == 1 && detection.classIndex == 2 && track.fallOrder == null) {
                currentFallCount++
                track.fallOrder = currentFallCount
                AppLog.i("PIN FELL! Track ID: ${track.id}, Fall Order: $currentFallCount")
            }

            track.apply {
                bbox = detection.bbox
                classIndex = detection.classIndex
                age = 0
                hits++
                
                // Path Tracking for Car (class 0)
                if (classIndex == 0) {
                    path.add(RectF(bbox))
                    if (path.size > 100) path.removeAt(0)
                }
            }
        }

        // 4. Create new tracks for unmatched detections
        for (dIdx in detections.indices) {
            if (dIdx !in matchedDetections) {
                val newTrack = Track(
                    id = nextId.getAndIncrement(),
                    bbox = detections[dIdx].bbox,
                    classIndex = detections[dIdx].classIndex
                )
                AppLog.d("New Track Created: ID=${newTrack.id}, Class=${newTrack.classIndex}")
                // If it's already fallen when first seen, maybe don't assign order or assign late?
                // Requirement says "Detect the exact moment each pin falls". 
                // We'll only assign fallOrder if we see the transition.
                
                if (newTrack.classIndex == 0) {
                    newTrack.path.add(RectF(newTrack.bbox))
                }
                
                tracks.add(newTrack)
            }
        }

        // 5. Remove dead tracks
        val beforeCount = tracks.size
        tracks.removeAll { it.age > maxAge }
        if (tracks.size < beforeCount) {
            AppLog.d("Removed ${beforeCount - tracks.size} dead tracks.")
        }

        return tracks.toList()
    }

    private fun calculateIoU(rect1: RectF, rect2: RectF): Float {
        val intersection = RectF()
        if (!intersection.setIntersect(rect1, rect2)) return 0f
        
        val interArea = intersection.width() * intersection.height()
        val unionArea = (rect1.width() * rect1.height()) + 
                        (rect2.width() * rect2.height()) - interArea
        
        return if (unionArea > 0) interArea / unionArea else 0f
    }
    
    fun reset() {
        tracks.clear()
        nextId.set(0)
        currentFallCount = 0
    }
}
