package com.example.androidobjecttracker

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.media.*
import android.net.Uri
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import com.example.modelengine.AppLog
import com.example.modelengine.ModelExecutor
import com.example.modelengine.SortTracker
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer

class FastVideoProcessor(
    private val context: Context,
    private val modelExecutor: ModelExecutor
) {
    private val TAG = "FastVideoProcessor"

    suspend fun processVideo(
        uri: Uri,
        undersamplingFactor: Int = 3,
        onProgress: (Float, Long) -> Unit,
        onFrameProcessed: (Bitmap, List<SortTracker.Track>) -> Unit
    ) {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, uri)
            AppLog.i("Starting video processing (Undersampling: $undersamplingFactor) for URI: $uri")
            
            val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            val durationMs = durationStr?.toLong() ?: 0L
            val frameCountStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)
            val totalFrames = frameCountStr?.toInt() ?: 0
            
            AppLog.d("Video Info: $totalFrames frames, $durationMs ms")

            if (totalFrames <= 0) {
                AppLog.e("Invalid frame count")
                return
            }

            val startTime = System.currentTimeMillis()
            
            for (i in 0 until totalFrames step undersamplingFactor) {
                try {
                    // Use getFrameAtTime with OPTION_CLOSEST for better stability than getFrameAtIndex
                    val timeUs = (i.toLong() * durationMs * 1000L) / totalFrames
                    val bitmap = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST)
                    
                    if (bitmap != null) {
                        val tracks = modelExecutor.detectAndTrack(bitmap)
                        
                        val progress = i.toFloat() / totalFrames
                        val elapsed = System.currentTimeMillis() - startTime
                        val etaMs = if (progress > 0.05) (elapsed / progress * (1 - progress)).toLong() else -1L

                        withContext(Dispatchers.Main) {
                            onFrameProcessed(bitmap, tracks)
                            onProgress(progress, etaMs)
                        }
                        
                        if ((i / undersamplingFactor) % 20 == 0) {
                            val etaStr = if (etaMs > 0) "${etaMs/1000}s" else "calculating..."
                            val detectionSummary = tracks.groupBy { it.classIndex }
                                .map { "ID ${it.key}: ${it.value.size}" }.joinToString()
                            AppLog.d("Frame $i/$totalFrames | Progress: ${(progress * 100).toInt()}% | ETA: $etaStr | Detections: [$detectionSummary]")
                        }
                    } else {
                        AppLog.e("Frame at $i is null, skipping...")
                    }
                } catch (e: Exception) {
                    AppLog.e("Failed to extract frame at $i", e)
                    // Continue to next frame instead of crashing
                }
            }

            withContext(Dispatchers.Main) {
                onProgress(1.0f, 0L)
            }

        } catch (e: Exception) {
            AppLog.e("Video processing failed", e)
        } finally {
            try {
                retriever.release()
            } catch (ignored: Exception) {
            }
        }
    }
}
