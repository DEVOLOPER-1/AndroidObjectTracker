package com.example.androidobjecttracker

import android.content.Context
import android.graphics.Bitmap
import android.media.*
import android.net.Uri
import com.example.modelengine.AppLog
import com.example.modelengine.ModelExecutor
import com.example.modelengine.SortTracker
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class FastVideoProcessor(
    private val context: Context,
    private val modelExecutor: ModelExecutor,
    private val videoEncoder: VideoEncoder
) {
    suspend fun processVideo(
        uri: Uri,
        onProgress: (Float, Long) -> Unit
    ) {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, uri)
            
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
            val frameCountStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)
            val totalFrames = frameCountStr?.toInt() ?: 0
            
            val width = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toInt() ?: 1280
            val height = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toInt() ?: 720

            AppLog.i("Processing video: $totalFrames frames, $durationMs ms, ${width}x${height}")

            videoEncoder.start(width, height)
            modelExecutor.reset()

            val startTime = System.currentTimeMillis()
            
            // Process at 30 FPS if possible, or step through total frames
            val step = 1 
            for (i in 0 until totalFrames step step) {
                val timeUs = (i.toLong() * durationMs * 1000L) / totalFrames
                val bitmap = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                
                if (bitmap != null) {
                    // 1. Run inference and tracking
                    val tracks = modelExecutor.detectAndTrack(bitmap)
                    
                    // 2. Encode frame with annotations
                    // Use a more memory-efficient way to get a mutable bitmap if needed
                    val mutableBitmap = if (bitmap.isMutable) bitmap else bitmap.copy(Bitmap.Config.ARGB_8888, true)
                    if (mutableBitmap != bitmap) bitmap.recycle()
                    
                    videoEncoder.encodeFrame(mutableBitmap, tracks)
                    mutableBitmap.recycle()
                    
                    val progress = i.toFloat() / totalFrames
                    val elapsed = System.currentTimeMillis() - startTime
                    val etaMs = if (progress > 0.05) (elapsed / progress * (1 - progress)).toLong() else -1L

                    withContext(Dispatchers.Main) {
                        onProgress(progress, etaMs)
                    }
                    
                    if (i % 30 == 0) {
                        AppLog.d("Progress: ${(progress * 100).toInt()}% | Frame: $i")
                    }
                }
            }

            val resultFile = videoEncoder.finish()
            if (resultFile != null) {
                videoEncoder.saveToGallery(resultFile)
            }

            withContext(Dispatchers.Main) {
                onProgress(1.0f, 0L)
            }

        } catch (e: Exception) {
            AppLog.e("Video processing failed", e)
        } finally {
            retriever.release()
        }
    }
}
