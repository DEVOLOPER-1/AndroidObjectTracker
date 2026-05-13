package com.example.androidobjecttracker.processor

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.provider.MediaStore
import com.example.androidobjecttracker.pipeline.PipelineOrchestrator
import com.example.androidobjecttracker.pipeline.PipelineParams
import com.example.androidobjecttracker.utils.AppLog
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

class FastVideoProcessor(
    private val context: Context,
    private val orchestrator: PipelineOrchestrator,
    private val encoder: VideoEncoder
) {
    suspend fun processVideo(
        uri: Uri,
        params: PipelineParams,
        onProgress: (Float) -> Unit
    ): Uri? = withContext(Dispatchers.Default) {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, uri)
            val width = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toInt() ?: 1280
            val height = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toInt() ?: 720
            val rotation = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toInt() ?: 0
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
            
            // Extract frame rate, default to 30. METADATA_KEY_VIDEO_FRAME_RATE is API 30+ 
            // but we can try the integer key 32 or just default to 30.
            val frameRate = 30
            
            // Swap width/height if rotated 90 or 270 degrees
            val finalWidth = if (rotation == 90 || rotation == 270) height else width
            val finalHeight = if (rotation == 90 || rotation == 270) width else height

            val tempFile = File(context.cacheDir, "processed_video.mp4")
            encoder.start(tempFile, finalWidth, finalHeight, frameRate)
            orchestrator.reset()

            val frameIntervalUs = 1000000L / frameRate
            var currentTimeUs = 0L
            val totalDurationUs = durationMs * 1000

            while (currentTimeUs < totalDurationUs) {
                // Get frame at specific time
                val frame = retriever.getFrameAtTime(currentTimeUs, MediaMetadataRetriever.OPTION_CLOSEST)
                if (frame != null) {
                    val processedFrame = orchestrator.processFrame(frame, params, currentTimeUs)
                    encoder.encodeFrame(processedFrame, currentTimeUs)
                    processedFrame.recycle()
                    frame.recycle()
                }
                
                currentTimeUs += frameIntervalUs
                onProgress(currentTimeUs.toFloat() / totalDurationUs)
            }

            // Final summary screen (3 seconds)
            val summaryFrame = orchestrator.generateSummaryFrame(finalWidth, finalHeight, currentTimeUs)
            repeat(frameRate * 3) {
                currentTimeUs += frameIntervalUs
                encoder.encodeFrame(summaryFrame, currentTimeUs)
            }
            summaryFrame.recycle()

            encoder.finish()
            return@withContext saveToGallery(tempFile)
        } catch (e: Exception) {
            AppLog.e("Video processing failed", e)
            null
        } finally {
            retriever.release()
        }
    }

    private fun saveToGallery(file: File): Uri? {
        val displayName = "Tracked_${System.currentTimeMillis()}.mp4"
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, displayName)
            put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
            put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/ObjectTracker")
        }

        val contentUri = context.contentResolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values)
        contentUri?.let { destUri ->
            context.contentResolver.openOutputStream(destUri)?.use { out ->
                file.inputStream().use { input ->
                    input.copyTo(out)
                }
            }
        }
        return contentUri
    }
}
