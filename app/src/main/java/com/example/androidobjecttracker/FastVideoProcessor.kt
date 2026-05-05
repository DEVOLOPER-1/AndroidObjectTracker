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
    suspend fun processVideo(uri: Uri, onProgress: (Float, Long) -> Unit) {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, uri)

            val durationMs =
                retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong()
                    ?: 0L
            val totalFrames =
                retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)
                    ?.toInt() ?: 0
            val srcWidth =
                retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toInt()
                    ?: 1280
            val srcHeight =
                retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toInt()
                    ?: 720

            // Downscale to 720p for both processing and encoding.
            // YOLO already rescales to 640x640 internally, so full 1080p buys nothing.
            // This halves the per-frame native + Java heap footprint.
            val maxDim = 1280
            val scaleFactor = minOf(maxDim.toFloat() / srcWidth, maxDim.toFloat() / srcHeight, 1.0f)
            val outWidth = (srcWidth * scaleFactor).toInt().let { if (it % 2 == 0) it else it - 1 }
            val outHeight =
                (srcHeight * scaleFactor).toInt().let { if (it % 2 == 0) it else it - 1 }

            AppLog.i("Processing video: $totalFrames frames @ ${srcWidth}x${srcHeight} → encoding at ${outWidth}x${outHeight}")

            videoEncoder.start(outWidth, outHeight)
            modelExecutor.reset()

            val startTime = System.currentTimeMillis()

            // step=2 — process every other frame, output at 15 FPS.
            // Halves YOLO calls and retriever native allocation rate on low-RAM devices.
            val step = 2

            for (i in 0 until totalFrames step step) {
                val timeUs = (i.toLong() * durationMs * 1000L) / totalFrames
                val raw =
                    retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)

                if (raw != null) {
                    // Scale down immediately — raw bitmap is recycled right after scaling.
                    // This frees the 1080p native buffer before YOLO runs.
                    val frame = if (raw.width == outWidth && raw.height == outHeight) {
                        raw
                    } else {
                        val scaled = Bitmap.createScaledBitmap(raw, outWidth, outHeight, true)
                        raw.recycle()   // <-- critical: release native memory NOW
                        scaled
                    }

                    val tracks = modelExecutor.detectAndTrack(frame)

                    try {
                        videoEncoder.encodeFrame(frame, tracks)
                    } catch (e: Exception) {
                        AppLog.e("FastVideoProcessor: encodeFrame failed on frame $i", e)
                    } finally {
                        frame.recycle()
                    }

                    val progress = i.toFloat() / totalFrames
                    val elapsed = System.currentTimeMillis() - startTime
                    val etaMs =
                        if (progress > 0.05f) (elapsed / progress * (1 - progress)).toLong() else -1L
                    withContext(Dispatchers.Main) { onProgress(progress, etaMs) }

                    if (i % 60 == 0) {
                        AppLog.d("Progress: ${(progress * 100).toInt()}% | Frame: $i")
                    }
                }
            }

            val resultFile = videoEncoder.finish()
            if (resultFile != null) videoEncoder.saveToGallery(resultFile)
            withContext(Dispatchers.Main) { onProgress(1.0f, 0L) }

        } catch (e: Exception) {
            AppLog.e("Video processing failed", e)
        } finally {
            retriever.release()
        }
    }
}