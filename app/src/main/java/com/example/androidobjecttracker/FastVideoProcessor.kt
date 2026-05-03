package com.example.androidobjecttracker

import android.content.Context
import android.graphics.*
import android.media.*
import android.net.Uri
import com.example.modelengine.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Handles the video frame stream. Corrects Coordinate Mapping and Missing Annotation bugs.
 */
class FastVideoProcessor(
    private val context: Context,
    private val modelExecutor: ModelExecutor,
    private val sotExecutor: SOTExecutor,
    private val staticTracker: StaticTracker,
    private val videoEncoder: VideoEncoder
) {
    // Annotation Paints
    private val carPaint = Paint().apply { color = Color.YELLOW; strokeWidth = 8f; style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND }
    private val pinPaint = Paint().apply { color = Color.GREEN; strokeWidth = 4f; style = Paint.Style.STROKE }
    private val textPaint = Paint().apply { color = Color.WHITE; textSize = 70f; typeface = Typeface.DEFAULT_BOLD; setShadowLayer(5f, 0f, 0f, Color.BLACK) }

    suspend fun processVideo(
        uri: Uri,
        onProgress: (Float, Long) -> Unit,
        onFrameProcessed: (Bitmap) -> Unit
    ) {
        val retriever = MediaMetadataRetriever()
        val extractor = MediaExtractor()
        
        try {
            retriever.setDataSource(context, uri)
            extractor.setDataSource(context, uri, null)
            
            val width = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toInt() ?: 0
            val height = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toInt() ?: 0
            val durationUs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong()?.times(1000) ?: 0L
            
            AppLog.i("FastVideoProcessor: Initializing encoder with resolution $width x $height")
            videoEncoder.start(width, height)
            val startTime = System.currentTimeMillis()

            val videoTrack = selectVideoTrack(extractor)
            if (videoTrack < 0) return
            extractor.selectTrack(videoTrack)

            var isInitialized = false
            var framesProcessed = 0

            while (true) {
                // 1. CAPTURE ORIGINAL PTS
                val ptsUs = extractor.sampleTime
                if (ptsUs < 0) break // End of file

                val frameStart = System.currentTimeMillis()
                val frame = retriever.getFrameAtTime(ptsUs, MediaMetadataRetriever.OPTION_CLOSEST)
                if (frame != null) {
                    framesProcessed++
                    if (framesProcessed % 10 == 0) {
                        AppLog.d("FastVideoProcessor: Processed $framesProcessed frames. Last frame retrieval took ${System.currentTimeMillis() - frameStart}ms")
                    }
                    // Create mutable bitmap for direct "baking" of annotations
                    val mutableBitmap = if (frame.isMutable) frame else {
                        val copy = frame.copy(Bitmap.Config.ARGB_8888, true)
                        frame.recycle()
                        copy
                    }
                    val canvas = Canvas(mutableBitmap)

                    if (!isInitialized) {
                        // Frame 0: Run initial object detection to bootstrap high-precision trackers
                        val detections = modelExecutor.detectAndTrack(mutableBitmap)
                        
                        // Map detections to trackers (Class 0,13,2 = Car, Class 1,4,7 = Pins)
                        val car = detections.find { it.classIndex in listOf(0, 13, 2) }
                        val pins = detections.filter { it.classIndex in listOf(1, 4, 7) }.take(5)
                        
                        car?.let { sotExecutor.init(mutableBitmap, it.bbox) }
                        staticTracker.initialize(mutableBitmap, pins.map { it.bbox })
                        isInitialized = true
                        AppLog.i("FastVideoProcessor: Trackers bootstrapped from YOLO detection on Frame 0")
                    }

                    // 2. RUN HYBRID TRACKERS
                    val carBbox = sotExecutor.update(mutableBitmap)
                    val pinStates = staticTracker.update(mutableBitmap)

                    // 3. RENDER ANNOTATIONS DIRECTLY ON BITMAP
                    // Draw Car Trajectory
                    val path = sotExecutor.carPath
                    if (path.size > 1) {
                        for (i in 1 until path.size) {
                            canvas.drawLine(path[i-1].x, path[i-1].y, path[i].x, path[i].y, carPaint)
                        }
                    }
                    
                    // Draw Car bounding box
                    carBbox?.let { canvas.drawRect(it, carPaint) }

                    // Draw Pins and Fall Order
                    pinStates.forEach { pin ->
                        if (pin.isFallen) {
                            textPaint.color = Color.RED
                            canvas.drawText(pin.fallOrder.toString(), pin.rect.centerX(), pin.rect.centerY(), textPaint)
                        } else {
                            canvas.drawRect(pin.rect, pinPaint)
                            textPaint.color = Color.WHITE
                            canvas.drawText("PIN", pin.rect.left, pin.rect.top - 10f, textPaint)
                        }
                    }

                    // 4. SUBMIT ANNOTATED FRAME WITH SOURCE PTS
                    videoEncoder.addFrame(mutableBitmap, ptsUs)

                    withContext(Dispatchers.Main) {
                        onFrameProcessed(mutableBitmap)
                        val progress = ptsUs.toFloat() / durationUs
                        val elapsed = System.currentTimeMillis() - startTime
                        val eta = if (progress > 0.01) ((elapsed / progress) * (1 - progress)).toLong() else -1L
                        onProgress(progress, eta)
                    }
                    // mutableBitmap.recycle() is now handled by the UI callback to prevent "Canvas: trying to use a recycled bitmap" crash
                }
                extractor.advance()
            }
            AppLog.i("FastVideoProcessor: Video processing finalized")
        } catch (e: Exception) {
            AppLog.e("FastVideoProcessor: Pipeline crashed", e)
        } finally {
            retriever.release()
            extractor.release()
        }
    }

    private fun selectVideoTrack(extractor: MediaExtractor): Int {
        for (i in 0 until extractor.trackCount) {
            val format = extractor.getTrackFormat(i)
            if (format.getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true) return i
        }
        return -1
    }
}
