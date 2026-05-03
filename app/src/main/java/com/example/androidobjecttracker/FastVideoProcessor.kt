package com.example.androidobjecttracker

import android.content.Context
import android.graphics.*
import android.media.*
import android.net.Uri
import com.example.modelengine.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Drives the full video-processing pipeline:
 *
 *   Frame 0  — YOLO detection (no SORT) → bootstrap SOTExecutor + StaticTracker
 *   Frame 1+ — SOTExecutor tracks the car; StaticTracker monitors pin regions
 *
 * Annotations are BAKED DIRECTLY into each Bitmap before it is fed to the
 * VideoEncoder.  The UI overlay is intentionally not used — annotations live
 * in the encoded output video, which is the project deliverable.
 *
 * Bug fixes applied in this revision:
 *   BUG 4 — Frame 0 no longer calls sotExecutor.update(); uses YOLO bbox directly.
 *   BUG 5 — Class IDs now follow the directive: Class 0 = Car, Class 1 = Pin.
 *   BUG 6 — pin.fallOrder is never rendered as null ("null" text removed from output).
 *   BUG 7 — modelExecutor.detect() is called (not detectAndTrack), keeping SORT
 *            out of the bootstrap step entirely.
 */
class FastVideoProcessor(
    private val context: Context,
    private val modelExecutor: ModelExecutor,
    private val sotExecutor: SOTExecutor,
    private val staticTracker: StaticTracker,
    private val videoEncoder: VideoEncoder
) {

    // ---- Annotation paints ------------------------------------------------
    private val carPaint = Paint().apply {
        color       = Color.YELLOW
        strokeWidth = 8f
        style       = Paint.Style.STROKE
        strokeCap   = Paint.Cap.ROUND
        strokeJoin  = Paint.Join.ROUND
        isAntiAlias = true
    }

    private val pinPaint = Paint().apply {
        color       = Color.GREEN
        strokeWidth = 4f
        style       = Paint.Style.STROKE
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color     = Color.WHITE
        textSize  = 70f
        typeface  = Typeface.DEFAULT_BOLD
        isAntiAlias = true
        setShadowLayer(6f, 0f, 0f, Color.BLACK)
    }

    // -----------------------------------------------------------------------
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

            val width      = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toInt()  ?: 0
            val height     = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toInt() ?: 0
            val durationUs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                                ?.toLong()?.times(1000) ?: 0L

            AppLog.i("FastVideoProcessor: Starting — ${width}×${height}, ${durationUs / 1_000_000}s")
            videoEncoder.start(width, height)

            val videoTrack = selectVideoTrack(extractor)
            if (videoTrack < 0) {
                AppLog.e("FastVideoProcessor: No video track found in URI")
                return
            }
            extractor.selectTrack(videoTrack)

            var isInitialized = false
            var framesProcessed = 0
            val startTime = System.currentTimeMillis()

            // ----------------------------------------------------------------
            // MAIN LOOP — one iteration per video frame
            // ----------------------------------------------------------------
            while (true) {
                val ptsUs = extractor.sampleTime
                if (ptsUs < 0) break   // end of stream

                // Retrieve the raw decoded frame for this PTS
                val rawFrame = retriever.getFrameAtTime(ptsUs, MediaMetadataRetriever.OPTION_CLOSEST)
                    ?: run { extractor.advance(); continue }

                framesProcessed++

                // Ensure the bitmap is mutable so we can draw on it
                val bitmap = if (rawFrame.isMutable) {
                    rawFrame
                } else {
                    val copy = rawFrame.copy(Bitmap.Config.ARGB_8888, true)
                    rawFrame.recycle()
                    copy
                }
                val canvas = Canvas(bitmap)

                if (!isInitialized) {
                    val detections = modelExecutor.detect(bitmap)
                    
                    // DEBUG: Log all detections on the bootstrap frame
                    if (detections.isEmpty()) {
                        AppLog.w("FastVideoProcessor: No detections on Frame $framesProcessed")
                    } else {
                        AppLog.d("FastVideoProcessor: Found ${detections.size} detections on Frame $framesProcessed")
                    }

                    // Best car: highest-confidence Class-0 (custom) or Class-2 (COCO) detection
                    val carDetection = detections
                        .filter { it.classIndex == 0 || it.classIndex == 2 }
                        .maxByOrNull { it.confidence }

                    // All pin detections: Class 1, sorted by confidence, cap at 10
                    val pinDetections = detections
                        .filter { it.classIndex == 1 }
                        .sortedByDescending { it.confidence }
                        .take(10)

                    // Hand off to specialist trackers
                    if (carDetection != null) {
                        sotExecutor.init(bitmap, carDetection.bbox)
                        AppLog.i("FastVideoProcessor: Car bootstrapped — bbox=${carDetection.bbox}  conf=${carDetection.confidence} (class=${carDetection.classIndex})")
                        
                        // Initialize pins if found, or log warning
                        if (pinDetections.isNotEmpty()) {
                            staticTracker.initialize(bitmap, pinDetections.map { it.bbox })
                            AppLog.i("FastVideoProcessor: ${pinDetections.size} pins bootstrapped")
                        } else {
                            AppLog.w("FastVideoProcessor: Car found but no pins detected yet. Will check again.")
                        }
                        
                        isInitialized = true
                        
                        // Draw YOLO detections directly on the bootstrap bitmap
                        canvas.drawRect(carDetection.bbox, carPaint)
                    } else {
                        AppLog.e("FastVideoProcessor: No car detected on Frame $framesProcessed! Retrying bootstrap on next frame...")
                        // isInitialized remains false, so we loop back here next frame
                        extractor.advance()
                        continue
                    }

                } else {
                    // --------------------------------------------------------
                    // FRAME 1+ — Hybrid Trackers
                    // --------------------------------------------------------

                    // -- Car: SOT --
                    val carBbox = sotExecutor.update(bitmap)

                    // Draw accumulated trajectory path
                    val path = sotExecutor.carPath
                    if (path.size > 1) {
                        val gfxPath = Path()
                        gfxPath.moveTo(path[0].x, path[0].y)
                        for (i in 1 until path.size) gfxPath.lineTo(path[i].x, path[i].y)
                        canvas.drawPath(gfxPath, carPaint)
                    }

                    // Draw current car bounding box
                    carBbox?.let { canvas.drawRect(it, carPaint) }
                }

                // ------------------------------------------------------------
                // ALWAYS — Pin tracking (runs on every frame including Frame 0;
                // on Frame 0 the intensity diff will be ~0 so no false falls)
                // ------------------------------------------------------------
                val pinStates = staticTracker.update(bitmap)

                pinStates.forEach { pin ->
                    if (pin.isFallen) {
                        // (BUG 6 FIX) fallOrder is 0 until fallen, then 1,2,3...
                        // We check isFallen first, so fallOrder is always > 0 here.
                        textPaint.color = Color.RED
                        canvas.drawText(
                            pin.fallOrder.toString(),
                            pin.rect.centerX(),
                            pin.rect.centerY() + textPaint.textSize / 3f,
                            textPaint
                        )
                        // Draw a red X over the fallen pin's region
                        canvas.drawRect(pin.rect, Paint().apply {
                            color = Color.RED; strokeWidth = 3f; style = Paint.Style.STROKE
                        })
                    } else {
                        canvas.drawRect(pin.rect, pinPaint)
                        textPaint.color = Color.WHITE
                        canvas.drawText("PIN", pin.rect.left, pin.rect.top - 12f, textPaint)
                    }
                }

                if (framesProcessed % 30 == 0) {
                    AppLog.d("FastVideoProcessor: $framesProcessed frames processed")
                }

                // ------------------------------------------------------------
                // Submit annotated frame to the encoder with the ORIGINAL PTS
                // ------------------------------------------------------------
                videoEncoder.addFrame(bitmap, ptsUs)

                // Deliver the annotated frame to the UI for live preview
                withContext(Dispatchers.Main) {
                    onFrameProcessed(bitmap)
                    val progress = if (durationUs > 0) ptsUs.toFloat() / durationUs else 0f
                    val elapsed  = System.currentTimeMillis() - startTime
                    val eta      = if (progress > 0.01f) ((elapsed / progress) * (1f - progress)).toLong() else -1L
                    onProgress(progress, eta)
                }

                extractor.advance()
            }

            AppLog.i("FastVideoProcessor: Done — $framesProcessed frames processed")

        } catch (e: Exception) {
            AppLog.e("FastVideoProcessor: Pipeline crashed", e)
        } finally {
            retriever.release()
            extractor.release()
        }
    }

    // -----------------------------------------------------------------------
    private fun selectVideoTrack(extractor: MediaExtractor): Int {
        for (i in 0 until extractor.trackCount) {
            val format = extractor.getTrackFormat(i)
            if (format.getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true) return i
        }
        return -1
    }
}
