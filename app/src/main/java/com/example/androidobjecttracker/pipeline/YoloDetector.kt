package com.example.androidobjecttracker.pipeline

import android.content.Context
import android.graphics.*
import ai.onnxruntime.*
import com.example.androidobjecttracker.utils.AppLog
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer
import kotlin.math.*

class YoloDetector(private val context: Context) {

    // Model constants
    private val inputSize = 640
    private val modelInputName = "images"         // check with Netron, user snippet says "images"
    private val modelOutputName = "output0"       // adjust if needed
    private val confThreshold = 0.4f
    private val iouThreshold = 0.2f  // for tracking, not NMS
    private val MATCH_RADIUS = 30f   // pixels, for matching fallen pins
    private val FALL_CONFIRM_FRAMES = 5

    // Classes
    private val PIN_CLASS = 1
    private val CAR_CLASS = 3

    // Fall detection thresholds (matched to Python)
    private val ASPECT_FALLEN_MAX = 0.6f

    // State
    private val pinTracks = mutableListOf<Pintrack>()
    private var nextPinId = 0
    private var pinFallOrder = 0
    private var currentFrameFallen = 0
    private val carPath = mutableListOf<PointF>()
    private var totalPinsDown = 0
    private var startTimeMs: Long = 0
    private var elapsedMs: Long = 0
    private val fallLog = mutableListOf<Triple<Long, Int, Int>>() // timestamp, id, order

    // ONNX Runtime
    private lateinit var ortSession: OrtSession
    private lateinit var ortEnv: OrtEnvironment

    // Pre-allocate arrays for preprocessing
    private val inputShape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())

    suspend fun load() {
        withContext(Dispatchers.IO) {
            try {
                ortEnv = OrtEnvironment.getEnvironment()
                val modelBytes = context.assets.open("bowling.onnx").readBytes()

                // Try enabling NNAPI for hardware acceleration
                val options = OrtSession.SessionOptions()
                options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                // Exynos 7904 has 8 cores, set threads to avoid overwhelming if fallback happens
                options.setIntraOpNumThreads(2) 
                
                try {
                    // Attempt NNAPI with FP16 if supported
                    options.addNnapi()
                    ortSession = ortEnv.createSession(modelBytes, options)
                    AppLog.i("ONNX Session created with NNAPI acceleration")
                } catch (e: OrtException) {
                    AppLog.e("NNAPI failed, using CPU with optimized threads", e)
                    val cpuOptions = OrtSession.SessionOptions()
                    cpuOptions.setIntraOpNumThreads(4) // Use more threads for pure CPU
                    ortSession = ortEnv.createSession(modelBytes, cpuOptions)
                }
            } catch (e: Exception) {
                AppLog.e("Critical error loading YoloDetector", e)
            }
        }
    }

    // Helper data class for detections
    private data class PinDetection(
        val bbox: FloatArray,
        val centroid: PointF,
        val aspect: Float
    )

    // Process one frame and return the annotated bitmap
    fun processFrame(sourceBitmap: Bitmap, currentTimeMs: Long): Bitmap {
        if (startTimeMs == 0L) startTimeMs = currentTimeMs
        elapsedMs = currentTimeMs - startTimeMs
        currentFrameFallen = 0

        // Preprocess and run inference
        val resized = Bitmap.createScaledBitmap(sourceBitmap, inputSize, inputSize, true)
        val floatBuffer = FloatBuffer.allocate(3 * inputSize * inputSize)
        bitmapToFloatBuffer(resized, floatBuffer)
        floatBuffer.rewind()

        // Run inference
        val inputTensor = OnnxTensor.createTensor(ortEnv, floatBuffer, inputShape)
        val pinDets = mutableListOf<FloatArray>()   // [x1,y1,x2,y2,conf]
        val carDets = mutableListOf<FloatArray>()

        inputTensor.use {
            ortSession.run(mapOf(modelInputName to inputTensor)).use { outputs ->
                val rawOutput = (outputs[0] as OnnxTensor).floatBuffer   // size: 300 * 6

                rawOutput.position(0)
                while (rawOutput.remaining() >= 6) {
                    val x1 = rawOutput.get()
                    val y1 = rawOutput.get()
                    val x2 = rawOutput.get()
                    val y2 = rawOutput.get()
                    val conf = rawOutput.get()
                    val cls = rawOutput.get()
                    if (conf < confThreshold) continue
                    when (cls.toInt()) {
                        PIN_CLASS -> pinDets.add(floatArrayOf(x1, y1, x2, y2, conf))
                        CAR_CLASS -> carDets.add(floatArrayOf(x1, y1, x2, y2, conf))
                    }
                }
            }
        }

        AppLog.d("Frame: ${pinDets.size} pins, ${carDets.size} cars detected. Active tracks: ${pinTracks.size}")

        // Scale coordinates back to original bitmap size
        val scaleX = sourceBitmap.width.toFloat() / inputSize
        val scaleY = sourceBitmap.height.toFloat() / inputSize
        for (d in pinDets) { d[0]*=scaleX; d[1]*=scaleY; d[2]*=scaleX; d[3]*=scaleY }
        for (d in carDets) { d[0]*=scaleX; d[1]*=scaleY; d[2]*=scaleX; d[3]*=scaleY }

        // --- Tracking & annotation ---
        val mutableBitmap = sourceBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)

        // Car path
        for (car in carDets) {
            val cx = (car[0] + car[2]) / 2f
            val cy = (car[1] + car[3]) / 2f
            carPath.add(PointF(cx, cy))
        }
        drawCarPath(canvas)

        // Pin tracking (Python-like logic)
        trackPins(pinDets)

        // Draw pins
        for (track in pinTracks) {
            if (track.bbox[0] == 0f && track.bbox[2] == 0f) continue // Skip empty bboxes
            val x1 = track.bbox[0]
            val y1 = track.bbox[1]
            val x2 = track.bbox[2]
            val y2 = track.bbox[3]
            
            val trackColor = if (!track.fallen) Color.RED else Color.GREEN
            val paint = Paint().apply {
                this.color = trackColor
                style = Paint.Style.STROKE
                strokeWidth = 4f
            }
            canvas.drawRect(x1, y1, x2, y2, paint)
            if (track.fallen && track.order > 0) {
                val textPaint = Paint().apply {
                    this.color = Color.WHITE
                    textSize = 36f
                    isFakeBoldText = true
                    setShadowLayer(4f, 2f, 2f, Color.BLACK)
                }
                canvas.drawText("#${track.order}", x1, y1 - 10f, textPaint)
            }
        }

        // Score overlay
        drawScoreOverlay(canvas)

        return mutableBitmap
    }

    private fun trackPins(pinDets: MutableList<FloatArray>) {
        val curDetections = pinDets.map {
            val w = it[2] - it[0]
            val h = it[3] - it[1]
            val aspect = if (w > 0) h / w else 0f
            AppLog.metric("pin_aspect", "%.2f".format(aspect))
            PinDetection(
                bbox = it,
                centroid = PointF((it[0] + it[2]) / 2f, (it[1] + it[3]) / 2f),
                aspect = aspect
            )
        }

        val handledDetections = BooleanArray(curDetections.size)

        // 1. Match with already fallen tracks (Proximity)
        for (track in pinTracks) {
            if (!track.fallen) continue
            var bestMatchIdx = -1
            var minDist = MATCH_RADIUS
            for (i in curDetections.indices) {
                if (handledDetections[i]) continue
                val d = dist(track.centroid, curDetections[i].centroid)
                if (d < minDist) {
                    minDist = d
                    bestMatchIdx = i
                }
            }
            if (bestMatchIdx != -1) {
                handledDetections[bestMatchIdx] = true
                updateTrackPosition(track, curDetections[bestMatchIdx])
            }
        }

        // 2. Match remaining with non-fallen tracks
        for (track in pinTracks) {
            if (track.fallen) continue
            var bestMatchIdx = -1
            var minDist = 100f // Larger threshold for moving/upright pins
            for (i in curDetections.indices) {
                if (handledDetections[i]) continue
                val d = dist(track.centroid, curDetections[i].centroid)
                if (d < minDist) {
                    minDist = d
                    bestMatchIdx = i
                }
            }
            if (bestMatchIdx != -1) {
                handledDetections[bestMatchIdx] = true
                updateTrackWithFallLogic(track, curDetections[bestMatchIdx])
            }
        }

        // 3. New tracks for unmatched detections
        for (i in curDetections.indices) {
            if (!handledDetections[i]) {
                val det = curDetections[i]
                pinTracks.add(Pintrack(
                    id = nextPinId++,
                    centroid = det.centroid,
                    bbox = det.bbox,
                    aspect = det.aspect,
                    fallen = false,
                    order = 0,
                    fallFrames = 0
                ))
                // Log discovery of new pin
                AppLog.d("New pin track ${nextPinId-1} at ${det.centroid}")
            }
        }
    }

    private fun updateTrackPosition(track: Pintrack, det: PinDetection) {
        track.centroid = det.centroid
        track.bbox = det.bbox
        track.aspect = det.aspect
        if (det.aspect <= ASPECT_FALLEN_MAX) {
            currentFrameFallen++
        }
    }

    private fun updateTrackWithFallLogic(track: Pintrack, det: PinDetection) {
        val aspect = det.aspect
        track.centroid = det.centroid
        track.bbox = det.bbox
        track.aspect = aspect

        if (aspect <= ASPECT_FALLEN_MAX) {
            currentFrameFallen++
            track.fallFrames++
            AppLog.d("Pin ${track.id} looks fallen (aspect: %.2f), confirm count: ${track.fallFrames}".format(aspect))
        } else {
            track.fallFrames = (track.fallFrames - 2).coerceAtLeast(0)
        }

        if (track.fallFrames >= FALL_CONFIRM_FRAMES) {
            track.fallen = true
            track.order = ++pinFallOrder
            totalPinsDown++
            fallLog.add(Triple(elapsedMs, track.id, track.order))
            AppLog.i("Pin ${track.id} CONFIRMED FALLEN. Order: ${track.order}")
        }
    }

    private fun dist(p1: PointF, p2: PointF): Float {
        val dx = p1.x - p2.x
        val dy = p1.y - p2.y
        return sqrt(dx * dx + dy * dy)
    }

    private fun drawCarPath(canvas: Canvas) {
        if (carPath.size < 2) return
        val paint = Paint().apply {
            color = Color.YELLOW
            style = Paint.Style.STROKE
            strokeWidth = 6f
            pathEffect = DashPathEffect(floatArrayOf(10f, 5f), 0f)
        }
        for (i in 1 until carPath.size) {
            canvas.drawLine(carPath[i-1].x, carPath[i-1].y, carPath[i].x, carPath[i].y, paint)
        }
    }

    private fun drawScoreOverlay(canvas: Canvas) {
        val elapsedSec = elapsedMs / 1000f
        val displayPins = max(totalPinsDown, currentFrameFallen)
        val text = "Time: %.1fs   Pins down: %d".format(elapsedSec, displayPins)
        val paint = Paint().apply {
            this.color = Color.WHITE
            textSize = 40f
            isFakeBoldText = true
            setShadowLayer(4f, 2f, 2f, Color.BLACK)
        }
        val backgroundPaint = Paint().apply { color = Color.argb(120, 0, 0, 0) }
        canvas.drawRect(10f, 10f, 550f, 70f, backgroundPaint)
        canvas.drawText(text, 20f, 55f, paint)
    }

    // Preprocessing helper: normalize and convert to CHW
    private fun bitmapToFloatBuffer(bitmap: Bitmap, out: FloatBuffer) {
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        for (i in pixels.indices) {
            val r = (pixels[i] shr 16) and 0xFF
            val g = (pixels[i] shr 8) and 0xFF
            val b = pixels[i] and 0xFF
            out.put(i, r / 255f)                       // R channel
            out.put(inputSize*inputSize + i, g / 255f) // G
            out.put(2*inputSize*inputSize + i, b / 255f) // B
        }
    }

    // Data class for pin tracks
    data class Pintrack(
        val id: Int,
        var centroid: PointF,
        var bbox: FloatArray,   // [x1,y1,x2,y2]
        var aspect: Float,
        var fallen: Boolean,
        var order: Int,
        var fallFrames: Int = 0
    )

    fun release() {
        if (::ortSession.isInitialized) ortSession.close()
        if (::ortEnv.isInitialized) ortEnv.close()
    }

    // Added reset method to support orchestrator reset
    fun reset() {
        pinTracks.clear()
        nextPinId = 0
        pinFallOrder = 0
        carPath.clear()
        totalPinsDown = 0
        startTimeMs = 0
        elapsedMs = 0
    }

    // Helper for summary frame
    fun getSummaryInfo(): Pair<Int, Int> {
        return Pair(totalPinsDown, carPath.size)
    }
}
