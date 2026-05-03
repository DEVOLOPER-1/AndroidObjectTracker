package com.example.modelengine

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.PyTorchAndroid
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * ModelExecutor handles the loading and execution of YOLOv8 TorchScript Lite models.
 */
class ModelExecutor(private val context: Context) {
    private var modelPath: String? = null
    private var module: Module? = null
    private var lastInferenceTime: Long = 0
    private val sortTracker = SortTracker()

    // YOLOv8 Constants
    private val inputWidth = 640
    private val inputHeight = 640
    private val confidenceThreshold = 0.35f
    private val nmsThreshold = 0.45f

    fun loadModel(assetName: String) {
        try {
            val numThreads = Runtime.getRuntime().availableProcessors()
            PyTorchAndroid.setNumThreads(numThreads)
            AppLog.i("Setting CPU threads to: $numThreads")

            modelPath = assetFilePath(context, assetName)
            module = Module.load(modelPath)
            AppLog.i("Model successfully loaded from: $modelPath")
        } catch (e: IOException) {
            AppLog.e("Model loading failed", e)
        }
    }

    fun reset() {
        sortTracker.reset()
        lastInferenceTime = 0
    }

    /**
     * Executes YOLO inference on the full frame and updates object tracks.
     */
    fun detectAndTrack(bitmap: Bitmap): List<SortTracker.Track> {
        val moduleInstance = module ?: return emptyList()
        val startTime = SystemClock.elapsedRealtime()

        // 1. Pre-processing: Resize and Normalize
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            resizedBitmap,
            floatArrayOf(0f, 0f, 0f), // YOLO typically expects 0-1, so mean 0, std 1
            floatArrayOf(1f, 1f, 1f)
        )
        
        // Recycle resized bitmap if it was a copy
        if (resizedBitmap != bitmap) {
            resizedBitmap.recycle()
        }

        // 2. Inference
        val output = moduleInstance.forward(IValue.from(inputTensor))
        val outputTensor = output.toTensor()
        val data = outputTensor.dataAsFloatArray

        // 3. Post-processing: Parse boxes and Apply NMS
        val detections = if (data.size == 1800) {
            parseEndToEndOutput(data)
        } else {
            val rawDetections = parseYoloOutput(data)
            applyNMS(rawDetections)
        }

        // 4. Update Tracker
        val tracks = sortTracker.update(detections)

        lastInferenceTime = SystemClock.elapsedRealtime() - startTime
        return tracks
    }

    private fun parseYoloOutput(data: FloatArray): List<SortTracker.Detection> {
        val detections = mutableListOf<SortTracker.Detection>()
        val numPredictions = 8400
        val numClasses = 3
        
        val expectedSize = (4 + numClasses) * numPredictions
        if (data.size != expectedSize) {
            AppLog.e("YOLO output data size mismatch: expected $expectedSize, got ${data.size}")
            return detections
        }

        for (i in 0 until numPredictions) {
            var maxProb = 0f
            var classIdx = -1
            for (c in 0 until numClasses) {
                val prob = data[ (4 + c) * numPredictions + i ]
                if (prob > maxProb) {
                    maxProb = prob
                    classIdx = c
                }
            }

            if (maxProb > confidenceThreshold) {
                val cx = data[ 0 * numPredictions + i ]
                val cy = data[ 1 * numPredictions + i ]
                val w = data[ 2 * numPredictions + i ]
                val h = data[ 3 * numPredictions + i ]

                val left = cx - w / 2f
                val top = cy - h / 2f
                
                detections.add(
                    SortTracker.Detection(
                        RectF(left, top, left + w, top + h),
                        classIdx,
                        maxProb
                    )
                )
            }
        }
        return detections
    }

    /**
     * Parses End-to-End models (e.g. YOLOv10) which output [1, 300, 6].
     * Data format: [x1, y1, x2, y2, score, class]
     */
    private fun parseEndToEndOutput(data: FloatArray): List<SortTracker.Detection> {
        val detections = mutableListOf<SortTracker.Detection>()
        val numDetections = 300
        val elementsPerDetection = 6

        for (i in 0 until numDetections) {
            val offset = i * elementsPerDetection
            val score = data[offset + 4]
            val classIdx = data[offset + 5].toInt()
            
            if (i < 5) { // Log top 5 detections for debugging
                AppLog.d("Top Detections [$i]: Class=$classIdx, Score=$score")
            }

            if (score > confidenceThreshold) {
                val x1 = data[offset + 0]
                val y1 = data[offset + 1]
                val x2 = data[offset + 2]
                val y2 = data[offset + 3]
                
                detections.add(
                    SortTracker.Detection(
                        RectF(x1, y1, x2, y2),
                        classIdx,
                        score
                    )
                )
            }
        }
        return detections
    }

    private fun applyNMS(detections: List<SortTracker.Detection>): List<SortTracker.Detection> {
        val sorted = detections.sortedByDescending { it.confidence }
        val result = mutableListOf<SortTracker.Detection>()
        val ignored = BooleanArray(sorted.size)

        for (i in sorted.indices) {
            if (ignored[i]) continue
            val d1 = sorted[i]
            result.add(d1)
            for (j in i + 1 until sorted.size) {
                if (ignored[j]) continue
                val d2 = sorted[j]
                if (calculateIoU(d1.bbox, d2.bbox) > nmsThreshold) {
                    ignored[j] = true
                }
            }
        }
        return result
    }

    private fun calculateIoU(rect1: RectF, rect2: RectF): Float {
        val intersection = RectF()
        if (!intersection.setIntersect(rect1, rect2)) return 0f
        val interArea = intersection.width() * intersection.height()
        val unionArea = (rect1.width() * rect1.height()) + (rect2.width() * rect2.height()) - interArea
        return if (unionArea > 0) interArea / unionArea else 0f
    }

    fun getLastInferenceTime(): Long = lastInferenceTime

    companion object {
        @Throws(IOException::class)
        fun assetFilePath(context: Context, assetName: String): String {
            val file = File(context.filesDir, assetName)
            if (file.exists() && file.length() > 0) return file.absolutePath
            context.assets.open(assetName).use { input ->
                FileOutputStream(file).use { output ->
                    val buffer = ByteArray(4096)
                    var read: Int
                    while (input.read(buffer).also { read = it } != -1) output.write(buffer, 0, read)
                }
            }
            return file.absolutePath
        }
    }
}

