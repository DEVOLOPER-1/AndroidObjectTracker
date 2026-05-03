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
 *
 * Two public entry-points:
 *  • detect()         – raw YOLO output, no SORT. Use for Bootstrap (Frame 0).
 *  • detectAndTrack() – YOLO + SORT. Use for live camera preview.
 */
class ModelExecutor(private val context: Context) {
    private var module: Module? = null
    private var lastInferenceTime: Long = 0
    private val sortTracker = SortTracker()

    // YOLOv8 input dimensions
    private val inputWidth = 640
    private val inputHeight = 640

    // Confidence threshold for initial detections.
    // Lowered to 0.15 to capture small/distant pins.
    private val confidenceThreshold = 0.15f
    private val nmsThreshold = 0.45f

    // YOLOv8-nano trained on 3 classes (Car=0, Pin=1, Fallen=2).
    private val numClasses = 3

    fun loadModel(assetName: String) {
        try {
            val numThreads = Runtime.getRuntime().availableProcessors()
            PyTorchAndroid.setNumThreads(numThreads)
            AppLog.i("ModelExecutor: Using $numThreads CPU threads")

            val path = assetFilePath(context, assetName)
            module = Module.load(path)
            AppLog.i("ModelExecutor: Model loaded from $path")
        } catch (e: IOException) {
            AppLog.e("ModelExecutor: Model loading failed", e)
        }
    }

    fun reset() {
        sortTracker.reset()
        lastInferenceTime = 0
    }

    // -------------------------------------------------------------------------
    // PUBLIC: Raw YOLO detection — NO SORT.  Use for Frame 0 bootstrap.
    // Returns detections scaled to the input bitmap's pixel space.
    // -------------------------------------------------------------------------
    fun detect(bitmap: Bitmap): List<SortTracker.Detection> {
        val moduleInstance = module ?: run {
            AppLog.e("ModelExecutor.detect: model not loaded")
            return emptyList()
        }

        val startTime = SystemClock.elapsedRealtime()

        // 1. Pre-process
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            resizedBitmap,
            floatArrayOf(0f, 0f, 0f),   // no mean subtraction (YOLO expects 0-1 range)
            floatArrayOf(1f, 1f, 1f)    // no std division
        )
        if (resizedBitmap != bitmap) resizedBitmap.recycle()

        // 2. Inference
        val outputTensor = moduleInstance.forward(IValue.from(inputTensor)).toTensor()
        val data = outputTensor.dataAsFloatArray

        // 3. Parse + NMS
        val rawDetections = when {
            // End-to-end format: [1, 300, 6]  (e.g. YOLOv10)
            data.size == 1800 -> parseEndToEndOutput(data)
            // Standard YOLOv8 transposed format: [1, (4+numClasses), 8400]
            else -> applyNMS(parseYoloOutput(data))
        }

        lastInferenceTime = SystemClock.elapsedRealtime() - startTime
        AppLog.d("ModelExecutor.detect: ${rawDetections.size} detections in ${lastInferenceTime}ms")

        // 4. Scale from model space (640×640) back to original bitmap space
        val scaleX = bitmap.width.toFloat() / inputWidth
        val scaleY = bitmap.height.toFloat() / inputHeight

        return rawDetections.map { det ->
            AppLog.d("  > Found [Class ${det.classIndex}] conf=${"%.3f".format(det.confidence)}")
            SortTracker.Detection(
                RectF(
                    det.bbox.left   * scaleX,
                    det.bbox.top    * scaleY,
                    det.bbox.right  * scaleX,
                    det.bbox.bottom * scaleY
                ),
                det.classIndex,
                det.confidence
            )
        }
    }

    // -------------------------------------------------------------------------
    // PUBLIC: YOLO + SORT.  Use for live camera preview only.
    // -------------------------------------------------------------------------
    fun detectAndTrack(bitmap: Bitmap): List<SortTracker.Track> {
        val scaledDetections = detect(bitmap)
        return sortTracker.update(scaledDetections)
    }

    // -------------------------------------------------------------------------
    // PRIVATE: Parsing helpers
    // -------------------------------------------------------------------------

    /**
     * Parses standard YOLOv8 output shape [1, (4+numClasses), 8400].
     * The tensor is stored in row-major order: data[row * 8400 + col].
     */
    private fun parseYoloOutput(data: FloatArray): List<SortTracker.Detection> {
        val numPredictions = 8400
        val expectedSize = (4 + numClasses) * numPredictions

        if (data.size != expectedSize) {
            AppLog.e("parseYoloOutput: size mismatch — expected $expectedSize, got ${data.size}. " +
                     "Check numClasses (currently $numClasses).")
            // Attempt a best-effort parse by inferring numClasses from actual data size
            val inferredClasses = (data.size / numPredictions) - 4
            if (inferredClasses < 1 || inferredClasses > 100) return emptyList()
            return parseYoloOutputWithClasses(data, inferredClasses, numPredictions)
        }

        return parseYoloOutputWithClasses(data, numClasses, numPredictions)
    }

    private fun parseYoloOutputWithClasses(
        data: FloatArray,
        nc: Int,
        numPred: Int
    ): List<SortTracker.Detection> {
        val detections = mutableListOf<SortTracker.Detection>()

        for (i in 0 until numPred) {
            var maxProb = 0f
            var classIdx = -1
            for (c in 0 until nc) {
                val prob = data[(4 + c) * numPred + i]
                if (prob > maxProb) {
                    maxProb = prob
                    classIdx = c
                }
            }

            if (maxProb > confidenceThreshold) {
                val cx = data[0 * numPred + i]
                val cy = data[1 * numPred + i]
                val w  = data[2 * numPred + i]
                val h  = data[3 * numPred + i]

                detections.add(
                    SortTracker.Detection(
                        RectF(cx - w / 2f, cy - h / 2f, cx + w / 2f, cy + h / 2f),
                        classIdx,
                        maxProb
                    )
                )
            }
        }
        return detections
    }

    /**
     * Parses end-to-end model output shape [1, 300, 6].
     * Format per detection: [x1, y1, x2, y2, score, classId].
     */
    private fun parseEndToEndOutput(data: FloatArray): List<SortTracker.Detection> {
        val detections = mutableListOf<SortTracker.Detection>()
        val numDetections = data.size / 6

        for (i in 0 until numDetections) {
            val offset = i * 6
            val score    = data[offset + 4]
            val classIdx = data[offset + 5].toInt()

            if (score > confidenceThreshold) {
                detections.add(
                    SortTracker.Detection(
                        RectF(data[offset], data[offset + 1], data[offset + 2], data[offset + 3]),
                        classIdx,
                        score
                    )
                )
            }
        }
        return detections
    }

    private fun applyNMS(detections: List<SortTracker.Detection>): List<SortTracker.Detection> {
        val sorted  = detections.sortedByDescending { it.confidence }
        val result  = mutableListOf<SortTracker.Detection>()
        val ignored = BooleanArray(sorted.size)

        for (i in sorted.indices) {
            if (ignored[i]) continue
            result.add(sorted[i])
            for (j in i + 1 until sorted.size) {
                if (ignored[j]) continue
                if (calculateIoU(sorted[i].bbox, sorted[j].bbox) > nmsThreshold) {
                    ignored[j] = true
                }
            }
        }
        return result
    }

    private fun calculateIoU(r1: RectF, r2: RectF): Float {
        val inter = RectF()
        if (!inter.setIntersect(r1, r2)) return 0f
        val interArea = inter.width() * inter.height()
        val unionArea = r1.width() * r1.height() + r2.width() * r2.height() - interArea
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
