package com.example.modelengine

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.util.Collections

/**
 * Executes YOLOv8 models using ONNX Runtime.
 * Handles preprocessing, inference, output parsing, NMS, and coordinate scaling.
 */
class YoloExecutor(context: Context, modelPath: String) {

    data class Detection(
        val bbox: RectF,
        val classIndex: Int,
        val confidence: Float
    )

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    private val inputSize = 640
    private val pixels = IntArray(inputSize * inputSize)
    private val floatArray = FloatArray(3 * inputSize * inputSize)

    init {
        val modelBytes = context.assets.open(modelPath).readBytes()
        val options = OrtSession.SessionOptions()
        try {
            options.addConfigEntry("session.use_nnapi", "1")
        } catch (e: Exception) {
            AppLog.e("NNAPI not available", e)
        }
        session = env.createSession(modelBytes, options)
        AppLog.i("YOLO Executor initialized with model: $modelPath (NNAPI enabled)")
    }

    /**
     * Runs detection on a given bitmap.
     * Returns a list of detections with coordinates scaled to original bitmap size.
     */
    fun detect(bitmap: Bitmap): List<Detection> {
        val startTime = System.currentTimeMillis()
        
        // 1. Preprocessing
        val resizedBitmap = if (bitmap.width == inputSize && bitmap.height == inputSize) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        }
        
        bitmapToFloatArray(resizedBitmap)
        if (resizedBitmap != bitmap) resizedBitmap.recycle()

        val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArray), longArrayOf(1, 3, 640, 640))

        // 2. Inference
        val results = session.run(Collections.singletonMap(session.inputNames.iterator().next(), inputTensor))
        val outputTensor = results[0] as OnnxTensor
        val outputShape = outputTensor.info.shape // e.g., [1, 8, 8400]
        val outputData = outputTensor.floatBuffer.array()

        // 3. Parsing (YOLOv8 output: [1, numRows, numColumns])
        // numRows = 4 (bbox) + numClasses
        val detections = mutableListOf<Detection>()
        val numRows = outputShape[1].toInt()
        val numColumns = outputShape[2].toInt()
        val numClasses = numRows - 4

        for (c in 0 until numColumns) {
            var maxConf = 0f
            var maxClass = -1
            
            for (classIdx in 0 until numClasses) {
                val conf = outputData[numColumns * (4 + classIdx) + c]
                if (conf > maxConf) {
                    maxConf = conf
                    maxClass = classIdx
                }
            }

            if (maxConf > 0.45f) {
                val cx = outputData[c]
                val cy = outputData[numColumns + c]
                val w = outputData[2 * numColumns + c]
                val h = outputData[3 * numColumns + c]

                val left = (cx - w / 2f)
                val top = (cy - h / 2f)
                val right = (cx + w / 2f)
                val bottom = (cy + h / 2f)

                detections.add(Detection(RectF(left, top, right, bottom), maxClass, maxConf))
            }
        }

        // 4. Non-Maximum Suppression
        val nmsDetections = applyNMS(detections)

        // 5. Scale coordinates back to original bitmap size
        val scaleX = bitmap.width.toFloat() / inputSize
        val scaleY = bitmap.height.toFloat() / inputSize
        
        val finalDetections = nmsDetections.map { det ->
            val scaledBbox = RectF(
                det.bbox.left * scaleX,
                det.bbox.top * scaleY,
                det.bbox.right * scaleX,
                det.bbox.bottom * scaleY
            )
            det.copy(bbox = scaledBbox)
        }

        val endTime = System.currentTimeMillis()
        AppLog.metric("InferenceTime", endTime - startTime)
        AppLog.d("Detected ${finalDetections.size} objects after NMS.")
        
        return finalDetections
    }

    private fun bitmapToFloatArray(bitmap: Bitmap) {
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        
        val channelSize = inputSize * inputSize
        for (i in pixels.indices) {
            val pixel = pixels[i]
            // NCHW format, RGB, normalized to [0, 1]
            floatArray[i] = ((pixel shr 16) and 0xFF) / 255.0f
            floatArray[i + channelSize] = ((pixel shr 8) and 0xFF) / 255.0f
            floatArray[i + 2 * channelSize] = (pixel and 0xFF) / 255.0f
        }
    }

    private fun applyNMS(detections: List<Detection>): List<Detection> {
        val sortedDetections = detections.sortedByDescending { it.confidence }.toMutableList()
        val selectedDetections = mutableListOf<Detection>()

        while (sortedDetections.isNotEmpty()) {
            val best = sortedDetections.removeAt(0)
            selectedDetections.add(best)
            
            val iterator = sortedDetections.iterator()
            while (iterator.hasNext()) {
                val next = iterator.next()
                if (calculateIoU(best.bbox, next.bbox) > 0.45f) {
                    iterator.remove()
                }
            }
        }
        return selectedDetections
    }

    private fun calculateIoU(rect1: RectF, rect2: RectF): Float {
        val intersection = RectF()
        if (!intersection.setIntersect(rect1, rect2)) return 0f
        
        val interArea = intersection.width() * intersection.height()
        val unionArea = (rect1.width() * rect1.height()) + 
                        (rect2.width() * rect2.height()) - interArea
        
        return if (unionArea > 0) interArea / unionArea else 0f
    }

    fun close() {
        session.close()
        env.close()
    }
}
