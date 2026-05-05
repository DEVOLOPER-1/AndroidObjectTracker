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
    private val CONFIDENCE_THRESHOLD = 0.45f
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

        // 1. Preprocessing — unchanged
        val resizedBitmap = if (bitmap.width == inputSize && bitmap.height == inputSize) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        }
        bitmapToFloatArray(resizedBitmap)
        if (resizedBitmap != bitmap) resizedBitmap.recycle()

        val inputTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(floatArray), longArrayOf(1, 3, 640, 640)
        )

        // 2. Inference — unchanged
        val results     = session.run(Collections.singletonMap(session.inputNames.iterator().next(), inputTensor))
        val outputTensor = results[0] as OnnxTensor
        val outputShape  = outputTensor.info.shape   // [1, 300, 6]
        val outputData   = outputTensor.floatBuffer.array()

        AppLog.i("YoloExecutor: output tensor shape = [${outputShape.joinToString(", ")}]")

        // 3. Parsing — REWRITTEN for post-NMS format [1, numSlots, 6]
        //
        // This model was exported with NMS already applied.
        // Each row i represents one detection:
        //   [x1, y1, x2, y2, confidence, class_id]
        // Coordinates are in the 640x640 input space and must be scaled back.
        // Empty slots have confidence ≈ 0 and are filtered out.
        //
        // The old parser assumed raw YOLOv8 format [1, 4+classes, 8400] which
        // is completely wrong for this model — it was reading x-coordinates as
        // confidence scores, producing values in the hundreds.

        val numSlots = outputShape[1].toInt()   // 300
        val numFields = outputShape[2].toInt()  // 6  →  x1,y1,x2,y2,conf,cls

        val scaleX = bitmap.width.toFloat()  / inputSize
        val scaleY = bitmap.height.toFloat() / inputSize

        val detections = mutableListOf<Detection>()

        for (i in 0 until numSlots) {
            val base = i * numFields

            val conf  = outputData[base + 4]
            if (conf < CONFIDENCE_THRESHOLD) continue   // skip empty / low-confidence slots

            val clsId = outputData[base + 5].toInt()
            if (clsId !in 0..3) continue                // guard against out-of-range class ids

            val x1 = outputData[base + 0] * scaleX
            val y1 = outputData[base + 1] * scaleY
            val x2 = outputData[base + 2] * scaleX
            val y2 = outputData[base + 3] * scaleY

            detections.add(Detection(RectF(x1, y1, x2, y2), clsId, conf))

            AppLog.d(
                "YoloExecutor: slot[$i] cls=$clsId conf=${"%.2f".format(conf)} " +
                        "bbox=(${x1.toInt()},${y1.toInt()},${x2.toInt()},${y2.toInt()})"
            )
        }

        // 4. No NMS — model already applied it. Skip applyNMS().

        AppLog.metric("InferenceTime", System.currentTimeMillis() - startTime)
        AppLog.d("Detected ${detections.size} objects after post-NMS parsing.")

        return detections
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
