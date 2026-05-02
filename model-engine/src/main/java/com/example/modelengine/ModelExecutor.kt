package com.example.modelengine

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.PyTorchAndroid
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * ModelExecutor handles the loading and execution of PyTorch Lite tracking models.
 * Supports multiple tracked objects by maintaining separate model instances.
 */
class ModelExecutor(private val context: Context) {
    private var modelPath: String? = null
    private var lastInferenceTime: Long = 0
    
    private val trackingInstances = mutableListOf<TrackingInstance>()

    private class TrackingInstance(
        val id: Int,
        val module: Module,
        var templateIValue: IValue,
        var lastBbox: RectF
    )

    fun loadModel(assetName: String) {
        try {
            // Enable CPU Multi-threading for fallback or hybrid execution
            val numThreads = Runtime.getRuntime().availableProcessors()
            PyTorchAndroid.setNumThreads(numThreads)
            Log.d(TAG, "CPU threads set to: $numThreads")

            modelPath = assetFilePath(context, assetName)
            Log.d(TAG, "Model path initialized: $modelPath")
        } catch (e: IOException) {
            Log.e(TAG, "Model path initialization failed", e)
        }
    }

    fun reset() {
        trackingInstances.clear()
        lastInferenceTime = 0
    }

    /**
     * Initializes a new tracker instance for a new object.
     */
    fun addTracker(bitmap: Bitmap, bbox: RectF): Int {
        val path = modelPath ?: return -1
        try {
            // Reverting to standard Module.load to avoid UnsatisfiedLinkError with Lite native libs
            val newModule = Module.load(path)
            
            val templateBitmap = cropTemplate(bitmap, bbox)
            val resizedTemplate = Bitmap.createScaledBitmap(templateBitmap, 128, 128, true)
            
            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                resizedTemplate,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
            )
            val templateIValue = IValue.from(inputTensor)

            // Initialize this specific module instance
            try {
                newModule.runMethod("initialize", templateIValue)
            } catch (e: Exception) {
                Log.d(TAG, "'initialize' method not found for instance, will use forward")
            }

            val id = if (trackingInstances.isEmpty()) 0 else trackingInstances.maxOf { it.id } + 1
            trackingInstances.add(TrackingInstance(id, newModule, templateIValue, bbox))
            Log.d(TAG, "Added tracker instance $id")
            return id
        } catch (e: Exception) {
            Log.e(TAG, "Failed to add tracker instance", e)
            return -1
        }
    }

    fun trackAll(bitmap: Bitmap): Map<Int, RectF> {
        val results = mutableMapOf<Int, RectF>()
        if (trackingInstances.isEmpty()) return results

        val startTime = SystemClock.elapsedRealtime()
        
        // Prepare search tensor once for all instances (assuming they use same search size)
        val resizedSearch = Bitmap.createScaledBitmap(bitmap, 256, 256, true)
        val searchTensor = TensorImageUtils.bitmapToFloat32Tensor(
            resizedSearch,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )
        val searchIValue = IValue.from(searchTensor)

        try {
            for (instance in trackingInstances) {
                val output: IValue = try {
                    instance.module.runMethod("track", searchIValue)
                } catch (e: Exception) {
                    instance.module.forward(instance.templateIValue, searchIValue)
                }

                val resultTensor = if (output.isTuple) {
                    val tuple = output.toTuple()
                    var found: org.pytorch.Tensor? = null
                    for (item in tuple) {
                        if (item.isTensor) {
                            val t = item.toTensor()
                            if (t.shape().last() == 4L) {
                                found = t
                                break
                            }
                        }
                    }
                    found ?: tuple[0].toTensor()
                } else {
                    output.toTensor()
                }

                val result = resultTensor.dataAsFloatArray
                Log.d(TAG, "Instance ${instance.id} model output: ${result.joinToString(", ")}")
                
                val newBbox = mapResult(result, bitmap.width, bitmap.height, instance.lastBbox)
                Log.d(TAG, "Instance ${instance.id} mapped bbox: $newBbox (img size: ${bitmap.width}x${bitmap.height})")

                instance.lastBbox = newBbox
                results[instance.id] = newBbox
            }
        } catch (e: Exception) {
            Log.e(TAG, "Tracking loop failed", e)
        }
        
        lastInferenceTime = SystemClock.elapsedRealtime() - startTime
        return results
    }

    fun getLastInferenceTime(): Long = lastInferenceTime

    private fun cropTemplate(bitmap: Bitmap, bbox: RectF): Bitmap {
        val x = bbox.left.toInt().coerceIn(0, bitmap.width - 1)
        val y = bbox.top.toInt().coerceIn(0, bitmap.height - 1)
        val w = bbox.width().toInt().coerceIn(1, bitmap.width - x)
        val h = bbox.height().toInt().coerceIn(1, bitmap.height - y)
        return Bitmap.createBitmap(bitmap, x, y, w, h)
    }

    private fun mapResult(result: FloatArray, imgW: Int, imgH: Int, lastBbox: RectF): RectF {
        if (result.size < 4) return lastBbox
        
        // AbaViTrack typically returns normalized [cx, cy, w, h] or [x, y, w, h] 
        // relative to the search image size (256x256).
        // Based on Python reference: cv.rectangle(frame_disp, (x, y), (x+w, y+h))
        
        val isNormalized = result.all { it <= 1.1f && it >= -0.1f }
        val scaleX = if (isNormalized) imgW.toFloat() else imgW.toFloat() / 256f
        val scaleY = if (isNormalized) imgH.toFloat() else imgH.toFloat() / 256f
        
        // If the values look like [center_x, center_y, width, height]
        // result[0] and result[1] are often ~0.5 in the search window
        val x: Float
        val y: Float
        val w: Float
        val h: Float

        // Heuristic: if it's AbaViTrack/Stark style, it might be [cx, cy, w, h] normalized
        if (isNormalized && result[0] > 0.1 && result[1] > 0.1) {
            // Try [cx, cy, w, h] to [x, y, w, h]
            w = result[2] * scaleX
            h = result[3] * scaleY
            x = (result[0] * scaleX) - (w / 2f)
            y = (result[1] * scaleY) - (h / 2f)
        } else {
            // Assume [x, y, w, h]
            x = result[0] * scaleX
            y = result[1] * scaleY
            w = result[2] * scaleX
            h = result[3] * scaleY
        }
        
        return RectF(x, y, x + w, y + h)
    }

    companion object {
        private const val TAG = "ModelExecutor"
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
