package com.example.modelengine

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * ModelExecutor handles the loading and execution of PyTorch Lite models.
 * This class follows the Separation of Concerns principle by isolating
 * the inference logic from the UI.
 */
class ModelExecutor(private val context: Context) {
    private var module: Module? = null
    private var lastInferenceTime: Long = 0

    /**
     * Loads the model from the assets folder.
     * @param assetName The name of the .ptl file in the assets directory.
     */
    fun loadModel(assetName: String) {
        try {
            val modelPath = assetFilePath(context, assetName)
            module = LiteModuleLoader.load(modelPath)
            Log.d("ModelExecutor", "Model loaded successfully: $assetName")
        } catch (e: IOException) {
            Log.e("ModelExecutor", "Error loading model: $assetName", e)
        }
    }

    /**
     * Executes the model on the provided Bitmap.
     * @param bitmap The input image to process.
     * @return The output Tensor from the model.
     */
    fun execute(bitmap: Bitmap): Tensor? {
        val currentModule = module ?: return null
        
        val startTime = SystemClock.elapsedRealtime()
        
        // 1. Pre-processing: Convert Bitmap to Tensor
        // Resizing to 224x224 as is common for many vision models, 
        // though AbaViTrack might require different dimensions.
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            resizedBitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )

        // 2. Inference: Forward pass through the model
        val outputTensor = currentModule.forward(IValue.from(inputTensor)).toTensor()
        
        lastInferenceTime = SystemClock.elapsedRealtime() - startTime
        
        return outputTensor
    }

    /**
     * Returns the time taken for the last inference in milliseconds.
     */
    fun getLastInferenceTime(): Long = lastInferenceTime

    companion object {
        /**
         * Copies the asset to the internal storage and returns the absolute path.
         * This is required because PyTorch Lite's native code cannot read directly from assets.
         */
        @Throws(IOException::class)
        fun assetFilePath(context: Context, assetName: String): String {
            val file = File(context.filesDir, assetName)
            if (file.exists() && file.length() > 0) {
                return file.absolutePath
            }

            context.assets.open(assetName).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (inputStream.read(buffer).also { read = it } != -1) {
                        outputStream.write(buffer, 0, read)
                    }
                    outputStream.flush()
                }
                return file.absolutePath
            }
        }
    }
}
