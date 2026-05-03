package com.example.modelengine

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.RectF
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.IOException
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Executor for SOT (Single Object Tracking) models like AbaViTrack.
 * Handles template initialization and search-region-based updates.
 */
class SOTExecutor(private val context: Context) {
    private var module: Module? = null
    private val templateSize = 128
    private val searchSize = 256
    
    private var templateTensor: Tensor? = null
    private var lastBbox: RectF? = null
    private var searchRegion: RectF? = null

    fun loadModel(assetName: String) {
        try {
            val path = ModelExecutor.assetFilePath(context, assetName)
            module = Module.load(path)
            AppLog.i("SOT Model loaded: $assetName")
        } catch (e: Exception) {
            AppLog.e("SOT Model loading failed", e)
        }
    }

    /**
     * Initializes the tracker on the first frame with the target bounding box.
     */
    fun init(frame: Bitmap, bbox: RectF) {
        lastBbox = RectF(bbox)
        val templateBitmap = cropSquare(frame, bbox, templateSize, 2.0f)
        templateTensor = TensorImageUtils.bitmapToFloat32Tensor(
            templateBitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )
        templateBitmap.recycle()
        AppLog.i("SOT initialized at $bbox")
    }

    /**
     * Updates the target position in the current frame.
     */
    fun update(frame: Bitmap): RectF? {
        val mod = module ?: return null
        val zTensor = templateTensor ?: return null
        val prevBbox = lastBbox ?: return null

        // 1. Prepare search region (4x target size)
        val region = getSearchRegion(prevBbox, frame.width, frame.height, 4.0f)
        searchRegion = region
        val xBitmap = cropSquare(frame, region, searchSize, 1.0f)
        val xTensor = TensorImageUtils.bitmapToFloat32Tensor(
            xBitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )

        // 2. Inference (AbaViTrack typically takes two inputs)
        try {
            val outputs = mod.forward(IValue.from(zTensor), IValue.from(xTensor))
            
            val outputTensors = mutableListOf<Tensor>()
            if (outputs.isTuple) {
                outputs.toTuple().forEach { if (it.isTensor) outputTensors.add(it.toTensor()) }
            } else if (outputs.isList) {
                outputs.toList().forEach { if (it.isTensor) outputTensors.add(it.toTensor()) }
            } else if (outputs.isDictStringKey) {
                val dict = outputs.toDictStringKey()
                dict["score_map"]?.let { if (it.isTensor) outputTensors.add(it.toTensor()) }
                dict["size_map"]?.let { if (it.isTensor) outputTensors.add(it.toTensor()) }
                dict["offset_map"]?.let { if (it.isTensor) outputTensors.add(it.toTensor()) }
            } else if (outputs.isTensor) {
                outputTensors.add(outputs.toTensor())
            }

            if (outputTensors.size >= 3) {
                decodeOutput(outputTensors[0], outputTensors[1], outputTensors[2], region)
            }
        } catch (e: Exception) {
            AppLog.e("SOT Inference failed", e)
        }

        xBitmap.recycle()
        return lastBbox
    }

    /**
     * Decodes [score_map, size_map, offset_map] to update lastBbox.
     */
    private fun decodeOutput(scoreMap: Tensor, sizeMap: Tensor, offsetMap: Tensor, region: RectF) {
        val scores = scoreMap.dataAsFloatArray
        val sizes = sizeMap.dataAsFloatArray
        val offsets = offsetMap.dataAsFloatArray
        
        val gridH = 16 // 256 / 16
        val gridW = 16
        
        var maxScore = -1f
        var bestIdx = 0
        
        for (i in scores.indices) {
            if (scores[i] > maxScore) {
                maxScore = scores[i]
                bestIdx = i
            }
        }

        if (maxScore < 0.1f) return // Low confidence

        val row = bestIdx / gridW
        val col = bestIdx % gridW
        
        // Size Map is [1, 2, 16, 16]
        val wNorm = sizes[0 * 256 + bestIdx]
        val hNorm = sizes[1 * 256 + bestIdx]
        
        // Offset Map is [1, 2, 16, 16]
        val dx = offsets[0 * 256 + bestIdx]
        val dy = offsets[1 * 256 + bestIdx]
        
        // Center relative to search patch (0-1)
        val cxNorm = (col + 0.5f + dx) / gridW
        val cyNorm = (row + 0.5f + dy) / gridH
        
        // Map back to image coordinates
        val cx = region.left + cxNorm * region.width()
        val cy = region.top + cyNorm * region.height()
        val w = wNorm * region.width()
        val h = hNorm * region.height()
        
        lastBbox = RectF(cx - w/2, cy - h/2, cx + w/2, cy + h/2)
    }

    private fun cropSquare(src: Bitmap, roi: RectF, targetSize: Int, paddingFactor: Float): Bitmap {
        val side = max(roi.width(), roi.height()) * paddingFactor
        val cx = roi.centerX()
        val cy = roi.centerY()
        
        val cropRect = RectF(cx - side/2, cy - side/2, cx + side/2, cy + side/2)
        
        val matrix = Matrix()
        matrix.postTranslate(-cropRect.left, -cropRect.top)
        matrix.postScale(targetSize / side, targetSize / side)
        
        val dst = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(dst)
        canvas.drawColor(Color.GRAY) // Padding color
        canvas.drawBitmap(src, matrix, null)
        return dst
    }

    private fun getSearchRegion(roi: RectF, imgW: Int, imgH: Int, factor: Float): RectF {
        val side = max(roi.width(), roi.height()) * factor
        val cx = roi.centerX()
        val cy = roi.centerY()
        return RectF(cx - side/2, cy - side/2, cx + side/2, cy + side/2)
    }
    
    fun reset() {
        templateTensor = null
        lastBbox = null
        searchRegion = null
    }
}
