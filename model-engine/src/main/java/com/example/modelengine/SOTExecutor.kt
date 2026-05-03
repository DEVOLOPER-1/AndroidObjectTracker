package com.example.modelengine

import android.content.Context
import android.graphics.*
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import kotlin.math.max

/**
 * High-precision Car Tracker using AbaViTrack.
 * Handles template/search crops and maps relative box predictions back to the full frame.
 */
class SOTExecutor {
    private var module: Module? = null
    // AbaViTrack configuration
    private val templateSize = 128
    private val searchSize = 256
    
    private var templateTensor: Tensor? = null
    private var lastBbox: RectF? = null
    val carPath = mutableListOf<PointF>()

    // AbaViTrack Normalization constants
    private val normMean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val normStd = floatArrayOf(0.229f, 0.224f, 0.225f)

    fun loadModel(assetName: String, context: Context) {
        try {
            val path = ModelExecutor.assetFilePath(context, assetName)
            module = Module.load(path)
            AppLog.i("SOT: Model loaded successfully")
        } catch (e: Exception) {
            AppLog.e("SOT: Model load failed", e)
        }
    }

    /**
     * Captures the target template on Frame 0.
     */
    fun init(frame: Bitmap, initialBox: RectF) {
        lastBbox = RectF(initialBox)
        carPath.clear()
        carPath.add(PointF(initialBox.centerX(), initialBox.centerY()))
        
        // FIX: Use a square region centered on the object for the template.
        // Stretching the object's non-square bounding box to a square tensor 
        // causes feature mismatch with the search patch.
        val side = max(initialBox.width(), initialBox.height()) * 2f
        val templateRegion = RectF(
            initialBox.centerX() - side/2,
            initialBox.centerY() - side/2,
            initialBox.centerX() + side/2,
            initialBox.centerY() + side/2
        )
        
        val templateBitmap = cropAndResize(frame, templateRegion, templateSize)
        templateTensor = TensorImageUtils.bitmapToFloat32Tensor(templateBitmap, normMean, normStd)
        templateBitmap.recycle()
        AppLog.i("SOT: Target initialized at $initialBox, Template Region: $templateRegion")
    }

    /**
     * Performs tracking on subsequent frames with coordinate mapping.
     */
    fun update(frame: Bitmap): RectF? {
        val mod = module ?: return null
        val z = templateTensor ?: return null
        val prev = lastBbox ?: return null

        // 1. Define Search Region (4x target size, centered on last known position)
        val side = max(prev.width(), prev.height()) * 4f
        val searchRegion = RectF(
            prev.centerX() - side/2,
            prev.centerY() - side/2,
            prev.centerX() + side/2,
            prev.centerY() + side/2
        )
        
        // 2. Extract and Normalize Search Patch
        val xBitmap = cropAndResize(frame, searchRegion, searchSize)
        val xTensor = TensorImageUtils.bitmapToFloat32Tensor(xBitmap, normMean, normStd)

        try {
            // 3. Inference: AbaViTrack expects [template, search]
            val outputs = mod.forward(IValue.from(z), IValue.from(xTensor))
            
            // Handle output tuple [score_map, size_map, offset_map]
            val outputTensors = when {
                outputs.isTuple -> outputs.toTuple()
                outputs.isList -> outputs.toList()
                else -> emptyArray()
            }
            
            if (outputTensors.size >= 3) {
                val scoreMap = outputTensors[0].toTensor()
                val sizeMap = outputTensors[1].toTensor()
                val offsetMap = outputTensors[2].toTensor()
                
                // FIX: Restore to 16. The model error (196 vs 256) indicates it expects 256 tokens (16x16)
                val gridDim = 16
                val scores = scoreMap.dataAsFloatArray
                var maxScore = -1000f // Handle raw logits or probabilities
                var bestIdx = 0
                for (i in scores.indices) {
                    if (scores[i] > maxScore) {
                        maxScore = scores[i]
                        bestIdx = i
                    }
                }

                if (maxScore > 0.1f || (maxScore > -5.0f && maxScore < 0.0f)) { // Flexible threshold
                    val row = bestIdx / gridDim
                    val col = bestIdx % gridDim
                    
                    val sizes = sizeMap.dataAsFloatArray
                    val offsets = offsetMap.dataAsFloatArray
                    
                    // Coordinates relative to the searchSize patch
                    val dx = offsets[bestIdx] 
                    val dy = offsets[gridDim * gridDim + bestIdx]
                    val wNorm = sizes[bestIdx]
                    val hNorm = sizes[gridDim * gridDim + bestIdx]
                    
                    val cxNorm = (col + 0.5f + dx) / gridDim
                    val cyNorm = (row + 0.5f + dy) / gridDim
                    
                    // CRITICAL: Map relative crop coordinates back to Full Frame
                    val cxFull = searchRegion.left + cxNorm * searchRegion.width()
                    val cyFull = searchRegion.top + cyNorm * searchRegion.height()
                    val wFull = wNorm * searchRegion.width()
                    val hFull = hNorm * searchRegion.height()
                    
                    lastBbox = RectF(cxFull - wFull/2, cyFull - hFull/2, cxFull + wFull/2, cyFull + hFull/2)
                    carPath.add(PointF(cxFull, cyFull))
                    AppLog.d("SOT: Tracked at ($cxFull, $cyFull) score=$maxScore")
                }
            }
        } catch (e: Exception) {
            AppLog.e("SOT: Tracking update failed", e)
        } finally {
            xBitmap.recycle()
        }

        return lastBbox
    }

    private fun cropAndResize(src: Bitmap, roi: RectF, targetSize: Int): Bitmap {
        val dst = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(dst)
        canvas.drawColor(Color.GRAY) // Gray padding for out-of-bounds
        
        val matrix = Matrix()
        val scale = targetSize.toFloat() / roi.width()
        matrix.postTranslate(-roi.left, -roi.top)
        matrix.postScale(scale, scale)
        
        canvas.drawBitmap(src, matrix, null)
        return dst
    }

    fun reset() {
        lastBbox = null
        templateTensor = null
        carPath.clear()
    }
}
