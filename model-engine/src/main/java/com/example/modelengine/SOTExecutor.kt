package com.example.modelengine

import android.content.Context
import android.graphics.*
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.PyTorchAndroid
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.PI
import kotlin.math.sqrt

/**
 * High-precision Car Tracker using AbaViTrack (Vision Transformer SOT).
 *
 * Architecture (Bootstrap pattern):
 *  • init()   – called ONCE on Frame 0 to capture the template crop from the YOLO bbox.
 *  • update() – called on every subsequent frame (Frame 1+).
 *
 * Key math:
 *  1. Template: 128×128 square crop centered on the target, extracted at init time.
 *  2. Search:   256×256 crop centered on the last known position, extracted each frame.
 *  3. Hanning Window: 16×16 cosine penalty applied element-wise to the score map BEFORE
 *     argmax. This strongly penalises locations far from the crop centre, preventing
 *     the tracker from snapping to distractors or background clutter.
 *  4. Coordinate mapping: the model predicts a box relative to the 256×256 search crop.
 *     update() maps those relative coordinates back to the full frame's pixel space.
 */
class SOTExecutor {

    private var module: Module? = null

    // ---- AbaViTrack dimensions (patch16_224) ----
    private val templateSize = TrackingConfig.AB_TEMPLATE_SIZE
    private val searchSize   = TrackingConfig.AB_SEARCH_SIZE
    private val gridDim      = 16          // default fallback grid dimension (16x16=256)

    // ---- Tracker state ----
    private var templateTensor: Tensor? = null
    private var lastBbox: RectF?         = null

    /** Persistent list of frame-centre points. Drawn as trajectory on the output video. */
    val carPath = mutableListOf<PointF>()

    // ---- ImageNet normalisation (AbaViTrack was ViT-pretrained on ImageNet) ----
    private val normMean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val normStd  = floatArrayOf(0.229f, 0.224f, 0.225f)

    // ---- Hanning Window (BUG 1 FIX) ----
    //
    // A 1-D Hanning window of length N is:
    //   w[i] = 0.5 * (1 – cos(2π·i / (N–1)))
    //
    // The 2-D window is the outer product w_row[r] * w_col[c], giving a
    // smooth bell-shaped surface whose maximum is at the centre cell (7,7).
    // Multiplying the flat score map by this surface before argmax means the
    // tracker is biased toward finding the target near the expected position,
    // and strong-but-peripheral responses (background clutter) are suppressed.
    private val hanningWindow: FloatArray = run {
        val hann1d = FloatArray(gridDim) { i ->
            (0.5 * (1.0 - cos(2.0 * PI * i / (gridDim - 1)))).toFloat()
        }
        // Outer product: hanningWindow[r * gridDim + c] = hann1d[r] * hann1d[c]
        FloatArray(gridDim * gridDim) { idx ->
            hann1d[idx / gridDim] * hann1d[idx % gridDim]
        }
    }

    // Confidence threshold AFTER sigmoid normalisation (range 0–1).
    // 0.25 is robust across models that output raw logits OR probabilities.
    private val confidenceThreshold = 0.25f

    // -------------------------------------------------------------------------
    fun loadModel(assetName: String, context: Context) {
        try {
            val numThreads = max(1, Runtime.getRuntime().availableProcessors() - 1)
            PyTorchAndroid.setNumThreads(numThreads)
            AppLog.i("SOTExecutor: Using $numThreads CPU threads for inference")

            val path = ModelExecutor.assetFilePath(context, assetName)
            module = Module.load(path)
            AppLog.i("SOTExecutor: Model loaded — $assetName")
        } catch (e: Exception) {
            AppLog.e("SOTExecutor: Model load failed", e)
        }
    }

    // -------------------------------------------------------------------------
    /**
     * Captures the template on Frame 0.
     * Must be called before any call to update().
     *
     * @param frame       The full camera/video frame bitmap.
     * @param initialBox  The bounding box returned by YOLO for the RC car (in frame space).
     */
    fun init(frame: Bitmap, initialBox: RectF) {
        lastBbox = RectF(initialBox)
        carPath.clear()
        carPath.add(PointF(initialBox.centerX(), initialBox.centerY()))

        AppLog.i("SOTExecutor.init: Using templateSize=$templateSize, searchSize=$searchSize")
        // Using a square avoids distorting the template crop when the
        // YOLO box has a non-1:1 aspect ratio, preventing feature mismatch
        // between template and search patches.
        val side = max(initialBox.width(), initialBox.height()) * 2f
        val cx   = initialBox.centerX()
        val cy   = initialBox.centerY()
        val templateRegion = RectF(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)

        val templateBitmap = cropAndResize(frame, templateRegion, templateSize)
        templateTensor = TensorImageUtils.bitmapToFloat32Tensor(templateBitmap, normMean, normStd)
        templateBitmap.recycle()

        AppLog.i("SOTExecutor.init: bbox=$initialBox  templateRegion=$templateRegion")
    }

    // -------------------------------------------------------------------------
    /**
     * Runs one tracking step on [frame].
     * Returns the updated bounding box in full-frame pixel coordinates,
     * or the previous bbox if the model has low confidence this frame.
     *
     * @param frame  The full camera/video frame bitmap for the CURRENT frame.
     */
    fun update(frame: Bitmap): RectF? {
        val mod  = module          ?: return lastBbox   // Return last known if model not loaded
        val z    = templateTensor  ?: return null
        val prev = lastBbox        ?: return null

        // ------------------------------------------------------------------
        // 1. Build search region: a square 4× the target size, centred on
        //    the last known position.  The larger the multiplier, the more
        //    motion the tracker can handle, at the cost of more distractor risk.
        // ------------------------------------------------------------------
        val side = max(prev.width(), prev.height()) * 4f
        val searchRegion = RectF(
            prev.centerX() - side / 2,
            prev.centerY() - side / 2,
            prev.centerX() + side / 2,
            prev.centerY() + side / 2
        )

        // ------------------------------------------------------------------
        // 2. Extract and normalise the search patch.
        // ------------------------------------------------------------------
        val xBitmap = cropAndResize(frame, searchRegion, searchSize)
        val xTensor = TensorImageUtils.bitmapToFloat32Tensor(xBitmap, normMean, normStd)

        try {
            // ------------------------------------------------------------------
            // 3. Inference: AbaViTrack takes [template, search] → (score, size, offset)
            // Some versions might output only (score, bbox) where bbox is [4, 16, 16]
            // ------------------------------------------------------------------
            val outputs = mod.forward(IValue.from(z), IValue.from(xTensor))

            val outputTensors: Array<IValue> = when {
                outputs.isTuple -> outputs.toTuple()
                outputs.isList  -> outputs.toList()
                else            -> arrayOf(outputs)
            }

            if (outputTensors.isEmpty()) {
                AppLog.e("SOTExecutor.update: no output tensors returned")
                return lastBbox
            }

            // Diagnostic: Log shapes of output tensors (only on first few frames)
            if (carPath.size < 5) {
                val shapes = outputTensors.map { it.toTensor().shape().contentToString() }
                AppLog.d("SOTExecutor.update: output shapes = $shapes")
            }

            var dx = 0f
            var dy = 0f
            var wNorm = 0f
            var hNorm = 0f
            var probability = 0f
            var row = 0
            var col = 0
            var currentGridDim = 16

            // Determine model output style
            // Diagnostic check for [1, 1, 4] style (common in some exports)
            val firstShape = outputTensors[0].toTensor().shape()
            if (firstShape.contentEquals(longArrayOf(1, 1, 4)) || firstShape.contentEquals(longArrayOf(1, 4))) {
                // Style D: Direct normalized bbox [x1, y1, x2, y2] relative to search crop
                val bboxArr = outputTensors[0].toTensor().dataAsFloatArray
                val x1 = bboxArr[0]; val y1 = bboxArr[1]; val x2 = bboxArr[2]; val y2 = bboxArr[3]
                
                // Map from [0,1] search-crop space to full-frame space
                val cxNorm = (x1 + x2) / 2f
                val cyNorm = (y1 + y2) / 2f
                val wNorm  = x2 - x1
                val hNorm  = y2 - y1
                
                val cxFull = searchRegion.left + cxNorm * searchRegion.width()
                val cyFull = searchRegion.top  + cyNorm * searchRegion.height()
                val wFull  = wNorm * searchRegion.width()
                val hFull  = hNorm * searchRegion.height()
                
                lastBbox = RectF(cxFull - wFull / 2, cyFull - hFull / 2,
                                 cxFull + wFull / 2, cyFull + hFull / 2)
                carPath.add(PointF(cxFull, cyFull))
                
                // If there's a second tensor, it's usually the confidence/score map
                var prob = 0.9f 
                if (outputTensors.size > 1) {
                    val confData = outputTensors[1].toTensor().dataAsFloatArray
                    if (confData.isNotEmpty()) {
                        val maxConf = confData.maxOrNull() ?: 0f
                        prob = 1f / (1f + exp(-maxConf))
                    }
                }
                
                AppLog.d("SOTExecutor.update: tracked (Direct) @ (%.1f, %.1f) prob=%.3f".format(cxFull, cyFull, prob))
                return lastBbox
            }

            // Heatmap-based logic (Styles A, B, C)
            val scoreMap = outputTensors[0].toTensor()
            val scores   = scoreMap.dataAsFloatArray
            val scoreCount = scores.size
            currentGridDim = sqrt(scoreCount.toDouble()).toInt()
            val spatialSize = currentGridDim * currentGridDim

            // ------------------------------------------------------------------
            // 4. Apply Hanning Window (Mixed with flat window to reduce "center-stuck" bias)
            // ------------------------------------------------------------------
            val windowSize = minOf(scores.size, hanningWindow.size)
            val alpha = 0.4f // 40% influence from Hanning, 60% raw scores
            val windowedScores = FloatArray(scores.size) { i ->
                val weight = if (i < windowSize) (1f - alpha) + alpha * hanningWindow[i] else 1f
                scores[i] * weight
            }

            var bestIdx  = 0
            var maxWin   = Float.NEGATIVE_INFINITY
            for (i in windowedScores.indices) {
                if (windowedScores[i] > maxWin) {
                    maxWin  = windowedScores[i]
                    bestIdx = i
                }
            }

            val rawScore = scores[bestIdx]
            probability = 1f / (1f + exp(-rawScore))

            if (probability < confidenceThreshold) {
                AppLog.d("SOTExecutor.update: low confidence (prob=${"%.3f".format(probability)}) — holding position")
                return lastBbox
            }

            row = bestIdx / currentGridDim
            col = bestIdx % currentGridDim

            when {
                // Case A: (score, w, h, dx, dy)
                outputTensors.size >= 5 -> {
                    val wMap  = outputTensors[1].toTensor().dataAsFloatArray
                    val hMap  = outputTensors[2].toTensor().dataAsFloatArray
                    val dxMap = outputTensors[3].toTensor().dataAsFloatArray
                    val dyMap = outputTensors[4].toTensor().dataAsFloatArray
                    dx    = if (bestIdx < dxMap.size) dxMap[bestIdx] else 0f
                    dy    = if (bestIdx < dyMap.size) dyMap[bestIdx] else 0f
                    wNorm = if (bestIdx < wMap.size)  wMap[bestIdx]  else 0f
                    hNorm = if (bestIdx < hMap.size)  hMap[bestIdx]  else 0f
                }
                // Case B: (score, size, offset)
                outputTensors.size >= 3 -> {
                    val sizes   = outputTensors[1].toTensor().dataAsFloatArray
                    val offsets = outputTensors[2].toTensor().dataAsFloatArray
                    wNorm = if (bestIdx < sizes.size) sizes[bestIdx] else 0f
                    hNorm = if (spatialSize + bestIdx < sizes.size) sizes[spatialSize + bestIdx] else 0f
                    dx    = if (bestIdx < offsets.size) offsets[bestIdx] else 0f
                    dy    = if (spatialSize + bestIdx < offsets.size) offsets[spatialSize + bestIdx] else 0f
                }
                // Case C: (score, bbox)
                outputTensors.size >= 2 -> {
                    val bboxes = outputTensors[1].toTensor().dataAsFloatArray
                    dx    = if (0 * spatialSize + bestIdx < bboxes.size) bboxes[0 * spatialSize + bestIdx] else 0f
                    dy    = if (1 * spatialSize + bestIdx < bboxes.size) bboxes[1 * spatialSize + bestIdx] else 0f
                    wNorm = if (2 * spatialSize + bestIdx < bboxes.size) bboxes[2 * spatialSize + bestIdx] else 0f
                    hNorm = if (3 * spatialSize + bestIdx < bboxes.size) bboxes[3 * spatialSize + bestIdx] else 0f
                }
            }

            val cxNorm = (col + 0.5f + dx) / currentGridDim
            val cyNorm = (row + 0.5f + dy) / currentGridDim

            val cxFull = searchRegion.left + cxNorm * searchRegion.width()
            val cyFull = searchRegion.top  + cyNorm * searchRegion.height()
            val wFull  = wNorm * searchRegion.width()
            val hFull  = hNorm * searchRegion.height()

            lastBbox = RectF(cxFull - wFull / 2, cyFull - hFull / 2,
                             cxFull + wFull / 2, cyFull + hFull / 2)
            carPath.add(PointF(cxFull, cyFull))

            AppLog.d("SOTExecutor.update: tracked @ (%.1f, %.1f) prob=${"%.3f".format(probability)}".format(cxFull, cyFull))

        } catch (e: Exception) {
            AppLog.e("SOTExecutor.update: inference failed", e)
        } finally {
            xBitmap.recycle()
        }

        return lastBbox
    }

    // -------------------------------------------------------------------------
    /**
     * Crops [src] to [roi] (which may extend beyond the bitmap bounds — padded
     * with neutral grey) and rescales the result to [targetSize]×[targetSize].
     */
    private fun cropAndResize(src: Bitmap, roi: RectF, targetSize: Int): Bitmap {
        val dst    = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(dst)
        canvas.drawColor(Color.rgb(128, 128, 128)) // neutral grey for out-of-bounds padding

        val scale = targetSize.toFloat() / roi.width()
        val matrix = Matrix()
        matrix.postTranslate(-roi.left, -roi.top)
        matrix.postScale(scale, scale)

        canvas.drawBitmap(src, matrix, null)
        return dst
    }

    fun reset() {
        lastBbox       = null
        templateTensor = null
        carPath.clear()
    }
}
