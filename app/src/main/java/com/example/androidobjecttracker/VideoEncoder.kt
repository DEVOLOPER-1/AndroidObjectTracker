package com.example.androidobjecttracker

import android.content.ContentValues
import android.content.Context
import android.graphics.*
import android.media.*
import android.net.Uri
import android.provider.MediaStore
import android.view.Surface
import com.example.modelengine.AppLog
import com.example.modelengine.StaticTracker
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Utility to encode annotated frames into a final MP4 video incrementally.
 * Fixes the presentationTimeUs bug to ensure video duration matches source.
 */
class VideoEncoder(private val context: Context) {
    private var codec: MediaCodec? = null
    private var muxer: MediaMuxer? = null
    private var inputSurface: Surface? = null
    private val bufferInfo = MediaCodec.BufferInfo()
    private var muxerStarted = false
    private var trackIndex = -1
    private var outFile: File? = null
    
    private var frameIntervalUs: Long = 0
    private var currentPtsUs: Long = 0

    private val paint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }
    
    private val linePaint = Paint().apply {
        color = Color.YELLOW
        strokeWidth = 4f
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 45f
        typeface = Typeface.DEFAULT_BOLD
        setShadowLayer(5f, 0f, 0f, Color.BLACK)
    }

    fun start(width: Int, height: Int, fps: Int, undersamplingFactor: Int = 1) {
        val fileName = "Annotated_Result_${System.currentTimeMillis()}.mp4"
        outFile = File(context.cacheDir, fileName)
        
        frameIntervalUs = (1_000_000L / fps) * undersamplingFactor
        currentPtsUs = 0

        val mime = MediaFormat.MIMETYPE_VIDEO_AVC
        val format = MediaFormat.createVideoFormat(mime, width, height)
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
        format.setInteger(MediaFormat.KEY_BIT_RATE, 8_000_000)
        format.setInteger(MediaFormat.KEY_FRAME_RATE, fps)
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)

        codec = MediaCodec.createEncoderByType(mime).apply {
            configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            inputSurface = createInputSurface()
            start()
        }

        muxer = MediaMuxer(outFile!!.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        muxerStarted = false
        trackIndex = -1
        AppLog.i("Encoder started: $width x $height @ $fps fps (Interval: $frameIntervalUs us)")
    }

    fun addFrame(bitmap: Bitmap, pins: List<StaticTracker.PinState>, carPath: List<PointF>) {
        val codec = this.codec ?: return
        val surface = this.inputSurface ?: return

        // 1. Create a mutable copy if necessary and draw annotations
        val annotatedBitmap = if (bitmap.isMutable) bitmap else bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(annotatedBitmap)
        
        // Draw Car Trace (Persistent line)
        if (carPath.size > 1) {
            val path = Path()
            path.moveTo(carPath[0].x, carPath[0].y)
            for (i in 1 until carPath.size) {
                path.lineTo(carPath[i].x, carPath[i].y)
            }
            canvas.drawPath(path, linePaint)
            
            // Draw Car Label at current position
            val last = carPath.last()
            canvas.drawText("RC CAR", last.x, last.y - 20f, textPaint)
        }

        // Draw Pins
        pins.forEach { pin ->
            if (pin.isFallen) {
                textPaint.color = Color.RED
                canvas.drawText("#${pin.fallOrder}", pin.currentCentroid.x, pin.currentCentroid.y, textPaint)
            } else {
                paint.color = Color.GREEN
                canvas.drawCircle(pin.currentCentroid.x, pin.currentCentroid.y, 30f, paint)
                textPaint.color = Color.WHITE
                canvas.drawText("PIN", pin.currentCentroid.x - 20f, pin.currentCentroid.y - 40f, textPaint)
            }
        }

        // 2. Feed annotated Bitmap to Surface with Correct Timestamp
        val displayCanvas = surface.lockCanvas(null)
        val src = Rect(0, 0, annotatedBitmap.width, annotatedBitmap.height)
        val dst = Rect(0, 0, displayCanvas.width, displayCanvas.height)
        displayCanvas.drawBitmap(annotatedBitmap, src, dst, null)
        surface.unlockCanvasAndPost(displayCanvas)

        // 3. Cleanup and Drain
        if (annotatedBitmap != bitmap) annotatedBitmap.recycle()
        drainEncoder(false)
        currentPtsUs += frameIntervalUs
    }

    fun finish(): File? {
        AppLog.i("Finishing video encoding...")
        codec?.signalEndOfInputStream()
        drainEncoder(true)
        
        codec?.stop()
        codec?.release()
        codec = null
        
        if (muxerStarted) {
            muxer?.stop()
        }
        muxer?.release()
        muxer = null
        
        inputSurface = null
        return outFile
    }

    private fun drainEncoder(endOfStream: Boolean) {
        val codec = this.codec ?: return
        val muxer = this.muxer ?: return

        while (true) {
            val encoderStatus = codec.dequeueOutputBuffer(bufferInfo, 10000)
            if (encoderStatus == MediaCodec.INFO_TRY_AGAIN_LATER) {
                if (!endOfStream) break
            } else if (encoderStatus == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                if (muxerStarted) throw RuntimeException("Format changed twice")
                trackIndex = muxer.addTrack(codec.outputFormat)
                muxer.start()
                muxerStarted = true
            } else if (encoderStatus >= 0) {
                val encodedData = codec.getOutputBuffer(encoderStatus) ?: throw RuntimeException("Buffer was null")
                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) {
                    bufferInfo.size = 0
                }
                if (bufferInfo.size != 0) {
                    if (!muxerStarted) throw RuntimeException("Muxer not started")
                    encodedData.position(bufferInfo.offset)
                    encodedData.limit(bufferInfo.offset + bufferInfo.size)
                    
                    // Inject fixed timestamp to maintain correct video duration
                    bufferInfo.presentationTimeUs = currentPtsUs

                    muxer.writeSampleData(trackIndex, encodedData, bufferInfo)
                }
                codec.releaseOutputBuffer(encoderStatus, false)
                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
            }
        }
    }

    fun saveToGallery(file: File): Uri? {
        val values = ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME, "Annotated_${file.name}")
            put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
            put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/TrackerResults")
        }
        val uri = context.contentResolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values)
        uri?.let {
            context.contentResolver.openOutputStream(it)?.use { output ->
                file.inputStream().use { input ->
                    input.copyTo(output)
                }
            }
        }
        AppLog.i("Video saved to gallery: $uri")
        return uri
    }
}
