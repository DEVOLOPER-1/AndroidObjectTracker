package com.example.androidobjecttracker

import android.content.ContentValues
import android.content.Context
import android.graphics.*
import android.media.*
import android.net.Uri
import android.provider.MediaStore
import android.view.Surface
import com.example.modelengine.AppLog
import com.example.modelengine.SortTracker
import java.io.File

/**
 * Utility to encode annotated frames into a final MP4 video incrementally.
 * Fixes the presentationTimeUs bug by enforcing strict monotonic PTS.
 */
class VideoEncoder(private val context: Context) {
    private var codec: MediaCodec? = null
    private var muxer: MediaMuxer? = null
    private var inputSurface: Surface? = null
    private val bufferInfo = MediaCodec.BufferInfo()
    private var muxerStarted = false
    private var trackIndex = -1
    private var outFile: File? = null
    
    private var frameCount = 0L
    private val frameIntervalUs = 1_000_000L / 15L  // 15 FPS to match step=2

    private var drawBuffer: Bitmap? = null
    private val pinPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }
    
    private val carPathPaint = Paint().apply {
        color = Color.CYAN // Persistent colored line for trajectory
        strokeWidth = 5f
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f
        typeface = Typeface.DEFAULT_BOLD
        setShadowLayer(5f, 0f, 0f, Color.BLACK)
    }

    fun start(width: Int, height: Int) {
        val fileName = "Processed_Result_${System.currentTimeMillis()}.mp4"
        outFile = File(context.cacheDir, fileName)
        frameCount = 0

        drawBuffer?.recycle()
        drawBuffer = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        val mime = MediaFormat.MIMETYPE_VIDEO_AVC
        val format = MediaFormat.createVideoFormat(mime, width, height)
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
        format.setInteger(MediaFormat.KEY_BIT_RATE, 6_000_000)   // 6 Mbps is plenty for 720p/15fps
        format.setInteger(MediaFormat.KEY_FRAME_RATE, 15)         // matches step=2 on 30fps source
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 2)

        codec = MediaCodec.createEncoderByType(mime).apply {
            configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            inputSurface = createInputSurface()
            start()
        }

        muxer = MediaMuxer(outFile!!.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        muxerStarted = false
        trackIndex = -1
        AppLog.i("VideoEncoder started: ${width}x${height} @ 15 FPS, 6 Mbps")
    }

    /**
     * Bakes annotations into the bitmap and sends it to the encoder.
     */
    fun encodeFrame(bitmap: Bitmap, tracks: List<SortTracker.TrackedObject>) {
        val surface = this.inputSurface ?: return
        val buffer  = this.drawBuffer  ?: return

        // Copy source frame into reusable buffer — no allocation
        val copyCanvas = Canvas(buffer)
        copyCanvas.drawBitmap(bitmap, 0f, 0f, null)

        // Draw annotations on top of the buffer
        tracks.forEach { track ->
            when (track.classIndex) {
                3 -> {
                    if (track.trajectory.size > 1) {
                        val path = Path()
                        path.moveTo(track.trajectory[0].x, track.trajectory[0].y)
                        for (i in 1 until track.trajectory.size) {
                            path.lineTo(track.trajectory[i].x, track.trajectory[i].y)
                        }
                        copyCanvas.drawPath(path, carPathPaint)
                    }
                    copyCanvas.drawText(
                        "CAR ID:${track.id}",
                        track.lastCx, track.lastCy - 20f, textPaint
                    )
                }
                1 -> {
                    if (track.state == SortTracker.State.FALLEN) {
                        pinPaint.color = Color.RED
                        copyCanvas.drawRect(track.bbox, pinPaint)
                        track.fallOrder?.let { order ->
                            copyCanvas.drawText(
                                "#$order",
                                track.bbox.centerX(), track.bbox.centerY(), textPaint
                            )
                        }
                    } else {
                        pinPaint.color = Color.GREEN
                        copyCanvas.drawRect(track.bbox, pinPaint)
                        copyCanvas.drawText(
                            "PIN", track.bbox.left, track.bbox.top - 10f, textPaint
                        )
                    }
                }
            }
        }

        // Send to encoder surface
        val surfaceCanvas = try {
            surface.lockCanvas(null)
        } catch (e: Exception) {
            AppLog.e("VideoEncoder: lockCanvas failed, skipping frame $frameCount", e)
            return
        }
        try {
            surfaceCanvas.drawBitmap(buffer, 0f, 0f, null)
        } finally {
            surface.unlockCanvasAndPost(surfaceCanvas)
        }

        drainEncoder(false)
        frameCount++
    }


    fun finish(): File? {
        AppLog.i("Finishing video encoding. Total frames: $frameCount")
        codec?.signalEndOfInputStream()
        drainEncoder(true)
        codec?.stop()
        codec?.release()
        codec = null
        if (muxerStarted) muxer?.stop()
        muxer?.release()
        muxer = null
        inputSurface = null
        // Release draw buffer
        drawBuffer?.recycle()
        drawBuffer = null
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
                val encodedData = codec.getOutputBuffer(encoderStatus) ?: throw RuntimeException("Buffer null")
                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) {
                    bufferInfo.size = 0
                }
                if (bufferInfo.size != 0) {
                    if (!muxerStarted) throw RuntimeException("Muxer not started")
                    encodedData.position(bufferInfo.offset)
                    encodedData.limit(bufferInfo.offset + bufferInfo.size)
                    
                    // Force strict monotonic PTS
                    bufferInfo.presentationTimeUs = frameCount * frameIntervalUs

                    muxer.writeSampleData(trackIndex, encodedData, bufferInfo)
                }
                codec.releaseOutputBuffer(encoderStatus, false)
                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
            }
        }
    }

    fun saveToGallery(file: File): Uri? {
        val values = ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME, file.name)
            put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
            put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/ObjectTracker")
        }
        val uri = context.contentResolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values)
        uri?.let {
            context.contentResolver.openOutputStream(it)?.use { output ->
                file.inputStream().use { input ->
                    input.copyTo(output)
                }
            }
        }
        AppLog.i("Video saved: $uri")
        return uri
    }
}
