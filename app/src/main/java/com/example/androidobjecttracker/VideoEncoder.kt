package com.example.androidobjecttracker

import android.content.ContentValues
import android.content.Context
import android.graphics.*
import android.media.*
import android.net.Uri
import android.provider.MediaStore
import android.view.Surface
import com.example.modelengine.AppLog
import java.io.File

/**
 * Utility to encode annotated frames into a final MP4 video.
 * Corrects the Timeline bug by strictly using original frame timestamps.
 */
class VideoEncoder(private val context: Context) {
    private var codec: MediaCodec? = null
    private var muxer: MediaMuxer? = null
    private var inputSurface: Surface? = null
    private val bufferInfo = MediaCodec.BufferInfo()
    private var muxerStarted = false
    private var trackIndex = -1
    private var outFile: File? = null
    private val ptsQueue = java.util.ArrayDeque<Long>()

    /**
     * Initializes the encoder. framerate is a baseline, but actual speed is controlled by PTS.
     */
    fun start(width: Int, height: Int, fps: Int = 30) {
        val fileName = "Annotated_${System.currentTimeMillis()}.mp4"
        outFile = File(context.cacheDir, fileName)
        
        val mime = MediaFormat.MIMETYPE_VIDEO_AVC
        val format = MediaFormat.createVideoFormat(mime, width, height)
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
        format.setInteger(MediaFormat.KEY_BIT_RATE, 12_000_000) // 12Mbps for high quality
        format.setInteger(MediaFormat.KEY_FRAME_RATE, fps)
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)

        codec = MediaCodec.createEncoderByType(mime).apply {
            configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            inputSurface = createInputSurface()
            start()
        }

        muxer = MediaMuxer(outFile!!.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        muxerStarted = false
        AppLog.i("VideoEncoder: Setup complete for $width x $height @ $fps FPS")
    }

    /**
     * Accepts an annotated frame and the exact PTS from the source video.
     */
    fun addFrame(bitmap: Bitmap, ptsUs: Long) {
        val surface = inputSurface ?: return
        
        // 1. Queue the PTS to handle encoder delay
        ptsQueue.add(ptsUs)

        // 2. Submit annotated Bitmap to the codec surface
        val canvas = surface.lockCanvas(null)
        val src = Rect(0, 0, bitmap.width, bitmap.height)
        val dst = Rect(0, 0, canvas.width, canvas.height)
        canvas.drawBitmap(bitmap, src, dst, null)
        surface.unlockCanvasAndPost(canvas)
        
        // 3. Drain encoder
        drainEncoder(false)
    }

    fun finish(): File? {
        AppLog.i("VideoEncoder: Finalizing encoding process")
        codec?.signalEndOfInputStream()
        drainEncoder(true)
        
        codec?.stop()
        codec?.release()
        codec = null
        
        if (muxerStarted) muxer?.stop()
        muxer?.release()
        muxer = null
        
        inputSurface = null
        return outFile
    }

    private fun drainEncoder(endOfStream: Boolean) {
        val codec = this.codec ?: return
        while (true) {
            val status = codec.dequeueOutputBuffer(bufferInfo, 10000)
            if (status == MediaCodec.INFO_TRY_AGAIN_LATER) {
                if (!endOfStream) break
            } else if (status == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                trackIndex = muxer!!.addTrack(codec.outputFormat)
                muxer!!.start()
                muxerStarted = true
            } else if (status >= 0) {
                val encodedData = codec.getOutputBuffer(status) ?: break
                if (bufferInfo.size != 0 && muxerStarted && trackIndex >= 0) {
                    encodedData.position(bufferInfo.offset)
                    encodedData.limit(bufferInfo.offset + bufferInfo.size)
                    
                    // CRITICAL FIX: Match output buffer with its original PTS from the queue
                    // This handles encoder latency (e.g. B-frame reordering or internal buffering)
                    val pts = ptsQueue.poll()
                    if (pts != null) {
                        bufferInfo.presentationTimeUs = pts
                    }
                    
                    try {
                        muxer!!.writeSampleData(trackIndex, encodedData, bufferInfo)
                    } catch (e: Exception) {
                        AppLog.e("VideoEncoder: Failed to write sample data", e)
                    }
                }
                codec.releaseOutputBuffer(status, false)
                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
            }
        }
    }

    fun saveToGallery(file: File): Uri? {
        val values = ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME, "Annotated_Bowling.mp4")
            put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
            put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/TrackerResults")
        }
        val uri = context.contentResolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values)
        uri?.let {
            context.contentResolver.openOutputStream(it)?.use { out ->
                file.inputStream().use { it.copyTo(out) }
            }
        }
        return uri
    }
}
