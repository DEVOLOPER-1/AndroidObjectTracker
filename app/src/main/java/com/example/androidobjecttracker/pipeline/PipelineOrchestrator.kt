package com.example.androidobjecttracker.pipeline

import android.content.Context
import android.graphics.*
import com.example.androidobjecttracker.utils.AppLog
import kotlinx.coroutines.runBlocking

class PipelineOrchestrator(context: Context) {
    private val detector = YoloDetector(context)
    
    init {
        AppLog.i("Initializing PipelineOrchestrator with YoloDetector...")
        // Using runBlocking here because init is not suspend, 
        // but in a real app this should be handled better.
        // For the sake of matching the "single-file pipeline" logic 
        // we'll ensure it's loaded.
        runBlocking {
            detector.load()
        }
        AppLog.i("PipelineOrchestrator initialized.")
    }

    fun processFrame(bitmap: Bitmap, params: PipelineParams, timestampUs: Long): Bitmap {
        // YoloDetector handles tracking, fall detection, and annotation in one go
        return detector.processFrame(bitmap, timestampUs / 1000)
    }

    fun generateSummaryFrame(width: Int, height: Int, timestampUs: Long): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(Color.BLACK)
        
        val elapsedSeconds = timestampUs / 1000000f
        val (fallenCount, pathLength) = detector.getSummaryInfo()
        
        val paint = Paint().apply {
            color = Color.WHITE
            textSize = 64f
            typeface = Typeface.DEFAULT_BOLD
            isAntiAlias = true
            textAlign = Paint.Align.CENTER
        }
        
        val lines = listOf(
            "Run Complete!",
            "Total Time: ${String.format(java.util.Locale.US, "%.1f", elapsedSeconds)}s",
            "Pins Knocked Down: $fallenCount",
            "Car Path Length: $pathLength points"
        )
        
        var y = height / 3f
        for (line in lines) {
            canvas.drawText(line, width / 2f, y, paint)
            y += 80f
        }
        
        return bitmap
    }

    fun reset() {
        detector.reset()
    }
}
