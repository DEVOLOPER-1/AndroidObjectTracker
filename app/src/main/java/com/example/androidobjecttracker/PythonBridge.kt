package com.example.androidobjecttracker

import android.graphics.Bitmap
import com.chaquo.python.Python
import java.io.ByteArrayOutputStream

object PythonBridge {
    private val python by lazy { Python.getInstance() }
    private val trackerModule by lazy { python.getModule("tracker_logic") }
    private val trackerInstance by lazy { 
        trackerModule.callAttr("PythonTracker", "yolo26n.onnx", "AbaViTrack.onnx")
    }

    fun processFrame(bitmap: Bitmap, isInit: Boolean, initialBbox: FloatArray? = null): Map<String, Any>? {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, stream)
        val imageBytes = stream.toByteArray()

        val result = trackerModule.callAttr("process_frame", trackerInstance, imageBytes, isInit, initialBbox)
        return result?.asMap()?.mapKeys { it.key.toString() }?.mapValues { it.value } as? Map<String, Any>
    }
}
