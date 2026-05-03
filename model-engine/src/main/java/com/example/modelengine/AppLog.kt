package com.example.modelengine

import android.util.Log

/**
 * Centralized logging utility for the Tracker Project.
 * Use this for consistent, filterable output.
 */
object AppLog {
    private const val PREFIX = ">>> TRACKER_DEBUG >>> "
    private const val TAG = "TrackerProject"
    
    var isEnabled = true

    fun d(message: String) {
        if (isEnabled) {
            val fullMsg = "$PREFIX$message"
            Log.d(TAG, fullMsg)
            println(fullMsg) // Also print to stdout for easy reading in some IDE views
        }
    }

    fun i(message: String) {
        if (isEnabled) {
            val fullMsg = "$PREFIX[INFO] $message"
            Log.i(TAG, fullMsg)
            println(fullMsg)
        }
    }

    fun e(message: String, throwable: Throwable? = null) {
        val fullMsg = "$PREFIX[ERROR] $message"
        Log.e(TAG, fullMsg, throwable)
        System.err.println(fullMsg)
        throwable?.printStackTrace()
    }
    
    fun metric(name: String, value: Any) {
        d("METRIC | $name: $value")
    }
}
