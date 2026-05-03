package com.example.modelengine

/**
 * Global configuration for the video processing pipeline.
 */
object TrackingConfig {
    /**
     * Number of frames to skip during video processing.
     * 0 = Process every frame
     * 1 = Process every 2nd frame
     * 2 = Process every 3rd frame (Default for performance)
     */
    var skipFrames: Int = 5

    // AbaViTrack (patch16_256) dimensions
    // Note: The model expects 256 tokens for search (16x16 grid), so search size must be 256.
    const val AB_TEMPLATE_SIZE = 128
    const val AB_SEARCH_SIZE   = 256
    
    // YOLOv8 dimensions
    const val YOLO_INPUT_SIZE = 640
}
