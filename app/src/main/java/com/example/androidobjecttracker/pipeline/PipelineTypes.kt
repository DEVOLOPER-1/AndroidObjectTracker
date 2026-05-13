package com.example.androidobjecttracker.pipeline

data class PipelineParams(
    val processEvery: Int = 1,
    val trailLength: Int = 200,
    val trailMinAlpha: Float = 0.25f,
    val preferredClass: String = "car",
    val nmsScoreThreshold: Float = 0.40f,
    val nmsIouThreshold: Float = 0.45f
)
