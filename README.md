# AndroidObjectTracker

**AndroidObjectTracker** is a high-performance, custom-engineered computer vision application for Android. It implements a sophisticated pipeline for real-time bowling pin detection, tracking, and causal event analysis.

Unlike standard implementations that rely on simple model-based classification to determine object states, this project utilizes a **classical CV engineering approach** to solve complex spatial and temporal problems.

---

## 🚀 Key Differentiators

### 1. Classical Engineering vs. Black-Box Classification
Most contemporary projects use deep learning models to provide binary answers (e.g., "pin up" vs. "pin down"). Our system prioritizes **CV engineering logic**:
*   Analyzes spatial orientation and physical movement.
*   Uses a classical logic pipeline to interpret the scene physics.
*   Provides deeper insights into *how* events occur rather than just detecting their final state.

### 2. Causal Tracking (Event-Based Logic)
Our pipeline distinguishes between a pin falling naturally and a pin being knocked down by a specific actor (the "car"):
*   **Temporal Tracking:** Tracks pins across frames to maintain identity.
*   **Spatial Interaction:** Calculates the intersection of the car's trajectory and pin locations.
*   **Causal Counting:** Only pins knocked down by the car's path are counted toward the score, ignoring external or unrelated falls.

### 3. Sophisticated Problem Solving
We address common CV challenges—occlusion, varying lighting, and perspective shifts—through rigorous engineering:
*   Refined **Non-Maximum Suppression (NMS)** thresholds.
*   Advanced **IOU (Intersection over Union)** analysis for object matching.
*   **Trail Tracking** logic to map trajectories over time.

---

## 🛠 Technical Architecture

### Core Pipeline
The application runs a multi-stage processing pipeline:
1.  **Frame Acquisition:** Captured via **CameraX** for high-efficiency video streaming.
2.  **Object Detection:** Uses a YOLO-based architecture running via **ONNX Runtime**.
3.  **Preprocessing:** Normalizes image data to Float32 for maximum numerical precision.
4.  **Tracking Engine:** A custom tracking algorithm that maintains state (id, centroid, bbox, aspect ratio) for every detected pin.
5.  **Heuristic Analysis:** Monitors the aspect ratio and movement of pins to detect "falls" with a confirmation buffer to eliminate noise.

### Performance & Hardware Optimization
*   **NNAPI Acceleration:** Utilizes the Android Neural Networks API to leverage on-device hardware (GPU/NPU).
*   **ONNX Graph Optimizations:** Enables level-3 runtime optimizations for maximum throughput.
*   **Precision-First Approach:** Avoids lossy static INT8 quantization in favor of runtime FP16/FP32 processing, ensuring the spatial analysis remains accurate.

---

## 📦 Tech Stack
*   **Language:** Kotlin
*   **UI Framework:** Jetpack Compose
*   **Inference Engine:** ONNX Runtime (Mobile)
*   **Camera API:** CameraX
*   **Hardware Acceleration:** Android NNAPI

---

## 🔗 Repository
[https://github.com/DEVOLOPER-1/AndroidObjectTracker.git](https://github.com/DEVOLOPER-1/AndroidObjectTracker.git)

---

## 📥 Download APK
You can download the latest pre-built APK directly from the repository:
**[Download PinTrackerPro.apk](./apk_release/PinTrackerPro.apk)**

---

## 👥 Project Team
**AndroidObjectTracker**
Created for the Course: *DSAI 352 - Spring 2026*
