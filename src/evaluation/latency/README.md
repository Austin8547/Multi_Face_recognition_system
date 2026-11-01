# ⚡ Multi-Face Recognition Latency Evaluation & Benchmarking

This directory contains the latency evaluation suite and comparative benchmark runners for the **AutoAttend** multi-face recognition pipeline. It benchmarks our custom optimized pipeline (**YOLOv8-Face + ArcFace ONNX**) against official **InsightFace** model packs (**Buffalo_S**, **Buffalo_M**, and **Buffalo_L**) under high-density real-world multi-face video streams.

---

## 📑 Table of Contents

1. [Overview & Objectives](#1-overview--objectives)
2. [Hardware & Test Environment](#2-hardware--test-environment)
3. [Thermal Artifact Mitigation & Benchmarking Isolation](#3-thermal-artifact-mitigation--benchmarking-isolation)
4. [Evaluation Methodology & Protocol](#4-evaluation-methodology--protocol)
5. [Benchmark Results](#5-benchmark-results)
7. [Directory Structure & Module Reference](#7-directory-structure--module-reference)
8. [Reproduction Guide](#8-reproduction-guide)

---

## 1. Overview & Objectives

This directory contains benchmarking scripts used to evaluate the computational performance of different real time multi face detection and recognition configurations in AutoAttend.

The objective is to compare model latency under a consistent video evaluation setup while reducing measurement variations caused by GPU thermal conditions.

- **High Concurrency & Crowd Density**: Processing frames containing 15–20 faces simultaneously without dropping video frames or stalling UI loops.
- **Latency Consistency**: Sustaining stable sub-100 ms inference latencies per processed frame to operate at real-time speeds (≥20–24 FPS with temporal frame skipping).

This benchmarking suite quantitatively measures the end-to-end latency and per-stage execution times of:

1. **Our Custom Pipeline**: Lightweight [YOLOv8_face](file:///home/austin/Multi_Face_recognition_system/src/utility/yolo_face.py#L8) detection + affine face alignment ([norm_crop](file:///home/austin/Multi_Face_recognition_system/src/utility/align.py#L27)) + batched [ArcFaceONNX](file:///home/austin/Multi_Face_recognition_system/src/utility/arcface.py#L7) embedding extraction + cosine similarity search against enrolled student matrices.
2. **InsightFace Buffalo_S**: Standard lightweight mobile model bundle (SCRFD detection + MobileFaceNet recognition).
3. **InsightFace Buffalo_M**: Balanced intermediate model bundle.
4. **InsightFace Buffalo_L**: High-accuracy large model bundle (ResNet-50 / ResNet-100 ArcFace backbone).

---

## 2. Hardware & Test Environment

All benchmarks were conducted under strict hardware and dataset controls:

| Parameter | Specification | Notes |
| :--- | :--- | :--- |
| **GPU / System Card** | **NVIDIA GeForce RTX 3050 (6 GB VRAM)** | Dedicated Laptop GPU |
| **Inference Engine** | **ONNX Runtime (with `CUDAExecutionProvider`)** | CUDA 12.x / cuDNN acceleration |
| **Input Video** | `vidssave.com action word for nursery class 720P.mp4` | Realistic nursery classroom crowd scene |
| **Video Resolution** | **720p HD (1280×720)** | Often formatted as 720p 30 FPS (`720x1240` / `1280x720`) |
| **Video Frame Rate** | **30.00 FPS** | Native stream frame rate |
| **Face Density** | **Mean: ~16.5 faces/frame (Max: 18 faces/frame)** | Dense multi-face stress test |
| **Frame Skip Interval** | `FRAME_SKIP = 2` | Runs inference every 2nd frame; cached drawing on skipped frames |
| **Warm-Up Frames** | `WARMUP_INFERENCE_FRAMES = 20` | Discarded prior to timing collection |
| **Measured Frames** | `MEASURED_INFERENCE_FRAMES = 500` | 1,040 total video frames parsed (520 inference passes) |

---

## 3. Thermal Artifact Mitigation & Benchmarking Isolation

### Why Every Model Is Benchmarked in a Separate File / Process

A critical engineering decision in this evaluation framework is separating each benchmark into its own standalone script:

- [yolo_arcface_eval.py](file:///home/austin/Multi_Face_recognition_system/src/evaluation/latency/yolo_arcface_eval.py)
- [insightface_s_eval.py](file:///home/austin/Multi_Face_recognition_system/src/evaluation/latency/insightface_s_eval.py)
- [insightface_m_eval.py](file:///home/austin/Multi_Face_recognition_system/src/evaluation/latency/insightface_m_eval.py)
- [insightface_l_eval.py](file:///home/austin/Multi_Face_recognition_system/src/evaluation/latency/insightface_l_eval.py)

### 🌡️ The Problem: Thermal Artifacts & GPU Clock Throttling

Modern GPUs—especially mobile GPUs like the **NVIDIA RTX 3050 6GB** in laptops—operate under strict Thermal Design Power (TGP) and dynamic boost clock algorithms (e.g., NVIDIA GPU Boost). Under continuous multi-face deep learning inference:

1. **Temperature Accumulation**: Sustained inference over 1,000+ frames drives GPU core and junction temperatures to high operating levels.
2. **Dynamic Thermal Throttling**: When GPU temperatures reach thermal thresholds, the hardware automatically scales down core boost clocks and memory frequencies to avoid overheating.
3. **Contamination of Subsequent Runs**: If all models are evaluated sequentially in one script or back-to-back without cooldown:
   - The first model runs on a **cold GPU** at maximum boost clock.
   - Subsequent models run on a **heat-saturated GPU** under aggressive thermal throttling (e.g., throttled to 1100–1300 MHz), artificially inflating their measured latencies by 15%–30%.
4. **VRAM Fragmentation & Context Leaks**: Sequential runs in a single Python runtime suffer from cumulative CUDA memory fragmentation, lingering ONNX Runtime session buffers, and garbage collection artifacts.

### ❄️ The Solution: Isolated Benchmark Execution with Cooldown Windows

By isolating each benchmark into an independent executable:

- Each model runs in a **fresh OS process** with clean CUDA contexts and zero residual memory allocations.
- A **cooling window** (several minutes) is enforced between runs, allowing the RTX 3050 GPU die and heatsink to return to ambient idle temperatures (~40°C–45°C).
- Every model receives the **same thermal headroom and sustained boost clock speeds**, ensuring scientifically valid, reproducible, and unbiased latency comparisons.

> *(This is reflected in the timestamped benchmark artifacts in `benchmark_results/`: runs were performed with distinct cooling intervals: `00:07:32` → `00:11:53` → `00:14:50` → `00:18:32`).*

---

## 4. Evaluation Methodology & Protocol

The benchmark implements rigorous performance measurement practices:

1. **High-Precision Monotonic Clock**: All timings use `time.perf_counter()`, capturing sub-millisecond hardware intervals without system clock drift.
2. **Warm-Up Phase**:
   - The first 20 inference calls are completely excluded from the measurement dataset.
   - This eliminates one-time cold-start latency artifacts: ONNX session graph optimization, CUDA kernel compilation, GPU memory allocation, and OpenCV video decoder initialization.
3. **Statistical Distribution**:
   - Mean, Median, 95th Percentile (P95), Minimum, and Maximum are computed over 500 steady-state inference frames.
   - P95 latency is particularly critical as it reflects worst-case response times during peak face clustering.
4. **End-to-End Metrics**:
   - **Full Inference Latency**: Detection + Landmark Alignment + Embedding Extraction + Identity Matching.
   - **Inference Throughput**: Theoretical processed frames per second ($1000 / \text{mean full latency}$).
   - **Observed Loop Throughput**: Real-world video processing frame rate including video decoding, frame skipping, bounding-box rendering, and display refresh.
5. **Per-Frame Audit Logging**:
   - Every single measured frame's stage timings and face count are exported to a timestamped CSV in `benchmark_results/`.

---

## 5. Benchmark Results

Below are the empirical benchmark results recorded on our **NVIDIA RTX 3050 6GB** system using the 720p 30 FPS test video.

### Summary Comparison

| Model Pipeline | Mean Faces / Frame | Mean Full Inference (ms) | Median Full Inference (ms) | P95 Full Inference (ms) | Inference Throughput (FPS) | Observed Video Loop (FPS) | Speedup vs Buffalo_L |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom YOLOv8 + ArcFace** | **16.49** | **71.284 ms** | **68.325 ms** | **87.848 ms** | **14.03 FPS** | **24.09 FPS** | **3.57× Faster** |
| **InsightFace Buffalo_S** | 16.35 | 168.507 ms | 167.535 ms | 185.364 ms | 5.93 FPS | 11.05 FPS | 1.51× Faster |
| **InsightFace Buffalo_M** | 16.45 | 246.429 ms | 250.027 ms | 258.371 ms | 4.06 FPS | 7.74 FPS | 1.03× Faster |
| **InsightFace Buffalo_L** | 16.58 | 254.720 ms | 257.643 ms | 266.376 ms | 3.93 FPS | 7.48 FPS | Baseline (1.0×) |

---

### Per-Model Granular Breakdown

#### 1. Custom YOLOv8 + ArcFace (Our Pipeline)

- **Status**: Completed (1,040 video frames, 520 inference frames, 500 measured)
- **Face Count**: Mean **16.49**, Max **18**

| Metric | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLO detect() total** | 10.789 | 10.631 | 11.863 | 9.658 | 16.069 |
| **Face alignment (all faces)** | 3.686 | 3.645 | 4.181 | 2.838 | 9.084 |
| **ArcFace (all faces, batched)** | 56.614 | 53.743 | 73.149 | 43.440 | 86.071 |
| **Embedding matching (all faces)** | 0.193 | 0.188 | 0.230 | 0.154 | 0.344 |
| **FULL inference pipeline** | **71.284** | **68.325** | **87.848** | **57.448** | **100.452** |
| Video read (context) | 1.042 | 0.850 | 2.482 | 0.667 | 4.265 |
| Full video loop (context) | 41.514 | 63.698 | 89.905 | 4.617 | 578.357 |

- **Inference Throughput**: **14.03 processed frames/s**
- **Observed Loop Throughput**: **24.09 video frames/s** (Near real-time 24 FPS!)

---

#### 2. InsightFace Buffalo_S (`buffallo_small`)

- **Status**: Completed (1,040 video frames, 520 inference frames, 500 measured)
- **Face Count**: Mean **16.35**, Max **18**

| Metric | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **InsightFace (detect + embed)** | 168.206 | 167.242 | 184.995 | 125.793 | 271.270 |
| **Embedding matching (all faces)** | 0.300 | 0.290 | 0.364 | 0.222 | 0.738 |
| **FULL inference pipeline** | **168.507** | **167.535** | **185.364** | **126.026** | **272.014** |
| Video read (context) | 0.995 | 0.826 | 2.277 | 0.689 | 4.659 |
| Full video loop (context) | 90.506 | 134.204 | 187.808 | 4.650 | 940.992 |

- **Inference Throughput**: **5.93 processed frames/s**
- **Observed Loop Throughput**: **11.05 video frames/s**

---

#### 3. InsightFace Buffalo_M (`buffallo_medium`)

- **Status**: Completed (1,040 video frames, 520 inference frames, 500 measured)
- **Face Count**: Mean **16.45**, Max **18**

| Metric | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **InsightFace (detect + embed)** | 246.156 | 249.758 | 258.099 | 193.701 | 282.490 |
| **Embedding matching (all faces)** | 0.271 | 0.269 | 0.305 | 0.218 | 0.526 |
| **FULL inference pipeline** | **246.429** | **250.027** | **258.371** | **193.930** | **282.818** |
| Video read (context) | 1.155 | 0.968 | 2.579 | 0.758 | 3.821 |
| Full video loop (context) | 129.278 | 171.372 | 261.866 | 4.673 | 1055.203 |

- **Inference Throughput**: **4.06 processed frames/s**
- **Observed Loop Throughput**: **7.74 video frames/s**

---

#### 4. InsightFace Buffalo_L (`buffallo large`)

- **Status**: Completed (1,040 video frames, 520 inference frames, 500 measured)
- **Face Count**: Mean **16.58**, Max **18**

| Metric | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **InsightFace (detect + embed)** | 254.429 | 257.359 | 266.069 | 220.084 | 286.542 |
| **Embedding matching (all faces)** | 0.290 | 0.288 | 0.331 | 0.236 | 0.556 |
| **FULL inference pipeline** | **254.720** | **257.643** | **266.376** | **220.321** | **286.911** |
| Video read (context) | 0.993 | 0.904 | 1.518 | 0.730 | 2.973 |
| Full video loop (context) | 133.605 | 186.844 | 269.966 | 4.558 | 1094.772 |

- **Inference Throughput**: **3.93 processed frames/s**
- **Observed Loop Throughput**: **7.48 video frames/s**

---

## 7. Directory Structure & Module Reference

```text
src/evaluation/latency/
├── README.md                 # Detailed evaluation documentation (this file)
├── settings.py               # Shared benchmark harness, stats computer, & InsightFace runner
├── yolo_arcface_eval.py      # Benchmark runner for custom YOLOv8 + ArcFace ONNX pipeline
├── insightface_s_eval.py     # Benchmark runner for InsightFace Buffalo_S
├── insightface_m_eval.py     # Benchmark runner for InsightFace Buffalo_M
└── insightface_l_eval.py     # Benchmark runner for InsightFace Buffalo_L
```

---

## 8. Reproduction Guide

To reproduce these benchmark numbers accurately on your system:

### 1. Prerequisites

Ensure your virtual environment is activated and dependencies are installed:

```bash

```

Verify that ONNX Runtime detects your CUDA execution provider:

```bash

```

### 2. Running Benchmarks with Thermal Cooldown

Always execute scripts as modules from the repository root, allowing at least **3–5 minutes of GPU idle cooldown between runs** to eliminate thermal artifacts:

#### Step 1: Benchmark Custom YOLOv8 + ArcFace

```bash
python -m src.evaluation.latency.yolo_arcface_eval
```

> ⏱️ *Wait 3–5 minutes for GPU temperature to settle back to idle.*

#### Step 2: Benchmark InsightFace Buffalo_S

```bash
python -m src.evaluation.latency.insightface_s_eval
```

> ⏱️ *Wait 3–5 minutes for GPU cooldown.*

#### Step 3: Benchmark InsightFace Buffalo_M

```bash
python -m src.evaluation.latency.insightface_m_eval
```

> ⏱️ *Wait 3–5 minutes for GPU cooldown.*

#### Step 4: Benchmark InsightFace Buffalo_L

```bash
python -m src.evaluation.latency.insightface_l_eval
```

### 3. Inspecting Saved Artifacts

Each benchmark run writes a detailed CSV with per-frame latencies to the `benchmark_results/` directory:

```text
benchmark_results/
```

The CSV files contain timestamps, frame indices, face counts, and sub-millisecond stage measurements for in-depth plotting and variance analysis.
