<div align="center">

# Multi-Face Recognition System (AutoAttend)

**Multi-Face Recognition System** engineered for dense crowd environments and powered by an optimized **YOLOv8-Face + ArcFace** pipeline.

</div>

This repository powers **AutoAttend**, a real-time smart attendance application that detects, aligns, identifies, and logs student attendance across scheduled timetable windows directly into a PostgreSQL database automatically

---

## 📑 Table of Contents

- [Overview](#-overview)
- [The AutoAttend Application](#-the-autoattend-application)
- [Core Architecture & Engineering Innovations](#-core-architecture)
- [Latency Benchmarks & Performance](#-latency-benchmarks--performance)
- [Repository Structure](#-repository-structure)
- [Prerequisites & Installation](#-prerequisites--installation)
- [Database Schema & Setup](#-database-schema--setup)
- [Workflow & Step-by-Step Usage](#-workflow--step-by-step-usage)
  - [1. Student Enrollment (Data Acquisition)](#1-student-enrollment-data-acquisition)
  - [2. Embedding Generation](#2-embedding-generation)
  - [3. Standalone Verification & Testing](#3-standalone-verification--testing)
  - [4. Running Real-Time Attendance](#4-running-real-time-attendance)
  - [5. Latency & Comparative Benchmarking](#5-latency--comparative-benchmarking)
- [Configuration Reference](#-configuration-reference)

---

## 🌟 Overview

Conventional facial recognition pipelines experience severe frame dropping and latency degradation when deployed in crowded scenes containing 15–20 individuals simultaneously. Standard face detectors and iterative single-face feature extractors quickly saturate system resources, causing inference delays well beyond acceptable interactive thresholds.

This **Multi-Face Recognition System** overcomes these bottlenecks through:

1. **Candidate-Only DFL Decoded YOLOv8-Face**: Pruning over 99% of grid anchors in logit space before decoding bounding boxes and facial landmarks.
2. **Batched ArcFace ONNX Extraction**: Stacking all detected and affine-aligned faces in a video frame into a single 4D tensor for one parallel GPU forward pass.
3. **Optimized Cosine Similarity Search**: Vectorized matrix multiplication against pre-computed identity feature maps.
4. **AutoAttend Integration**: An end-to-end automated attendance manager that synchronizes camera streams with institutional schedules, applying multi-tier caching and idempotent atomic database updates.

---

## 🎓 The AutoAttend Application

**AutoAttend** is the automated institutional attendance management application built atop this recognition engine.

```text
       [ Video Camera Stream ]
                 │
                 ▼
       [ AutoAttend Engine ]
                 │
  ┌──────────────┴──────────────┐
  │ Check Scheduled Time Window │ ── Outside Window ──► [ Skip Frame ]
  └──────────────┬──────────────┘
                 │ Inside Active Window (e.g. h1: 09:40 - 09:50)
                 ▼
  ┌─────────────────────────────┐
  │ Multi-Face Recognition (AI) │ ── "Unknown" ──────► [ Highlight Red ]
  └──────────────┬──────────────┘
                 │ Recognized Student ID
                 ▼
  ┌─────────────────────────────┐
  │ In-Memory Window Dedup      │ ── Already Processed ──► [ Use Cached Score ]
  └──────────────┬──────────────┘
                 │ New Verification
                 ▼
  ┌─────────────────────────────┐
  │ 10s Cooldown & DB Update    │
  │ • Mark active hour (h1)     │
  │ • Recalculate Sections      │
  │ • Evaluate Full/Half Day    │
  └──────────────┬──────────────┘
                 ▼
       [ PostgreSQL Database ]
```

### Key AutoAttend Attendence logic

- **Timetable Window Enforcement**: Attendance is strictly registered only during designated class windows:
  - **Hour 1 (`h1`)**: 09:40 – 09:50
  - **Hour 2 (`h2`)**: 10:40 – 10:50
  - **Hour 3 (`h3`)**: 11:55 – 12:05
  - **Hour 4 (`h4`)**: 14:10 – 14:20
  - **Hour 5 (`h5`)**: 15:10 – 15:20
- **Automated Academic Session & Score Aggregation**:
  - **Morning Section (`mrg_section`)**: Evaluated as `h1 AND h2 AND h3`.
  - **Evening Section (`eveg_section`)**: Evaluated as `h4 AND h5`.
  - **Daily Score (`fullday`)**:
    - `1.0` credit if both Morning and Evening sections are complete.
    - `0.5` credit if either Morning or Evening section is complete.
    - `0.0` if neither section condition is met.

---

## ⚡ Core Architecture

```text
[ Input Frame: 640x480 / 720p ]
            │
            ▼
┌──────────────────────────────────────┐
│  1. YOLOv8-Face Detection (ONNX)     │
│  • Resizing & Aspect Letterbox       │
│  • Logit-Space Candidate Filtering   │ ◄── Discards 99% of grid points before sigmoid
│  • Selective DFL Bounding Box Decode │
│  • 5-Point Landmark Extraction       │
│  • OpenCV NMS Boxes                  │
└──────────────────┬───────────────────┘
                   │ Bounding Boxes & 5-Point Landmarks
                   ▼
┌──────────────────────────────────────┐
│  2. 5-Point Affine Normalization     │
│  • Rigid Similarity Transform        │ ◄── skimage.transform.SimilarityTransform
│  • cv2.warpAffine -> 112x112 crops   │
└──────────────────┬───────────────────┘
                   │ Stack of N Face Crops: (N, 3, 112, 112)
                   ▼
┌──────────────────────────────────────┐
│  3. Batched ArcFace Feature Extractor│
│  • Single Parallel GPU Forward Pass  │ ◄── ONNX Runtime (CUDA / DirectML)
│  • Vectorized L2 Normalization       │
└──────────────────┬───────────────────┘
                   │ Normalized 512-D Embeddings: (N, 512)
                   ▼
┌──────────────────────────────────────┐
│  4. Vector Similarity Matching       │
│  • Cosine Matrix Dot Product         │ ◄── flat_matrix (M, 512) @ embedding
│  • Argmax & Similarity Thresholding  │
└──────────────────┬───────────────────┘
                   │ Identity IDs & Similarity Scores
                   ▼
┌──────────────────────────────────────┐
│  5. AutoAttend Attendance Engine     │
│  • Timetable Validation              │
│  • In-Memory Deduplication Cache     │
│  • PostgreSQL Atomic Upsert          │
└──────────────────────────────────────┘
```

---

## 📊 Latency Benchmarks & Performance

Rigorous benchmarking was conducted using an actual dense nursery classroom video stream (`vidssave.com action word for nursery class 720P.mp4`) containing **mean 16.5 faces/frame** (peak 18 faces/frame) at 720p 30 FPS.

### Test Environment

- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)
- **Runtime**: ONNX Runtime with `CUDAExecutionProvider` / DirectML
- **Workload**: 1,040 video frames parsed, 520 inference passes, 500 measured passes (post 20 warm-up frames), `FRAME_SKIP = 2`.

### Summary Comparison

| Model Pipeline              | Mean Faces / Frame | Mean Full Inference (ms) | Median Full Inference (ms) | P95 Full Inference (ms) | Inference Throughput (FPS) | Observed Video Loop (FPS) | Speedup vs Buffalo_L   |
| :-------------------------- | :----------------: | :----------------------: | :------------------------: | :---------------------: | :------------------------: | :-----------------------: | :--------------------: |
| **Custom YOLOv8 + ArcFace** |       **16.49**    |        **71.284 ms**     |        **68.325 ms**       |      **87.848 ms**      |        **14.03 FPS**       |        **24.09 FPS**      |   **3.57× Faster**     |
| **InsightFace Buffalo_S**   |        16.35       |        168.507 ms        |        167.535 ms          |        185.364 ms       |          5.93 FPS          |          11.05 FPS        |      1.51× Faster      |
| **InsightFace Buffalo_M**   |        16.45       |        246.429 ms        |        250.027 ms          |        258.371 ms       |          4.06 FPS          |           7.74 FPS        |      1.03× Faster      |
| **InsightFace Buffalo_L**   |        16.58       |        254.720 ms        |        257.643 ms          |        266.376 ms       |          3.93 FPS          |           7.48 FPS        | Baseline (1.0×)        |

|   |   |   |   |   |
| :-: | :-: | :-: | :-: | :-: |
| **Total Full Inference** | **71.28 ms** | **68.33 ms** | **87.85 ms** | **57.45 ms** |

> [!NOTE]
> **Thermal Artifact Mitigation**: Benchmarks are structured as isolated processes with physical cooldown intervals to prevent GPU dynamic clock throttling from biasing consecutive test runs.

---

## 🗂️ Repository Structure

```text
Multi_Face_recognition_system/
├── benchmark_results/             # Output CSV latency logs from benchmark executions
├── data/
│   ├── embeddings/                # Serialized identity embeddings (.pkl)
│   ├── enrollment/                # Raw student face crop sets grouped by student name
│   └── *.mp4                      # Sample 720p evaluation video
├── documents/                     # Research papers (ArcFace, InsightFace, YOLOv8)
├── weights/                       # Pre-trained & converted model weights
│   ├── arcface.onnx               # ArcFace feature extractor
│   ├── w600k_r50.onnx             # Glint360k ResNet-50 ArcFace ONNX model
│   ├── yolov8n-face.onnx          # High-speed YOLOv8n-Face ONNX model
│   └── yolov8n-face.pt            # PyTorch checkpoint
├── src/
│   ├── config/
│   │   ├── config.py              # Configuration loader module
│   │   └── config.yaml            # Central configuration file
│   ├── evaluation/
│   │   └── latency/               # Isolated comparative latency benchmark runners
│   │       ├── insightface_s_eval.py
│   │       ├── insightface_m_eval.py
│   │       ├── insightface_l_eval.py
│   │       ├── yolo_arcface_eval.py
│   │       ├── settings.py
│   │       └── README.md
│   ├── student_enroll/
│   │   ├── data_collection.py     # Interactive multi-batch face photo capture CLI
│   │   └── embedding.py           # Batch enrollment face embedding generator
│   ├── testing/
│   │   ├── inference_test.py      # Standalone webcam recognition runner
│   │   ├── yolov8_face_test.py    # YOLOv8 face & 5-landmark visualizer
│   │   └── test_trt.py            # TensorRT validation test
│   ├── utility/
│   │   ├── align.py               # Affine similarity transformation & norm_crop
│   │   ├── arcface.py             # Batched ArcFace ONNX wrapper
│   │   ├── attendance.py          # AutoAttend time-window manager & deduplication
│   │   ├── database.py            # PostgreSQL connection pool & atomic SQL logic
│   │   ├── webcam.py              # Threaded camera stream utility
│   │   └── yolo_face.py           # Optimized YOLOv8-Face detector with logit pruning
│   ├── face_recognition.py        # Main AutoAttend live recognition & attendance app
│   └── requirements.txt           # Project Python dependencies
└── README.md                      # Project documentation
```

---

## 🛠️ Prerequisites & Installation

### 1. System Requirements

- **Python**: Version 3.10 or 3.11
- **CUDA/cuDNN**: CUDA 12.x compatible drivers for GPU acceleration
- **PostgreSQL**: PostgreSQL 14+ installed and running locally

### 2. Environment Setup

Clone the repository and create an isolated virtual environment:

```bash
cd /home/austin/Multi_Face_recognition_system
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install PyTorch with CUDA support and the project dependencies:

```bash
pip install --upgrade pip
pip install -r src/requirements.txt
```

> [!NOTE]
> **DirectML Note (Windows / AMD GPUs)**: If running on Windows with DirectX 12 or DirectML execution provider, install:
>
> ```bash
> pip install onnxruntime-directml
> ```

---

## 🗄️ Database Schema & Setup

AutoAttend requires a PostgreSQL database named `attendence_db`.

### 1. Database Initialization

Access your PostgreSQL prompt and create the database:

```sql
CREATE DATABASE attendence_db;
\c attendence_db;
```

### 2. Create Schema

Run the following SQL DDL statements:

```sql
-- Students Registry
CREATE TABLE IF NOT EXISTS students (
    roll_no VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    department VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Daily Attendance Log
CREATE TABLE IF NOT EXISTS attendance (
    id SERIAL PRIMARY KEY,
    roll_no VARCHAR(50) NOT NULL REFERENCES students(roll_no) ON DELETE CASCADE,
    attendance_date DATE DEFAULT CURRENT_DATE,
    h1 BOOLEAN DEFAULT FALSE,
    h2 BOOLEAN DEFAULT FALSE,
    h3 BOOLEAN DEFAULT FALSE,
    h4 BOOLEAN DEFAULT FALSE,
    h5 BOOLEAN DEFAULT FALSE,
    mrg_section BOOLEAN DEFAULT FALSE,
    eveg_section BOOLEAN DEFAULT FALSE,
    fullday NUMERIC(3, 1) DEFAULT 0.0,
    CONSTRAINT unique_student_daily UNIQUE (roll_no, attendance_date)
);
```

### 3. Update Database Credentials

Configure your database connection parameters in `src/utility/database.py`:

```python
def get_connection():
    return psycopg2.connect(
        dbname   = "attendence_db",
        user     = "your_username",
        password = "your_password",
        host     = "localhost",
        port     = "5432",
    )
```

---

## 🚀 Workflow & Step-by-Step Usage

### 1. Student Enrollment (Data Acquisition)

To enroll a new student, run the guided interactive capture script. It captures multiple pose batches (e.g., `far1`, `far2`, `far3`) with configurable intervals:

```bash
python -m src.student_enroll.data_collection
```

- Select inference device (`cpu` or `gpu`).
- Enter the student's name (or Roll Number).
- Face detections will show live bounding boxes. Head crops are saved automatically to `data/enrollment/<name>/`.

### 2. Embedding Generation

Process enrolled image sets into 512-dimensional L2-normalized feature maps:

```bash
python -m src.student_enroll.embedding
```

- Scans `data/enrollment/`.
- Skips already-processed identities.
- Generates aligned embeddings and serializes them to `data/embeddings/<name>.pkl`.

### 3. Standalone Verification & Testing

To verify face detection, 5-point landmark tracking, and real-time FPS without writing to the database:

```bash
# Test YOLOv8-Face detection & 5 landmarks visualization
python -m src.testing.yolov8_face_test

# Test live webcam face recognition
python -m src.testing.inference_test
```

### 4. Running Real-Time Attendance

Launch the complete AutoAttend application with camera feeds, live identity labeling, active schedule checking, and automatic database updates:

```bash
python -m src.face_recognition
```

- Press `q` to terminate the video feed and close database connections cleanly.

### 5. Latency & Comparative Benchmarking

To reproduce empirical performance benchmarks against InsightFace model packs:

```bash
# Run custom optimized YOLOv8 + ArcFace benchmark
python -m src.evaluation.latency.yolo_arcface_eval

# Run InsightFace benchmarks
python -m src.evaluation.latency.insightface_s_eval
python -m src.evaluation.latency.insightface_m_eval
python -m src.evaluation.latency.insightface_l_eval
```

- Per-frame latency statistics and throughput summaries are exported to `benchmark_results/`.
