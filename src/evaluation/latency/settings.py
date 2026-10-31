import csv
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

from src.config.config import config


VIDEO_PATH = config["directory"]["VIDEO_DIR"]
FRAME_SKIP = config["test"]["FRAME_SKIP"]
WARMUP_INFERENCE_FRAMES = config["test"]["WARMUP_INFERENCE_FRAMES"]
MEASURED_INFERENCE_FRAMES = config["test"]["MEASURED_INFERENCE_FRAMES"]
EMBEDDINGS_DIR = config["directory"]["EMBEDDINGS_DIR"]
SIMILARITY_THRESHOLD = float(config["model"]["ARCFACE_SIM_THRESH"])
RESULTS_DIR = Path("benchmark_results")


def load_embeddings(directory):
    database = {}
    if not os.path.isdir(directory):
        return database
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".pkl"):
            continue
        filepath = os.path.join(directory, filename)
        with open(filepath, "rb") as handle:
            embeddings = pickle.load(handle)
        if len(embeddings):
            database[os.path.splitext(filename)[0]] = np.asarray(
                np.stack(embeddings), dtype=np.float32
            )
    return database


def recognize(embedding, database):
    if not database:
        return "Unknown", 0.0
    best_name, best_score = "Unknown", -1.0
    for name, stored_matrix in database.items():
        score = float(np.dot(stored_matrix, embedding).max())
        if score > best_score:
            best_name, best_score = name, score
    if best_score < SIMILARITY_THRESHOLD:
        return "Unknown", best_score
    return best_name, best_score


def stats(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def print_stats(label, values):
    result = stats(values)
    if result is None:
        print(f"{label:36s} no measurements")
        return
    print(
        f"{label:36s} mean {result['mean']:8.3f} ms | "
        f"median {result['median']:8.3f} ms | "
        f"P95 {result['p95']:8.3f} ms | "
        f"min {result['min']:8.3f} ms | max {result['max']:8.3f} ms"
    )


def run_benchmark(model_name: str, display_title: str, csv_prefix: str):
    print("=" * 90)
    print(f"AUTOATTEND — INSIGHTFACE ({model_name.upper()}) VIDEO FILE LATENCY BENCHMARK")
    print("=" * 90)
    print(f"OpenCV: {cv2.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"Available ORT providers: {ort.get_available_providers()}")
    print(f"Video Source: {VIDEO_PATH}")
    print(f"Frame skip: {FRAME_SKIP}")
    print(f"Warm-up processed frames: {WARMUP_INFERENCE_FRAMES}")
    print(f"Target measured processed frames: {MEASURED_INFERENCE_FRAMES}")

    print(f"\n[1/3] Initializing InsightFace ({model_name} / CUDA)...")
    app = FaceAnalysis(name=model_name)
    app.prepare(ctx_id=0, det_size=(640, 640))

    print("\n[2/3] Loading student embeddings...")
    database = load_embeddings(EMBEDDINGS_DIR)
    print(f"Loaded identities: {len(database)}")
    if not database:
        print("WARNING: no embeddings loaded; matching latency will not represent real attendance use.")

    print("\n[3/3] Opening video file...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: could not open video file {VIDEO_PATH}")
        return

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video Info: {total_video_frames} total frames, {fps:.2f} FPS")

    frame_index = 0
    processed_index = 0
    measured_rows = []
    cached_results = []
    total_loop_times = []
    capture_times = []
    stop_reason = "completed"
    show_preview = True

    try:
        while len(measured_rows) < MEASURED_INFERENCE_FRAMES:
            loop_start = time.perf_counter()
            read_start = time.perf_counter()
            ok, frame = cap.read()
            read_ms = (time.perf_counter() - read_start) * 1000.0
            
            if not ok or frame is None:
                print("\nReached end of video file.")
                stop_reason = "completed (end of video reached)"
                break

            frame_index += 1
            should_infer = (frame_index % FRAME_SKIP == 0)

            if should_infer:
                processed_index += 1
                full_start = time.perf_counter()

                faces = app.get(frame)
                insightface_ms = (time.perf_counter() - full_start) * 1000.0

                matching_start = time.perf_counter()
                frame_results = []
                cached_results = []
                
                for face in faces:
                    embedding = face.normed_embedding
                    name, similarity = recognize(embedding, database)
                    frame_results.append((name, similarity))
                    box = face.bbox.astype(int)
                    cached_results.append((box, name, similarity))

                matching_ms = (time.perf_counter() - matching_start) * 1000.0
                full_ms = (time.perf_counter() - full_start) * 1000.0

                if processed_index > WARMUP_INFERENCE_FRAMES:
                    measured_rows.append({
                        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                        "camera_frame": frame_index,
                        "processed_frame": processed_index,
                        "face_count": len(faces),
                        "video_read_ms": read_ms,
                        "insightface_pipeline_ms": insightface_ms,
                        "matching_all_faces_ms": matching_ms,
                        "full_inference_ms": full_ms,
                    })

                    n = len(measured_rows)
                    if n == 1 or n % 10 == 0:
                        print(
                            f"{n:3d}/{MEASURED_INFERENCE_FRAMES} | faces={len(faces):2d} | "
                            f"InsightFace={insightface_ms:7.2f} ms | match={matching_ms:7.2f} ms | "
                            f"FULL={full_ms:7.2f} ms"
                        )

            if show_preview:
                for box, name, similarity in cached_results:
                    x1, y1, x2, y2 = box
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame, f"{name} {similarity:.2f}",
                        (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2
                    )
                loop_elapsed = max(time.perf_counter() - loop_start, 1e-9)
                cv2.putText(
                    frame, f"Loop FPS: {1.0 / loop_elapsed:.1f} | Faces: {len(cached_results)}",
                    (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 1
                )
                cv2.imshow(display_title, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_reason = "stopped early by user"
                    break

            total_loop_times.append((time.perf_counter() - loop_start) * 1000.0)
            capture_times.append(read_ms)

    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()

    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"Status: {stop_reason}")
    print(f"Video frames read: {frame_index}")
    print(f"Inference frames processed: {processed_index}")
    print(f"Measured frames: {len(measured_rows)} (after {min(processed_index, WARMUP_INFERENCE_FRAMES)} warm-up calls)")
    if not measured_rows:
        print("No timed samples. Check video path and configuration.")
        return

    print(f"Mean faces per measured frame: {np.mean([r['face_count'] for r in measured_rows]):.2f}")
    print(f"Maximum faces in a measured frame: {max(r['face_count'] for r in measured_rows)}")
    print()
    print_stats("InsightFace pipeline (detect+embed)", [r["insightface_pipeline_ms"] for r in measured_rows])
    print_stats("Embedding matching (all faces)", [r["matching_all_faces_ms"] for r in measured_rows])
    print_stats("FULL inference pipeline", [r["full_inference_ms"] for r in measured_rows])
    print_stats("Video read (context)", [r["video_read_ms"] for r in measured_rows])
    print_stats("Full video loop (context)", total_loop_times)

    full_stats = stats([r["full_inference_ms"] for r in measured_rows])
    print(f"\nApproximate inference throughput (1000 / mean full latency): {1000.0 / full_stats['mean']:.2f} processed frames/s")
    if total_loop_times:
        print(f"Observed loop throughput (includes frame skip / display): {1000.0 / np.mean(total_loop_times):.2f} video frames/s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{csv_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(measured_rows[0].keys()))
        writer.writeheader()
        writer.writerows(measured_rows)
    print(f"\nPer-frame CSV: {output_path.resolve()}")