
import csv
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from src.utility.yolo_face import YOLOv8_face
from src.utility.align import norm_crop
from src.utility.arcface import ArcFaceONNX
from src.config.config import config
from src.evaluation.latency.settings import load_embeddings,recognize,stats,print_stats


VIDEO_PATH = config["directory"]["VIDEO_DIR"]

FRAME_SKIP = config['test']['FRAME_SKIP']
WARMUP_INFERENCE_FRAMES = config['test']['WARMUP_INFERENCE_FRAMES']
MEASURED_INFERENCE_FRAMES = config['test']['MEASURED_INFERENCE_FRAMES']
SHOW_PREVIEW = True

YOLO_MODEL_PATH = config["model"]["YOLO_ONNX_PATH"]
ARCFACE_MODEL_PATH = config["model"]["ARCFACE_MODEL_PATH"]
EMBEDDINGS_DIR = config["directory"]["EMBEDDINGS_DIR"]
FACE_IMAGE_SIZE = config["model"]["IMAGE_SIZE"]
SIMILARITY_THRESHOLD = float(config["model"]["ARCFACE_SIM_THRESH"])
YOLO_CONF_THRESHOLD = float(config["model"]["YOLO_CONF_THRESH"])

RESULTS_DIR = Path("benchmark_results")




def landmarks_5x2(flat_landmarks):
    return flat_landmarks.reshape(5, 3)[:, :2].astype(np.float32)



def main():
    print("=" * 90)
    print("AUTOATTEND — VIDEO FILE FULL INFERENCE LATENCY BENCHMARK")
    print("=" * 90)
    print(f"OpenCV: {cv2.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"Available ORT providers: {ort.get_available_providers()}")
    print(f"Video Source: {VIDEO_PATH}")
    print(f"Frame skip: {FRAME_SKIP}")
    print(f"Warm-up processed frames: {WARMUP_INFERENCE_FRAMES}")
    print(f"Target measured processed frames: {MEASURED_INFERENCE_FRAMES}")
    print("\nNote: CUDA being available does not prove every ONNX node runs on GPU.")

    print("\n[1/4] Loading YOLO...")
    detector = YOLOv8_face(
        YOLO_MODEL_PATH,
        conf_thres=YOLO_CONF_THRESHOLD,
        iou_thres=0.5,
    )

    print("\n[2/4] Loading ArcFace...")
    arcface = ArcFaceONNX(ARCFACE_MODEL_PATH)

    print("\n[3/4] Loading student embeddings...")
    database = load_embeddings(EMBEDDINGS_DIR)
    print(f"Loaded identities: {len(database)}")
    if not database:
        print("WARNING: no embeddings loaded; matching latency will not represent real attendance use.")

    print("\n[4/4] Opening video file...")
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

                boxes, scores, class_ids, keypoints = detector.detect(
                    frame, show_timing=False
                )
                yolo_total_ms = (time.perf_counter() - full_start) * 1000.0

                align_start = time.perf_counter()
                aligned_faces = [
                    norm_crop(frame, landmarks_5x2(kp), image_size=FACE_IMAGE_SIZE)
                    for kp in keypoints
                ]
                alignment_ms = (time.perf_counter() - align_start) * 1000.0

                arcface_start = time.perf_counter()
                embeddings = arcface.get_embeddings(aligned_faces)

                arcface_ms = (time.perf_counter() - arcface_start) * 1000.0

                matching_start = time.perf_counter()
                frame_results = [
                    recognize(embedding, database) for embedding in embeddings
                ]
                matching_ms = (time.perf_counter() - matching_start) * 1000.0

                full_ms = (time.perf_counter() - full_start) * 1000.0
                cached_results = [
                    (box, name, similarity)
                    for box, (name, similarity) in zip(boxes, frame_results)
                ]

                # Exclude first warm-up inference calls.
                if processed_index > WARMUP_INFERENCE_FRAMES:
                    measured_rows.append({
                        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                        "camera_frame": frame_index,
                        "processed_frame": processed_index,
                        "face_count": len(boxes),
                        "camera_read_ms": read_ms,
                        "yolo_total_ms": yolo_total_ms,
                        "alignment_all_faces_ms": alignment_ms,
                        "arcface_all_faces_ms": arcface_ms,
                        "matching_all_faces_ms": matching_ms,
                        "full_inference_ms": full_ms,
                    })

                    n = len(measured_rows)
                    if n == 1 or n % 10 == 0:
                        print(
                            f"{n:3d}/{MEASURED_INFERENCE_FRAMES} | faces={len(boxes):2d} | "
                            f"YOLO={yolo_total_ms:7.2f} ms | align={alignment_ms:7.2f} ms | "
                            f"ArcFace={arcface_ms:7.2f} ms | match={matching_ms:7.2f} ms | "
                            f"FULL={full_ms:7.2f} ms"
                        )

            if SHOW_PREVIEW:
                for box, name, similarity in cached_results:
                    x, y, w, h = box.astype(int)
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(
                        frame, f"{name} {similarity:.2f}",
                        (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2
                    )
                loop_elapsed = max(time.perf_counter() - loop_start, 1e-9)
                cv2.putText(
                    frame, f"Loop FPS: {1.0 / loop_elapsed:.1f} | Faces: {len(cached_results)}",
                    (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 1
                )
                cv2.imshow("AutoAttend — video latency benchmark (q to stop)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_reason = "stopped early by user"
                    break

            total_loop_times.append((time.perf_counter() - loop_start) * 1000.0)
            capture_times.append(read_ms)

    finally:
        cap.release()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()

    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"Status: {stop_reason}")
    print(f"Video frames read: {frame_index}")
    print(f"Inference frames processed: {processed_index}")
    print(f"Measured frames: {len(measured_rows)} (after {min(processed_index, WARMUP_INFERENCE_FRAMES)} warm-up calls)")
    if not measured_rows:
        print("No timed samples. Check video path, models and configuration.")
        return

    print(f"Mean faces per measured frame: {np.mean([r['face_count'] for r in measured_rows]):.2f}")
    print(f"Maximum faces in a measured frame: {max(r['face_count'] for r in measured_rows)}")
    print()
    print_stats("YOLO detect() total", [r["yolo_total_ms"] for r in measured_rows])
    print_stats("Face alignment (all faces)", [r["alignment_all_faces_ms"] for r in measured_rows])
    print_stats("ArcFace (all faces)", [r["arcface_all_faces_ms"] for r in measured_rows])
    print_stats("Embedding matching (all faces)", [r["matching_all_faces_ms"] for r in measured_rows])
    print_stats("FULL inference pipeline", [r["full_inference_ms"] for r in measured_rows])
    print_stats("Video read (context)", [r["camera_read_ms"] for r in measured_rows])
    print_stats("Full video loop (context)", total_loop_times)

    full_stats = stats([r["full_inference_ms"] for r in measured_rows])
    print(f"\nApproximate inference throughput (1000 / mean full latency): {1000.0 / full_stats['mean']:.2f} processed frames/s")
    if total_loop_times:
        print(f"Observed loop throughput (includes frame skip / display): {1000.0 / np.mean(total_loop_times):.2f} video frames/s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"autoattend_video_latency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(measured_rows[0].keys()))
        writer.writeheader()
        writer.writerows(measured_rows)
    print(f"\nPer-frame CSV: {output_path.resolve()}")
    print("\nFor comparison runs, keep the same lighting, video source, enrolled database,")
    print("frame skip, model files and background workload.")


if __name__ == "__main__":
    main()