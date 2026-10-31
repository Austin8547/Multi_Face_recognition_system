import cv2
import os
import sys
import time
import torch
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config.config import config


def find_webcam(preferred_index):
    """Try preferred index first, then scan 0-5 as fallback."""
    cap = cv2.VideoCapture(preferred_index)
    if cap.isOpened():
        print(f"Webcam found at index {preferred_index}")
        return cap

    print(f"Webcam not found at index {preferred_index}. Scanning...")
    for i in range(6):
        if i == preferred_index:
            continue
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Webcam found at index {i}")
            return cap

    return None


def main():
    # ── Load config ───────────────────────────────────────────────────────────
    # Config is loaded from src.config.config
    # Data acquisition params
    batches          = config['data_acquisition']['BATCHES']
    photos_per_batch = config['data_acquisition']['PHOTOS_PER_BATCH']
    capture_interval = config['data_acquisition']['CAPTURE_INTERVAL']

    # Model params
    yolo_model_path = config['model']['YOLO_MODEL_PATH']
    conf_threshold  = config['model']['YOLO_CONF_THRESH']
    padding         = config['model']['PADDING']

    # Directories
    enrollment_dir = config['directory']['ENROLLMENT_DIR']

    # Webcam index — add WEBCAM_INDEX under [hardware] in config.yaml
    # If the key doesn't exist yet, defaults to 0
    webcam_index = config.get('hardware', {}).get('WEBCAM_INDEX', 0)

    # ── Load YOLO model ───────────────────────────────────────────────────────
    while True:
        device_choice = input("Select device to run the YOLO model [cpu/gpu]: ").strip().lower()
        if device_choice in ['cpu', 'gpu']:
            break
        print("Invalid choice. Please enter 'cpu' or 'gpu'.")

    print(f"Loading YOLO model from {yolo_model_path}...")
    try:
        model = YOLO(yolo_model_path)
        if device_choice == 'gpu':
            if torch.cuda.is_available():
                print("CUDA is available! Moving model to GPU...")
                model.to('cuda')
            else:
                print("CUDA not available. Falling back to CPU...")
        else:
            print("Using CPU.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # ── Person name ───────────────────────────────────────────────────────────
    person_name = input("Enter the name of the person: ").strip()
    if not person_name:
        print("Name cannot be empty. Exiting.")
        return

    person_dir = os.path.join(enrollment_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)

    # ── Open webcam ───────────────────────────────────────────────────────────
    cap = find_webcam(webcam_index)
    if cap is None:
        print("Error: Could not open any webcam. "
              "Check WEBCAM_INDEX in config.yaml or verify camera connection.")
        return

    print("\nStarting Data Collection Pipeline...")

    total_photos_taken = 0

    for batch_idx, batch_name in enumerate(batches):
        print(f"\n--- Batch {batch_idx + 1}/{len(batches)}: '{batch_name}' ---")
        print("Please prepare your position/expression for this batch.")
        input("Press [Enter] to start capturing...")

        photos_taken_in_batch = 0
        last_capture_time     = time.time()

        while photos_taken_in_batch < photos_per_batch:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam.")
                break

            display_frame = frame.copy()

            # Detect faces
            results = model(frame, conf=conf_threshold, verbose=False)

            # Draw bounding boxes for feedback
            for r in results:
                for box in r.boxes:
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = b
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.putText(display_frame, f"Batch: {batch_name}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Captured: {photos_taken_in_batch}/{photos_per_batch}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Data Collection - Face Detection", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Capture interrupted by user ('q' pressed).")
                cap.release()
                cv2.destroyAllWindows()
                return

            # Capture on interval
            current_time = time.time()
            if current_time - last_capture_time >= capture_interval:
                face_captured = False

                for r in results:
                    if len(r.boxes) > 0:
                        b = r.boxes[0].xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = b

                        box_w = x2 - x1
                        box_h = y2 - y1

                        pad_top    = int(box_h * 0.2) + padding
                        pad_bottom = int(box_h * 0.15) + padding
                        pad_sides  = int(box_w * 0.15) + padding

                        h, w = frame.shape[:2]
                        x1 = max(0, x1 - pad_sides)
                        y1 = max(0, y1 - pad_top)
                        x2 = min(w, x2 + pad_sides)
                        y2 = min(h, y2 + pad_bottom)

                        head_crop = frame[y1:y2, x1:x2]

                        if head_crop.size > 0:
                            photo_filename = f"{person_name}_{batch_name}_{photos_taken_in_batch + 1}.jpg"
                            photo_path     = os.path.join(person_dir, photo_filename)
                            cv2.imwrite(photo_path, head_crop)
                            print(f"Captured: {photo_filename}")

                            photos_taken_in_batch += 1
                            total_photos_taken    += 1
                            last_capture_time      = current_time
                            face_captured          = True

                            # Flash green to confirm capture
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
                            cv2.imshow("Data Collection - Face Detection", display_frame)
                            cv2.waitKey(100)

                        break  # one face per frame

        print(f"Batch '{batch_name}' complete. ({photos_taken_in_batch} photos saved)")

    print(f"\nData collection finished for '{person_name}'.")
    print(f"Total photos taken: {total_photos_taken}/{len(batches) * photos_per_batch}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()