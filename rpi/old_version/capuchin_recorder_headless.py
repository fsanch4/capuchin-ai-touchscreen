import cv2
import torch
import argparse
import time
import csv
from pathlib import Path
from datetime import datetime
from collections import deque

def estimate_fps(time_deque):
    if len(time_deque) < 2:
        return 30.0  # default fallback
    elapsed_time = time_deque[-1] - time_deque[0]
    return len(time_deque) / elapsed_time if elapsed_time > 0 else 30.0

def run(weights='best.pt', source=0, img_size=416, conf_thres=0.5):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=False)
    model.conf = conf_thres
    model.iou = 0.4

    cap = cv2.VideoCapture(int(source)) if str(source).isdigit() else cv2.VideoCapture(source)
    assert cap.isOpened(), f"Failed to open video source {source}"

    record_dir = Path("recordings")
    record_dir.mkdir(exist_ok=True)
    log_path = record_dir / "recordings_log.csv"
    log_exists = log_path.exists()

    log_file = open(log_path, "a", newline="")
    csv_writer = csv.writer(log_file)
    if not log_exists:
        csv_writer.writerow(["filename", "start_time", "end_time"])

    recording = False
    out = None
    last_detection_time = 0
    recording_start_time = None

    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    time_deque = deque(maxlen=30)
    estimated_fps = 30.0

    print("[INFO] Starting detection and recording... Press Ctrl+C to quit.")

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            time_deque.append(start_time)
            estimated_fps = estimate_fps(time_deque)

            results = model(frame, size=img_size)
            detections = results.xyxy[0]
            has_detection = len(detections) > 0
            current_time = time.time()

            if has_detection:
                last_detection_time = current_time

                # Write timestamp to shared file for reward interface
                with open("/tmp/last_detection.txt", "w") as f:
                    f.write(str(current_time))

                if not recording:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"capuchin_{timestamp}.mp4"
                    out_path = record_dir / filename
                    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'),
                                          estimated_fps, (FRAME_WIDTH, FRAME_HEIGHT))
                    recording_start_time = datetime.now().isoformat()
                    print(f"[INFO] Started recording: {filename}")
                    recording = True

            if recording and out:
                out.write(frame)

            if recording and (current_time - last_detection_time > 10):
                recording_end_time = datetime.now().isoformat()
                print(f"[INFO] Stopped recording: {filename}")
                out.release()
                csv_writer.writerow([filename, recording_start_time, recording_end_time])
                recording = False
                out = None

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        if out:
            out.release()
        cap.release()
        log_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt', help='path to model weights')
    parser.add_argument('--source', type=str, default='0', help='camera index or video file path')
    parser.add_argument('--img', type=int, default=416, help='inference image size')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    args = parser.parse_args()

    run(weights=args.weights, source=args.source, img_size=args.img, conf_thres=args.conf)
