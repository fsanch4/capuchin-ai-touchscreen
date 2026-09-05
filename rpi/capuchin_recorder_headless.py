"""
YOLO detection and video-recording worker.

``run_recorder`` is called by ``run_capuchinai.py`` in its own child process.
"""

import csv
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path


LOGGER = logging.getLogger(__name__)

def estimate_fps(sample_times: deque[float]) -> float:
    if len(sample_times) < 2:
        return 30.0
    elapsed = sample_times[-1] - sample_times[0]
    return (len(sample_times) - 1) / elapsed if elapsed > 0 else 30.0


def run_recorder(
    stop_event,
    last_detection,
    *,
    weights: str,
    source: str = "0",
    img_size: int = 416,
    conf_thres: float = 0.5,
    detection_timeout: float = 10.0,
    record_dir: str = "recordings",
) -> None:
    # Heavy/native dependencies are imported only inside the recorder child.
    import cv2
    import torch

    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=weights,
        force_reload=False,
    )
    model.conf = conf_thres
    model.iou = 0.4

    source_text = str(source)
    video_source = int(source_text) if source_text.isdigit() else source_text
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video source {source!r}")

    output_dir = Path(record_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "recordings_log.csv"
    log_exists = log_path.exists() and log_path.stat().st_size > 0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_times: deque[float] = deque(maxlen=30)

    out = None
    filename: str | None = None
    recording_start_time: str | None = None
    last_local_detection: float | None = None

    LOGGER.info("Starting detection and recording")

    try:
        with log_path.open("a", newline="", encoding="utf-8") as log_file:
            csv_writer = csv.writer(log_file)
            if not log_exists:
                csv_writer.writerow(["filename", "start_time", "end_time"])
                log_file.flush()

            def finish_recording() -> None:
                nonlocal out, filename, recording_start_time
                if out is None:
                    return

                out.release()
                out = None
                recording_end_time = datetime.now().isoformat()
                csv_writer.writerow(
                    [filename, recording_start_time, recording_end_time]
                )
                log_file.flush()
                LOGGER.info("Stopped recording: %s", filename)
                filename = None
                recording_start_time = None

            try:
                while not stop_event.is_set():
                    sample_time = time.monotonic()
                    ok, frame = cap.read()
                    if not ok:
                        LOGGER.warning("Video source stopped producing frames")
                        break

                    sample_times.append(sample_time)
                    results = model(frame, size=img_size)
                    has_detection = len(results.xyxy[0]) > 0
                    now = time.monotonic()

                    if has_detection:
                        last_local_detection = now
                        with last_detection.get_lock():
                            last_detection.value = now

                        if out is None:
                            timestamp = datetime.now().strftime(
                                "%Y%m%d_%H%M%S_%f"
                            )
                            filename = f"capuchin_{timestamp}.mp4"
                            output_path = output_dir / filename
                            out = cv2.VideoWriter(
                                str(output_path),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                estimate_fps(sample_times),
                                (frame_width, frame_height),
                            )
                            if not out.isOpened():
                                out.release()
                                out = None
                                raise RuntimeError(
                                    f"Failed to open video writer for {output_path}"
                                )
                            recording_start_time = datetime.now().isoformat()
                            LOGGER.info("Started recording: %s", filename)

                    if out is not None:
                        out.write(frame)

                    if (
                        out is not None
                        and last_local_detection is not None
                        and now - last_local_detection > detection_timeout
                    ):
                        finish_recording()

                    stop_event.wait(0.005)
            finally:
                finish_recording()
    finally:
        cap.release()
