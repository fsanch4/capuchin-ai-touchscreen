"""
YOLO detection and video-recording worker.

``run_recorder`` is called by ``run_capuchinai.py`` in its own child process.
"""

import csv
import logging
import time
from datetime import datetime
from pathlib import Path


LOGGER = logging.getLogger(__name__)


class _RealtimeVideoWriter:
    """Resample live frames onto a fixed-FPS timeline, holding the last frame.

    Timestamps are monotonic camera-read completion times, not inference times.
    This preserves elapsed time but cannot recover frames missed during inference
    or compensate for frames already buffered by the camera backend.
    """

    def __init__(self, writer, fps: float) -> None:
        self.writer = writer
        self.fps = fps
        self.start_time = None
        self.last_frame = None
        self.frame_count = 0

    def _fill_until(self, timestamp: float) -> None:
        if self.start_time is None:
            return
        elapsed = timestamp - self.start_time
        while self.frame_count / self.fps < elapsed:
            self.writer.write(self.last_frame)
            self.frame_count += 1

    def write(self, frame, captured_at: float) -> None:
        if self.start_time is None:
            self.start_time = captured_at
            self.writer.write(frame)
            self.frame_count = 1
        else:
            # Fill earlier playback slots with the earlier frame, not this one.
            self._fill_until(captured_at)
        self.last_frame = frame

    def release(self, stopped_at: float) -> None:
        try:
            self._fill_until(stopped_at)
        finally:
            self.writer.release()


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
    record_fps: float = 15.0,
) -> None:
    if record_fps <= 0:
        raise ValueError("record_fps must be finite and positive")
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

                stopped_at = time.monotonic()
                recording_end_time = datetime.now().isoformat()
                out.release(stopped_at)
                LOGGER.info(
                    "Recording timing: %.3fs elapsed, %.3fs playback (%d frames at %.2f FPS)",
                    stopped_at - out.start_time if out.start_time is not None else 0.0,
                    out.frame_count / record_fps,
                    out.frame_count,
                    record_fps,
                )
                out = None
                csv_writer.writerow(
                    [filename, recording_start_time, recording_end_time]
                )
                log_file.flush()
                LOGGER.info("Stopped recording: %s", filename)
                filename = None
                recording_start_time = None

            try:
                while not stop_event.is_set():
                    ok, frame = cap.read()
                    captured_at = time.monotonic()
                    captured_wall_time = datetime.now().isoformat()
                    if not ok:
                        LOGGER.warning("Video source stopped producing frames")
                        break

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
                            writer = cv2.VideoWriter(
                                str(output_path),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                record_fps,
                                (frame_width, frame_height),
                            )
                            if not writer.isOpened():
                                writer.release()
                                raise RuntimeError(
                                    f"Failed to open video writer for {output_path}"
                                )
                            out = _RealtimeVideoWriter(writer, record_fps)
                            recording_start_time = captured_wall_time
                            LOGGER.info("Started recording: %s", filename)

                    if out is not None:
                        out.write(frame, captured_at)

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
