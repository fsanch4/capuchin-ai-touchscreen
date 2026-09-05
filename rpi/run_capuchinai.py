"""
Start and supervise the recorder and touchscreen processes.
"""

import argparse
import logging
import multiprocessing as mp
import signal
import time
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(processName)s %(levelname)s %(message)s",
    )


def recorder_entry(stop_event, last_detection, options: dict) -> None:
    # Let the parent process handle Ctrl+C.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    configure_logging()

    try:
        # Import heavy/native libraries only inside this child.
        from capuchin_recorder_headless import run_recorder

        run_recorder(stop_event, last_detection, **options)
    finally:
        # If either worker finishes or crashes, stop the whole application.
        stop_event.set()


def touchscreen_entry(stop_event, last_detection, options: dict) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    configure_logging()

    try:
        from touchscreen_reward_interface import run_reward_interface

        run_reward_interface(stop_event, last_detection, **options)
    finally:
        stop_event.set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(APP_DIR / "best.pt"))
    parser.add_argument("--source", default="0")
    parser.add_argument("--img", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--detection-timeout", type=float, default=10.0)
    parser.add_argument(
        "--record-dir",
        default=str(APP_DIR / "recordings"),
    )
    parser.add_argument("--relay-pin", type=int, default=17)
    parser.add_argument("--relay-duration", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()

    # Parallelization parameters
    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()

    # Synchronized double containing the most recent detection timestamp.
    last_detection = ctx.Value("d", 0.0)

    recorder = ctx.Process(
        name="recorder",
        target=recorder_entry,
        args=(
            stop_event,
            last_detection,
            {
                "weights": str(Path(args.weights).expanduser().resolve()),
                "source": args.source,
                "img_size": args.img,
                "conf_thres": args.conf,
                "detection_timeout": args.detection_timeout,
                "record_dir": str(
                    Path(args.record_dir).expanduser().resolve()
                ),
            },
        ),
    )

    touchscreen = ctx.Process(
        name="touchscreen",
        target=touchscreen_entry,
        args=(
            stop_event,
            last_detection,
            {
                "detection_timeout": args.detection_timeout,
                "relay_pin": args.relay_pin,
                "relay_duration": args.relay_duration,
            },
        ),
    )

    processes = [recorder, touchscreen]

    def request_stop(signum, _frame) -> None:
        LOGGER.info("Received signal %s; stopping", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        for process in processes:
            process.start()
            LOGGER.info("Started %s, pid=%s", process.name, process.pid)

        while not stop_event.wait(0.25):
            # This also catches native crashes that bypass a worker's finally.
            if any(p.exitcode is not None for p in processes):
                stop_event.set()
    finally:
        stop_event.set()

        deadline = time.monotonic() + 10
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))

        # Last resort after cooperative shutdown.
        for process in processes:
            if process.is_alive():
                LOGGER.warning("Force-stopping %s", process.name)
                process.terminate()
                process.join(2)

    failures = [p for p in processes if p.exitcode != 0]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
