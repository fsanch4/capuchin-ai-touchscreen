"""
Touchscreen and reward-relay worker.
"""

import logging
import time


LOGGER = logging.getLogger(__name__)
BLUE = (0, 0, 255)


def trigger_reward(relay, stop_event, duration: float) -> None:
    LOGGER.info("Triggering reward")
    relay.on()
    try:
        stop_event.wait(duration)
    finally:
        relay.off()
    LOGGER.info("Reward delivered")


def run_reward_interface(
    stop_event,
    last_detection,
    *,
    detection_timeout: float = 10.0,
    relay_pin: int = 17,
    relay_duration: float = 0.5,
) -> None:
    # Hardware and display initialization must happen inside this child process,
    # never at module import time.
    import pygame
    from gpiozero import OutputDevice

    relay = OutputDevice(
        relay_pin,
        active_high=False,
        initial_value=False,
    )

    try:
        pygame.init()
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("Touchscreen Reward System")
        screen.fill(BLUE)
        pygame.display.flip()
        clock = pygame.time.Clock()

        LOGGER.info("Touchscreen reward interface is running")

        while not stop_event.is_set():
            for event in pygame.event.get():
                if stop_event.is_set():
                    break

                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN
                    and event.key == pygame.K_ESCAPE
                ):
                    stop_event.set()
                    break

                if event.type == pygame.MOUSEBUTTONDOWN:
                    with last_detection.get_lock():
                        detected_at = last_detection.value

                    detection_is_recent = (
                        detected_at > 0
                        and time.monotonic() - detected_at
                        <= detection_timeout
                    )
                    if detection_is_recent:
                        trigger_reward(relay, stop_event, relay_duration)
                    else:
                        LOGGER.info(
                            "Touch ignored: no detection in the last %.1f seconds",
                            detection_timeout,
                        )

            clock.tick(60)
    finally:
        relay.off()
        relay.close()
        pygame.quit()
