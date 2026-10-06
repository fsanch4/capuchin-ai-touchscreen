"""
Touchscreen and reward-relay worker.

Keyboard: 1 = blue screen, 2 = reduced target, 3 = moving target.
Switches start a fresh test after any reward/flash, preserving the cooldown.
"""


from dataclasses import replace
import logging
import math
import time

from cognitive_tests import TestSettings, make_test


LOGGER = logging.getLogger(__name__)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class RewardCycle:
    """Advance the relay pulse and success flash without blocking event handling."""

    def __init__(self, relay, test, settings: TestSettings) -> None:
        self.relay = relay
        self.test = test
        self.settings = settings
        self.phase = "ready"
        self.deadline = 0.0
        self.cooldown_until = 0.0

    def ready(self, now: float) -> bool:
        return self.phase == "ready" and now >= self.cooldown_until

    def start(self, now: float) -> None:
        if not self.ready(now):
            return
        LOGGER.info("Triggering reward")
        self.relay.on()
        self.phase = "reward"
        self.deadline = now + self.settings.relay_duration

    def update(self, now: float) -> None:
        if self.phase == "reward" and now >= self.deadline:
            self.relay.off()
            LOGGER.info("Reward delivered")
            # Cooldown begins when the motor is turned off, as in test_moving.py.
            self.cooldown_until = now + self.settings.cooldown_duration
            self.phase = "feedback"
            self.deadline = now + self.settings.flash_duration

        if self.phase == "feedback" and now >= self.deadline:
            self.test.on_reward()
            self.phase = "ready"


def run_reward_interface(
    stop_event,
    last_detection,
    *,
    detection_timeout: float = 10.0,
    relay_pin: int = 17,
    relay_duration: float | None = None,
    test_name: str = "blue",
    target_size: tuple[int, int] | None = None,
    cooldown_duration: float | None = None,
    flash_duration: float | None = None,
) -> None:
    test = make_test(test_name, target_size)
    overrides = {
        name: value
        for name, value in (
            ("relay_duration", relay_duration),
            ("cooldown_duration", cooldown_duration),
            ("flash_duration", flash_duration),
        )
        if value is not None
    }
    settings = replace(test.settings, **overrides)
    for name, value in (
        ("detection_timeout", detection_timeout),
        ("relay_duration", settings.relay_duration),
        ("cooldown_duration", settings.cooldown_duration),
        ("flash_duration", settings.flash_duration),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if settings.relay_duration == 0:
        raise ValueError("relay_duration must be positive")

    # Hardware and display initialization must happen inside this child process,
    # never at module import time.
    import pygame
    from gpiozero import OutputDevice

    relay = None
    try:
        pygame.init()
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption(f"CapuchinAI: {test_name}")
        test.start(screen.get_size())
        test.draw(screen)
        pygame.display.flip()
        clock = pygame.time.Clock()

        relay = OutputDevice(relay_pin, active_high=False, initial_value=False)
        cycle = RewardCycle(relay, test, settings)
        accept_touches = True
        pending_test_name = None
        test_keys = {
            pygame.K_1: "blue",
            pygame.K_2: "reduced",
            pygame.K_3: "moving",
            pygame.K_KP1: "blue",
            pygame.K_KP2: "reduced",
            pygame.K_KP3: "moving",
        }

        LOGGER.info("Cognitive test %s is running: %s", test_name, settings)
        LOGGER.info("Switch tests with 1=blue, 2=reduced, 3=moving; Esc quits")

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

                if event.type == pygame.KEYDOWN and event.key in test_keys:
                    selected = test_keys[event.key]
                    # Selecting the current test cancels a pending switch.
                    # Repeated presses must not reset an already active trial.
                    pending_test_name = selected if selected != test_name else None
                    # Ignore touches from this batch: they refer to the screen
                    # displayed before the requested change.
                    accept_touches = False
                    continue

                # Use the existing touchscreen-as-mouse path only; also handling
                # FINGERDOWN can double-count a touch on SDL touchscreens.
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    now = time.monotonic()
                    if not accept_touches or not cycle.ready(now):
                        continue
                    with last_detection.get_lock():
                        detected_at = last_detection.value

                    detection_is_recent = (
                        detected_at > 0
                        and 0 <= now - detected_at <= detection_timeout
                    )
                    if detection_is_recent:
                        if test.on_touch(event.pos) and not stop_event.is_set():
                            cycle.start(time.monotonic())
                    else:
                        LOGGER.info(
                            "Touch ignored: no detection in the last %.1f seconds",
                            detection_timeout,
                        )

            if stop_event.is_set():
                break

            # Drain this frame's events before advancing to a new trial, so a
            # touch queued during feedback/cooldown cannot answer the new trial.
            now = time.monotonic()
            cycle.update(now)
            if pending_test_name is not None and cycle.phase == "ready":
                try:
                    next_test = make_test(
                        pending_test_name,
                        target_size if pending_test_name != "blue" else None,
                    )
                    next_test.start(screen.get_size())
                except ValueError as error:
                    # For example, the requested target may not fit this display.
                    LOGGER.warning("Cannot switch to %s: %s", pending_test_name, error)
                else:
                    test = next_test
                    test_name = pending_test_name
                    settings = replace(test.settings, **overrides)
                    # Keep the same cycle so switching cannot erase a cooldown.
                    cycle.test = test
                    cycle.settings = settings
                    pygame.display.set_caption(f"CapuchinAI: {test_name}")
                    LOGGER.info("Switched cognitive test to %s: %s", test_name, settings)
                pending_test_name = None

            if cycle.phase == "ready":
                test.update(now)
            if cycle.phase == "feedback":
                screen.fill(WHITE)
            elif now < cycle.cooldown_until:
                screen.fill(BLACK)
            else:
                test.draw(screen)
            pygame.display.flip()
            accept_touches = cycle.ready(now)
            clock.tick(60)
    finally:
        try:
            if relay is not None:
                try:
                    relay.off()
                finally:
                    relay.close()
        finally:
            pygame.quit()
