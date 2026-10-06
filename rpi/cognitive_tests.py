"""
A generic way to implement pygame tests, independent of the back-end.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
import pygame

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)


@dataclass(frozen=True)
class TestSettings:
    relay_duration: float = 0.5
    cooldown_duration: float = 0.0
    flash_duration: float = 0.0


class CognitiveTest(ABC):
    settings = TestSettings()

    @abstractmethod
    def start(self, screen_size: tuple[int, int]) -> None:
        """Initialize the first trial after the display has opened."""

    @abstractmethod
    def draw(self, screen: pygame.Surface) -> None:
        """Draw the task."""

    @abstractmethod
    def on_touch(self, position: tuple[int, int]) -> bool:
        """Interpret an eligible touch. Return True to request a reward."""

    def on_reward(self) -> None:
        """Called once after the relay pulse and optional success flash finish."""

    def update(self, now: float) -> None:
        """
        Optional per-frame update using monotonic time (also during cooldown).
        Not called during the relay pulse or success flash.
        """


class BlueScreenTest(CognitiveTest):
    def start(self, screen_size: tuple[int, int]) -> None:
        self.screen_size = screen_size

    def draw(self, screen: pygame.Surface) -> None:
        screen.fill(BLUE)

    def on_touch(self, position: tuple[int, int]) -> bool:
        x, y = position
        width, height = self.screen_size
        return 0 <= x < width and 0 <= y < height


class ReducedTargetTest(CognitiveTest):
    settings = TestSettings(relay_duration=0.4, cooldown_duration=4.0)

    def __init__(self, target_size: tuple[int, int] = (300, 300)) -> None:
        self.target_size = target_size

    def start(self, screen_size: tuple[int, int]) -> None:
        width, height = screen_size
        target_width, target_height = self.target_size
        if not (0 < target_width <= width and 0 < target_height <= height):
            raise ValueError(
                f"Target {self.target_size} must fit inside display {screen_size}"
            )
        self.screen_size = screen_size
        self.target = pygame.Rect(
            (width - target_width) // 2,
            (height - target_height) // 2,
            target_width,
            target_height,
        )

    def draw(self, screen: pygame.Surface) -> None:
        screen.fill(BLACK)
        pygame.draw.rect(screen, BLUE, self.target)

    def on_touch(self, position: tuple[int, int]) -> bool:
        return self.target.collidepoint(position)


class MovingTargetTest(ReducedTargetTest):
    """Choose a random position initially and after each completed reward."""

    settings = TestSettings(
        relay_duration=0.5, cooldown_duration=4.0, flash_duration=0.5
    )

    def __init__(self, target_size: tuple[int, int] = (250, 250)) -> None:
        super().__init__(target_size)

    def start(self, screen_size: tuple[int, int]) -> None:
        super().start(screen_size)
        self._move_target()

    def on_reward(self) -> None:
        self._move_target()

    def _move_target(self) -> None:
        width, height = self.screen_size
        self.target.topleft = (
            random.randint(0, width - self.target.width),
            random.randint(0, height - self.target.height),
        )


# Add a new task here to make it available through --test in the launcher.
TEST_TYPES = {
    "blue": BlueScreenTest,
    "reduced": ReducedTargetTest,
    "moving": MovingTargetTest,
}


def make_test(name: str, target_size: tuple[int, int] | None = None) -> CognitiveTest:
    try:
        test_type = TEST_TYPES[name]
    except KeyError:
        raise ValueError(f"Unknown cognitive test: {name!r}") from None
    if target_size is not None:
        if not issubclass(test_type, ReducedTargetTest):
            raise ValueError("--target-size is only supported by target tests")
        return test_type(target_size=target_size)
    return test_type()
