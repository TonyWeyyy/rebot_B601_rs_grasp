"""Small UI/FPS helpers shared across live OpenCV preview scripts."""

from __future__ import annotations

import time


class FpsMeter:
    """Sliding 1-second FPS counter."""

    def __init__(self) -> None:
        self._counter = 0
        self._timer = time.perf_counter()
        self.fps = 0.0

    def tick(self) -> float:
        self._counter += 1
        now = time.perf_counter()
        elapsed = now - self._timer
        if elapsed >= 1.0:
            self.fps = self._counter / elapsed
            self._counter = 0
            self._timer = now
        return self.fps
