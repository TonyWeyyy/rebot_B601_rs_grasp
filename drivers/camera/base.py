"""Shared RGB-D camera interface.

CameraDriver:
  open() / close(): start or stop camera streams.
  get_frame(): return one BGR color frame and one depth-mm frame.
  get_fused_frame(): capture multiple frames and return median-filtered depth.
  K / D: camera matrix and distortion coefficients.
  warm_up(): drop early frames before inference or calibration.
  setup_aruco() / detect_aruco() / draw_aruco() / detect_and_draw_aruco():
    optional ArUco helpers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class CameraFrameError(RuntimeError):
    """Raised after repeated camera frame acquisition failures."""


def fuse_depth_frames(depth_frames: list[np.ndarray]) -> np.ndarray:
    """Compute pixel-wise median over non-zero uint16 depth frames."""
    if not depth_frames:
        raise ValueError("depth_frames must not be empty")
    if len(depth_frames) == 1:
        return depth_frames[0].copy()

    stack = np.stack([np.asarray(d, dtype=np.float32) for d in depth_frames], axis=0)
    valid = np.isfinite(stack) & (stack > 0.0)
    has_valid = np.any(valid, axis=0)
    result = np.zeros(depth_frames[0].shape, dtype=np.uint16)
    if not np.any(has_valid):
        return result

    valid_cols = np.where(valid[:, has_valid], stack[:, has_valid], np.nan)
    med = np.nanmedian(valid_cols, axis=0)
    result[has_valid] = np.clip(np.rint(med), 0.0, float(np.iinfo(np.uint16).max)).astype(np.uint16)
    return result


class CameraDriver(ABC):
    """Common color and depth camera interface."""

    _FRAME_FAIL_WARN = 10
    _FRAME_FAIL_LIMIT = 60

    def _reset_frame_failures(self) -> None:
        self._frame_failures = 0

    def _record_frame_failure(self, reason: str) -> None:
        count = int(getattr(self, "_frame_failures", 0)) + 1
        self._frame_failures = count
        if count == self._FRAME_FAIL_WARN:
            print(f"[Camera] {count} consecutive frame failures: {reason}")
        if count >= self._FRAME_FAIL_LIMIT:
            raise CameraFrameError(
                f"Camera failed for {count} consecutive frames: {reason}\n"
                "  Check connection, USB permissions, or other camera clients."
            )

    # Lifecycle

    @abstractmethod
    def open(self) -> None:
        """Open camera streams."""

    @abstractmethod
    def close(self) -> None:
        """Stop streams and release resources."""

    # Frames

    @abstractmethod
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return one color/depth frame pair.

        Returns:
            color_bgr: uint8 BGR image, or None.
            depth_mm: uint16 depth image in millimeters, or None.
        """

    def get_fused_frame(
        self,
        n_frames: int = 3,
        max_attempts: Optional[int] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture ``n_frames`` RGB-D frames and return the latest color + median depth."""
        n_frames = max(1, int(n_frames))
        if n_frames == 1:
            return self.get_frame()

        attempts = max(n_frames * 2, int(max_attempts)) if max_attempts is not None else n_frames * 3
        last_color: Optional[np.ndarray] = None
        depths: list[np.ndarray] = []
        for _ in range(attempts):
            color_bgr, depth_mm = self.get_frame()
            if color_bgr is None or depth_mm is None:
                continue
            last_color = color_bgr
            if not depths or depth_mm.shape == depths[0].shape:
                depths.append(depth_mm)
            if len(depths) >= n_frames:
                break

        if last_color is None or not depths:
            return None, None
        return last_color, fuse_depth_frames(depths)

    # Intrinsics

    @property
    @abstractmethod
    def K(self) -> np.ndarray:
        """Camera matrix, shape (3, 3), float64."""

    @property
    @abstractmethod
    def D(self) -> np.ndarray:
        """Distortion coefficients, shape (1, N), float64."""

    # Helpers

    def warm_up(self, n_frames: int = 20) -> None:
        """Drop early frames while exposure settles."""
        for _ in range(n_frames):
            self.get_frame()

    def setup_aruco(
        self,
        marker_length_m: float,
        dict_id: int = 0,
        target_marker_id: Optional[int] = None,
    ) -> None:
        """Create an ArUco detector using this camera calibration.

        Args:
            marker_length_m: marker side length in meters.
            dict_id: cv2.aruco dictionary id.
            target_marker_id: detect only this id; None selects the nearest one.
        """
        from calibration.aruco_pose import ArUcoDetector
        self._aruco = ArUcoDetector(marker_length_m, dict_id, target_marker_id)

    def detect_aruco(self, bgr: np.ndarray):
        """Detect an ArUco marker. Call setup_aruco first."""
        return self._aruco.detect(bgr, self.K, self.D)

    def draw_aruco(self, bgr: np.ndarray) -> np.ndarray:
        """Draw detected ArUco markers. Call setup_aruco first."""
        return self._aruco.draw_detected(bgr, self.K, self.D)

    def detect_and_draw_aruco(self, bgr: np.ndarray, axis_length: float = 0.03):
        """Detect target ArUco marker and draw all detected markers in a single pass."""
        return self._aruco.detect_and_draw(bgr, self.K, self.D, axis_length=axis_length)

    # Context manager

    def __enter__(self) -> "CameraDriver":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()
