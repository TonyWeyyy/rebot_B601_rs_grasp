from __future__ import annotations

from pathlib import Path

from .base import CameraDriver, CameraFrameError, fuse_depth_frames
from .orbbec_gemini2 import OrbbecGemini2
from .realsense import RealsenseCamera

__all__ = [
    "CameraDriver",
    "CameraFrameError",
    "OrbbecGemini2",
    "RealsenseCamera",
    "fuse_depth_frames",
    "make_camera",
]


def make_camera(cfg: dict) -> CameraDriver:
    """Create a camera driver from config/default.yaml."""
    cam_cfg = cfg.get("camera", {})
    cam_type = str(cam_cfg.get("type", "")).lower()
    w = int(cam_cfg.get("color_width", 1280))
    h = int(cam_cfg.get("color_height", 720))
    fps = int(cam_cfg.get("fps", 30))
    serial = cam_cfg.get("serial")
    serial_str = str(serial).strip() if serial is not None and str(serial).strip() else None

    _root = Path(__file__).resolve().parent.parent.parent
    calib_dir = str(_root / "config" / "calibration" / cam_type)

    if "orbbec" in cam_type:
        return OrbbecGemini2(w, h, fps, calib_dir=calib_dir, serial=serial_str)
    if "realsense" in cam_type:
        return RealsenseCamera(w, h, fps, calib_dir=calib_dir, serial=serial_str)
    raise ValueError(
        f"Unsupported camera type: {cam_type!r}\n"
        f"Set camera.type in config/default.yaml to:\n"
        f"  orbbec_gemini2 | realsense_d435i | realsense_d405"
    )
