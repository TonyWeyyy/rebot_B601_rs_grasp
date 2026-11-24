"""Pure helpers for target-to-container placement.

No camera, SDK, CAN, or robot connection is created in this module.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from utils.target_selector import select_container
from utils.yolo_utils import YoloDetection

__all__ = [
    "build_place_poses",
    "detection_center_cam",
    "select_container",
    "validate_base_pose",
    "validate_target_container_separation",
]


def detection_center_cam(
    detection: YoloDetection,
    depth_mm: np.ndarray,
    K: np.ndarray,
    *,
    min_depth_m: float = 0.05,
    max_depth_m: float = 1.0,
    depth_quantile: float = 0.5,
    min_depth_pixels: int = 30,
    use_mask_centroid: bool = False,
) -> np.ndarray:
    """Estimate a detection's metric center from masked depth and mask centroid or bbox center.

    When ``use_mask_centroid=True``, the 2D center ``(u, v)`` is computed as the
    mean pixel coordinate of valid masked depth pixels (plus 0.5-pixel box
    convention alignment only when falling back to bbox center), which reduces
    lateral shift for angled or partially occluded containers.
    """
    depth = np.asarray(depth_mm)
    if depth.ndim != 2:
        raise ValueError("depth_mm must be a single-channel image")
    mask = np.asarray(detection.mask)
    if mask.shape != depth.shape:
        raise ValueError(f"Detection mask {mask.shape} does not match depth {depth.shape}")

    valid = (
        (mask > 0)
        & np.isfinite(depth)
        & (depth > float(min_depth_m) * 1000.0)
        & (depth < float(max_depth_m) * 1000.0)
    )
    count = int(valid.sum())
    if count < int(min_depth_pixels):
        raise ValueError(f"Container has only {count} valid depth pixels; need {min_depth_pixels}")

    z = float(np.quantile(depth[valid].astype(np.float32) / 1000.0, float(depth_quantile)))
    if use_mask_centroid:
        vs, us = np.nonzero(valid)
        u = float(np.mean(us))
        v = float(np.mean(vs))
    else:
        x1, y1, x2, y2 = [float(value) for value in detection.bbox_xyxy]
        u = 0.5 * (x1 + x2)
        v = 0.5 * (y1 + y2)

    K = np.asarray(K, dtype=np.float64)
    x = (u - float(K[0, 2])) * z / float(K[0, 0])
    y = (v - float(K[1, 2])) * z / float(K[1, 1])
    point = np.array([x, y, z], dtype=np.float64)
    if not np.isfinite(point).all():
        raise ValueError("Container center is non-finite")
    return point


def build_place_poses(
    container_center_cam: np.ndarray,
    T_cam2base: np.ndarray,
    place_cfg: dict[str, Any],
) -> tuple[tuple[float, ...], tuple[float, ...], np.ndarray]:
    """Return place, pre-place and measured container base center.

    The release XY comes from the container. Z is an independently configured
    TCP release height, not the noisy container-bottom depth.
    """
    center_cam = np.asarray(container_center_cam, dtype=np.float64)
    T = np.asarray(T_cam2base, dtype=np.float64)
    if center_cam.shape != (3,) or T.shape != (4, 4):
        raise ValueError("Invalid container center or camera-to-base transform")
    center_base_h = T @ np.append(center_cam, 1.0)
    center_base = center_base_h[:3]

    place_z = float(place_cfg.get("height_m", 0.15))
    pre_offset = float(place_cfg.get("pre_place_offset_m", 0.08))
    if pre_offset <= 0.0:
        raise ValueError("place.pre_place_offset_m must be positive")
    default_rpy = [place_cfg.get("roll", 0.0), place_cfg.get("pitch", 1.2), place_cfg.get("yaw", 0.0)]
    rpy = np.asarray(place_cfg.get("orientation_rpy", default_rpy), dtype=np.float64)
    if rpy.shape != (3,) or not np.isfinite(rpy).all():
        raise ValueError("place.orientation_rpy must contain three finite angles")
    roll, pitch, yaw = [float(value) for value in rpy]

    place = (float(center_base[0]), float(center_base[1]), place_z, roll, pitch, yaw)
    pre_place = (float(center_base[0]), float(center_base[1]), place_z + pre_offset, roll, pitch, yaw)
    return place, pre_place, center_base


def validate_base_pose(
    pose: tuple[float, ...],
    safety: dict[str, Any],
    *,
    min_base_z_m: float,
    name: str,
) -> None:
    """Raise instead of executing a non-finite, low, or out-of-workspace pose."""
    values = np.asarray(pose, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError(f"{name} is not a finite 6D pose")
    xyz = values[:3]
    if xyz[2] < float(min_base_z_m):
        raise ValueError(f"{name} z={xyz[2]:.3f}m below floor={min_base_z_m:.3f}m")

    lower = np.asarray(safety.get("base_workspace_min", [-np.inf] * 3), dtype=np.float64)
    upper = np.asarray(safety.get("base_workspace_max", [np.inf] * 3), dtype=np.float64)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
        raise ValueError("Invalid base workspace config")
    if np.any(xyz < lower) or np.any(xyz > upper):
        raise ValueError(
            f"{name} xyz={np.round(xyz, 4).tolist()} outside "
            f"workspace min={lower.tolist()} max={upper.tolist()}"
        )


def validate_target_container_separation(
    grasp6d: tuple[float, ...],
    place6d: tuple[float, ...],
    *,
    min_horizontal_distance_m: float = 0.05,
) -> None:
    """Prevent target/container duplicates from becoming a zero-transfer task."""
    delta = np.asarray(place6d[:2], dtype=np.float64) - np.asarray(grasp6d[:2], dtype=np.float64)
    distance = float(np.linalg.norm(delta))
    if distance < float(min_horizontal_distance_m):
        raise ValueError(
            f"Target and container XY are only {distance:.3f}m apart; "
            "check duplicate or incorrect YOLO detections"
        )
