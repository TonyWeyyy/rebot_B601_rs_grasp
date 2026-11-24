"""Shared 6D grasp data structures and estimator-agnostic helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from utils.transforms import (
    graspnet_rotation_to_rebot_tcp_rotation,
    transform_grasp_pose_to_base_with_retreat,
)
from utils.yolo_utils import YoloDetection


@dataclass
class GraspFrameResult:
    """Unified output of a single RGB-D 6D grasp inference call (AnyGrasp or GraspNet)."""
    grasps: Any
    pre_bbox_grasps: Any
    bbox_grasps: Any
    best: Optional[Any]
    status: str
    target_status: str
    detections: list[YoloDetection]
    selected_target: Optional[YoloDetection]
    o3d_cloud: Any
    raw_cloud: np.ndarray
    already_nms: bool = False


def copy_grasp_group(grasps: Any) -> Any:
    """Shallow-copy a graspnetAPI.GraspGroup (or compatible stub)."""
    cls = type(grasps)
    arr = grasps.grasp_group_array
    return cls(arr.copy() if hasattr(arr, "copy") else list(arr))


def rank_grasps(grasps: Any, *, run_nms: bool = True) -> Any:
    """Return a copy of ``grasps`` optionally NMS-filtered and sorted by descending score."""
    ranked = copy_grasp_group(grasps)
    if len(ranked) == 0:
        return ranked
    if run_nms:
        try:
            ranked = ranked.nms()
        except Exception as exc:
            print(f"[WARN] Grasp NMS skipped: {exc}")
    sorted_group = ranked.sort_by_score()
    return sorted_group if sorted_group is not None else ranked


def visualization_grasps(result: GraspFrameResult, mode: str) -> Any:
    """Return the grasp subset selected for Open3D diagnostics."""
    if mode == "pre-bbox":
        return result.pre_bbox_grasps
    if mode == "bbox":
        return result.bbox_grasps
    return result.grasps


def grasp_to_base_poses(
    grasp: Any,
    T_cam2base: np.ndarray,
    pregrasp_offset_m: float,
    retreat_offset_m: float,
    insertion_depth_m: float = 0.0,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Convert a GraspNet/AnyGrasp candidate to base-frame (grasp, pregrasp, retreat) 6D poses."""
    return transform_grasp_pose_to_base_with_retreat(
        np.asarray(grasp.translation, dtype=np.float64),
        graspnet_rotation_to_rebot_tcp_rotation(grasp.rotation_matrix),
        T_cam2base,
        pregrasp_offset_m,
        retreat_offset_m,
        insertion_depth_m,
    )
