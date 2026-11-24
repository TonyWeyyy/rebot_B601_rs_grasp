"""AnyGrasp SDK inference and safety helpers for the grasp pipeline.

The detector returns graspnetAPI.GraspGroup, so downstream TCP conversion, IK
checks, grasp execution, and gripper control remain shared with GraspNet.
"""

from __future__ import annotations

import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from utils.grasp_types import GraspFrameResult, copy_grasp_group
from utils.pointcloud import (
    bbox_region_mask,
    build_point_cloud,
    mask_region_mask,
    target_region_mask,
    workspace_mask,
)
from utils.target_selector import select_target, target_status_text
from utils.yolo_utils import YoloDetection, detect_objects

try:
    from graspnetAPI import Grasp, GraspGroup
except ImportError:  # Allow offline geometry/safety imports without graspnetAPI installed.
    Grasp = Any  # type: ignore[assignment,misc]
    GraspGroup = None  # type: ignore[assignment,misc]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANYGRASP_SDK_ROOT = PROJECT_ROOT / "sdk" / "anygrasp_sdk"

# Backward-compatible alias.
AnyGraspFrameResult = GraspFrameResult

__all__ = [
    "ANYGRASP_SDK_ROOT",
    "AnyGraspFrameResult",
    "GraspFrameResult",
    "PROJECT_ROOT",
    "bbox_region_mask",
    "build_detector",
    "build_point_cloud",
    "check_grasp_base_geometry",
    "infer_frame",
    "infer_grasps",
    "mask_region_mask",
    "resolve_anygrasp_paths",
    "select_target",
    "target_region_mask",
    "target_status_text",
    "workspace_mask",
]


def _make_empty_grasp_group():
    if GraspGroup is None:
        from graspnetAPI import GraspGroup as _GraspGroup
        return _GraspGroup()
    return GraspGroup()


def _empty_result(
    status: str,
    target_status: str,
    detections=None,
    *,
    build_o3d: bool = False,
) -> GraspFrameResult:
    empty = _make_empty_grasp_group()
    empty_cloud = None
    if build_o3d:
        try:
            import open3d as o3d
            empty_cloud = o3d.geometry.PointCloud()
        except ImportError:
            empty_cloud = None
    return GraspFrameResult(
        grasps=empty,
        pre_bbox_grasps=_make_empty_grasp_group(),
        bbox_grasps=_make_empty_grasp_group(),
        best=None,
        status=status,
        target_status=target_status,
        detections=list(detections or []),
        selected_target=None,
        o3d_cloud=empty_cloud,
        raw_cloud=np.empty((0, 3), dtype=np.float32),
        already_nms=True,
    )


def resolve_anygrasp_paths(
    cfg: dict[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
    checkpoint_override: str | None = None,
) -> tuple[Path, Path]:
    sdk_root = Path(str(cfg.get("sdk_root", ANYGRASP_SDK_ROOT))).expanduser()
    if not sdk_root.is_absolute():
        sdk_root = project_root / sdk_root
    sdk_root = sdk_root.resolve()
    detection_dir = sdk_root if sdk_root.name == "grasp_detection" else sdk_root / "grasp_detection"

    checkpoint_value = checkpoint_override or cfg.get("checkpoint", "log/checkpoint_detection.tar")
    checkpoint = Path(str(checkpoint_value)).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = detection_dir / checkpoint if len(checkpoint.parts) <= 2 else project_root / checkpoint
    return detection_dir.resolve(), checkpoint.resolve()


def build_detector(cfg: dict[str, Any], checkpoint_override: str | None = None):
    """Load the current AnyGrasp SDK; no camera or robot connection is created."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("AnyGrasp requires CUDA, but torch.cuda.is_available() is False")

    detection_dir, checkpoint = resolve_anygrasp_paths(cfg, checkpoint_override=checkpoint_override)
    gsnet_so = detection_dir / "gsnet.so"
    license_dir = detection_dir / "license"
    for path in (gsnet_so, license_dir, checkpoint):
        if not path.exists():
            raise FileNotFoundError(f"AnyGrasp required path missing: {path}")

    max_width = float(cfg.get("max_gripper_width", 0.09))
    gripper_height = float(cfg.get("gripper_height", 0.03))
    if not 0.0 < max_width <= 0.1:
        raise ValueError("AnyGrasp max_gripper_width must be in (0, 0.1] meters")
    if not 0.0 < gripper_height <= 0.2:
        raise ValueError("AnyGrasp gripper_height must be in (0, 0.2] meters")
    if not bool(cfg.get("collision_detection", True)):
        raise ValueError("collision_detection=false is forbidden in this robot project")

    detection_str = str(detection_dir)
    if detection_str in sys.path:
        sys.path.remove(detection_str)
    sys.path.insert(0, detection_str)
    from gsnet import create_detector

    detector_cfg = Namespace(
        checkpoint_path=str(checkpoint),
        max_gripper_width=max_width,
        gripper_height=gripper_height,
    )
    detector = create_detector(detector_cfg)
    if detector is None:
        raise RuntimeError(
            "AnyGrasp create_detector() returned None; check this machine's license, "
            "feature id, checkpoint, and matching gsnet.so"
        )
    print(f"[AnyGrasp] detector loaded: {checkpoint}")
    return detector


def infer_grasps(
    detector,
    points: np.ndarray,
    region: np.ndarray,
    cfg: dict[str, Any],
    *,
    preferred_approach_cam=None,
):
    min_region_points = int(cfg.get("min_region_points", 300))
    if int(region.sum()) < min_region_points:
        raise RuntimeError(f"Only {int(region.sum())} region points; need at least {min_region_points}")

    if preferred_approach_cam is None:
        preferred_approach_cam = cfg.get("preferred_approach_cam", [0.0, 0.0, 1.0])
    approach = np.asarray(preferred_approach_cam, dtype=np.float32)
    if approach.shape != (3,) or not np.isfinite(approach).all() or np.linalg.norm(approach) < 1e-6:
        raise ValueError("preferred_approach_cam must be a finite non-zero 3D vector")

    dense = bool(cfg.get("dense_grasp", False))
    grasps = detector.get_grasp(points.astype(np.float32), {
        "dense_grasp": dense,
        "collision_detection": True,
        "region_steering": np.asarray(region, dtype=bool),
        "approach_steering": approach,
        "approach_thresh": np.deg2rad(float(cfg.get("approach_thresh_deg", 45.0))),
    })
    if grasps is None or len(grasps) == 0:
        return _make_empty_grasp_group()
    if not dense:
        grasps = grasps.nms()
    sorted_grasps = grasps.sort_by_score()
    if sorted_grasps is not None:
        grasps = sorted_grasps
    threshold = float(cfg.get("score_threshold", 0.0))
    grasps = grasps[np.asarray(grasps.scores) > threshold]
    return grasps


def infer_frame(
    detector,
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    *,
    cfg: dict[str, Any],
    min_depth: float,
    max_depth: float,
    yolo_model=None,
    yolo_opts=None,
    detections: Optional[list[YoloDetection]] = None,
    target_class: Optional[str] = None,
    target_margin_px: int = 12,
    target_expand_ratio: float = 1.35,
    max_grasp_width_m: Optional[float] = 0.09,
    preferred_approach_cam=None,
    build_o3d: bool = True,
) -> GraspFrameResult:
    detections_provided = detections is not None
    detections = list(detections or [])
    selected = None
    if yolo_model is not None:
        if not detections_provided:
            _, detections = detect_objects(yolo_model, color_bgr, yolo_opts or {})
        selected = select_target(detections, target_class)
        if selected is None:
            target_status = target_status_text(selected, detections, target_class)
            return _empty_result(
                f"inference skipped: {target_status}",
                target_status,
                detections,
                build_o3d=build_o3d,
            )

    tic = time.time()
    cloud, points, _ = build_point_cloud(
        color_bgr,
        depth_mm,
        K,
        min_depth_m=min_depth,
        max_depth_m=max_depth,
        max_input_points=int(cfg.get("max_input_points", 100000)),
        point_seed=cfg.get("point_seed", 0),
        build_o3d=build_o3d,
    )
    ws_mask = workspace_mask(points, cfg.get("lims", [-0.3, 0.3, -0.3, 0.3, 0.05, 1.0]))
    region_mode = "full"
    if selected is not None:
        region, region_mode = target_region_mask(
            points,
            selected,
            K,
            image_shape=color_bgr.shape[:2],
            base_mask=ws_mask,
            use_mask=bool(cfg.get("use_mask_steering", True)),
            mask_dilate_px=int(cfg.get("mask_dilate_px", 10)),
            margin_px=target_margin_px,
            expand_ratio=target_expand_ratio,
            min_points=int(cfg.get("min_region_points", 300)),
        )
    else:
        region = ws_mask

    grasps = infer_grasps(
        detector, points, region, cfg, preferred_approach_cam=preferred_approach_cam
    )
    pre_bbox = copy_grasp_group(grasps)
    if max_grasp_width_m is not None and len(grasps) > 0:
        grasps = grasps[np.asarray(grasps.widths) <= float(max_grasp_width_m)]
    bbox_grasps = copy_grasp_group(grasps)
    best = grasps[0] if len(grasps) else None
    elapsed = time.time() - tic
    target_status = (
        target_status_text(selected, detections, target_class)
        if yolo_model is not None else "YOLO disabled: full-scene AnyGrasp"
    )
    status = (
        f"AnyGrasp grasps={len(grasps)} region({region_mode})={int(region.sum())} "
        f"cloud={len(points)} inference={elapsed:.2f}s"
    )
    return GraspFrameResult(
        grasps=grasps,
        pre_bbox_grasps=pre_bbox,
        bbox_grasps=bbox_grasps,
        best=best,
        status=status,
        target_status=target_status,
        detections=detections,
        selected_target=selected,
        o3d_cloud=cloud,
        raw_cloud=points,
        already_nms=not bool(cfg.get("dense_grasp", False)),
    )


def check_grasp_base_geometry(
    grasp: Any,
    T_cam2base: np.ndarray,
    pre6d: tuple[float, ...],
    grasp6d: tuple[float, ...],
    retreat6d: tuple[float, ...],
    safety: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Geometric safety checks in base frame before IK verification.

    This does not prove full-arm collision safety. It rejects unsafe geometry and
    leaves reachability to the IK planner and execution controller.
    """
    reasons: list[str] = []
    R = np.asarray(grasp.rotation_matrix, dtype=np.float64)
    T = np.asarray(T_cam2base, dtype=np.float64)
    if (
        R.shape != (3, 3)
        or not np.isfinite(R).all()
        or not np.allclose(R.T @ R, np.eye(3), atol=1e-4)
        or not np.isclose(np.linalg.det(R), 1.0, atol=1e-4)
    ):
        reasons.append("invalid grasp rotation")
        return reasons, {}
    if T.shape != (4, 4) or not np.isfinite(T).all():
        reasons.append("invalid T_cam2base")
        return reasons, {}

    width = float(grasp.width)
    max_width = float(safety.get("max_gripper_width", 0.09))
    margin = float(safety.get("width_margin_m", 0.003))
    if not 0.0 < width <= max_width - margin:
        reasons.append(f"width={width:.3f}m exceeds safe limit={max_width - margin:.3f}m")

    approach = T[:3, :3] @ R[:, 0]
    approach /= max(np.linalg.norm(approach), 1e-12)
    downward = np.asarray(safety.get("downward_base", [0.0, 0.0, -1.0]), dtype=np.float64)
    downward /= max(np.linalg.norm(downward), 1e-12)
    tilt = float(np.rad2deg(np.arccos(np.clip(np.dot(approach, downward), -1.0, 1.0))))
    max_tilt = float(safety.get("max_tilt_deg", 45.0))
    if tilt > max_tilt:
        reasons.append(f"approach tilt={tilt:.1f}deg exceeds {max_tilt:.1f}deg")

    center_cam = np.asarray(grasp.translation, dtype=np.float64)
    tip_cam = center_cam + float(grasp.depth) * R[:, 0]
    center_base = T[:3, :3] @ center_cam + T[:3, 3]
    tip_base = T[:3, :3] @ tip_cam + T[:3, 3]
    positions = {
        "pre": np.asarray(pre6d[:3], dtype=np.float64),
        "grasp": np.asarray(grasp6d[:3], dtype=np.float64),
        "retreat": np.asarray(retreat6d[:3], dtype=np.float64),
        "tip": tip_base,
        "center": center_base,
    }
    if any(not np.isfinite(p).all() for p in positions.values()):
        reasons.append("non-finite base position")
        return reasons, {"tilt_deg": tilt}

    min_z = float(safety.get("min_base_z_m", 0.01))
    for name in ("pre", "grasp", "retreat", "tip"):
        if positions[name][2] < min_z:
            reasons.append(f"{name} z={positions[name][2]:.3f}m below floor={min_z:.3f}m")

    lower = np.asarray(safety.get("base_workspace_min", [-np.inf] * 3), dtype=np.float64)
    upper = np.asarray(safety.get("base_workspace_max", [np.inf] * 3), dtype=np.float64)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
        reasons.append("invalid base workspace config")
    else:
        for name, position in positions.items():
            if np.any(position < lower) or np.any(position > upper):
                reasons.append(f"{name} position outside base workspace")

    details = {"tilt_deg": tilt, "tip_base": tip_base, "center_base": center_base, "positions": positions}
    return reasons, details
