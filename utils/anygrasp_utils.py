"""AnyGrasp SDK inference and safety helpers for the existing grasp pipeline.

The detector returns graspnetAPI.GraspGroup, so downstream TCP conversion, IK
checks, grasp execution, and gripper control remain shared with GraspNet.
"""

from __future__ import annotations

import sys
import time
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import open3d as o3d
import torch
from graspnetAPI import Grasp, GraspGroup

from utils.yolo_utils import YoloDetection, detect_objects

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANYGRASP_SDK_ROOT = PROJECT_ROOT / "sdk" / "anygrasp_sdk"


@dataclass
class AnyGraspFrameResult:
    grasps: GraspGroup
    pre_bbox_grasps: GraspGroup
    bbox_grasps: GraspGroup
    best: Optional[Grasp]
    status: str
    target_status: str
    detections: list[YoloDetection]
    selected_target: Optional[YoloDetection]
    o3d_cloud: o3d.geometry.PointCloud
    raw_cloud: np.ndarray


def _copy_grasps(grasps: GraspGroup) -> GraspGroup:
    return GraspGroup(grasps.grasp_group_array.copy())


def _empty_result(status: str, target_status: str, detections=None) -> AnyGraspFrameResult:
    empty = GraspGroup()
    return AnyGraspFrameResult(
        grasps=empty,
        pre_bbox_grasps=GraspGroup(),
        bbox_grasps=GraspGroup(),
        best=None,
        status=status,
        target_status=target_status,
        detections=list(detections or []),
        selected_target=None,
        o3d_cloud=o3d.geometry.PointCloud(),
        raw_cloud=np.empty((0, 3), dtype=np.float32),
    )


def resolve_anygrasp_paths(
    cfg: dict[str, Any], *, project_root: Path = PROJECT_ROOT, checkpoint_override: str | None = None
) -> tuple[Path, Path]:
    sdk_root = Path(str(cfg.get("sdk_root", ANYGRASP_SDK_ROOT))).expanduser()
    if not sdk_root.is_absolute():
        sdk_root = project_root / sdk_root
    sdk_root = sdk_root.resolve()
    detection_dir = sdk_root if sdk_root.name == "grasp_detection" else sdk_root / "grasp_detection"

    checkpoint_value = checkpoint_override or cfg.get("checkpoint", "log/checkpoint_detection.tar")
    checkpoint = Path(str(checkpoint_value)).expanduser()
    if not checkpoint.is_absolute():
        # 纯文件名和 log/... 都相对于 grasp_detection；其他相对路径相对于项目根目录。
        checkpoint = detection_dir / checkpoint if len(checkpoint.parts) <= 2 else project_root / checkpoint
    return detection_dir.resolve(), checkpoint.resolve()


def build_detector(cfg: dict[str, Any], checkpoint_override: str | None = None):
    """Load the current AnyGrasp SDK; no camera or robot connection is created."""
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


def build_point_cloud(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
    max_input_points: int = 100000,
    point_seed: Optional[int] = 0,
) -> tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray]:
    """Create a metric camera-frame cloud using the project's existing RGB-D convention."""
    if color_bgr is None or depth_mm is None:
        raise ValueError("color/depth frame is None")
    if color_bgr.shape[:2] != depth_mm.shape[:2]:
        depth_mm = cv2.resize(
            depth_mm, (color_bgr.shape[1], color_bgr.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    depth = np.asarray(depth_mm)
    mask = (
        np.isfinite(depth)
        & (depth > max(0.0, float(min_depth_m)) * 1000.0)
        & (depth < float(max_depth_m) * 1000.0)
    )
    if int(mask.sum()) < 100:
        raise RuntimeError("Too few valid depth pixels in the configured depth range")

    z = depth.astype(np.float32) / 1000.0
    h, w = depth.shape
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = (us - float(K[0, 2])) * z / float(K[0, 0])
    y = (vs - float(K[1, 2])) * z / float(K[1, 1])
    points = np.stack((x, y, z), axis=-1)[mask].astype(np.float32)
    colors = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)[mask] / 255.0

    if max_input_points > 0 and len(points) > int(max_input_points):
        rng = np.random.default_rng(point_seed)
        indices = rng.choice(len(points), int(max_input_points), replace=False)
        points = points[indices]
        colors = colors[indices]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return cloud, points, colors


def workspace_mask(points: np.ndarray, lims) -> np.ndarray:
    limits = np.asarray(lims, dtype=np.float32)
    if limits.shape != (6,) or not np.isfinite(limits).all():
        raise ValueError("anygrasp.lims must be [xmin,xmax,ymin,ymax,zmin,zmax]")
    x0, x1, y0, y1, z0, z1 = limits
    if not (x0 < x1 and y0 < y1 and z0 < z1):
        raise ValueError("anygrasp.lims min/max ordering is invalid")
    return (
        (points[:, 0] >= x0) & (points[:, 0] <= x1)
        & (points[:, 1] >= y0) & (points[:, 1] <= y1)
        & (points[:, 2] >= z0) & (points[:, 2] <= z1)
    )


def bbox_region_mask(
    points: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    K: np.ndarray,
    *,
    margin_px: int,
    expand_ratio: float,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Project cloud points into the selected YOLO bbox for region_steering."""
    z = points[:, 2]
    valid = z > 1e-6
    u = np.full(len(points), np.nan, dtype=np.float64)
    v = np.full(len(points), np.nan, dtype=np.float64)
    u[valid] = float(K[0, 0]) * points[valid, 0] / z[valid] + float(K[0, 2])
    v[valid] = float(K[1, 1]) * points[valid, 1] / z[valid] + float(K[1, 2])

    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    expand_ratio = max(1.0, float(expand_ratio))
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    half_w = 0.5 * max(1.0, x2 - x1) * expand_ratio
    half_h = 0.5 * max(1.0, y2 - y1) * expand_ratio
    h, w = image_shape
    x1 = np.clip(cx - half_w - margin_px, 0, w - 1)
    x2 = np.clip(cx + half_w + margin_px, 0, w - 1)
    y1 = np.clip(cy - half_h - margin_px, 0, h - 1)
    y2 = np.clip(cy + half_h + margin_px, 0, h - 1)
    return valid & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)


def infer_grasps(
    detector,
    points: np.ndarray,
    region: np.ndarray,
    cfg: dict[str, Any],
    *,
    preferred_approach_cam=None,
) -> GraspGroup:
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
        return GraspGroup()
    if not dense:
        grasps = grasps.nms()
    grasps = grasps.sort_by_score()
    threshold = float(cfg.get("score_threshold", 0.0))
    grasps = grasps[np.asarray(grasps.scores) > threshold]
    return grasps


def select_target(detections: list[YoloDetection], target_class: Optional[str]) -> Optional[YoloDetection]:
    if not detections:
        return None
    candidates = detections
    if target_class:
        target_norm = target_class.casefold()
        exact = [target for target in detections if target.class_name.casefold() == target_norm]
        contains = [target for target in detections if target_norm in target.class_name.casefold()]
        candidates = exact or contains
    if not candidates:
        return None
    return max(candidates, key=lambda target: target.conf)


def target_status_text(selected, detections, target_class) -> str:
    if selected is not None:
        return f"target={selected.class_name} {selected.conf:.2f} detections={len(detections)}"
    if target_class:
        return f"target={target_class} not found detections={len(detections)}"
    return f"target not found detections={len(detections)}"


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
) -> AnyGraspFrameResult:
    detections_provided = detections is not None
    detections = list(detections or [])
    selected = None
    if yolo_model is not None:
        if not detections_provided:
            _, detections = detect_objects(yolo_model, color_bgr, yolo_opts or {})
        selected = select_target(detections, target_class)
        if selected is None:
            target_status = target_status_text(selected, detections, target_class)
            return _empty_result(f"inference skipped: {target_status}", target_status, detections)

    tic = time.time()
    cloud, points, _ = build_point_cloud(
        color_bgr,
        depth_mm,
        K,
        min_depth_m=min_depth,
        max_depth_m=max_depth,
        max_input_points=int(cfg.get("max_input_points", 100000)),
        point_seed=cfg.get("point_seed", 0),
    )
    region = workspace_mask(points, cfg.get("lims", [-0.3, 0.3, -0.3, 0.3, 0.05, 1.0]))
    if selected is not None:
        region &= bbox_region_mask(
            points,
            selected.bbox_xyxy,
            K,
            margin_px=target_margin_px,
            expand_ratio=target_expand_ratio,
            image_shape=color_bgr.shape[:2],
        )

    grasps = infer_grasps(
        detector, points, region, cfg, preferred_approach_cam=preferred_approach_cam
    )
    pre_bbox = _copy_grasps(grasps)
    if max_grasp_width_m is not None and len(grasps) > 0:
        grasps = grasps[np.asarray(grasps.widths) <= float(max_grasp_width_m)]
    bbox_grasps = _copy_grasps(grasps)
    best = grasps[0] if len(grasps) else None
    elapsed = time.time() - tic
    target_status = (
        target_status_text(selected, detections, target_class)
        if yolo_model is not None else "YOLO disabled: full-scene AnyGrasp"
    )
    status = (
        f"AnyGrasp grasps={len(grasps)} region_points={int(region.sum())} "
        f"cloud={len(points)} inference={elapsed:.2f}s"
    )
    return AnyGraspFrameResult(
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
    )


def check_grasp_base_geometry(
    grasp: Grasp,
    T_cam2base: np.ndarray,
    pre6d: tuple[float, ...],
    grasp6d: tuple[float, ...],
    retreat6d: tuple[float, ...],
    safety: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Additional AnyGrasp-only checks before the existing IK checks.

    This does not prove full-arm collision safety. It rejects unsafe geometry and
    leaves reachability to the existing IK checker and execution controller.
    """
    reasons: list[str] = []
    R = np.asarray(grasp.rotation_matrix, dtype=np.float64)
    T = np.asarray(T_cam2base, dtype=np.float64)
    if R.shape != (3, 3) or not np.isfinite(R).all() or not np.allclose(R.T @ R, np.eye(3), atol=1e-4) \
            or not np.isclose(np.linalg.det(R), 1.0, atol=1e-4):
        reasons.append("invalid grasp rotation")
        return reasons, {}
    if T.shape != (4, 4) or not np.isfinite(T).all():
        reasons.append("invalid T_cam2base")
        return reasons, {}

    width = float(grasp.width)
    max_width = float(safety.get("max_gripper_width", 0.09))
    margin = float(safety.get("width_margin_m", 0.003))
    if not 0.0 < width <= max_width - margin:
        reasons.append(f"width={width:.3f}m exceeds safe limit={max_width-margin:.3f}m")

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
