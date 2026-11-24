"""GraspNet helpers that sit above camera frames and YOLO detections.

GraspNet/pointnet2 SDK modules are imported lazily on demand so importing shared
visualization or target helpers never triggers CUDA/pointnet2 dependencies.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

from utils.grasp_types import (
    GraspFrameResult,
    copy_grasp_group,
    grasp_to_base_poses,
    rank_grasps,
    visualization_grasps,
)
from utils.pointcloud import make_o3d_point_cloud, project_points_to_pixels
from utils.target_selector import select_target, selected_target_text, target_status_text
from utils.visualization import (
    DISPLAY_FLIP_X,
    Open3DGraspWindow,
    draw_best_grasp_projection,
    draw_detections_overlay,
    draw_status,
)
from utils.yolo_utils import YoloDetection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRASPNET_ROOT = PROJECT_ROOT / "sdk" / "graspnet-baseline"
DEFAULT_NUM_VIEW = 300
DEFAULT_VOXEL_SIZE = 0.01
DEFAULT_WARMUP_FRAMES = 20

# Backward-compatible alias.
GraspNetFrameResult = GraspFrameResult

__all__ = [
    "DEFAULT_NUM_VIEW",
    "DEFAULT_VOXEL_SIZE",
    "DEFAULT_WARMUP_FRAMES",
    "DISPLAY_FLIP_X",
    "GRASPNET_ROOT",
    "GraspFrameResult",
    "GraspNetFrameResult",
    "Open3DGraspWindow",
    "PROJECT_ROOT",
    "build_end_points",
    "build_net",
    "copy_grasp_group",
    "draw_best_grasp_projection",
    "draw_detections_overlay",
    "draw_status",
    "filter_grasps_by_bbox",
    "filter_grasps_by_width",
    "grasp_to_base_poses",
    "infer_frame",
    "infer_grasps",
    "prepare_graspnet_imports",
    "resolve_checkpoint_path",
    "select_best_grasp",
    "select_target",
    "selected_target_text",
    "target_status_text",
    "visualization_grasps",
]


def prepare_graspnet_imports(graspnet_root: Path = GRASPNET_ROOT) -> None:
    for subdir in ("models", "dataset", "utils", "pointnet2", "graspnetAPI"):
        path = str(graspnet_root / subdir)
        if path not in sys.path:
            sys.path.insert(0, path)
    root = str(graspnet_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_checkpoint_path(
    checkpoint: str,
    *,
    project_root: Path = PROJECT_ROOT,
    graspnet_root: Path = GRASPNET_ROOT,
) -> Path:
    checkpoint_path = Path(str(checkpoint)).expanduser()
    if checkpoint_path.is_absolute():
        return checkpoint_path
    if len(checkpoint_path.parts) > 1:
        return project_root / checkpoint_path
    return graspnet_root / "checkpoints" / checkpoint_path


def build_net(checkpoint_path: str | Path, num_view: int = DEFAULT_NUM_VIEW):
    import torch

    prepare_graspnet_imports()
    from graspnet import GraspNet

    checkpoint_path = resolve_checkpoint_path(str(checkpoint_path))
    if not torch.cuda.is_available():
        raise RuntimeError("GraspNet pointnet2 operators require CUDA, but torch.cuda is unavailable.")

    net = GraspNet(
        input_feature_dim=0,
        num_view=num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    device = torch.device("cuda:0")
    net.to(device)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    print(f"Loaded checkpoint {checkpoint_path} (epoch: {checkpoint['epoch']})")
    return net


def build_end_points(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    num_point: int,
    min_depth_m: float,
    max_depth_m: float,
    *,
    build_o3d: bool = True,
) -> tuple[dict, Any, np.ndarray]:
    import torch

    prepare_graspnet_imports()
    from data_utils import CameraInfo, create_point_cloud_from_depth_image

    if color_bgr.shape[:2] != depth_mm.shape[:2]:
        depth_mm = cv2.resize(depth_mm, (color_bgr.shape[1], color_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    min_mm = int(max(0.0, min_depth_m) * 1000.0)
    max_mm = int(max_depth_m * 1000.0)
    depth = depth_mm.astype(np.uint16, copy=False)
    mask = (depth > min_mm) & (depth < max_mm)
    if int(mask.sum()) == 0:
        raise RuntimeError("No valid depth pixels in the configured depth range.")

    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = depth.shape
    camera = CameraInfo(w, h, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), 1000.0)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    cloud_masked = cloud[mask]
    color_masked = color_rgb[mask]

    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs = np.concatenate(
            [
                np.arange(len(cloud_masked)),
                np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True),
            ],
            axis=0,
        )

    end_points = {
        "point_clouds": torch.from_numpy(cloud_masked[idxs].astype(np.float32)[np.newaxis]).cuda(non_blocking=True),
        "cloud_colors": color_masked[idxs],
    }

    o3d_cloud = make_o3d_point_cloud(cloud_masked, color_masked) if build_o3d else None
    return end_points, o3d_cloud, cloud_masked


def infer_grasps(
    net: Any,
    end_points: dict,
    raw_cloud: np.ndarray,
    collision_thresh: float,
    voxel_size: float = DEFAULT_VOXEL_SIZE,
) -> tuple[Any, dict[str, int]]:
    import torch

    prepare_graspnet_imports()
    from collision_detector import ModelFreeCollisionDetector
    from graspnet import pred_decode
    from graspnetAPI import GraspGroup

    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)

    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
    decoded_count = len(gg)
    collision_removed = 0
    if len(gg) > 0 and collision_thresh > 0:
        detector = ModelFreeCollisionDetector(raw_cloud, voxel_size=voxel_size)
        collision_mask = detector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
        collision_removed = int(np.count_nonzero(collision_mask))
        gg = gg[~collision_mask]

    return gg, {
        "decoded": decoded_count,
        "pre_collision": decoded_count,
        "collision_removed": collision_removed,
        "final": len(gg),
    }


def filter_grasps_by_bbox(
    grasps: Any,
    bbox_xyxy: tuple[int, int, int, int],
    K: np.ndarray,
    *,
    margin_px: int = 0,
    expand_ratio: float = 1.0,
    image_shape: Optional[tuple[int, int]] = None,
) -> Any:
    if len(grasps) == 0:
        return grasps

    translations = np.asarray(grasps.translations, dtype=np.float64)
    valid_z, u, v = project_points_to_pixels(translations, K)

    x1, y1, x2, y2 = [float(val) for val in bbox_xyxy]
    expand_ratio = max(1.0, float(expand_ratio))
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    half_w = 0.5 * max(1.0, x2 - x1) * expand_ratio
    half_h = 0.5 * max(1.0, y2 - y1) * expand_ratio
    x1 = int(round(cx - half_w)) - int(margin_px)
    y1 = int(round(cy - half_h)) - int(margin_px)
    x2 = int(round(cx + half_w)) + int(margin_px)
    y2 = int(round(cy + half_h)) + int(margin_px)
    if image_shape is not None:
        h, w = image_shape
        x1 = int(np.clip(x1, 0, max(0, w - 1)))
        x2 = int(np.clip(x2, 0, max(0, w - 1)))
        y1 = int(np.clip(y1, 0, max(0, h - 1)))
        y2 = int(np.clip(y2, 0, max(0, h - 1)))

    keep = valid_z & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    return grasps[keep]


def filter_grasps_by_width(grasps: Any, max_width_m: Optional[float]) -> Any:
    if max_width_m is None or len(grasps) == 0:
        return grasps
    return grasps[np.asarray(grasps.widths, dtype=np.float64) <= float(max_width_m)]


def select_best_grasp(grasps: Any) -> Optional[Any]:
    if len(grasps) == 0:
        return None
    ranked = rank_grasps(grasps, run_nms=True)
    return ranked[0] if len(ranked) > 0 else None


def infer_frame(
    net: Any,
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    *,
    num_point: int,
    min_depth: float,
    max_depth: float,
    collision_thresh: float,
    voxel_size: float = DEFAULT_VOXEL_SIZE,
    yolo_model: Optional[Any] = None,
    yolo_opts: Optional[dict[str, Any]] = None,
    detections: Optional[list[YoloDetection]] = None,
    target_class: Optional[str] = None,
    target_margin_px: int = 0,
    target_expand_ratio: float = 1.0,
    max_grasp_width_m: Optional[float] = None,
    build_o3d: bool = True,
) -> GraspFrameResult:
    prepare_graspnet_imports()
    from graspnetAPI import GraspGroup
    from utils.yolo_utils import detect_objects

    detections_provided = detections is not None
    detections_list: list[YoloDetection] = list(detections or [])
    selected_target: Optional[YoloDetection] = None
    target_label = "full scene"

    if yolo_model is not None:
        if not detections_provided:
            _, detections_list = detect_objects(yolo_model, color_bgr, yolo_opts or {})
        selected_target = select_target(detections_list, target_class)
        if selected_target is None:
            target_status = target_status_text(selected_target, detections_list, target_class)
            empty = GraspGroup()
            return GraspFrameResult(
                grasps=empty,
                pre_bbox_grasps=empty,
                bbox_grasps=empty,
                best=None,
                status=f"inference skipped: {target_status}",
                target_status=target_status,
                detections=detections_list,
                selected_target=None,
                o3d_cloud=None,
                raw_cloud=np.empty((0, 3), dtype=np.float32),
                already_nms=False,
            )
        target_label = f"{selected_target.class_name} {selected_target.conf:.2f}"

    tic = time.time()
    end_points, o3d_cloud, raw_cloud = build_end_points(
        color_bgr, depth_mm, K, num_point, min_depth, max_depth, build_o3d=build_o3d
    )
    grasps, counts = infer_grasps(net, end_points, raw_cloud, collision_thresh, voxel_size)
    pre_bbox_grasps = copy_grasp_group(grasps)
    if selected_target is not None:
        grasps = filter_grasps_by_bbox(
            grasps,
            selected_target.bbox_xyxy,
            K,
            margin_px=target_margin_px,
            expand_ratio=target_expand_ratio,
            image_shape=color_bgr.shape[:2],
        )
    bbox_grasps = copy_grasp_group(grasps)
    grasps = filter_grasps_by_width(grasps, max_grasp_width_m)
    best = select_best_grasp(grasps)
    elapsed = time.time() - tic

    if yolo_model is None:
        status = f"grasps={len(grasps)} decoded={counts['decoded']} inference={elapsed:.2f}s"
        target_status = "YOLO disabled: full-scene GraspNet"
    else:
        status = (
            f"{target_label} grasps={len(grasps)}/{len(bbox_grasps)}/{len(pre_bbox_grasps)} decoded={counts['decoded']} "
            f"collide={counts['collision_removed']}/{counts['pre_collision']} inference={elapsed:.2f}s"
        )
        target_status = target_status_text(selected_target, detections_list, target_class)

    return GraspFrameResult(
        grasps=grasps,
        pre_bbox_grasps=pre_bbox_grasps,
        bbox_grasps=bbox_grasps,
        best=best,
        status=status,
        target_status=target_status,
        detections=detections_list,
        selected_target=selected_target,
        o3d_cloud=o3d_cloud,
        raw_cloud=raw_cloud,
        already_nms=False,
    )
