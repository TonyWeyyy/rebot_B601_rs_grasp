"""RGB-D point cloud projection and target region-steering masks."""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from utils.yolo_utils import YoloDetection

_PIXEL_GRID_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _get_pixel_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    key = (int(height), int(width))
    cached = _PIXEL_GRID_CACHE.get(key)
    if cached is None:
        us, vs = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        cached = (us, vs)
        # Bound cache size for safety.
        if len(_PIXEL_GRID_CACHE) >= 8:
            _PIXEL_GRID_CACHE.clear()
        _PIXEL_GRID_CACHE[key] = cached
    return cached


def make_o3d_point_cloud(points: np.ndarray, colors: Optional[np.ndarray] = None) -> Any:
    """Build an Open3D PointCloud lazily (only imports open3d when called)."""
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    if len(points) > 0:
        cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        if colors is not None and len(colors) == len(points):
            cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return cloud


def build_point_cloud(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
    max_input_points: int = 100000,
    point_seed: Optional[int] = 0,
    build_o3d: bool = True,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Create a metric camera-frame cloud from aligned BGR and uint16 depth-mm images."""
    if color_bgr is None or depth_mm is None:
        raise ValueError("color/depth frame is None")
    if color_bgr.shape[:2] != depth_mm.shape[:2]:
        depth_mm = cv2.resize(
            depth_mm,
            (color_bgr.shape[1], color_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
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
    us, vs = _get_pixel_grid(h, w)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    z_valid = z[mask]
    x_valid = (us[mask] - cx) * z_valid / fx
    y_valid = (vs[mask] - cy) * z_valid / fy
    points = np.stack((x_valid, y_valid, z_valid), axis=-1).astype(np.float32)
    colors = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)[mask] / 255.0

    if max_input_points > 0 and len(points) > int(max_input_points):
        rng = np.random.default_rng(point_seed)
        indices = rng.choice(len(points), int(max_input_points), replace=False)
        points = points[indices]
        colors = colors[indices]

    cloud = make_o3d_point_cloud(points, colors) if build_o3d else None
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


def project_points_to_pixels(points: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project (N, 3) camera-frame points to (u, v) pixel coordinates."""
    z = points[:, 2]
    valid = z > 1e-6
    u = np.full(len(points), np.nan, dtype=np.float64)
    v = np.full(len(points), np.nan, dtype=np.float64)
    u[valid] = float(K[0, 0]) * points[valid, 0] / z[valid] + float(K[0, 2])
    v[valid] = float(K[1, 1]) * points[valid, 1] / z[valid] + float(K[1, 2])
    return valid, u, v


def bbox_region_mask(
    points: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    K: np.ndarray,
    *,
    margin_px: int,
    expand_ratio: float,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Project cloud points into the expanded YOLO bbox."""
    valid, u, v = project_points_to_pixels(points, K)

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


def mask_region_mask(
    points: np.ndarray,
    binary_mask: np.ndarray,
    K: np.ndarray,
    *,
    dilate_px: int = 10,
    image_shape: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Project cloud points onto a dilated 2D segmentation mask."""
    mask_u8 = (np.asarray(binary_mask) > 0).astype(np.uint8)
    if image_shape is not None and mask_u8.shape[:2] != image_shape:
        mask_u8 = cv2.resize(
            mask_u8, (int(image_shape[1]), int(image_shape[0])), interpolation=cv2.INTER_NEAREST
        )
    if int(dilate_px) > 0:
        ksize = 2 * int(dilate_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)

    h, w = mask_u8.shape[:2]
    valid, u, v = project_points_to_pixels(points, K)
    in_bounds = valid & (u >= 0.0) & (u <= float(w - 1)) & (v >= 0.0) & (v <= float(h - 1))
    result = np.zeros(len(points), dtype=bool)
    if np.any(in_bounds):
        ui = np.rint(u[in_bounds]).astype(np.int32)
        vi = np.rint(v[in_bounds]).astype(np.int32)
        result[in_bounds] = mask_u8[vi, ui] > 0
    return result


def target_region_mask(
    points: np.ndarray,
    target: YoloDetection,
    K: np.ndarray,
    *,
    image_shape: tuple[int, int],
    base_mask: Optional[np.ndarray] = None,
    use_mask: bool = True,
    mask_dilate_px: int = 10,
    margin_px: int = 12,
    expand_ratio: float = 1.35,
    min_points: int = 300,
) -> tuple[np.ndarray, str]:
    """Compute target region steering mask; prefers dilated segmentation mask with bbox fallback."""
    ws = np.ones(len(points), dtype=bool) if base_mask is None else np.asarray(base_mask, dtype=bool)
    if use_mask and target.mask is not None and np.any(target.mask > 0):
        seg_region = ws & mask_region_mask(
            points,
            target.mask,
            K,
            dilate_px=mask_dilate_px,
            image_shape=image_shape,
        )
        if int(seg_region.sum()) >= int(min_points):
            return seg_region, "mask"

    bbox_region = ws & bbox_region_mask(
        points,
        target.bbox_xyxy,
        K,
        margin_px=margin_px,
        expand_ratio=expand_ratio,
        image_shape=image_shape,
    )
    return bbox_region, "bbox"
