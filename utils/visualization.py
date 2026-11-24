"""2D OpenCV overlay and 3D Open3D grasp visualization helpers.

This module intentionally does NOT import GraspNet, pointnet2, torch, or
AnyGrasp so that all demos and scripts can share visualization without
pulling in heavy or unrelated SDK dependencies.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import cv2
import numpy as np

from utils.grasp_types import copy_grasp_group, visualization_grasps
from utils.yolo_utils import YoloDetection

DISPLAY_FLIP_X = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)

__all__ = [
    "DISPLAY_FLIP_X",
    "Open3DGraspWindow",
    "draw_best_grasp_projection",
    "draw_container_overlay",
    "draw_detections_overlay",
    "draw_status",
    "visualization_grasps",
]


class Open3DGraspWindow:
    """Interactive Open3D point-cloud and grasp candidate preview window."""

    def __init__(self, title: str, top_k: int) -> None:
        import open3d as o3d

        self._o3d = o3d
        self._top_k = int(top_k)
        self._vis = o3d.visualization.Visualizer()
        if not self._vis.create_window(title, width=1280, height=720):
            self._vis.destroy_window()
            raise RuntimeError("Open3D visualizer window could not be created")
        self._geometries: list[Any] = []
        self._initialized = False

    def update(self, cloud: Any, grasps: Any) -> None:
        o3d = self._o3d
        for geom in self._geometries:
            self._vis.remove_geometry(geom, reset_bounding_box=False)
        self._geometries = []

        geometries: list[Any] = []
        if cloud is not None:
            cloud_vis = o3d.geometry.PointCloud(cloud)
            cloud_vis.transform(DISPLAY_FLIP_X)
            geometries.append(cloud_vis)

        if grasps is not None and len(grasps) > 0:
            grasps_vis = copy_grasp_group(grasps)
            try:
                grasps_vis = grasps_vis.nms()
            except Exception as exc:
                print(f"Grasp NMS skipped: {exc}")
            sorted_vis = grasps_vis.sort_by_score()
            if sorted_vis is not None:
                grasps_vis = sorted_vis
            grasps_vis = grasps_vis[: self._top_k]
            grasps_vis.transform(DISPLAY_FLIP_X)
            geometries.extend(grasps_vis.to_open3d_geometry_list())

        for geom in geometries:
            self._vis.add_geometry(geom, reset_bounding_box=not self._initialized)
        self._geometries = geometries
        self._initialized = True
        self.poll()

    def poll(self) -> bool:
        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return alive

    def close(self) -> None:
        self._vis.destroy_window()


def draw_detections_overlay(
    frame: np.ndarray,
    detections: Sequence[YoloDetection],
    selected: Optional[YoloDetection],
    target_class: Optional[str],
) -> np.ndarray:
    display = frame.copy()
    selected_key = (
        (selected.result_index, selected.detection_index) if selected is not None else None
    )
    for target in detections:
        is_selected = selected_key == (target.result_index, target.detection_index)
        color = (0, 255, 80) if is_selected else (0, 185, 255)
        thickness = 3 if is_selected else 2
        x1, y1, x2, y2 = target.bbox_xyxy
        cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
        label = f"{target.class_name} {target.conf:.2f}"
        if target_class and is_selected:
            label = f"TARGET {label}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bg_y1 = max(0, y1 - label_size[1] - 8)
        cv2.rectangle(display, (x1, bg_y1), (x1 + label_size[0] + 8, y1), (0, 0, 0), -1)
        cv2.putText(display, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return display


def draw_container_overlay(
    display: np.ndarray,
    container: Optional[YoloDetection],
    label_prefix: str = "CONTAINER",
) -> None:
    if container is None:
        return
    x1, y1, x2, y2 = container.bbox_xyxy
    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 180, 0), 3)
    cv2.putText(
        display,
        f"{label_prefix}: {container.class_name} {container.conf:.2f}",
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 180, 0),
        2,
        cv2.LINE_AA,
    )


def draw_best_grasp_projection(display: np.ndarray, grasp: Optional[Any], K: np.ndarray) -> None:
    if grasp is None:
        return
    x, y, z = [float(v) for v in grasp.translation]
    if z <= 1e-6:
        return
    u = int(round(float(K[0, 0]) * x / z + float(K[0, 2])))
    v = int(round(float(K[1, 1]) * y / z + float(K[1, 2])))
    if 0 <= u < display.shape[1] and 0 <= v < display.shape[0]:
        cv2.drawMarker(display, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        label = f"best score={grasp.score:.2f} width={grasp.width * 100:.1f}cm"
        cv2.putText(display, label, (u + 10, max(24, v - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


def draw_status(
    frame: np.ndarray,
    status: str,
    target_status: str = "",
    frozen: bool = False,
    title: str = "Grasp Demo",
    keys_hint: str = "G/SPACE: infer   R: resume   Q/ESC: quit",
) -> np.ndarray:
    display = frame.copy()
    lines = [title, keys_hint]
    if target_status:
        lines.append(target_status)
    if status:
        lines.append(status)
    y = 28
    for line in lines:
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
    if frozen:
        cv2.putText(display, "[FROZEN]", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
    return display
