"""
Live camera AnyGrasp preview; no CAN, arm SDK, or gripper connection.

Keys:
  G/Space: infer the current RGB-D frame.
  R: resume live preview.
  Q/Esc: exit.

Usage:
    python scripts/anygrasp_camera_demo.py
    python scripts/anygrasp_camera_demo.py --target-class toy
    python scripts/anygrasp_camera_demo.py --no-yolo --no-open3d
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRASPNET_ROOT = PROJECT_ROOT / "sdk" / "graspnet-baseline"


def _prepare_imports() -> None:
    project_root = str(PROJECT_ROOT)
    if project_root in sys.path:
        sys.path.remove(project_root)
    sys.path.insert(0, project_root)
    # Only needed for shared visualization/UI helpers in graspnet_utils.
    for subdir in ("models", "dataset", "utils", "pointnet2", "graspnetAPI"):
        path = str(GRASPNET_ROOT / subdir)
        if path not in sys.path:
            sys.path.insert(1, path)
    root = str(GRASPNET_ROOT)
    if root not in sys.path:
        sys.path.insert(1, root)


_prepare_imports()

from drivers.camera import make_camera  # noqa: E402
from utils.camera_utils import configure_camera, load_config  # noqa: E402
import utils.anygrasp_utils as anygrasp_utils  # noqa: E402
import utils.graspnet_utils as graspnet_utils  # noqa: E402
from utils.yolo_utils import detect_objects, load_yolo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AnyGrasp live camera demo; never connects to the arm")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--camera-type", choices=("realsense_d435i", "realsense_d405", "orbbec_gemini2"), default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--checkpoint", default=None, help="override anygrasp.checkpoint")
    parser.add_argument("--target-class", default=None)
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--no-open3d", action="store_true")
    return parser.parse_args()


def _print_candidates(result: anygrasp_utils.AnyGraspFrameResult, limit: int = 5) -> None:
    for index, grasp in enumerate(result.grasps[:limit]):
        tip = np.asarray(grasp.translation) + float(grasp.depth) * np.asarray(grasp.rotation_matrix)[:, 0]
        print(
            f"[AnyGrasp #{index}] score={grasp.score:.3f} width={grasp.width:.3f}m "
            f"center={np.round(grasp.translation, 3).tolist()} tip={np.round(tip, 3).tolist()}"
        )


def main() -> None:
    args = parse_args()
    cfg = configure_camera(load_config(args.config), args)
    ag_cfg = dict(cfg.get("anygrasp", {}))
    detector = anygrasp_utils.build_detector(ag_cfg, args.checkpoint)
    yolo_model = None
    yolo_opts = {}
    if not args.no_yolo:
        yolo_model, yolo_opts = load_yolo(cfg, project_root=PROJECT_ROOT)

    cam = make_camera(cfg)
    vis: Optional[graspnet_utils.Open3DGraspWindow] = None
    status = "warming up camera..."
    target_status = "target detector warming up..."
    frozen = False
    last_display: Optional[np.ndarray] = None
    last_detections = []
    selected_target = None
    frame_index = 0
    target_class = args.target_class or ag_cfg.get("target_class")
    window_name = "AnyGrasp Live Camera (no robot)"

    print(
        f"Using camera: {cfg['camera']['type']} "
        f"{cfg['camera'].get('color_width')}x{cfg['camera'].get('color_height')}@{cfg['camera'].get('fps')}"
    )
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(cfg["camera"].get("color_width", 1280)), int(cfg["camera"].get("color_height", 720)))

    try:
        cam.open()
        cam.warm_up(graspnet_utils.DEFAULT_WARMUP_FRAMES)
        print("Camera intrinsics:\n", cam.K)
        print("Camera-only mode: G/SPACE infer, R resume, Q/ESC quit")

        while True:
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                continue

            frame_index += 1
            if not frozen and yolo_model is not None and (frame_index == 1 or frame_index % int(yolo_opts["infer_every"]) == 0):
                try:
                    _, last_detections = detect_objects(yolo_model, color_bgr, yolo_opts)
                    selected_target = anygrasp_utils.select_target(last_detections, target_class)
                    target_status = anygrasp_utils.target_status_text(selected_target, last_detections, target_class)
                except Exception as exc:
                    last_detections = []
                    selected_target = None
                    target_status = f"YOLO failed: {exc}"

            if frozen and last_display is not None:
                display = graspnet_utils.draw_status(last_display, status, target_status, frozen=True)
            else:
                display_base = color_bgr
                if yolo_model is not None:
                    display_base = graspnet_utils.draw_detections_overlay(
                        color_bgr, last_detections, selected_target, target_class
                    )
                display = graspnet_utils.draw_status(display_base, status, target_status)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                frozen = False
                last_display = None
                status = "live preview"
                continue
            if key in (ord("g"), ord("G"), ord(" ")):
                try:
                    result = anygrasp_utils.infer_frame(
                        detector,
                        color_bgr,
                        depth_mm,
                        cam.K,
                        cfg=ag_cfg,
                        min_depth=float(ag_cfg.get("min_depth", 0.05)),
                        max_depth=float(ag_cfg.get("max_depth", 1.0)),
                        yolo_model=yolo_model,
                        yolo_opts=yolo_opts,
                        target_class=target_class,
                        target_margin_px=int(ag_cfg.get("target_margin_px", 12)),
                        target_expand_ratio=float(ag_cfg.get("target_expand_ratio", 1.35)),
                        max_grasp_width_m=float(ag_cfg.get("max_gripper_width", 0.09)),
                    )
                    status = result.status
                    target_status = result.target_status
                    last_detections = result.detections
                    selected_target = result.selected_target
                    print(status)
                    _print_candidates(result)
                    display_base = color_bgr.copy()
                    if yolo_model is not None:
                        display_base = graspnet_utils.draw_detections_overlay(
                            display_base, last_detections, selected_target, target_class
                        )
                    graspnet_utils.draw_best_grasp_projection(display_base, result.best, cam.K)
                    last_display = display_base
                    frozen = True

                    if not args.no_open3d:
                        if vis is None:
                            vis = graspnet_utils.Open3DGraspWindow("AnyGrasp Grasps", int(ag_cfg.get("top_k", 20)))
                        vis.update(result.o3d_cloud, result.grasps)
                except Exception as exc:
                    status = f"inference failed: {exc}"
                    print(status)

            if vis is not None and not vis.poll():
                vis.close()
                vis = None
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cam.close()
        if vis is not None:
            vis.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
