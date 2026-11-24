"""
Ordinary visual grasping demo based on YOLO.

Keys:
  G: capture the current RGB-D frame and execute the best grasp.
  R: resume live preview.
  Q/Esc: exit (releases gripper and homes in real mode; zero motion in --dry-run).

Usage:
    python scripts/main.py
    python scripts/main.py --dry-run
    python scripts/main.py --dry-run --move-ready
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.preview_app import FpsMeter  # noqa: E402
from core.robot_session import RobotSession, execute_grasp_sequence  # noqa: E402
from drivers.camera import make_camera  # noqa: E402
from utils.camera_utils import compose_cam_to_base_transform, load_config, load_hand_eye  # noqa: E402
from utils.ordinary_grasp import GraspPose, draw_grasp, estimate_grasps, select_best_grasp  # noqa: E402
from utils.transforms import (  # noqa: E402
    canonicalize_parallel_gripper_tcp_rotation,
    rotation_matrix_to_euler_zyx,
    transform_grasp_pose_to_base_with_retreat,
)
from utils.yolo_utils import load_yolo  # noqa: E402


def _render_display(
    image: np.ndarray,
    grasps: list[GraspPose],
    best: Optional[GraspPose],
    status_text: str,
) -> np.ndarray:
    display = image.copy()
    for grasp in grasps:
        draw_grasp(display, grasp)

    cv2.putText(display, status_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    if best is not None:
        x_m, y_m, z_m = best.position.tolist()
        best_text = (
            f"best={best.class_name} conf={best.conf:.2f} "
            f"xyz=({x_m:+.3f},{y_m:+.3f},{z_m:+.3f}) jaw={best.jaw_width_m * 100:.1f}cm"
        )
        cv2.putText(
            display,
            best_text,
            (10, display.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (120, 255, 140),
            2,
        )
    return display


def _print_best_grasp(grasp: GraspPose) -> None:
    tcp_rotation = canonicalize_parallel_gripper_tcp_rotation(grasp.tcp_rotation)
    print("\n[G] Best grasp:")
    print(f"  class={grasp.class_name} conf={grasp.conf:.3f}")
    print(f"  center_px={grasp.center_px} angle_deg={grasp.angle_deg:.2f}")
    print(f"  jaw_width_m={grasp.jaw_width_m:.4f} object_length_m={grasp.object_length_m:.4f}")
    print(f"  position_xyz={grasp.position.tolist()}")
    print(f"  grasp_rpy={rotation_matrix_to_euler_zyx(grasp.rotation).tolist()}")
    print(f"  tcp_rpy={rotation_matrix_to_euler_zyx(tcp_rotation).tolist()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ordinary short-axis grasp demo")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dry-run", action="store_true", help="estimate only; never move the arm or gripper")
    parser.add_argument("--move-ready", action="store_true", help="with --dry-run, move only to ready at startup")
    parser.add_argument("--confirm-motion", action="store_true", help="require typing YES before motion")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(config_path)

    cam_cfg = cfg.get("camera", {})
    print(f"=== Camera: {cam_cfg.get('type')} ===")
    cam = make_camera(cfg)
    robot = RobotSession(cfg, dry_run=args.dry_run, move_ready_in_dry_run=args.move_ready, init_ik=False)

    last_results: list[Any] = []
    last_grasps: list[GraspPose] = []
    frozen = False
    last_display: Optional[np.ndarray] = None
    frame_index = 0
    fps_meter = FpsMeter()

    window_name = "Main - Ordinary Grasp"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print("\n[Keys]  G=grasp  R=resume  Q/ESC=quit\n")

    T_hand_eye: Optional[np.ndarray] = None

    try:
        cam.open()
        cam.warm_up(15)
        K = cam.K.astype(np.float32)

        cam_type = str(cam_cfg.get("type", "")).lower()
        T_hand_eye, hand_eye_mode = load_hand_eye(PROJECT_ROOT, cam_type)
        if T_hand_eye is None or hand_eye_mode != "eye_in_hand":
            print("[WARN] Hand-eye calibration unavailable; grasp execution disabled")
            T_hand_eye = None

        gp_cfg = cfg.get("grasp_pipeline", {})
        grasp_cfg = gp_cfg.get("grasp", {})
        pregrasp_offset_m = float(grasp_cfg.get("pregrasp_offset_m", 0.08))
        retreat_offset_m = float(grasp_cfg.get("retreat_offset_m", pregrasp_offset_m))
        depth_quantile = float(grasp_cfg.get("depth_quantile", 0.75))
        approach_mode = str(grasp_cfg.get("ordinary_approach_mode", "camera_z"))
        fusion_frames = max(1, int(grasp_cfg.get("depth_fusion_frames", 3)))
        infer_every = max(1, int(gp_cfg.get("infer_every_live", 2)))

        model, yolo_opts = load_yolo(cfg, project_root=PROJECT_ROOT)

        print("=== Init robot ===")
        robot.start()

        while True:
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                continue

            frame_index += 1
            fps_value = fps_meter.tick()

            if not frozen and (frame_index % infer_every == 0 or not last_results):
                last_results = model.predict(
                    color_bgr,
                    verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )
                last_grasps = estimate_grasps(
                    last_results,
                    depth_mm,
                    K,
                    depth_quantile=depth_quantile,
                    approach_mode=approach_mode,
                )

            status = f"{'FROZEN' if frozen else 'LIVE'} {fps_value:.1f}fps | G=grasp R=resume Q=quit"
            best_live = select_best_grasp(last_grasps)
            if frozen and last_display is not None:
                display = last_display.copy()
                cv2.putText(display, "[FROZEN]", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
            else:
                display = _render_display(color_bgr, last_grasps, best_live, status)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                frozen = False
                last_display = None
                continue

            if key in (ord("g"), ord("G")):
                print(f"\n[G] Capture ({fusion_frames} frames) and estimate grasp")
                snap_color, snap_depth = cam.get_fused_frame(fusion_frames)
                if snap_color is None or snap_depth is None:
                    print("[G] Frame capture failed")
                    continue

                # Freeze TCP pose at capture time before YOLO inference.
                T_cam2base = (
                    compose_cam_to_base_transform(robot.get_tcp_pose(), T_hand_eye, cfg)
                    if T_hand_eye is not None
                    else None
                )

                snap_results = model.predict(
                    snap_color,
                    verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )
                snap_grasps = estimate_grasps(
                    snap_results,
                    snap_depth,
                    K,
                    depth_quantile=depth_quantile,
                    approach_mode=approach_mode,
                )
                best = select_best_grasp(snap_grasps)
                if best is None:
                    print("[G] No valid grasp")
                    continue

                _print_best_grasp(best)

                snap_display = _render_display(snap_color, snap_grasps, best, "SNAPSHOT")
                frozen = True
                last_display = snap_display
                last_results = snap_results
                last_grasps = snap_grasps

                if T_cam2base is None:
                    print("[G] Hand-eye calibration unavailable")
                    continue

                grasp6d, pre6d, retreat6d = transform_grasp_pose_to_base_with_retreat(
                    best.position,
                    best.tcp_rotation,
                    T_cam2base,
                    pregrasp_offset_m,
                    retreat_offset_m,
                )
                execute_grasp_sequence(
                    robot.controller,
                    robot.grasp_driver,
                    grasp6d,
                    pre6d,
                    retreat6d,
                    robot.ready_cfg,
                    motion_cfg=grasp_cfg,
                    dry_run=args.dry_run,
                    confirm_motion=args.confirm_motion,
                    arm=robot.arm,
                    n_joints=robot.n_joints,
                )

    finally:
        robot.close()
        try:
            cam.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Done.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
