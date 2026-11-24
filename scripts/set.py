"""
Parameterized OBB pick-and-place demo based on YOLO.

Workflow:
  1. Detect target object and container in the same RGB-D frame
  2. Grasp target and lift safely to ready position
  3. Move to pre-place -> release pose above container, release, retract vertically
  4. Return to ready position

Keys:
  G: capture and execute pick-and-place
  R: resume live preview
  Q/Esc: exit (safe-homes arm before disabling motors; releases gripper only in real mode)

Usage:
    python scripts/set.py
    python scripts/set.py --target-keywords toy,red --container-keywords box,container --place-height 0.15
    python scripts/set.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.preview_app import FpsMeter  # noqa: E402
from core.robot_session import RobotSession, execute_pick_and_place_sequence  # noqa: E402
from drivers.camera import make_camera  # noqa: E402
from utils.camera_utils import compose_cam_to_base_transform, load_config, load_hand_eye  # noqa: E402
from utils.ordinary_grasp import GraspPose, estimate_grasps  # noqa: E402
from utils.target_selector import select_by_keywords  # noqa: E402
from utils.transforms import transform_grasp_pose_to_base_with_retreat  # noqa: E402
from utils.yolo_utils import load_yolo  # noqa: E402


def _parse_keywords(raw: str) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in raw.split(",") if item.strip())


def _render_display(
    image: np.ndarray,
    grasps: list[GraspPose],
    best_target: Optional[GraspPose],
    best_container: Optional[GraspPose],
    status_text: str,
) -> np.ndarray:
    display = image.copy()
    for grasp in grasps:
        color = (0, 255, 0) if grasp.is_valid else (0, 165, 255)
        x1, y1, x2, y2 = [int(v) for v in grasp.bbox_xyxy]
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            display,
            f"{grasp.class_name} {grasp.conf:.2f}",
            (x1, max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    cv2.putText(display, status_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    if best_target is not None:
        x_m, y_m, z_m = best_target.position.tolist()
        cv2.putText(
            display,
            f"target({best_target.class_name}): xyz=({x_m:+.3f},{y_m:+.3f},{z_m:+.3f})",
            (10, display.shape[0] - 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (120, 255, 140),
            2,
        )

    if best_container is not None:
        x_m, y_m, z_m = best_container.position.tolist()
        cv2.putText(
            display,
            f"container({best_container.class_name}): xyz=({x_m:+.3f},{y_m:+.3f},{z_m:+.3f})",
            (10, display.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 200, 120),
            2,
        )

    return display


def parse_args(
    argv: Optional[Sequence[str]] = None,
    *,
    default_target_keywords: str = "banana",
    default_container_keywords: str = "box",
    default_place_height: float = 0.08,
    window_title: str = "Set - OBB Pick and Place",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=window_title)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--target-keywords", default=default_target_keywords, help="comma-separated target keywords")
    parser.add_argument("--container-keywords", default=default_container_keywords, help="comma-separated container keywords")
    parser.add_argument("--place-height", type=float, default=default_place_height, help="TCP release Z in base frame (m)")
    parser.add_argument("--dry-run", action="store_true", help="estimate only; never move arm or gripper")
    parser.add_argument("--move-ready", action="store_true", help="with --dry-run, move only to ready at startup")
    parser.add_argument("--confirm-motion", action="store_true", help="require YES / PLACE confirmation")
    args = parser.parse_args(argv)
    args.window_title = window_title
    return args


def run_obb_place(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(config_path)

    target_keywords = _parse_keywords(args.target_keywords)
    container_keywords = _parse_keywords(args.container_keywords)

    cam_cfg = cfg.get("camera", {})
    print(f"=== Camera: {cam_cfg.get('type')} ===")
    print(f"=== Target keywords: {target_keywords} | Container keywords: {container_keywords} ===")
    cam = make_camera(cfg)
    robot = RobotSession(cfg, dry_run=args.dry_run, move_ready_in_dry_run=args.move_ready, init_ik=False)

    last_results: list[Any] = []
    last_grasps: list[GraspPose] = []
    frozen = False
    last_display: Optional[np.ndarray] = None
    frame_index = 0
    fps_meter = FpsMeter()

    window_name = args.window_title
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print("\n[Keys]  G=place  R=resume  Q/ESC=quit\n")

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
        place_cfg = dict(cfg.get("place", {}))
        place_cfg["height_m"] = float(args.place_height)

        pregrasp_offset_m = float(grasp_cfg.get("pregrasp_offset_m", 0.08))
        retreat_offset_m = float(grasp_cfg.get("retreat_offset_m", pregrasp_offset_m))
        pre_place_offset_m = float(place_cfg.get("pre_place_offset_m", 0.08))
        place_rpy = place_cfg.get("orientation_rpy", [0.0, 1.2, 0.0])
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

            best_target = select_by_keywords(last_grasps, target_keywords)
            best_container = select_by_keywords(last_grasps, container_keywords, exclude=best_target)

            status = f"{'FROZEN' if frozen else 'LIVE'} {fps_value:.1f}fps | G=place R=resume Q=quit"
            if frozen and last_display is not None:
                display = last_display.copy()
                cv2.putText(display, "[FROZEN]", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
            else:
                display = _render_display(color_bgr, last_grasps, best_target, best_container, status)

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
                print(f"\n[G] Capture ({fusion_frames} frames) and execute pick-and-place")
                snap_color, snap_depth = cam.get_fused_frame(fusion_frames)
                if snap_color is None or snap_depth is None:
                    print("[G] Frame capture failed")
                    continue

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

                snap_target = select_by_keywords(snap_grasps, target_keywords)
                snap_container = select_by_keywords(snap_grasps, container_keywords, exclude=snap_target)

                if snap_target is None:
                    print("[G] No valid target detected")
                    continue
                if snap_container is None:
                    print("[G] No valid container detected")
                    continue

                print(f"\n[G] Target: class={snap_target.class_name} conf={snap_target.conf:.3f} xyz={snap_target.position.tolist()}")
                print(f"[G] Container: class={snap_container.class_name} conf={snap_container.conf:.3f} xyz={snap_container.position.tolist()}")

                snap_display = _render_display(snap_color, snap_grasps, snap_target, snap_container, "SNAPSHOT")
                frozen = True
                last_display = snap_display
                last_results = snap_results
                last_grasps = snap_grasps

                if T_cam2base is None:
                    print("[G] Hand-eye calibration unavailable")
                    continue

                grasp6d, pre6d, retreat6d = transform_grasp_pose_to_base_with_retreat(
                    snap_target.position,
                    snap_target.tcp_rotation,
                    T_cam2base,
                    pregrasp_offset_m,
                    retreat_offset_m,
                )
                container_base = T_cam2base @ np.append(np.asarray(snap_container.position, dtype=np.float64), 1.0)
                place_z = float(args.place_height)
                roll, pitch, yaw = [float(v) for v in place_rpy]
                place6d = (float(container_base[0]), float(container_base[1]), place_z, roll, pitch, yaw)
                pre_place6d = (
                    float(container_base[0]),
                    float(container_base[1]),
                    place_z + pre_place_offset_m,
                    roll,
                    pitch,
                    yaw,
                )

                ok = execute_pick_and_place_sequence(
                    robot.controller,
                    robot.grasp_driver,
                    grasp6d,
                    pre6d,
                    retreat6d,
                    pre_place6d,
                    place6d,
                    robot.ready_cfg,
                    place_cfg,
                    dry_run=args.dry_run,
                    confirm_motion=args.confirm_motion,
                    arm=robot.arm,
                    n_joints=robot.n_joints,
                )
                print("[G] Place completed successfully!" if ok else "[G] Place stopped/failed")

    finally:
        robot.close()
        try:
            cam.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Done.")

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_obb_place(parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
