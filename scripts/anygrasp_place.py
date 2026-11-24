"""
AnyGrasp pick-and-place for the reBot B601-RS project.

Workflow:
  target + container in one RGB-D frame -> target/place safety & chained IK checks
  -> grasp -> retreat -> ready -> pre-place -> place -> release -> pre-place -> ready.

Keys:
  G/Space: capture and plan one pick-and-place task.
  R: resume live preview.
  Q/Esc: exit (releases gripper and homes in real mode; zero motion in --dry-run).

Safety:
  --dry-run never executes grasp/place/gripper motions and never moves at exit.
  --move-ready explicitly permits only the initial ready motion in dry-run.
  --confirm-motion requires YES before grasp and PLACE before transfer/release.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ik_planner import select_executable_grasp  # noqa: E402
from core.preview_app import FpsMeter  # noqa: E402
from core.robot_session import (  # noqa: E402
    RobotSession,
    execute_pick_and_place_sequence,
)
from drivers.camera import make_camera  # noqa: E402
from drivers.robot.grasp_driver import GRIPPER_MAX_DISTANCE_M  # noqa: E402
import utils.anygrasp_utils as anygrasp_utils  # noqa: E402
import utils.place_utils as place_utils  # noqa: E402
from utils.camera_utils import compose_cam_to_base_transform, configure_camera, load_config, load_hand_eye  # noqa: E402
from utils.target_selector import select_container, select_target  # noqa: E402
from utils.visualization import (  # noqa: E402
    Open3DGraspWindow,
    draw_best_grasp_projection,
    draw_container_overlay,
    draw_detections_overlay,
    draw_status,
)
from utils.yolo_utils import detect_objects, load_yolo  # noqa: E402


# Backward-compatible export for tests.
_execute_place_sequence = execute_pick_and_place_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AnyGrasp pick-and-place with target/container and chained IK prechecks")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--target-class", default="toy")
    parser.add_argument("--container-class", default=None, help="default: place.container_class in YAML")
    parser.add_argument("--extra-yolo-class", action="append", default=[], help="add open-vocabulary YOLO class")
    parser.add_argument("--camera-type", choices=("realsense_d435i", "realsense_d405", "orbbec_gemini2"), default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--yolo-conf", type=float, default=None)
    parser.add_argument("--target-margin-px", type=int, default=None)
    parser.add_argument("--target-expand-ratio", type=float, default=None)
    parser.add_argument("--fusion-frames", type=int, default=None, help="depth median fusion frames at capture")
    parser.add_argument("--anygrasp-checkpoint", default=None)
    parser.add_argument("--min-base-z", type=float, default=None)
    parser.add_argument("--place-height", type=float, default=None, help="TCP release height in base frame, meters")
    parser.add_argument("--dry-run", action="store_true", help="plan and check IK only; no grasp/place/exit motion")
    parser.add_argument("--move-ready", action="store_true", help="with --dry-run, move only to ready at startup")
    parser.add_argument("--confirm-motion", action="store_true", help="require YES before grasp and PLACE before release")
    parser.add_argument("--no-open3d", action="store_true")
    return parser.parse_args()


def _check_place_ik_chain(
    ik_checker: Any,
    pre_place6d: tuple[float, ...],
    place6d: tuple[float, ...],
    retract_to_pre_place: bool,
) -> None:
    """Verify chained IK for pre-place -> place (and optional retract -> pre-place)."""
    poses = [pre_place6d, place6d]
    names = ["pre-place", "place"]
    if retract_to_pre_place:
        poses.append(pre_place6d)
        names.append("post-release-retract")

    if hasattr(ik_checker, "check_chain"):
        ok, errors, _ = ik_checker.check_chain(poses)
        for name, err in zip(names, errors):
            step_ok = ok or (len(errors) > names.index(name) + 1)
            print(f"[Place IK] {name}: success={step_ok} error={err:.6f}")
        if not ok:
            failed_name = names[len(errors) - 1]
            raise RuntimeError(f"Place IK chain failed before any grasp motion at {failed_name} (err={errors[-1]:.6f})")
        return

    failed = []
    for name, pose in zip(names[:2], poses[:2]):
        ok, err = ik_checker.check(*pose)
        print(f"[Place IK] {name}: success={ok} error={err:.6f}")
        if not ok:
            failed.append((name, err))
    if failed:
        raise RuntimeError(f"Place IK failed before any grasp motion: {failed}")


def main() -> int:
    args = parse_args()
    cfg = configure_camera(load_config(args.config), args)
    grasp_cfg = cfg.get("grasp_pipeline", {}).get("grasp", {})
    anygrasp_cfg = dict(cfg.get("anygrasp", {}))
    place_cfg = dict(cfg.get("place", {}))
    if args.place_height is not None:
        place_cfg["height_m"] = float(args.place_height)

    container_class = args.container_class or str(place_cfg.get("container_class", "box"))
    min_base_z = float(
        args.min_base_z if args.min_base_z is not None else grasp_cfg.get("min_base_z_m", 0.01)
    )
    pregrasp_offset = float(grasp_cfg.get("pregrasp_offset_m", 0.08))
    retreat_offset = float(grasp_cfg.get("retreat_offset_m", pregrasp_offset))
    insertion_depth = float(grasp_cfg.get("insertion_depth_m", 0.02))
    fusion_frames = max(
        1,
        int(args.fusion_frames if args.fusion_frames is not None else grasp_cfg.get("depth_fusion_frames", 3)),
    )
    max_ik_candidates = int(grasp_cfg.get("max_ik_candidates", 20))
    target_margin_px = int(
        args.target_margin_px if args.target_margin_px is not None else anygrasp_cfg.get("target_margin_px", 12)
    )
    target_expand_ratio = float(
        args.target_expand_ratio
        if args.target_expand_ratio is not None
        else anygrasp_cfg.get("target_expand_ratio", 1.35)
    )
    retract_to_pre_place = bool(place_cfg.get("retract_to_pre_place", True))
    use_mask_centroid = bool(place_cfg.get("use_mask_centroid", True))

    safety = dict(anygrasp_cfg.get("safety", {}))
    safety["max_gripper_width"] = min(
        GRIPPER_MAX_DISTANCE_M,
        float(anygrasp_cfg.get("max_gripper_width", GRIPPER_MAX_DISTANCE_M)),
    )

    cam = make_camera(cfg)
    robot = RobotSession(cfg, dry_run=args.dry_run, move_ready_in_dry_run=args.move_ready, init_ik=True)
    vis: Optional[Open3DGraspWindow] = None
    frozen = False
    last_display: Optional[np.ndarray] = None
    frame_index = 0
    fps_meter = FpsMeter()
    live_detections = []
    live_target = None
    live_container = None
    window_name = "AnyGrasp Pick and Place"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        window_name,
        int(cfg["camera"].get("color_width", 1280)),
        int(cfg["camera"].get("color_height", 720)),
    )
    print("[Keys] G/SPACE=plan pick-place  R=resume  Q/ESC=quit")

    try:
        cam.open()
        cam.warm_up(20)
        K = cam.K.astype(np.float64)
        print("Camera intrinsics:\n", K)
        cam_type = str(cfg["camera"].get("type", "")).lower()
        T_hand_eye, hand_eye_mode = load_hand_eye(PROJECT_ROOT, cam_type)
        if T_hand_eye is None or hand_eye_mode != "eye_in_hand":
            raise RuntimeError("Eye-in-hand calibration is required for pick-and-place")

        extra_classes = list(args.extra_yolo_class or [])
        for cls_name in (args.target_class, container_class):
            if cls_name and cls_name not in extra_classes:
                extra_classes.append(cls_name)

        yolo_model, yolo_opts = load_yolo(
            cfg,
            project_root=PROJECT_ROOT,
            conf_override=args.yolo_conf,
            extra_classes=extra_classes,
        )
        detector = anygrasp_utils.build_detector(anygrasp_cfg, args.anygrasp_checkpoint)

        print("=== Init robot ===")
        robot.start()

        while True:
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                continue

            fps_val = fps_meter.tick()
            if not frozen:
                frame_index += 1
                infer_every = max(1, int(yolo_opts.get("infer_every", 3)))
                if frame_index == 1 or frame_index % infer_every == 0:
                    try:
                        _, live_detections = detect_objects(yolo_model, color_bgr, yolo_opts)
                        live_target = select_target(live_detections, args.target_class)
                        live_container = select_container(
                            live_detections, container_class, exclude=live_target
                        )
                    except Exception as exc:
                        print(f"[Live YOLO] detection failed: {exc}")

            if frozen and last_display is not None:
                display = last_display.copy()
            else:
                display = draw_detections_overlay(
                    color_bgr, live_detections, live_target, args.target_class
                )
                draw_container_overlay(display, live_container, label_prefix="CONTAINER")
                display = draw_status(
                    display,
                    f"LIVE {fps_val:.1f}fps | target={args.target_class} container={container_class}",
                    title="AnyGrasp Pick and Place",
                    keys_hint="G/SPACE: plan pick-place   R: resume   Q/ESC: quit",
                )
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                frozen = False
                last_display = None
                frame_index = 0
                continue
            if key not in (ord("g"), ord("G"), ord(" ")):
                if vis is not None and not vis.poll():
                    vis.close()
                    vis = None
                continue

            print(f"\n[G] Capture target and container ({fusion_frames} frames)")
            snap_color, snap_depth = cam.get_fused_frame(fusion_frames)
            if snap_color is None or snap_depth is None:
                print("[G] Frame capture failed")
                continue

            try:
                T_cam2base = compose_cam_to_base_transform(robot.get_tcp_pose(), T_hand_eye, cfg)
                _, detections = detect_objects(yolo_model, snap_color, yolo_opts)
                target = select_target(detections, args.target_class)
                container = select_container(detections, container_class, exclude=target)

                snap_display = draw_detections_overlay(snap_color, detections, target, args.target_class)
                draw_container_overlay(snap_display, container, label_prefix="PLACE")
                last_display = snap_display
                frozen = True

                if target is None:
                    raise RuntimeError(f"Target class not found: {args.target_class}")
                if container is None:
                    raise RuntimeError(f"Container class not found: {container_class}")

                print(f"[Detection] target={target.class_name} conf={target.conf:.3f} bbox={target.bbox_xyxy}")
                print(f"[Detection] container={container.class_name} conf={container.conf:.3f} bbox={container.bbox_xyxy}")
                container_cam = place_utils.detection_center_cam(
                    container,
                    snap_depth,
                    K,
                    min_depth_m=float(anygrasp_cfg.get("min_depth", 0.05)),
                    max_depth_m=float(anygrasp_cfg.get("max_depth", 1.0)),
                    depth_quantile=float(place_cfg.get("container_depth_quantile", 0.5)),
                    use_mask_centroid=use_mask_centroid,
                )
                place6d, pre_place6d, container_base = place_utils.build_place_poses(
                    container_cam, T_cam2base, place_cfg
                )
                print(
                    f"[Container] camera_xyz={np.round(container_cam, 4).tolist()} "
                    f"base_measured_xyz={np.round(container_base, 4).tolist()}"
                )
                place_utils.validate_base_pose(place6d, safety, min_base_z_m=min_base_z, name="place")
                place_utils.validate_base_pose(pre_place6d, safety, min_base_z_m=min_base_z, name="pre-place")
                _check_place_ik_chain(robot.ik_checker, pre_place6d, place6d, retract_to_pre_place)

                preferred_approach_cam = T_cam2base[:3, :3].T @ np.array([0.0, 0.0, -1.0])
                result = anygrasp_utils.infer_frame(
                    detector,
                    snap_color,
                    snap_depth,
                    K,
                    cfg=anygrasp_cfg,
                    min_depth=float(anygrasp_cfg.get("min_depth", 0.05)),
                    max_depth=float(anygrasp_cfg.get("max_depth", 1.0)),
                    yolo_model=yolo_model,
                    yolo_opts=yolo_opts,
                    detections=detections,
                    target_class=args.target_class,
                    target_margin_px=target_margin_px,
                    target_expand_ratio=target_expand_ratio,
                    max_grasp_width_m=GRIPPER_MAX_DISTANCE_M,
                    preferred_approach_cam=preferred_approach_cam,
                    build_o3d=not args.no_open3d,
                )
                print(f"[G] {result.status}")
                if result.best is None:
                    raise RuntimeError("No valid AnyGrasp target candidate")

                selected_grasp = select_executable_grasp(
                    robot.ik_checker,
                    result.grasps,
                    T_cam2base,
                    pregrasp_offset,
                    retreat_offset,
                    insertion_depth,
                    min_base_z,
                    anygrasp_safety=safety,
                    already_nms=result.already_nms,
                    max_ik_candidates=max_ik_candidates,
                )
                if selected_grasp is None:
                    raise RuntimeError("No target grasp passes safety and IK")
                best, grasp6d, pre6d, retreat6d = selected_grasp
                place_utils.validate_target_container_separation(grasp6d, place6d)
                print(
                    f"[Target] score={float(best.score):.4f} width={float(best.width):.4f}m "
                    f"depth={float(best.depth):.4f}m"
                )

                snap_display = draw_detections_overlay(snap_color, detections, target, args.target_class)
                draw_container_overlay(snap_display, container, label_prefix="PLACE")
                draw_best_grasp_projection(snap_display, best, K)
                snap_display = draw_status(
                    snap_display,
                    f"SNAPSHOT | {result.status}",
                    f"target={target.class_name}; container={container.class_name}",
                    frozen=True,
                    title="AnyGrasp Pick and Place",
                )
                last_display = snap_display
                frozen = True

                if not args.no_open3d and result.o3d_cloud is not None:
                    if vis is None:
                        vis = Open3DGraspWindow("AnyGrasp Pick Candidates", int(anygrasp_cfg.get("top_k", 20)))
                    vis.update(result.o3d_cloud, result.grasps)
            except Exception as exc:
                print(f"[G] Planning rejected before motion: {exc}")
                continue

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
            print("[G] Pick-and-place completed" if ok else "[G] Pick-and-place stopped/cancelled")

    finally:
        robot.close()
        try:
            cam.close()
        except Exception:
            pass
        if vis is not None:
            vis.close()
        cv2.destroyAllWindows()
        print("Done.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
