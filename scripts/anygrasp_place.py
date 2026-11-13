"""
AnyGrasp pick-and-place for the existing reBot B601-RS project.

Workflow:
  target + container in one RGB-D frame -> target/place safety and IK checks
  -> grasp -> retreat -> ready -> pre-place -> place -> release -> ready.

Keys:
  G/Space: capture and plan one pick-and-place task.
  R: resume live preview.
  Q/Esc: exit. In real mode the existing exit path releases the gripper.

Safety:
  --dry-run never executes grasp/place/gripper motions and never moves at exit.
  --move-ready explicitly permits only the initial ready motion in dry-run.
  --confirm-motion requires YES before grasp and PLACE before transfer/release.
  Without --dry-run, startup ready motion still occurs as in the original demo.

Example:
  python scripts/anygrasp_place.py --target-class toy --container-class box \
      --min-base-z 0.05 --dry-run --move-ready
  python scripts/anygrasp_place.py --target-class toy --container-class box \
      --min-base-z 0.05 --confirm-motion
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the exact target-side IK and coordinate-selection path already validated
# by scripts/grasp.py. Importing it does not run its main() or connect hardware.
from grasp import IkChecker, _move_ready, _select_executable_grasp, _wait_motion  # noqa: E402
from drivers.camera import make_camera  # noqa: E402
from drivers.robot.grasp_driver import GRIPPER_MAX_DISTANCE_M, GraspDriver, selected_arm_config  # noqa: E402
from reBotArm_control_py.actuator import RebotArm  # noqa: E402
from reBotArm_control_py.controllers import RebotArmEndPose  # noqa: E402
import utils.anygrasp_utils as anygrasp_utils  # noqa: E402
import utils.graspnet_utils as graspnet_utils  # noqa: E402
import utils.place_utils as place_utils  # noqa: E402
from utils.camera_utils import compose_cam_to_base_transform, load_config, load_hand_eye  # noqa: E402
from utils.yolo_utils import detect_objects, load_yolo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AnyGrasp pick-and-place with target/container and IK prechecks")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--target-class", default="toy")
    parser.add_argument("--container-class", default=None, help="default: place.container_class in YAML")
    parser.add_argument("--anygrasp-checkpoint", default=None)
    parser.add_argument("--min-base-z", type=float, default=None)
    parser.add_argument("--place-height", type=float, default=None, help="TCP release height in base frame, meters")
    parser.add_argument("--dry-run", action="store_true", help="plan and check IK only; no grasp/place/gripper motion")
    parser.add_argument("--move-ready", action="store_true", help="with --dry-run, move only to ready at startup")
    parser.add_argument("--confirm-motion", action="store_true", help="require YES before grasp and PLACE before release")
    parser.add_argument("--no-open3d", action="store_true")
    return parser.parse_args()


def _ready_with_duration(controller: RebotArmEndPose, ready_cfg: dict[str, Any], duration: float) -> None:
    cfg = dict(ready_cfg)
    cfg["duration"] = float(duration)
    _move_ready(controller, cfg)


def _confirm(expected: str, prompt: str, enabled: bool) -> bool:
    if not enabled:
        return True
    return input(prompt).strip() == expected


def _execute_place_sequence(
    controller: RebotArmEndPose,
    grasp_driver: GraspDriver,
    grasp6d: tuple[float, ...],
    pre6d: tuple[float, ...],
    retreat6d: tuple[float, ...],
    pre_place6d: tuple[float, ...],
    place6d: tuple[float, ...],
    ready_cfg: dict[str, Any],
    place_cfg: dict[str, Any],
    *,
    dry_run: bool,
    confirm_motion: bool,
) -> bool:
    """Execute pick -> retreat -> ready -> pre-place -> release -> ready."""
    labels = (
        ("pregrasp", pre6d),
        ("grasp", grasp6d),
        ("retreat", retreat6d),
        ("pre-place", pre_place6d),
        ("place/release", place6d),
    )
    for name, pose in labels:
        print(
            f"[Place plan] {name:13s} xyz=({pose[0]:+.3f},{pose[1]:+.3f},{pose[2]:+.3f}) "
            f"rpy=({pose[3]:+.3f},{pose[4]:+.3f},{pose[5]:+.3f})"
        )

    if dry_run:
        print("[Place plan] dry run; skip all grasp/place/gripper motions")
        return True

    if not _confirm(
        "YES",
        "[Safety] Type YES to grasp and lift; anything else cancels before gripper motion: ",
        confirm_motion,
    ):
        print("[Safety] Grasp cancelled before gripper motion")
        return False

    pre_duration = float(place_cfg.get("pregrasp_duration_s", 4.0))
    grasp_duration = float(place_cfg.get("grasp_duration_s", 3.0))
    retreat_duration = float(place_cfg.get("retreat_duration_s", 3.0))
    ready_duration = float(place_cfg.get("ready_duration_s", 4.0))
    pre_place_duration = float(place_cfg.get("pre_place_duration_s", 4.0))
    place_duration = float(place_cfg.get("place_duration_s", 3.0))

    print("[Pick] Open gripper")
    grasp_driver.open_gripper()
    print("[Pick] Move to pregrasp")
    if not controller.move_to_traj(*pre6d, duration=pre_duration):
        print("[Pick] Pregrasp controller/IK failed")
        return False
    _wait_motion(controller, pre_duration)

    print("[Pick] Move to grasp")
    if not controller.move_to_traj(*grasp6d, duration=grasp_duration):
        print("[Pick] Grasp controller/IK failed")
        return False
    _wait_motion(controller, grasp_duration)

    print("[Pick] Close gripper")
    if not grasp_driver.grasp():
        print("[Pick] Empty grasp; placement is forbidden")
        _ready_with_duration(controller, ready_cfg, ready_duration)
        return False

    print("[Pick] Holding object; retreat")
    if not controller.move_to_traj(*retreat6d, duration=retreat_duration):
        print("[Pick] Retreat failed; object remains held, inspect the arm before continuing")
        return False
    _wait_motion(controller, retreat_duration)
    print("[Pick] Transfer to ready with object")
    _ready_with_duration(controller, ready_cfg, ready_duration)

    if not _confirm(
        "PLACE",
        "[Safety] Object is held at ready. Type PLACE to move to container and release; anything else cancels: ",
        confirm_motion,
    ):
        print("[Safety] Place cancelled; object remains held at ready and is NOT released automatically")
        return False

    print("[Place] Move to pre-place")
    if not controller.move_to_traj(*pre_place6d, duration=pre_place_duration):
        print("[Place] Pre-place failed; object remains held at ready")
        return False
    _wait_motion(controller, pre_place_duration)

    print("[Place] Move to release pose")
    if not controller.move_to_traj(*place6d, duration=place_duration):
        print("[Place] Place failed; return to ready with object")
        _ready_with_duration(controller, ready_cfg, ready_duration)
        return False
    _wait_motion(controller, place_duration)

    print("[Place] Release object into container")
    grasp_driver.open_gripper(timeout=2.0)
    time.sleep(float(place_cfg.get("release_wait_s", 1.0)))
    print("[Place] Return ready")
    _ready_with_duration(controller, ready_cfg, ready_duration)
    grasp_driver.open_gripper(distance_m=0.0, timeout=2.0)
    return True


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    robot_cfg = cfg.get("robot", {})
    ready_cfg = robot_cfg.get("ready_pose", {})
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

    safety = dict(anygrasp_cfg.get("safety", {}))
    safety["max_gripper_width"] = min(
        GRIPPER_MAX_DISTANCE_M,
        float(anygrasp_cfg.get("max_gripper_width", GRIPPER_MAX_DISTANCE_M)),
    )

    cam = make_camera(cfg)
    controller: Optional[RebotArmEndPose] = None
    rebotarm: Optional[RebotArm] = None
    grasp_driver: Optional[GraspDriver] = None
    vis: Optional[graspnet_utils.Open3DGraspWindow] = None
    robot_ready = False
    frozen = False
    last_display: Optional[np.ndarray] = None
    window_name = "AnyGrasp Pick and Place"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(cfg["camera"].get("color_width", 1280)), int(cfg["camera"].get("color_height", 720)))
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

        yolo_model, yolo_opts = load_yolo(cfg, project_root=PROJECT_ROOT)
        detector = anygrasp_utils.build_detector(anygrasp_cfg, args.anygrasp_checkpoint)

        print("=== Init robot ===")
        selected = selected_arm_config(robot_cfg.get("repo_root"))
        rebotarm = RebotArm()
        controller = RebotArmEndPose(rebotarm, arm_control_mode=selected.controller_mode)
        grasp_driver = GraspDriver(
            rebotarm,
            controller,
            gripper_config=robot_cfg.get("gripper"),
            repo_root=robot_cfg.get("repo_root"),
        )
        grasp_driver.start()
        robot_ready = True
        ik_checker = IkChecker(rebotarm)
        print(f"[Robot] mode: {selected.controller_mode}")

        if args.dry_run:
            if args.move_ready:
                print("[Robot] dry-run: initial ready motion only; all grasp/place/gripper motions disabled")
                _move_ready(controller, ready_cfg)
            else:
                print("[Robot] dry-run: stay at current observation pose; no ready motion")
        else:
            print("[Robot] Move ready")
            _move_ready(controller, ready_cfg)

        while True:
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                continue

            if frozen and last_display is not None:
                display = last_display.copy()
            else:
                display = color_bgr.copy()
                cv2.putText(display, "G=plan pick-place R=resume Q=quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(display, f"target={args.target_class} container={container_class}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                frozen = False
                last_display = None
                continue
            if key not in (ord("g"), ord("G"), ord(" ")):
                if vis is not None and not vis.poll():
                    vis.close()
                    vis = None
                continue

            print("\n[G] Capture target and container")
            snap_color, snap_depth = cam.get_frame()
            if snap_color is None or snap_depth is None:
                print("[G] Frame capture failed")
                continue

            try:
                T_cam2base = compose_cam_to_base_transform(
                    grasp_driver.get_tcp_pose(), T_hand_eye, cfg
                )
                _, detections = detect_objects(yolo_model, snap_color, yolo_opts)
                target = anygrasp_utils.select_target(detections, args.target_class)
                container = place_utils.select_container(
                    detections, container_class, exclude=target
                )
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
                )
                place6d, pre_place6d, container_base = place_utils.build_place_poses(
                    container_cam, T_cam2base, place_cfg
                )
                print(f"[Container] camera_xyz={np.round(container_cam, 4).tolist()} base_measured_xyz={np.round(container_base, 4).tolist()}")
                place_utils.validate_base_pose(place6d, safety, min_base_z_m=min_base_z, name="place")
                place_utils.validate_base_pose(pre_place6d, safety, min_base_z_m=min_base_z, name="pre-place")

                place_ik_results = []
                for name, pose in (("pre-place", pre_place6d), ("place", place6d)):
                    ok, error = ik_checker.check(*pose)
                    print(f"[Place IK] {name}: success={ok} error={error:.6f}")
                    place_ik_results.append((name, ok, error))

                failed_place_ik = [
                    (name, error)
                    for name, ok, error in place_ik_results
                    if not ok
                ]
                if failed_place_ik:
                    raise RuntimeError(
                        f"Place IK failed before any grasp motion: {failed_place_ik}"
                    )

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
                    target_margin_px=int(anygrasp_cfg.get("target_margin_px", 12)),
                    target_expand_ratio=float(anygrasp_cfg.get("target_expand_ratio", 1.35)),
                    max_grasp_width_m=GRIPPER_MAX_DISTANCE_M,
                    preferred_approach_cam=preferred_approach_cam,
                )
                print(f"[G] {result.status}")
                if result.best is None:
                    raise RuntimeError("No valid AnyGrasp target candidate")

                selected_grasp = _select_executable_grasp(
                    ik_checker,
                    result.grasps,
                    T_cam2base,
                    pregrasp_offset,
                    retreat_offset,
                    insertion_depth,
                    min_base_z,
                    anygrasp_safety=safety,
                )
                if selected_grasp is None:
                    raise RuntimeError("No target grasp passes safety and IK")
                best, grasp6d, pre6d, retreat6d = selected_grasp
                place_utils.validate_target_container_separation(grasp6d, place6d)
                print(
                    f"[Target] score={float(best.score):.4f} width={float(best.width):.4f}m "
                    f"depth={float(best.depth):.4f}m"
                )

                snap_display = graspnet_utils.draw_detections_overlay(
                    snap_color, detections, target, args.target_class
                )
                x1, y1, x2, y2 = container.bbox_xyxy
                cv2.rectangle(snap_display, (x1, y1), (x2, y2), (255, 180, 0), 3)
                cv2.putText(snap_display, f"PLACE: {container.class_name}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
                graspnet_utils.draw_best_grasp_projection(snap_display, best, K)
                snap_display = graspnet_utils.draw_status(
                    snap_display,
                    f"SNAPSHOT | {result.status}",
                    f"target={target.class_name}; container={container.class_name}",
                    frozen=True,
                    title="AnyGrasp Pick and Place",
                )
                last_display = snap_display
                frozen = True

                if not args.no_open3d:
                    if vis is None:
                        vis = graspnet_utils.Open3DGraspWindow(
                            "AnyGrasp Pick Candidates", int(anygrasp_cfg.get("top_k", 20))
                        )
                    vis.update(result.o3d_cloud, result.grasps)
            except Exception as exc:
                print(f"[G] Planning rejected before motion: {exc}")
                continue

            ok = _execute_place_sequence(
                controller,
                grasp_driver,
                grasp6d,
                pre6d,
                retreat6d,
                pre_place6d,
                place6d,
                ready_cfg,
                place_cfg,
                dry_run=args.dry_run,
                confirm_motion=args.confirm_motion,
            )
            print("[G] Pick-and-place completed" if ok else "[G] Pick-and-place stopped/cancelled")

    finally:
        print("\n[Exit] Pick-and-place cleanup")
        try:
            if not args.dry_run and robot_ready and grasp_driver is not None and controller is not None and getattr(controller, "_running", False):
                grasp_driver.release_gripper()
        except Exception as exc:
            print(f"[Exit] {exc}")
        try:
            if controller is not None and getattr(controller, "_running", False):
                controller.end()
            elif rebotarm is not None:
                rebotarm.disconnect()
        except Exception as exc:
            print(f"[Exit] {exc}")
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
