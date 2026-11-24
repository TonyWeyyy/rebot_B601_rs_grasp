"""
Visual grasping demo based on GraspNet-baseline or AnyGrasp, with YOLO.

Keys:
  G/Space: run the selected estimator on the current RGB-D frame and execute a grasp.
  R: resume live preview.
  Q/Esc: exit (safe-homes arm before disabling motors; releases gripper only in real mode).

Usage:
    python scripts/grasp.py --estimator anygrasp --dry-run
    python scripts/grasp.py --estimator anygrasp --target-class cup
    python scripts/grasp.py --estimator graspnet --dry-run
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

from core.ik_planner import IkChecker, select_executable_grasp as _select_executable_grasp  # noqa: E402
from core.preview_app import FpsMeter  # noqa: E402
from core.robot_session import (  # noqa: E402
    RobotSession,
    execute_grasp_sequence,
    move_ready as _move_ready,
    wait_motion as _wait_motion,
)
from drivers.camera import make_camera  # noqa: E402
from drivers.robot.grasp_driver import GRIPPER_MAX_DISTANCE_M  # noqa: E402
from utils.camera_utils import compose_cam_to_base_transform, configure_camera, load_config, load_hand_eye  # noqa: E402
from utils.target_selector import select_target, target_status_text  # noqa: E402
from utils.transforms import (  # noqa: E402
    canonicalize_parallel_gripper_tcp_rotation,
    graspnet_rotation_to_rebot_tcp_rotation,
    rotation_matrix_to_euler_zyx,
)
from utils.visualization import (  # noqa: E402
    Open3DGraspWindow,
    draw_best_grasp_projection,
    draw_detections_overlay,
    draw_status,
    visualization_grasps,
)
from utils.yolo_utils import YoloDetection, detect_objects, load_yolo as load_yolo_from_config  # noqa: E402


def _print_grasp(grasp: Any, estimator_name: str = "GraspNet") -> None:
    tcp_rotation = canonicalize_parallel_gripper_tcp_rotation(
        graspnet_rotation_to_rebot_tcp_rotation(grasp.rotation_matrix)
    )
    print(f"\n[G] Best {estimator_name} grasp:")
    print(f"  score={grasp.score:.4f} width={grasp.width:.4f} height={grasp.height:.4f} depth={grasp.depth:.4f}")
    print(f"  position_xyz={grasp.translation.tolist()}")
    print(f"  graspnet_rpy={rotation_matrix_to_euler_zyx(grasp.rotation_matrix).tolist()}")
    print(f"  tcp_rpy={rotation_matrix_to_euler_zyx(tcp_rotation).tolist()}")


def resolve_estimator_params(
    args: argparse.Namespace,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Merge CLI overrides with YAML config, ensuring explicit CLI flags always win."""
    grasp_cfg = cfg.get("grasp_pipeline", {}).get("grasp", {})
    estimator_name = str(args.estimator or grasp_cfg.get("estimator", "graspnet")).lower()
    if estimator_name not in ("graspnet", "anygrasp"):
        raise ValueError(f"Unsupported estimator: {estimator_name}")

    est_cfg = dict(cfg.get(estimator_name, {}))
    min_depth = float(args.min_depth if args.min_depth is not None else est_cfg.get("min_depth", 0.05))
    max_depth = float(args.max_depth if args.max_depth is not None else est_cfg.get("max_depth", 1.0))
    target_margin_px = int(
        args.target_margin_px if args.target_margin_px is not None else est_cfg.get("target_margin_px", 12)
    )
    target_expand_ratio = float(
        args.target_expand_ratio
        if args.target_expand_ratio is not None
        else est_cfg.get("target_expand_ratio", 1.35)
    )
    top_k = int(args.top_k if args.top_k is not None else est_cfg.get("top_k", 20 if estimator_name == "anygrasp" else 50))
    fusion_frames = max(
        1,
        int(args.fusion_frames if args.fusion_frames is not None else grasp_cfg.get("depth_fusion_frames", 3)),
    )
    pregrasp_offset_m = float(
        args.pregrasp_offset if args.pregrasp_offset is not None else grasp_cfg.get("pregrasp_offset_m", 0.08)
    )
    retreat_offset_m = float(
        args.retreat_offset if args.retreat_offset is not None else grasp_cfg.get("retreat_offset_m", pregrasp_offset_m)
    )
    insertion_depth_m = float(grasp_cfg.get("insertion_depth_m", 0.0))
    min_base_z_m = float(
        args.min_base_z if args.min_base_z is not None else grasp_cfg.get("min_base_z_m", 0.03)
    )
    max_ik_candidates = int(grasp_cfg.get("max_ik_candidates", 20))
    return {
        "estimator_name": estimator_name,
        "est_cfg": est_cfg,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "target_margin_px": target_margin_px,
        "target_expand_ratio": target_expand_ratio,
        "top_k": top_k,
        "fusion_frames": fusion_frames,
        "pregrasp_offset_m": pregrasp_offset_m,
        "retreat_offset_m": retreat_offset_m,
        "insertion_depth_m": insertion_depth_m,
        "min_base_z_m": min_base_z_m,
        "max_ik_candidates": max_ik_candidates,
        "motion_cfg": grasp_cfg,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraspNet / AnyGrasp robot grasp demo")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--estimator", choices=("graspnet", "anygrasp"), default=None,
                        help="override grasp_pipeline.grasp.estimator")
    parser.add_argument("--checkpoint", default=None, help="GraspNet checkpoint override")
    parser.add_argument("--anygrasp-checkpoint", default=None,
                        help="AnyGrasp checkpoint override; defaults to anygrasp.checkpoint in YAML")
    parser.add_argument("--dry-run", action="store_true",
                        help="estimate and check IK only; never execute grasp/gripper motions (safe-homes on exit)")
    parser.add_argument(
        "--move-ready",
        action="store_true",
        help="with --dry-run, move only to ready_pose at startup; grasp and gripper motions remain disabled",
    )
    parser.add_argument(
        "--confirm-motion",
        action="store_true",
        help="after pose selection, require typing YES before grasp/gripper motion",
    )
    parser.add_argument("--camera-type", choices=("realsense_d435i", "realsense_d405", "orbbec_gemini2"), default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--fusion-frames", type=int, default=None, help="depth median fusion frame count at capture")
    parser.add_argument("--num-point", type=int, default=None)
    parser.add_argument("--num-view", type=int, default=300)
    parser.add_argument("--collision-thresh", type=float, default=None)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--min-depth", type=float, default=None, help="meters (overrides YAML)")
    parser.add_argument("--max-depth", type=float, default=None, help="meters (overrides YAML)")
    parser.add_argument("--top-k", type=int, default=None, help="Open3D top-k grasps (overrides YAML)")
    parser.add_argument("--target-class", default=None)
    parser.add_argument("--extra-yolo-class", action="append", default=[], help="add open-vocabulary YOLO class")
    parser.add_argument("--target-margin-px", type=int, default=None)
    parser.add_argument("--target-expand-ratio", type=float, default=None, help="YOLO bbox expansion ratio")
    parser.add_argument("--no-yolo", action="store_true", help="disable YOLO and use full-scene inference")
    parser.add_argument("--yolo-model", default=None)
    parser.add_argument("--yolo-device", default=None)
    parser.add_argument("--yolo-conf", type=float, default=None)
    parser.add_argument("--yolo-iou", type=float, default=None)
    parser.add_argument("--infer-every-live", type=int, default=None)
    parser.add_argument("--pregrasp-offset", type=float, default=None, help="meters")
    parser.add_argument("--retreat-offset", type=float, default=None, help="meters")
    parser.add_argument("--min-base-z", type=float, default=None, help="minimum executable TCP z in base frame, meters")
    parser.add_argument("--no-open3d", action="store_true", help="do not open Open3D after inference")
    parser.add_argument(
        "--open3d-grasps",
        choices=("final", "bbox", "pre-bbox"),
        default="final",
        help="Open3D grasp set: final, bbox, or pre-bbox",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = configure_camera(load_config(Path(args.config)), args)
    params = resolve_estimator_params(args, cfg)

    estimator_name = params["estimator_name"]
    anygrasp_cfg = dict(cfg.get("anygrasp", {}))
    graspnet_cfg = dict(cfg.get("graspnet", {}))
    anygrasp_safety = dict(anygrasp_cfg.get("safety", {}))
    anygrasp_safety["max_gripper_width"] = min(
        GRIPPER_MAX_DISTANCE_M,
        float(anygrasp_cfg.get("max_gripper_width", GRIPPER_MAX_DISTANCE_M)),
    )

    cam_cfg = cfg["camera"]
    print(f"=== Init camera: {cam_cfg['type']} {cam_cfg.get('color_width')}x{cam_cfg.get('color_height')}@{cam_cfg.get('fps')} ===")
    cam = make_camera(cfg)

    last_detections: list[YoloDetection] = []
    selected_target: Optional[YoloDetection] = None
    last_target_status = "target detector warming up..."
    status = "warming up camera..."
    frozen = False
    last_display: Optional[np.ndarray] = None
    frame_index = 0
    fps_meter = FpsMeter()
    window_name = f"Main - {estimator_name} Grasp"
    vis: Optional[Open3DGraspWindow] = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(cam_cfg.get("color_width", 1280)), int(cam_cfg.get("color_height", 720)))
    print("\n[Keys] G/SPACE=grasp  R=resume  Q/ESC=quit\n")

    robot = RobotSession(cfg, dry_run=args.dry_run, move_ready_in_dry_run=args.move_ready, init_ik=True)
    T_hand_eye: Optional[np.ndarray] = None

    try:
        cam.open()
        cam.warm_up(args.warmup)
        K = cam.K.astype(np.float64)
        print("Camera intrinsics:\n", K)

        cam_type = str(cam_cfg.get("type", "")).lower()
        T_hand_eye, hand_eye_mode = load_hand_eye(PROJECT_ROOT, cam_type)
        if T_hand_eye is None or hand_eye_mode != "eye_in_hand":
            print("[WARN] Hand-eye calibration unavailable; grasp execution disabled")
            T_hand_eye = None

        extra_classes = list(args.extra_yolo_class or [])
        if args.target_class:
            extra_classes.append(args.target_class)

        print("=== Load models ===")
        yolo_model, yolo_opts = load_yolo_from_config(
            cfg,
            project_root=PROJECT_ROOT,
            no_yolo=args.no_yolo,
            model_override=args.yolo_model,
            device_override=args.yolo_device,
            conf_override=args.yolo_conf,
            iou_override=args.yolo_iou,
            infer_every_override=args.infer_every_live,
            extra_classes=extra_classes,
        )
        last_target_status = (
            f"YOLO disabled: full-scene {estimator_name}"
            if yolo_model is None
            else "target detector warming up..."
        )

        print(f"=== Load estimator: {estimator_name} ===")
        if estimator_name == "anygrasp":
            import utils.anygrasp_utils as est_mod
            net = est_mod.build_detector(anygrasp_cfg, args.anygrasp_checkpoint)
        else:
            import utils.graspnet_utils as est_mod
            checkpoint = args.checkpoint or graspnet_cfg.get("checkpoint", "checkpoint-rs.tar")
            net = est_mod.build_net(checkpoint, args.num_view)

        print("=== Init robot ===")
        robot.start()

        while True:
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                continue

            frame_index += 1
            fps_value = fps_meter.tick()

            if not frozen and yolo_model is not None and (frame_index == 1 or frame_index % int(yolo_opts["infer_every"]) == 0):
                try:
                    _, last_detections = detect_objects(yolo_model, color_bgr, yolo_opts)
                    selected_target = select_target(last_detections, args.target_class)
                    last_target_status = target_status_text(selected_target, last_detections, args.target_class)
                except Exception as exc:
                    last_detections = []
                    selected_target = None
                    last_target_status = f"YOLO failed: {exc}"

            if frozen and last_display is not None:
                display = last_display.copy()
            else:
                display_base = color_bgr
                if yolo_model is not None:
                    display_base = draw_detections_overlay(color_bgr, last_detections, selected_target, args.target_class)
                display = draw_status(
                    display_base,
                    f"LIVE {fps_value:.1f}fps | {status}",
                    last_target_status,
                    title=f"Main - {estimator_name} Grasp",
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
                status = "live preview"
                continue

            if key in (ord("g"), ord("G"), ord(" ")):
                print(f"\n[G] Capture ({params['fusion_frames']} frames) and run {estimator_name}")
                snap_color, snap_depth = cam.get_fused_frame(params["fusion_frames"])
                if snap_color is None or snap_depth is None:
                    print("[G] Frame capture failed")
                    continue

                T_cam2base_capture = None
                preferred_approach_cam = None
                if T_hand_eye is not None:
                    T_cam2base_capture = compose_cam_to_base_transform(robot.get_tcp_pose(), T_hand_eye, cfg)
                    if estimator_name == "anygrasp":
                        preferred_approach_cam = T_cam2base_capture[:3, :3].T @ np.array([0.0, 0.0, -1.0])

                try:
                    if estimator_name == "anygrasp":
                        result = est_mod.infer_frame(
                            net,
                            snap_color,
                            snap_depth,
                            K,
                            cfg=anygrasp_cfg,
                            min_depth=params["min_depth"],
                            max_depth=params["max_depth"],
                            yolo_model=yolo_model,
                            yolo_opts=yolo_opts,
                            target_class=args.target_class,
                            target_margin_px=params["target_margin_px"],
                            target_expand_ratio=params["target_expand_ratio"],
                            max_grasp_width_m=GRIPPER_MAX_DISTANCE_M,
                            preferred_approach_cam=preferred_approach_cam,
                            build_o3d=not args.no_open3d,
                        )
                    else:
                        num_point = int(args.num_point if args.num_point is not None else graspnet_cfg.get("num_point", 20000))
                        collision_thresh = float(
                            args.collision_thresh
                            if args.collision_thresh is not None
                            else graspnet_cfg.get("collision_thresh", 0.01)
                        )
                        result = est_mod.infer_frame(
                            net,
                            snap_color,
                            snap_depth,
                            K,
                            num_point=num_point,
                            min_depth=params["min_depth"],
                            max_depth=params["max_depth"],
                            collision_thresh=collision_thresh,
                            voxel_size=args.voxel_size,
                            yolo_model=yolo_model,
                            yolo_opts=yolo_opts,
                            target_class=args.target_class,
                            target_margin_px=params["target_margin_px"],
                            target_expand_ratio=params["target_expand_ratio"],
                            max_grasp_width_m=GRIPPER_MAX_DISTANCE_M,
                            build_o3d=not args.no_open3d,
                        )
                except Exception as exc:
                    status = f"inference failed: {exc}"
                    print(f"[G] {status}")
                    continue

                status = result.status
                last_target_status = result.target_status
                last_detections = result.detections
                selected_target = result.selected_target
                vis_grasps = visualization_grasps(result, args.open3d_grasps)

                print(f"[G] {status}")
                if not args.no_open3d and result.o3d_cloud is not None:
                    try:
                        if vis is None:
                            vis = Open3DGraspWindow(f"{estimator_name} Grasps", params["top_k"])
                        vis.update(result.o3d_cloud, vis_grasps)
                        print(f"[G] Open3D {args.open3d_grasps} candidates={len(vis_grasps)}")
                    except Exception as exc:
                        print(f"[G] Open3D failed: {exc}")
                        if vis is not None:
                            vis.close()
                            vis = None

                if result.best is None:
                    print(f"[G] No valid {estimator_name} grasp")
                    continue

                frozen = True
                display_base = snap_color
                if yolo_model is not None:
                    display_base = draw_detections_overlay(snap_color, last_detections, selected_target, args.target_class)
                snap_display = draw_status(
                    display_base,
                    f"SNAPSHOT | {status}",
                    last_target_status,
                    frozen=True,
                    title=f"Main - {estimator_name} Grasp",
                )
                last_display = snap_display

                if T_hand_eye is None:
                    draw_best_grasp_projection(snap_display, result.best, K)
                    last_display = snap_display
                    print("[G] Hand-eye calibration unavailable")
                    continue

                selected_exec = _select_executable_grasp(
                    robot.ik_checker,
                    result.grasps,
                    T_cam2base_capture,
                    params["pregrasp_offset_m"],
                    params["retreat_offset_m"],
                    params["insertion_depth_m"],
                    params["min_base_z_m"],
                    anygrasp_safety=anygrasp_safety if estimator_name == "anygrasp" else None,
                    already_nms=result.already_nms,
                    max_ik_candidates=params["max_ik_candidates"],
                )
                if selected_exec is None:
                    print(f"[G] No IK-reachable grasp above min_base_z={params['min_base_z_m']:.3f}m")
                    continue
                best, grasp6d, pre6d, retreat6d = selected_exec

                _print_grasp(best, estimator_name)
                draw_best_grasp_projection(snap_display, best, K)
                last_display = snap_display

                execute_grasp_sequence(
                    robot.controller,
                    robot.grasp_driver,
                    grasp6d,
                    pre6d,
                    retreat6d,
                    robot.ready_cfg,
                    motion_cfg=params["motion_cfg"],
                    dry_run=args.dry_run,
                    confirm_motion=args.confirm_motion,
                    arm=robot.arm,
                    n_joints=robot.n_joints,
                )

            if vis is not None and not vis.poll():
                vis.close()
                vis = None

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
