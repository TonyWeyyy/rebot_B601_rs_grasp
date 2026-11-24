"""Unified robot session lifecycle, motion settling, and pick/place execution."""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from core.ik_planner import IkChecker


def wait_motion(
    controller: Any,
    duration: float,
    extra: float = 0.6,
    *,
    arm: Any = None,
    n_joints: Optional[int] = None,
    settle_tol: Optional[float] = 0.03,
    settle_timeout: float = 1.0,
) -> bool:
    """Wait for trajectory send thread to finish, plus optional joint settling check."""
    thread = getattr(controller, "_send_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=float(duration) + float(extra) + 2.0)
    else:
        time.sleep(max(0.0, float(duration) + float(extra)))

    if arm is None or settle_tol is None:
        return True

    q_target = getattr(controller, "_q_target", None)
    if q_target is None:
        return True
    n = int(n_joints) if n_joints is not None else len(q_target)
    deadline = time.monotonic() + max(0.0, float(settle_timeout))
    while True:
        try:
            q_now = np.asarray(arm.get_state(request_feedback=False)[0][:n], dtype=np.float64)
            if float(np.max(np.abs(q_now - np.asarray(q_target[:n], dtype=np.float64)))) <= float(settle_tol):
                return True
        except Exception:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)


def move_ready(
    controller: Any,
    ready_cfg: dict[str, Any],
    *,
    duration_override: Optional[float] = None,
    arm: Any = None,
    n_joints: Optional[int] = None,
) -> bool:
    duration = float(duration_override if duration_override is not None else ready_cfg.get("duration", 3.0))
    ok = controller.move_to_traj(
        x=float(ready_cfg.get("x", 0.25)),
        y=float(ready_cfg.get("y", 0.0)),
        z=float(ready_cfg.get("z", 0.35)),
        roll=float(ready_cfg.get("roll", 0.0)),
        pitch=float(ready_cfg.get("pitch", 1.2)),
        yaw=float(ready_cfg.get("yaw", 0.0)),
        duration=duration,
    )
    if ok:
        wait_motion(controller, duration, arm=arm, n_joints=n_joints)
    return bool(ok)


def _confirm(expected: str, prompt: str, enabled: bool) -> bool:
    if not enabled:
        return True
    return input(prompt).strip() == expected


def _gripper_still_holding(grasp_driver: Any) -> bool:
    if hasattr(grasp_driver, "is_holding"):
        return bool(grasp_driver.is_holding())
    return True


def execute_grasp_sequence(
    controller: Any,
    grasp_driver: Any,
    grasp6d: tuple[float, ...],
    pre6d: tuple[float, ...],
    retreat6d: Optional[tuple[float, ...]],
    ready_cfg: dict[str, Any],
    *,
    motion_cfg: Optional[dict[str, Any]] = None,
    dry_run: bool = False,
    confirm_motion: bool = False,
    arm: Any = None,
    n_joints: Optional[int] = None,
) -> bool:
    """Execute open -> pregrasp -> grasp -> close -> retreat -> ready."""
    xg, yg, zg, rxg, ryg, rzg = grasp6d
    xp, yp, zp, rxp, ryp, rzp = pre6d
    print(f"[Grasp] pregrasp xyz=({xp:+.3f},{yp:+.3f},{zp:+.3f}) rpy=({rxp:+.3f},{ryp:+.3f},{rzp:+.3f})")
    print(f"[Grasp] grasp    xyz=({xg:+.3f},{yg:+.3f},{zg:+.3f}) rpy=({rxg:+.3f},{ryg:+.3f},{rzg:+.3f})")
    if retreat6d is not None:
        xr, yr, zr, rxr, ryr, rzr = retreat6d
        print(f"[Grasp] retreat  xyz=({xr:+.3f},{yr:+.3f},{zr:+.3f}) rpy=({rxr:+.3f},{ryr:+.3f},{rzr:+.3f})")

    if dry_run:
        print("[Grasp] dry run; skip motion")
        return False

    if not _confirm(
        "YES",
        "[Safety] Type YES to allow grasp/gripper motion; anything else cancels: ",
        confirm_motion,
    ):
        print("[Grasp] Motion cancelled by user")
        return False

    mcfg = motion_cfg or {}
    pre_duration = float(mcfg.get("pregrasp_duration_s", 2.5))
    grasp_duration = float(mcfg.get("grasp_duration_s", 2.0))
    retreat_duration = float(mcfg.get("retreat_duration_s", 2.0))
    ready_duration = float(mcfg.get("ready_duration_s", ready_cfg.get("duration", 3.0)))

    print("[Grasp] Open gripper")
    grasp_driver.open_gripper()

    print("[Grasp] Move to pregrasp")
    if not controller.move_to_traj(xp, yp, zp, rxp, ryp, rzp, duration=pre_duration):
        print("[Grasp] Pregrasp IK failed")
        return False
    wait_motion(controller, pre_duration, arm=arm, n_joints=n_joints)

    print("[Grasp] Move to grasp")
    if not controller.move_to_traj(xg, yg, zg, rxg, ryg, rzg, duration=grasp_duration):
        print("[Grasp] Grasp IK failed; returning to ready")
        move_ready(controller, ready_cfg, duration_override=ready_duration, arm=arm, n_joints=n_joints)
        return False
    wait_motion(controller, grasp_duration, arm=arm, n_joints=n_joints)

    print("[Grasp] Closing")
    ok = bool(grasp_driver.grasp())
    print("[Grasp] Holding object" if ok else "[Grasp] Empty grasp")

    if retreat6d is not None:
        xr, yr, zr, rxr, ryr, rzr = retreat6d
        print("[Grasp] Retreat")
        if controller.move_to_traj(xr, yr, zr, rxr, ryr, rzr, duration=retreat_duration):
            wait_motion(controller, retreat_duration, arm=arm, n_joints=n_joints)

    if ok and not _gripper_still_holding(grasp_driver):
        print("[Grasp] Object slipped during retreat")
        ok = False

    print("[Grasp] Return ready")
    move_ready(controller, ready_cfg, duration_override=ready_duration, arm=arm, n_joints=n_joints)
    return ok


def execute_pick_and_place_sequence(
    controller: Any,
    grasp_driver: Any,
    grasp6d: tuple[float, ...],
    pre6d: tuple[float, ...],
    retreat6d: tuple[float, ...],
    pre_place6d: tuple[float, ...],
    place6d: tuple[float, ...],
    ready_cfg: dict[str, Any],
    place_cfg: dict[str, Any],
    *,
    dry_run: bool = False,
    confirm_motion: bool = False,
    arm: Any = None,
    n_joints: Optional[int] = None,
) -> bool:
    """Execute pick -> retreat -> ready -> pre-place -> place -> release -> pre-place -> ready."""
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

    pre_duration = float(place_cfg.get("pregrasp_duration_s", 1.5))
    grasp_duration = float(place_cfg.get("grasp_duration_s", 1.2))
    retreat_duration = float(place_cfg.get("retreat_duration_s", 1.5))
    ready_duration = float(place_cfg.get("ready_duration_s", 2.0))
    pre_place_duration = float(place_cfg.get("pre_place_duration_s", 2.0))
    place_duration = float(place_cfg.get("place_duration_s", 1.2))
    retract_to_pre_place = bool(place_cfg.get("retract_to_pre_place", True))

    print("[Pick] Open gripper")
    grasp_driver.open_gripper()

    print("[Pick] Move to pregrasp")
    if not controller.move_to_traj(*pre6d, duration=pre_duration):
        print("[Pick] Pregrasp controller/IK failed")
        return False
    wait_motion(controller, pre_duration, arm=arm, n_joints=n_joints)

    print("[Pick] Move to grasp")
    if not controller.move_to_traj(*grasp6d, duration=grasp_duration):
        print("[Pick] Grasp controller/IK failed; returning to ready")
        move_ready(controller, ready_cfg, duration_override=ready_duration, arm=arm, n_joints=n_joints)
        return False
    wait_motion(controller, grasp_duration, arm=arm, n_joints=n_joints)

    print("[Pick] Close gripper")
    if not grasp_driver.grasp():
        print("[Pick] Empty grasp; retreating vertically before returning to ready")
        if controller.move_to_traj(*retreat6d, duration=retreat_duration):
            wait_motion(controller, retreat_duration, arm=arm, n_joints=n_joints)
        move_ready(controller, ready_cfg, duration_override=ready_duration, arm=arm, n_joints=n_joints)
        return False

    print("[Pick] Holding object; retreat")
    if not controller.move_to_traj(*retreat6d, duration=retreat_duration):
        print("[Pick] Retreat failed; object remains held, inspect the arm before continuing")
        return False
    wait_motion(controller, retreat_duration, arm=arm, n_joints=n_joints)

    if not _gripper_still_holding(grasp_driver):
        print("[Pick] Object lost during retreat; returning to ready without placement")
        move_ready(controller, ready_cfg, duration_override=ready_duration, arm=arm, n_joints=n_joints)
        return False

    print("[Pick] Transfer to ready with object")
    move_ready(controller, ready_cfg, duration_override=ready_duration, arm=arm, n_joints=n_joints)

    if not _gripper_still_holding(grasp_driver):
        print("[Pick] Object lost while moving to ready; placement aborted")
        return False

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
    wait_motion(controller, pre_place_duration, arm=arm, n_joints=n_joints)

    print("[Place] Move to release pose")
    if not controller.move_to_traj(*place6d, duration=place_duration):
        print("[Place] Place failed; return to ready with object")
        move_ready(controller, ready_cfg, duration_override=ready_duration, arm=arm, n_joints=n_joints)
        return False
    wait_motion(controller, place_duration, arm=arm, n_joints=n_joints)

    print("[Place] Release object into container")
    grasp_driver.open_gripper(timeout=2.0)
    time.sleep(float(place_cfg.get("release_wait_s", 0.5)))

    if retract_to_pre_place:
        print("[Place] Retract vertically to pre-place")
        if controller.move_to_traj(*pre_place6d, duration=place_duration):
            wait_motion(controller, place_duration, arm=arm, n_joints=n_joints)

    print("[Place] Return ready")
    move_ready(controller, ready_cfg, duration_override=ready_duration, arm=arm, n_joints=n_joints)
    grasp_driver.open_gripper(distance_m=0.0, timeout=2.0)
    return True


class RobotSession:
    """Managed lifecycle for reBotArm + RebotArmEndPose + GraspDriver + IkChecker.

    Guarantees that ``dry_run=True`` never triggers unexpected arm or gripper
    motion at startup (unless ``move_ready_in_dry_run=True``) or at exit.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        dry_run: bool = False,
        move_ready_in_dry_run: bool = False,
        init_ik: bool = True,
    ) -> None:
        self._cfg = cfg
        self._robot_cfg = cfg.get("robot", {})
        self._ready_cfg = self._robot_cfg.get(
            "ready_pose",
            {"x": 0.25, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.2, "yaw": 0.0, "duration": 3.0},
        )
        self.dry_run = bool(dry_run)
        self.move_ready_in_dry_run = bool(move_ready_in_dry_run)
        self._init_ik = bool(init_ik)

        self.arm: Any = None
        self.controller: Any = None
        self.grasp_driver: Any = None
        self.ik_checker: Optional[IkChecker] = None
        self.controller_mode: str = ""
        self._started = False

    @property
    def ready_cfg(self) -> dict[str, Any]:
        return self._ready_cfg

    @property
    def n_joints(self) -> Optional[int]:
        if self.arm is None:
            return None
        arm_group = self.arm.groups.get("arm")
        return arm_group.num_joints if arm_group is not None else None

    def start(self) -> "RobotSession":
        from drivers.robot.grasp_driver import GraspDriver, prioritize_env_pinocchio, selected_arm_config

        prioritize_env_pinocchio()
        from reBotArm_control_py.actuator import RebotArm
        from reBotArm_control_py.controllers import RebotArmEndPose

        selected = selected_arm_config(self._robot_cfg.get("repo_root"))
        self.controller_mode = selected.controller_mode
        self.arm = RebotArm()
        self.controller = RebotArmEndPose(self.arm, arm_control_mode=selected.controller_mode)
        self.grasp_driver = GraspDriver(
            self.arm,
            self.controller,
            gripper_config=self._robot_cfg.get("gripper"),
            repo_root=self._robot_cfg.get("repo_root"),
        )
        self.grasp_driver.start()
        self._started = True
        if self._init_ik:
            self.ik_checker = IkChecker(self.arm)
        print(f"[Robot] mode: {selected.controller_mode}")

        if self.dry_run:
            if self.move_ready_in_dry_run:
                print("[Robot] dry-run: move to ready observation pose only; grasp and gripper motions disabled")
                self.move_ready()
            else:
                print("[Robot] dry-run: stay at current observation pose; no startup or exit motion")
        else:
            print("[Robot] Move ready")
            self.move_ready()
        return self

    def move_ready(self, duration_override: Optional[float] = None) -> bool:
        return move_ready(
            self.controller,
            self._ready_cfg,
            duration_override=duration_override,
            arm=self.arm,
            n_joints=self.n_joints,
        )

    def get_tcp_pose(self) -> np.ndarray:
        return self.grasp_driver.get_tcp_pose()

    def close(self) -> None:
        if not self._started and self.arm is None:
            return
        safe_home = not self.dry_run
        release = not self.dry_run
        if self.dry_run:
            print("\n[Exit] Dry-run cleanup (stopping control loop without homing or gripper motion)")
        else:
            print("\n[Exit] Release gripper and home")

        try:
            if self.grasp_driver is not None:
                self.grasp_driver.stop(safe_home=safe_home, release=release)
            elif self.controller is not None and getattr(self.controller, "_running", False):
                if safe_home:
                    self.controller.end()
                else:
                    self.arm.stop_control_loop()
                    self.controller._running = False
                    self.arm.disconnect()
            elif self.arm is not None:
                self.arm.disconnect()
        except Exception as exc:
            print(f"[Exit] {exc}")
        finally:
            self._started = False

    def __enter__(self) -> "RobotSession":
        return self.start()

    def __exit__(self, *_) -> None:
        self.close()
