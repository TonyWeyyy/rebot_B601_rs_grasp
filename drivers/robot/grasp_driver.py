"""Small grasp-side helper for reBotArm visual grasping.

The SDK owns arm connection, mode switching, Cartesian planning, gravity
compensation, and the control loop. This module provides only the extra
gripper and pose helpers used by the vision workflows.

selected_arm_config(): read the SDK hardware YAML and choose controller mode.

GraspDriver:
  start(): start SDK control and attach gripper tick handling.
  stop(): cleanly stop control loop, optionally releasing and homing.
  open_gripper(): open to a requested jaw distance.
  set_gripper_distance_async(): non-blocking position command.
  close_gripper_async(): optional delayed non-blocking close to rest.
  grasp(): close with force control and report object contact.
  is_holding(): verify the gripper is still holding an object above hard stop.
  release_gripper(): open and return the gripper to closed rest.
  get_gripper_state(): return cached position, velocity, and torque.
  get_tcp_pose(): return the current TCP pose as a 4x4 matrix.
"""

from __future__ import annotations

import site
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml


def prioritize_env_pinocchio() -> None:
    """Prevent user-site cmeel Pinocchio from shadowing a conda-installed Pinocchio.

    When ``~/.local/.../cmeel.prefix`` is on ``sys.path`` ahead of the active
    environment's ``site-packages`` while ``LD_LIBRARY_PATH`` points to the
    environment's ``lib/``, Python loads the user-site ``pinocchio`` binding
    against the environment's ``libpinocchio_default.so`` and fails with an
    ``undefined symbol`` error. If the active environment provides its own
    ``pinocchio`` package, demote ``cmeel.prefix`` entries to the end of
    ``sys.path`` while keeping other user-site packages (e.g. ``pyorbbecsdk``)
    untouched.
    """
    if "pinocchio" in sys.modules:
        return
    try:
        env_site_dirs = [Path(p) for p in site.getsitepackages()]
    except Exception:
        return
    has_env_pinocchio = any((d / "pinocchio").is_dir() for d in env_site_dirs)
    if not has_env_pinocchio:
        return
    cmeel_entries = [p for p in sys.path if "cmeel.prefix" in p]
    if not cmeel_entries:
        return
    sys.path[:] = [p for p in sys.path if "cmeel.prefix" not in p] + cmeel_entries


_CAMERAWS_ROOT = Path(__file__).resolve().parents[2]
_REBOT_REPO_NAME = "reBotArm_control_py"
_DEFAULT_REBOT_REPO = _CAMERAWS_ROOT / "sdk" / _REBOT_REPO_NAME

GRIPPER_MAX_DISTANCE_M = 0.09

_DEFAULT_GRIPPER_BY_ARM = {
    "dm": {
        "angle_open": 5.0,
        "counterclockwise": True,
        "tau_max": 1.5,
        "close_torque": 1.0,
        "default_force": 0.30,
        "post_contact_close_angle": 0.0,
        "hard_stop_angle": 0.05,
        "stall_vel": 0.05,
        "contact_torque_ratio": 0.45,
        "post_contact_timeout_s": 0.30,
        "debug": False,
    },
    "rs": {
        "angle_open": 5.0,
        "counterclockwise": False,
        "tau_max": 1.5,
        "close_torque": 1.5,
        "default_force": 0.80,
        "post_contact_close_angle": 0.12,
        # 实测（RS + 柔性夹爪）：空夹机械限位在 pos≈-0.75，留 0.10 rad 余量。
        "hard_stop_angle": -0.65,
        "stall_vel": 0.20,
        "contact_torque_ratio": 0.35,
        "post_contact_timeout_s": 0.30,
        "debug": False,
    },
}


@dataclass(frozen=True)
class SelectedArmConfig:
    arm_type: str
    controller_mode: str


def _is_rebot_repo_root(path: Path) -> bool:
    pkg = path / _REBOT_REPO_NAME
    return (
        path.is_dir()
        and (pkg / "actuator" / "rebotarm.py").is_file()
        and (path / "config" / "rebotarm.yaml").is_file()
    )


def find_rebot_repo_root(hint: Optional[str] = None) -> Path:
    repo = Path(hint).expanduser() if hint else _DEFAULT_REBOT_REPO
    if not repo.is_absolute():
        repo = (_CAMERAWS_ROOT / repo).resolve()
    else:
        repo = repo.resolve()
    if _is_rebot_repo_root(repo):
        return repo
    raise FileNotFoundError(f"reBotArm_control_py repo not found: {repo}")


def ensure_rebot_sdk_in_syspath(hint: Optional[str] = None) -> Path:
    repo = find_rebot_repo_root(hint)
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a YAML mapping")
    return data


def selected_hardware_yaml(repo_root: Optional[str] = None) -> Path:
    repo = find_rebot_repo_root(repo_root)
    config_dir = repo / "config"
    global_cfg = _read_yaml(config_dir / "rebotarm.yaml")
    hw_yaml = global_cfg.get("hardware_yaml")
    if not hw_yaml:
        raise ValueError(f"{config_dir / 'rebotarm.yaml'} missing hardware_yaml")

    hw_path = Path(str(hw_yaml))
    if not hw_path.is_absolute():
        hw_path = config_dir / hw_path
    hw_path = hw_path.resolve()
    if not hw_path.is_file():
        raise FileNotFoundError(f"Hardware config not found: {hw_path}")
    return hw_path


def selected_arm_config(repo_root: Optional[str] = None) -> SelectedArmConfig:
    """Return the selected arm type and matching SDK controller mode."""
    hw_path = selected_hardware_yaml(repo_root)
    stem = hw_path.stem.lower()
    if stem.endswith("_dm") or stem == "dm":
        return SelectedArmConfig(arm_type="dm", controller_mode="posvel")
    if stem.endswith("_rs") or stem == "rs":
        return SelectedArmConfig(arm_type="rs", controller_mode="mit")
    raise ValueError(f"Cannot infer arm type from hardware config: {hw_path}")


class GraspDriver:
    MAX_DISTANCE_M = GRIPPER_MAX_DISTANCE_M
    _STATE_IDLE = "idle"
    _STATE_POSITION = "position"
    _STATE_CLOSING = "closing"
    _STATE_TIGHTENING = "tightening"
    _STATE_HOLDING = "holding"
    _contact_torque_ratio = 0.35

    def __init__(
        self,
        arm: Any,
        controller: Any,
        gripper_config: Optional[dict] = None,
        repo_root: Optional[str] = None,
    ) -> None:
        self._arm = arm
        self._controller = controller
        self._arm_group = arm.groups.get("arm")
        self._gripper_group = arm.groups.get("gripper")
        if self._arm_group is None:
            raise ValueError("Hardware config missing groups.arm")
        if self._gripper_group is None or not arm.has_gripper:
            raise ValueError("Hardware config missing groups.gripper")
        gripper_jcfgs = getattr(self._gripper_group, "_jcfgs", [])
        if not gripper_jcfgs:
            raise ValueError("groups.gripper has no joints")
        self._gripper_name = gripper_jcfgs[0].name
        self._gripper_motor: Any = None

        prioritize_env_pinocchio()
        from reBotArm_control_py.kinematics import compute_fk, load_robot_model, pad_q_for_model

        self._compute_fk = compute_fk
        self._pad_q_for_model = pad_q_for_model
        self._model = load_robot_model()
        self._n = self._arm_group.num_joints

        selected = selected_arm_config(repo_root)
        defaults = _DEFAULT_GRIPPER_BY_ARM[selected.arm_type]
        gcfg = {**defaults, **((gripper_config or {}).get(selected.arm_type) or {})}
        self._configure_gripper_params(gcfg)

        self._state_lock = threading.Lock()
        self._state = self._STATE_IDLE
        self._target_pos = 0.0
        self._start_pos = 0.0
        self._contact_pos = 0.0
        self._hold_torque = self._default_force
        self._position_reached = True
        self._grasp_result: Optional[bool] = None
        self._last_gripper_state: Optional[tuple[float, float, float]] = None
        self._debug_last_print = 0.0

    def _configure_gripper_params(self, gcfg: dict[str, Any]) -> None:
        post_contact_close_angle = float(gcfg["post_contact_close_angle"])
        if not np.isfinite(post_contact_close_angle) or not 0.0 <= post_contact_close_angle <= 0.30:
            raise ValueError("gripper post_contact_close_angle must be within [0.0, 0.30] rad")

        contact_torque_ratio = float(gcfg.get("contact_torque_ratio", 0.35))
        if not np.isfinite(contact_torque_ratio) or not 0.0 < contact_torque_ratio < 1.0:
            raise ValueError("gripper contact_torque_ratio must be within (0.0, 1.0)")

        stall_vel = float(gcfg.get("stall_vel", 0.20))
        if not np.isfinite(stall_vel) or stall_vel <= 0.0:
            raise ValueError("gripper stall_vel must be positive")

        post_contact_timeout_s = float(gcfg.get("post_contact_timeout_s", 0.30))
        if not np.isfinite(post_contact_timeout_s) or post_contact_timeout_s < 0.0:
            raise ValueError("gripper post_contact_timeout_s must be non-negative")

        motion_sign = 1.0 if bool(gcfg.get("counterclockwise")) else -1.0
        self._angle_open = -motion_sign * abs(float(gcfg["angle_open"]))
        self._tau_max = abs(float(gcfg["tau_max"]))
        self._open_sign = 1.0 if self._angle_open >= 0.0 else -1.0
        self._close_sign = motion_sign
        self._close_torque = self._close_sign * abs(float(gcfg["close_torque"]))
        self._default_force = self._close_sign * abs(float(gcfg["default_force"]))
        self._open_soft_limit = 0.98 * self._angle_open
        self._open_lo = min(self._open_soft_limit, 0.0)
        self._open_hi = max(self._open_soft_limit, 0.0)
        self._hard_stop_angle = self._open_sign * float(gcfg.get("hard_stop_angle", -0.65))
        self._arrive_tol = 0.12
        self._kp_move = 5.0
        self._kd_move = 1.0
        self._kd_close = 0.5
        self._stall_vel = stall_vel
        self._contact_torque_ratio = contact_torque_ratio
        self._startup_dist = 0.30
        self._post_contact_close_angle = post_contact_close_angle
        self._post_contact_arrive_tol = 0.03
        self._post_contact_timeout_s = post_contact_timeout_s
        self._tighten_deadline = 0.0
        self._debug = bool(gcfg.get("debug", False))

    def start(self) -> None:
        """Start the SDK arm controller and let this driver own the gripper."""
        if getattr(self._controller, "_running", False):
            return

        self._controller._has_gripper = False
        self._arm.connect()
        self._gripper_motor = self._gripper_group._mm[self._gripper_name]
        if self._arm_group:
            if self._controller._arm_control_mode == "mit":
                self._arm_group.mode_mit(
                    kp=self._arm_group._mit_kp,
                    kd=self._arm_group._mit_kd,
                )
            else:
                self._arm_group.mode_pos_vel()
            self._arm_group.enable()

        self._gripper_group.mode_mit()
        self._gripper_group.enable()
        self._prime_arm_target()
        self._prime_gripper_state()
        self._arm.start_control_loop(self._loop_cb)
        self._controller._running = True

    def stop(self, *, safe_home: bool = True, release: bool = False) -> None:
        """Stop control and disconnect.

        When ``safe_home=False`` (e.g. ``--dry-run``), the arm and gripper do
        not execute any exit motion: the control loop is stopped and the bus is
        disconnected directly.
        """
        running = bool(getattr(self._controller, "_running", False))
        if release and running:
            try:
                self.release_gripper()
            except Exception as exc:
                print(f"[GraspDriver] release_gripper failed during stop: {exc}")

        if running and safe_home:
            self._controller.end()
            return

        if running:
            stop_send = getattr(self._controller, "_stop_send", None)
            if stop_send is not None:
                stop_send.set()
            send_thread = getattr(self._controller, "_send_thread", None)
            if send_thread is not None and send_thread.is_alive():
                send_thread.join(timeout=2.0)
            try:
                self._arm.stop_control_loop()
            except Exception:
                pass
            self._controller._running = False

        if self._arm is not None:
            try:
                self._arm.disconnect()
            except Exception as exc:
                print(f"[GraspDriver] disconnect failed during stop: {exc}")

    def _loop_cb(self, r: Any, dt: float) -> None:
        self._controller._loop_cb(r, dt)
        self.gripper_tick(dt)

    def _ensure_running(self) -> None:
        if not getattr(self._controller, "_running", False):
            raise RuntimeError("GraspDriver is not started; call grasp_driver.start() first")

    def _send_gripper_mit(
        self,
        pos: float,
        vel: float = 0.0,
        kp: float = 0.0,
        kd: float = 0.0,
        tau: float = 0.0,
    ) -> None:
        pos_cmd = float(np.clip(pos, self._open_lo, self._open_hi))
        tau_cmd = float(np.clip(tau, -self._tau_max, self._tau_max))
        self._gripper_group.send_mit(
            np.array([pos_cmd], dtype=np.float64),
            vel=np.array([vel], dtype=np.float64),
            kp=np.array([kp], dtype=np.float64),
            kd=np.array([kd], dtype=np.float64),
            tau=np.array([tau_cmd], dtype=np.float64),
        )

    def _prime_arm_target(self) -> None:
        q_now = self._arm.get_state()[0][: self._n]
        self._controller._q_target[:] = q_now
        self._controller._qd_target[:] = 0.0

    def _prime_gripper_state(self, timeout: float = 1.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._gripper_group._request_feedback()
            state = self._read_gripper_state_cached()
            if state is not None:
                with self._state_lock:
                    self._target_pos = state[0]
                    self._contact_pos = state[0]
                    self._state = self._STATE_IDLE
                    self._position_reached = True
                    self._grasp_result = None
                return
            time.sleep(0.02)

    def _read_gripper_state_cached(self) -> Optional[tuple[float, float, float]]:
        if self._gripper_motor is None:
            return self._last_gripper_state
        st = self._gripper_motor.get_state()
        if st is None:
            return self._last_gripper_state
        self._last_gripper_state = (float(st.pos), float(st.vel), float(st.torq))
        return self._last_gripper_state

    def _wait_gripper_state(self, timeout: float = 1.0) -> tuple[float, float, float]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self._read_gripper_state_cached()
            if state is not None:
                return state
            time.sleep(0.02)
        raise RuntimeError("Gripper feedback is not ready")

    def get_gripper_state(self) -> tuple[float, float, float]:
        state = self._read_gripper_state_cached()
        if state is None:
            raise RuntimeError("Gripper feedback is not ready")
        return state

    def is_holding(self, margin_rad: float = 0.05) -> bool:
        """Return True if the gripper is in HOLDING state and not collapsed to hard stop."""
        pos_vel_torq = self._read_gripper_state_cached()
        with self._state_lock:
            if self._state != self._STATE_HOLDING:
                return False
        if pos_vel_torq is None:
            return True
        pos = pos_vel_torq[0]
        threshold = self._open_sign * self._hard_stop_angle + max(0.0, float(margin_rad))
        return (self._open_sign * pos) > threshold

    def _distance_to_target_pos(self, distance_m: float) -> float:
        d = float(np.clip(distance_m, 0.0, self.MAX_DISTANCE_M))
        raw_target = (d / self.MAX_DISTANCE_M) * self._angle_open
        return float(np.clip(raw_target, self._open_lo, self._open_hi))

    def _set_position_target(self, target: float) -> None:
        with self._state_lock:
            self._target_pos = float(target)
            self._state = self._STATE_POSITION
            self._position_reached = False
            self._grasp_result = None

    def set_gripper_distance_async(self, distance_m: float = 0.0) -> None:
        """Command the gripper to move to ``distance_m`` without blocking."""
        self._ensure_running()
        self._set_position_target(self._distance_to_target_pos(distance_m))

    def close_gripper_async(self, delay_s: float = 0.0) -> threading.Thread:
        """Close gripper to rest position asynchronously after an optional delay."""
        self._ensure_running()

        def _worker() -> None:
            if delay_s > 0.0:
                time.sleep(float(delay_s))
            if getattr(self._controller, "_running", False):
                self._set_position_target(self._distance_to_target_pos(0.0))

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return thread

    def _wait_until(self, predicate, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()

    def _position_done(self) -> bool:
        with self._state_lock:
            return self._position_reached

    def _grasp_done(self) -> bool:
        with self._state_lock:
            return self._grasp_result is not None

    def _post_contact_target(self, contact_pos: float) -> float:
        """Return a limited position target slightly past first contact."""
        target = contact_pos + self._close_sign * self._post_contact_close_angle
        if self._close_sign < 0.0:
            close_limit = max(self._open_lo, self._hard_stop_angle)
            target = max(target, close_limit)
        else:
            close_limit = min(self._open_hi, self._hard_stop_angle)
            target = min(target, close_limit)
        return float(np.clip(target, self._open_lo, self._open_hi))

    def _start_post_contact_tightening(
        self, pos: float, now: float
    ) -> tuple[float, float, float, float, float]:
        """Start a short tightening phase, or preserve legacy behavior at 0."""
        target = self._post_contact_target(pos)
        self._target_pos = target
        command = (target, 0.0, self._kp_move, self._kd_move, self._hold_torque)
        if self._debug:
            print(f"[gripper] contact pos={pos:.3f} -> tighten target={target:.3f}")

        if self._post_contact_close_angle <= 0.0 or abs(target - pos) <= 1e-9:
            self._state = self._STATE_HOLDING
            self._grasp_result = True
        else:
            self._state = self._STATE_TIGHTENING
            self._tighten_deadline = now + self._post_contact_timeout_s
            self._grasp_result = None
        return command

    def gripper_tick(self, dt: float = 0.0) -> None:
        del dt
        pos_vel_torq = self._read_gripper_state_cached()
        with self._state_lock:
            state = self._state
            target = self._target_pos
            command: Optional[tuple[float, float, float, float, float]] = None

            if state == self._STATE_POSITION:
                command = (target, 0.0, self._kp_move, self._kd_move, 0.0)
                if pos_vel_torq is not None and abs(pos_vel_torq[0] - target) < self._arrive_tol:
                    self._position_reached = True

            elif state == self._STATE_CLOSING:
                command = (0.0, 0.0, 0.0, self._kd_close, self._close_torque)

                if pos_vel_torq is not None:
                    pos, vel, torq = pos_vel_torq
                    self._contact_pos = pos
                    moved = abs(pos - self._start_pos) >= self._startup_dist
                    at_hard_stop = self._open_sign * pos < self._open_sign * self._hard_stop_angle
                    torque_loaded = abs(torq) > self._contact_torque_ratio * abs(self._close_torque)
                    stopped = abs(vel) < self._stall_vel

                    now = time.monotonic()
                    if self._debug and now - self._debug_last_print >= 0.1:
                        self._debug_last_print = now
                        print(f"[gripper] pos={pos:.3f} vel={vel:.3f} torq={torq:.3f}")

                    if moved and torque_loaded and not at_hard_stop:
                        command = self._start_post_contact_tightening(pos, now)
                    elif moved and stopped:
                        if at_hard_stop:
                            self._target_pos = 0.0
                            self._state = self._STATE_POSITION
                            self._position_reached = False
                            self._grasp_result = False
                            command = (0.0, 0.0, self._kp_move, self._kd_move, 0.0)
                        else:
                            command = self._start_post_contact_tightening(pos, now)

            elif state == self._STATE_TIGHTENING:
                command = (target, 0.0, self._kp_move, self._kd_move, self._hold_torque)
                arrived = False
                if pos_vel_torq is not None:
                    pos = pos_vel_torq[0]
                    self._contact_pos = pos
                    arrived = abs(pos - target) <= self._post_contact_arrive_tol
                if arrived or time.monotonic() >= self._tighten_deadline:
                    self._state = self._STATE_HOLDING
                    self._grasp_result = True

            elif state == self._STATE_HOLDING:
                command = (self._target_pos, 0.0, self._kp_move, self._kd_move, self._hold_torque)

        if command is not None:
            pos, vel, kp, kd, tau = command
            self._send_gripper_mit(pos, vel=vel, kp=kp, kd=kd, tau=tau)

    def open_gripper(self, distance_m: float = GRIPPER_MAX_DISTANCE_M, timeout: float = 3.0) -> None:
        self._ensure_running()
        target = self._distance_to_target_pos(distance_m)
        self._set_position_target(target)
        self._wait_until(self._position_done, timeout)

    def grasp(self, force: Optional[float] = None, timeout: float = 5.0) -> bool:
        self._ensure_running()
        start_pos, _, _ = self._wait_gripper_state()
        hold_torque = self._close_sign * float(
            np.clip(abs(force if force is not None else self._default_force), 0.05, self._tau_max)
        )
        with self._state_lock:
            self._start_pos = start_pos
            self._contact_pos = start_pos
            self._target_pos = 0.0
            self._tighten_deadline = 0.0
            self._hold_torque = hold_torque
            self._state = self._STATE_CLOSING
            self._position_reached = False
            self._grasp_result = None

        if not self._wait_until(self._grasp_done, timeout):
            with self._state_lock:
                if self._grasp_result is None:
                    self._target_pos = self._contact_pos
                    self._state = self._STATE_POSITION
                    self._position_reached = False
                    self._grasp_result = False

        with self._state_lock:
            return bool(self._grasp_result)

    def release_gripper(self, timeout: float = 4.0) -> None:
        self._ensure_running()
        self.open_gripper(timeout=min(2.0, timeout))
        self._set_position_target(0.0)
        self._wait_until(self._position_done, timeout)

    def get_tcp_pose(self) -> np.ndarray:
        q_arm = self._arm.get_state(request_feedback=False)[0][: self._n]
        q = self._pad_q_for_model(self._model, q_arm, self._n)
        pos, rot, _ = self._compute_fk(self._model, q)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T
