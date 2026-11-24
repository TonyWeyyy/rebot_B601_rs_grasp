"""Chained IK reachability checker and multi-criteria 6D grasp selection."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from utils import anygrasp_utils
from utils.grasp_types import grasp_to_base_poses, rank_grasps


class IkChecker:
    """Pinocchio-based endpoint and chained-trajectory IK checker."""

    def __init__(self, arm: Any) -> None:
        from reBotArm_control_py.kinematics import (
            get_end_effector_frame_id,
            load_robot_model,
            pad_q_for_model,
            pos_rot_to_se3,
            solve_ik,
        )
        from reBotArm_control_py.kinematics.inverse_kinematics import IKParams

        self._arm = arm
        self._arm_group = arm.groups.get("arm")
        if self._arm_group is None:
            raise ValueError("Hardware config missing groups.arm")
        self._n = self._arm_group.num_joints
        self._model = load_robot_model()
        self._data = self._model.createData()
        self._end_frame_id = get_end_effector_frame_id(self._model)
        self._params = IKParams(max_iter=200, tolerance=1e-4, step_size=0.5, damping=1e-6)
        self._pad_q_for_model = pad_q_for_model
        self._pos_rot_to_se3 = pos_rot_to_se3
        self._solve_ik = solve_ik

    def current_q(self) -> np.ndarray:
        return np.asarray(self._arm.get_state(request_feedback=False)[0][: self._n], dtype=np.float64)

    def check_with_q(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        q_init: Optional[np.ndarray] = None,
    ) -> tuple[bool, float, Optional[np.ndarray]]:
        q_seed = self.current_q() if q_init is None else np.asarray(q_init, dtype=np.float64)[: self._n]
        q_padded = self._pad_q_for_model(self._model, q_seed, self._n)
        target = self._pos_rot_to_se3(
            np.array([x, y, z], dtype=np.float64), roll=roll, pitch=pitch, yaw=yaw
        )
        result = self._solve_ik(
            self._model,
            self._data,
            self._end_frame_id,
            target,
            q_padded,
            self._params,
            controlled_joints=self._n,
        )
        q_sol = np.asarray(result.q[: self._n], dtype=np.float64) if result.success else None
        return bool(result.success), float(result.error), q_sol

    def check(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> tuple[bool, float]:
        ok, err, _ = self.check_with_q(x, y, z, roll, pitch, yaw)
        return ok, err

    def check_chain(
        self,
        poses: Sequence[tuple[float, ...]],
        q_init: Optional[np.ndarray] = None,
    ) -> tuple[bool, list[float], list[np.ndarray]]:
        """Solve a sequence of 6D poses sequentially, seeding each step with the previous solution."""
        q_curr = self.current_q() if q_init is None else np.asarray(q_init, dtype=np.float64)[: self._n]
        errors: list[float] = []
        solutions: list[np.ndarray] = []
        for pose in poses:
            ok, err, q_sol = self.check_with_q(*pose, q_init=q_curr)
            errors.append(err)
            if not ok or q_sol is None:
                return False, errors, solutions
            solutions.append(q_sol)
            q_curr = q_sol
        return True, errors, solutions


def _pose_z_ok(pose6d: tuple[float, ...], min_z: float) -> bool:
    return float(pose6d[2]) >= float(min_z)


def _check_pose_sequence(
    ik_checker: Any,
    poses: Sequence[tuple[float, ...]],
) -> tuple[bool, float, list[np.ndarray]]:
    """Run chained IK when supported, or fall back to independent .check() for minimal stubs."""
    if hasattr(ik_checker, "check_chain"):
        ok, errors, solutions = ik_checker.check_chain(poses)
        worst = max(errors) if errors else 0.0
        return ok, worst, solutions

    worst = 0.0
    for pose in poses:
        ok, err = ik_checker.check(*pose)
        worst = max(worst, float(err))
        if not ok:
            return False, worst, []
    return True, worst, []


def _joint_travel_cost(ik_checker: Any, solutions: Sequence[np.ndarray]) -> float:
    if not solutions:
        return 0.0
    start = None
    if hasattr(ik_checker, "current_q"):
        try:
            start = np.asarray(ik_checker.current_q(), dtype=np.float64)
        except Exception:
            start = None
    total = 0.0
    prev = start
    for q in solutions:
        if prev is not None and len(prev) == len(q):
            total += float(np.linalg.norm(q - prev))
        prev = q
    return total


def select_executable_grasp(
    ik_checker: Any,
    grasps: Any,
    T_cam2base: np.ndarray,
    pregrasp_offset_m: float,
    retreat_offset_m: float,
    insertion_depth_m: float,
    min_base_z_m: float,
    anygrasp_safety: Optional[dict[str, Any]] = None,
    *,
    already_nms: bool = False,
    max_ik_candidates: int = 20,
    feasible_pool_size: int = 3,
    tilt_penalty_weight: float = 0.05,
    travel_penalty_weight: float = 0.02,
) -> Optional[tuple[Any, tuple[float, ...], tuple[float, ...], tuple[float, ...]]]:
    """Filter, safety-check, chained-IK verify, and select the best executable 6D grasp."""
    ranked = rank_grasps(grasps, run_nms=not already_nms)
    skipped_low = 0
    skipped_ik = 0
    skipped_geometry = 0
    ik_checked = 0
    worst_err = 0.0

    feasible: list[tuple[float, int, Any, tuple[float, ...], tuple[float, ...], tuple[float, ...]]] = []

    for idx in range(len(ranked)):
        if max_ik_candidates > 0 and ik_checked >= int(max_ik_candidates):
            break

        grasp = ranked[idx]
        grasp6d, pre6d, retreat6d = grasp_to_base_poses(
            grasp,
            T_cam2base,
            pregrasp_offset_m,
            retreat_offset_m,
            insertion_depth_m,
        )
        if not (
            _pose_z_ok(pre6d, min_base_z_m)
            and _pose_z_ok(grasp6d, min_base_z_m)
            and _pose_z_ok(retreat6d, min_base_z_m)
        ):
            skipped_low += 1
            continue

        tilt_deg = 0.0
        if anygrasp_safety is not None:
            safety = dict(anygrasp_safety)
            safety["min_base_z_m"] = min_base_z_m
            reasons, details = anygrasp_utils.check_grasp_base_geometry(
                grasp, T_cam2base, pre6d, grasp6d, retreat6d, safety
            )
            if reasons:
                skipped_geometry += 1
                if idx == 0:
                    print(f"[AnyGrasp debug] workspace_min={np.asarray(safety.get('base_workspace_min', [])).tolist()}")
                    print(f"[AnyGrasp debug] workspace_max={np.asarray(safety.get('base_workspace_max', [])).tolist()}")
                    print(f"[AnyGrasp debug] camera_base_xyz={np.round(T_cam2base[:3, 3], 4).tolist()}")
                    for name, position in details.get("positions", {}).items():
                        print(f"[AnyGrasp debug] {name}_base_xyz={np.round(position, 4).tolist()}")
                print(f"[AnyGrasp safety] reject rank={idx + 1}: {'; '.join(reasons)}")
                continue
            tilt_deg = float(details.get("tilt_deg", 0.0))
            print(f"[AnyGrasp safety] pass rank={idx + 1} tilt={tilt_deg:.1f}deg")

        ik_checked += 1
        ok, err, solutions = _check_pose_sequence(ik_checker, (pre6d, grasp6d, retreat6d))
        worst_err = max(worst_err, err)
        if not ok:
            skipped_ik += 1
            continue

        travel = _joint_travel_cost(ik_checker, solutions)
        composite_score = (
            float(grasp.score)
            - float(tilt_penalty_weight) * (tilt_deg / 90.0)
            - float(travel_penalty_weight) * travel
        )
        feasible.append((composite_score, idx, grasp, grasp6d, pre6d, retreat6d))
        if len(feasible) >= max(1, int(feasible_pool_size)):
            break

    if feasible:
        best_score, best_idx, best_grasp, best_grasp6d, best_pre6d, best_retreat6d = max(
            feasible, key=lambda item: item[0]
        )
        print(
            f"[G] Executable rank={best_idx + 1}/{len(ranked)} "
            f"score={float(best_grasp.score):.4f} composite={best_score:.4f} "
            f"(evaluated {len(feasible)} feasible)"
        )
        if skipped_low or skipped_ik or skipped_geometry:
            print(f"[G] Skipped low_z={skipped_low} geometry={skipped_geometry} ik_fail={skipped_ik}")
        return best_grasp, best_grasp6d, best_pre6d, best_retreat6d

    print(
        f"[G] No IK-reachable grasp: low_z={skipped_low} geometry={skipped_geometry} "
        f"ik_fail={skipped_ik} max_err={worst_err:.4f}"
    )
    return None
