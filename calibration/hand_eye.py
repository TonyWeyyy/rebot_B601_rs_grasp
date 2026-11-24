"""手眼标定 — 基于 OpenCV calibrateHandEye，支持残差评估、离群样本剔除与多求解器对比。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np


class CalibMode(Enum):
    EYE_IN_HAND = "eye_in_hand"   # 相机在末端，随末端运动


_METHOD_MAP = {
    "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
    "PARK":       cv2.CALIB_HAND_EYE_PARK,
    "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass
class CalibMetrics:
    """手眼标定一致性残差指标（T_marker2base = T_g2b @ T_c2g @ T_m2c 的跨样本离散度）。"""
    trans_rmse_m: float
    trans_max_m: float
    rot_rmse_deg: float
    rot_max_deg: float
    per_sample_trans_m: np.ndarray
    per_sample_rot_deg: np.ndarray
    rejected_indices: list[int] = field(default_factory=list)


@dataclass
class CalibResult:
    T_result: np.ndarray    # (4, 4) 手眼变换矩阵
    mode: str               # CalibMode.value
    n_samples: int
    method: str
    metrics: Optional[CalibMetrics] = None


@dataclass
class _Sample:
    T_gripper2base: np.ndarray   # (4, 4)
    T_marker2cam:   np.ndarray   # (4, 4)


def _mean_rotation_so3(rotations: Sequence[np.ndarray]) -> np.ndarray:
    """Compute chordal L2 mean rotation on SO(3) via SVD projection."""
    R_sum = np.sum(rotations, axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0.0:
        U[:, -1] *= -1.0
        R_mean = U @ Vt
    return R_mean.astype(np.float64)


def _rotation_error_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    trace = float(np.trace(R_a.T @ R_b))
    cos_theta = float(np.clip(0.5 * (trace - 1.0), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


class HandEyeCalibrator:
    """
    手眼标定器。

    Eye-in-Hand 模式：
        求解 T_cam2gripper，使得
            T_marker2base = T_gripper2base @ T_cam2gripper @ T_marker2cam
        在所有姿态下恒成立。
    """

    def __init__(
        self,
        mode: CalibMode = CalibMode.EYE_IN_HAND,
        method: str = "TSAI",
    ) -> None:
        if mode != CalibMode.EYE_IN_HAND:
            raise ValueError("Only eye-in-hand calibration is supported")
        self._mode = mode
        self._method = method.upper()
        self._samples: List[_Sample] = []

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    def add_sample(
        self,
        T_gripper2base: np.ndarray,
        T_marker2cam: np.ndarray,
    ) -> None:
        """添加一个标定样本。"""
        self._samples.append(_Sample(
            T_gripper2base=np.asarray(T_gripper2base, dtype=np.float64),
            T_marker2cam=np.asarray(T_marker2cam, dtype=np.float64),
        ))

    def _solve_subset(self, samples: Sequence[_Sample], method: str) -> np.ndarray:
        cv_method = _METHOD_MAP.get(method.upper(), cv2.CALIB_HAND_EYE_TSAI)
        R_g2b = [s.T_gripper2base[:3, :3] for s in samples]
        t_g2b = [s.T_gripper2base[:3,  3].reshape(3, 1) for s in samples]
        R_t2c = [s.T_marker2cam[:3, :3] for s in samples]
        t_t2c = [s.T_marker2cam[:3,  3].reshape(3, 1) for s in samples]

        R_c2g, t_c2g = cv2.calibrateHandEye(
            R_g2b, t_g2b, R_t2c, t_t2c, method=cv_method
        )
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_c2g
        T[:3,  3] = t_c2g.flatten()
        return T

    def evaluate(
        self,
        T_cam2gripper: np.ndarray,
        indices: Optional[Sequence[int]] = None,
        rejected_indices: Optional[Sequence[int]] = None,
    ) -> CalibMetrics:
        """评估手眼矩阵在给定样本上的标记位姿一致性残差。"""
        sample_list = (
            [self._samples[i] for i in indices]
            if indices is not None
            else self._samples
        )
        if not sample_list:
            raise ValueError("Cannot evaluate calibration with 0 samples")

        T_c2g = np.asarray(T_cam2gripper, dtype=np.float64)
        marker_poses = [s.T_gripper2base @ T_c2g @ s.T_marker2cam for s in sample_list]
        translations = np.array([T[:3, 3] for T in marker_poses], dtype=np.float64)
        rotations = [T[:3, :3] for T in marker_poses]

        mean_t = np.mean(translations, axis=0)
        mean_R = _mean_rotation_so3(rotations)

        trans_errs = np.linalg.norm(translations - mean_t.reshape(1, 3), axis=1)
        rot_errs = np.array([_rotation_error_deg(mean_R, R) for R in rotations], dtype=np.float64)

        return CalibMetrics(
            trans_rmse_m=float(np.sqrt(np.mean(trans_errs ** 2))),
            trans_max_m=float(np.max(trans_errs)),
            rot_rmse_deg=float(np.sqrt(np.mean(rot_errs ** 2))),
            rot_max_deg=float(np.max(rot_errs)),
            per_sample_trans_m=trans_errs,
            per_sample_rot_deg=rot_errs,
            rejected_indices=list(rejected_indices or []),
        )

    def calibrate(
        self,
        min_samples: int = 5,
        *,
        method: Optional[str] = None,
        reject_outliers: bool = False,
        outlier_sigma: float = 2.5,
        max_outlier_rounds: int = 2,
    ) -> CalibResult:
        """计算手眼变换并评估一致性残差，可选自动剔除离群样本。"""
        if self.n_samples < min_samples:
            raise ValueError(
                f"样本不足：{self.n_samples} < {min_samples}，请继续采集"
            )

        solver_name = (method or self._method).upper()
        active_indices = list(range(self.n_samples))
        rejected: list[int] = []

        T = self._solve_subset([self._samples[i] for i in active_indices], solver_name)
        metrics = self.evaluate(T, indices=active_indices, rejected_indices=rejected)

        if reject_outliers and len(active_indices) > min_samples:
            for _ in range(max(1, int(max_outlier_rounds))):
                errs = metrics.per_sample_trans_m
                median = float(np.median(errs))
                mad = float(np.median(np.abs(errs - median)))
                # 1.4826 * MAD estimates std for normal distribution; floor at 2mm to avoid over-pruning.
                robust_sigma = max(1.4826 * mad, 0.002)
                threshold = median + float(outlier_sigma) * robust_sigma
                keep_local = [idx for idx, e in enumerate(errs) if e <= threshold]
                if len(keep_local) == len(active_indices) or len(keep_local) < min_samples:
                    break
                for local_idx in range(len(active_indices)):
                    if local_idx not in keep_local:
                        rejected.append(active_indices[local_idx])
                active_indices = [active_indices[i] for i in keep_local]
                T = self._solve_subset([self._samples[i] for i in active_indices], solver_name)
                metrics = self.evaluate(T, indices=active_indices, rejected_indices=sorted(rejected))

        return CalibResult(
            T_result=T,
            mode=self._mode.value,
            n_samples=len(active_indices),
            method=solver_name,
            metrics=metrics,
        )

    def compare_methods(
        self,
        min_samples: int = 5,
        *,
        reject_outliers: bool = False,
    ) -> dict[str, CalibResult]:
        """运行 OpenCV 全部 5 种手眼标定求解器并对比残差。"""
        results: dict[str, CalibResult] = {}
        for name in _METHOD_MAP:
            try:
                results[name] = self.calibrate(
                    min_samples=min_samples,
                    method=name,
                    reject_outliers=reject_outliers,
                )
            except Exception:
                continue
        return results

    def save_samples(self, path: Union[str, Path]) -> None:
        """保存原始采样的 (T_gripper2base, T_marker2cam) 序列，便于离线复算与排查。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        T_g2b = np.stack([s.T_gripper2base for s in self._samples], axis=0) if self._samples else np.empty((0, 4, 4))
        T_m2c = np.stack([s.T_marker2cam for s in self._samples], axis=0) if self._samples else np.empty((0, 4, 4))
        np.savez(str(path), T_gripper2base=T_g2b, T_marker2cam=T_m2c)

    @classmethod
    def load_samples(
        cls,
        path: Union[str, Path],
        *,
        mode: CalibMode = CalibMode.EYE_IN_HAND,
        method: str = "TSAI",
    ) -> "HandEyeCalibrator":
        """从离线样本文件恢复标定器。"""
        data = np.load(str(path), allow_pickle=False)
        calibrator = cls(mode=mode, method=method)
        for T_g2b, T_m2c in zip(data["T_gripper2base"], data["T_marker2cam"]):
            calibrator.add_sample(T_g2b, T_m2c)
        return calibrator

    @staticmethod
    def save(result: CalibResult, path: Union[str, Path]) -> None:
        """保存标定结果为 .npz 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "T_result": result.T_result,
            "mode": np.array([result.mode]),
            "n_samples": np.array([result.n_samples]),
            "method": np.array([result.method]),
        }
        if result.metrics is not None:
            payload["trans_rmse_m"] = np.array([result.metrics.trans_rmse_m])
            payload["trans_max_m"] = np.array([result.metrics.trans_max_m])
            payload["rot_rmse_deg"] = np.array([result.metrics.rot_rmse_deg])
            payload["rot_max_deg"] = np.array([result.metrics.rot_max_deg])
        np.savez(str(path), **payload)

    @staticmethod
    def load(path: Union[str, Path]) -> CalibResult:
        """从 .npz 文件加载标定结果（向下兼容无 metrics 的旧文件）。"""
        data = np.load(str(path), allow_pickle=False)
        metrics = None
        if "trans_rmse_m" in data and "rot_rmse_deg" in data:
            metrics = CalibMetrics(
                trans_rmse_m=float(data["trans_rmse_m"][0]),
                trans_max_m=float(data["trans_max_m"][0]) if "trans_max_m" in data else 0.0,
                rot_rmse_deg=float(data["rot_rmse_deg"][0]),
                rot_max_deg=float(data["rot_max_deg"][0]) if "rot_max_deg" in data else 0.0,
                per_sample_trans_m=np.empty((0,), dtype=np.float64),
                per_sample_rot_deg=np.empty((0,), dtype=np.float64),
            )
        return CalibResult(
            T_result=data["T_result"],
            mode=str(data["mode"][0]),
            n_samples=int(data["n_samples"][0]),
            method=str(data["method"][0]) if "method" in data else "TSAI",
            metrics=metrics,
        )
