"""Unit tests for calibration/hand_eye.py (metrics, outlier rejection, solver comparison, I/O)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration.hand_eye import CalibMode, HandEyeCalibrator
from utils.transforms import pose6d_to_mat4


def _synthetic_samples(n: int = 18, seed: int = 42, noise_m: float = 0.0):
    rng = np.random.default_rng(seed)
    T_cam2gripper_true = pose6d_to_mat4(0.05, -0.02, 0.07, 0.15, -0.25, 0.30)
    T_marker2base_true = pose6d_to_mat4(0.35, 0.04, 0.01, 0.05, -0.02, 0.40)

    samples = []
    for _ in range(n):
        T_g2b = pose6d_to_mat4(
            rng.uniform(0.18, 0.38),
            rng.uniform(-0.18, 0.18),
            rng.uniform(0.22, 0.42),
            rng.uniform(-0.8, 0.8),
            rng.uniform(0.2, 1.0),
            rng.uniform(-1.0, 1.0),
        )
        # T_marker2base = T_g2b @ T_c2g @ T_m2c  =>  T_m2c = inv(T_c2g) @ inv(T_g2b) @ T_marker2base
        T_m2c = np.linalg.inv(T_cam2gripper_true) @ np.linalg.inv(T_g2b) @ T_marker2base_true
        if noise_m > 0.0:
            T_m2c = T_m2c.copy()
            T_m2c[:3, 3] += rng.normal(0.0, noise_m, size=3)
        samples.append((T_g2b, T_m2c))
    return T_cam2gripper_true, samples


class HandEyeTest(unittest.TestCase):
    def test_exact_recovery_and_zero_residual(self):
        T_true, samples = _synthetic_samples(n=16, noise_m=0.0)
        calib = HandEyeCalibrator(CalibMode.EYE_IN_HAND, method="TSAI")
        for T_g2b, T_m2c in samples:
            calib.add_sample(T_g2b, T_m2c)

        result = calib.calibrate()
        np.testing.assert_allclose(result.T_result, T_true, atol=1e-5)
        self.assertIsNotNone(result.metrics)
        self.assertLess(result.metrics.trans_rmse_m, 1e-5)
        self.assertLess(result.metrics.rot_rmse_deg, 1e-3)

    def test_outlier_rejection_improves_accuracy(self):
        T_true, samples = _synthetic_samples(n=18, noise_m=0.001)
        # Corrupt 2 samples severely (10 cm PnP jump)
        corrupted = list(samples)
        for bad_idx in (3, 11):
            T_g2b, T_m2c = corrupted[bad_idx]
            T_bad = T_m2c.copy()
            T_bad[:3, 3] += np.array([0.12, -0.09, 0.08])
            corrupted[bad_idx] = (T_g2b, T_bad)

        calib = HandEyeCalibrator(CalibMode.EYE_IN_HAND, method="PARK")
        for T_g2b, T_m2c in corrupted:
            calib.add_sample(T_g2b, T_m2c)

        raw = calib.calibrate(reject_outliers=False)
        clean = calib.calibrate(reject_outliers=True)
        self.assertIn(3, clean.metrics.rejected_indices)
        self.assertIn(11, clean.metrics.rejected_indices)
        self.assertLess(clean.metrics.trans_rmse_m, raw.metrics.trans_rmse_m * 0.25)
        raw_err = float(np.linalg.norm(raw.T_result[:3, 3] - T_true[:3, 3]))
        clean_err = float(np.linalg.norm(clean.T_result[:3, 3] - T_true[:3, 3]))
        self.assertLess(clean_err, raw_err)

    def test_compare_methods_and_save_load_roundtrip(self):
        _, samples = _synthetic_samples(n=14, noise_m=0.0005)
        calib = HandEyeCalibrator(CalibMode.EYE_IN_HAND, method="TSAI")
        for T_g2b, T_m2c in samples:
            calib.add_sample(T_g2b, T_m2c)

        compared = calib.compare_methods()
        self.assertGreaterEqual(len(compared), 4)
        for res in compared.values():
            self.assertIsNotNone(res.metrics)

        with tempfile.TemporaryDirectory() as tmpdir:
            res_path = Path(tmpdir) / "hand_eye.npz"
            smp_path = Path(tmpdir) / "samples.npz"
            res = calib.calibrate()
            HandEyeCalibrator.save(res, res_path)
            loaded = HandEyeCalibrator.load(res_path)
            np.testing.assert_allclose(loaded.T_result, res.T_result)
            self.assertAlmostEqual(loaded.metrics.trans_rmse_m, res.metrics.trans_rmse_m, places=7)

            calib.save_samples(smp_path)
            reloaded = HandEyeCalibrator.load_samples(smp_path)
            self.assertEqual(reloaded.n_samples, calib.n_samples)


if __name__ == "__main__":
    unittest.main(verbosity=2)
