"""Unit tests for utils/ordinary_grasp.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.ordinary_grasp import estimate_grasp, estimate_grasps


class _FakeBox:
    def __init__(self, xyxy, cls_id, conf):
        self.xyxy = np.array([xyxy], dtype=np.float32)
        self.cls = np.array([cls_id], dtype=np.float32)
        self.conf = np.array([conf], dtype=np.float32)


def _make_box_result(xyxy, cls_id=0, conf=0.9, name="block"):
    return SimpleNamespace(
        names={cls_id: name},
        obb=None,
        masks=None,
        boxes=[_FakeBox(xyxy, cls_id, conf)],
    )


class OrdinaryGraspTest(unittest.TestCase):
    def test_object_length_uses_long_axis_with_anisotropic_intrinsics(self):
        # Rectangle width=20px (along u), height=100px (along v) -> short=u, long=v.
        # Choose fx=100, fy=200 at z=1.0m:
        #   jaw_width_m = 20 * 1.0 / 100 = 0.20 m
        #   object_length_m = 100 * 1.0 / 200 = 0.50 m (old bug gave 100/100 = 1.00 m!)
        K = np.array([[100.0, 0.0, 100.0], [0.0, 200.0, 100.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        depth_mm = np.full((200, 200), 1000, dtype=np.uint16)
        res = _make_box_result([90, 50, 110, 150])
        grasp = estimate_grasp(res, 0, depth_mm, K, depth_quantile=0.5, approach_mode="camera_z")

        self.assertTrue(grasp.is_valid)
        # Mask refinement measures the 5th-95th percentile cross-section span (~90% of 20px = 18px -> 0.18m).
        self.assertAlmostEqual(grasp.jaw_width_m, 0.18, places=2)
        self.assertAlmostEqual(grasp.object_length_m, 0.50, places=2)

    def test_camera_z_vs_ray_approach_off_center(self):
        K = np.array([[200.0, 0.0, 100.0], [0.0, 200.0, 100.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        depth_mm = np.full((200, 200), 500, dtype=np.uint16)
        # Off-center object near top-right corner
        res = _make_box_result([150, 10, 190, 30])

        g_camz = estimate_grasp(res, 0, depth_mm, K, approach_mode="camera_z")
        g_ray = estimate_grasp(res, 0, depth_mm, K, approach_mode="ray")
        self.assertTrue(g_camz.is_valid and g_ray.is_valid)
        # camera_z approach is strictly [0, 0, -1] (TCP X = [0, 0, +1])
        np.testing.assert_allclose(g_camz.rotation[:, 2], [0.0, 0.0, -1.0], atol=1e-6)
        # ray approach tilts away from Z axis for off-center objects
        self.assertGreater(abs(float(g_ray.rotation[0, 2])), 0.2)

    def test_zero_depth_is_rejected_cleanly(self):
        K = np.array([[200.0, 0.0, 50.0], [0.0, 200.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        depth_mm = np.zeros((100, 100), dtype=np.uint16)
        grasps = estimate_grasps([_make_box_result([20, 20, 40, 60])], depth_mm, K)
        self.assertEqual(len(grasps), 1)
        self.assertFalse(grasps[0].is_valid)
        self.assertEqual(grasps[0].rejected_reason, "no_valid_depth_or_rect")


if __name__ == "__main__":
    unittest.main(verbosity=2)
