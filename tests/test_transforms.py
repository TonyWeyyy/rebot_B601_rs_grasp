"""Unit tests for utils/transforms.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.transforms import (
    canonicalize_parallel_gripper_tcp_rotation,
    grasp_axes_to_rebot_tcp_rotation,
    graspnet_rotation_to_rebot_tcp_rotation,
    mat4_to_pose6d,
    pose6d_to_mat4,
    quat_to_mat4,
    rotation_matrix_to_euler_zyx,
    transform_grasp_pose_to_base,
    transform_grasp_pose_to_base_with_retreat,
)


class TransformsTest(unittest.TestCase):
    def test_pose6d_roundtrip(self):
        pose = (0.31, -0.12, 0.24, 0.35, 0.72, -0.88)
        T = pose6d_to_mat4(*pose)
        recovered = mat4_to_pose6d(T)
        np.testing.assert_allclose(recovered, pose, atol=1e-7)

    def test_quat_to_mat4_identity_and_90deg_z(self):
        T_id = quat_to_mat4(0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0)
        np.testing.assert_allclose(T_id[:3, :3], np.eye(3), atol=1e-8)
        np.testing.assert_allclose(T_id[:3, 3], [0.1, 0.2, 0.3], atol=1e-8)

        s = np.sin(np.pi / 4)
        T_z = quat_to_mat4(0, 0, 0, 0, 0, s, s)
        R_expected = pose6d_to_mat4(0, 0, 0, 0, 0, np.pi / 2)[:3, :3]
        np.testing.assert_allclose(T_z[:3, :3], R_expected, atol=1e-8)

    def test_canonicalize_prefers_smaller_abs_roll(self):
        R_big_roll = pose6d_to_mat4(0, 0, 0, 2.9, 0.4, 0.2)[:3, :3]
        R_canon = canonicalize_parallel_gripper_tcp_rotation(R_big_roll)
        roll_before = abs(rotation_matrix_to_euler_zyx(R_big_roll)[0])
        roll_after = abs(rotation_matrix_to_euler_zyx(R_canon)[0])
        self.assertLess(roll_after, roll_before)
        # Tool X (approach) axis must stay identical.
        np.testing.assert_allclose(R_canon[:, 0], R_big_roll[:, 0], atol=1e-8)

    def test_grasp_axes_to_rebot_tcp_orthonormal_right_handed(self):
        R = grasp_axes_to_rebot_tcp_rotation(
            grip_axis=np.array([0.0, 1.0, 0.0]),
            open_axis=np.array([1.0, 0.0, 0.0]),
            approach_axis=np.array([0.0, 0.0, -1.0]),
        )
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-8)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=7)
        np.testing.assert_allclose(R[:, 0], [0.0, 0.0, 1.0], atol=1e-8)

    def test_transform_grasp_pose_insertion_and_offsets(self):
        # TCP X pointing straight down (-Z in base)
        R_down = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])
        tcp_rot = graspnet_rotation_to_rebot_tcp_rotation(R_down)
        grasp6d, pre6d, retreat6d = transform_grasp_pose_to_base_with_retreat(
            position_cam=np.array([0.25, 0.0, 0.12]),
            tcp_rotation_cam=tcp_rot,
            T_cam2base=np.eye(4),
            pregrasp_offset_m=0.08,
            retreat_offset_m=0.10,
            insertion_depth_m=0.02,
        )
        # Insertion pushes 2cm deeper (-Z) -> 0.10; pregrasp lifts 8cm -> 0.18; retreat lifts 10cm -> 0.20
        self.assertAlmostEqual(grasp6d[2], 0.10, places=6)
        self.assertAlmostEqual(pre6d[2], 0.18, places=6)
        self.assertAlmostEqual(retreat6d[2], 0.20, places=6)

        grasp2, pre2 = transform_grasp_pose_to_base(
            np.array([0.25, 0.0, 0.12]), tcp_rot, np.eye(4), 0.08, 0.02
        )
        np.testing.assert_allclose(grasp2, grasp6d, atol=1e-8)
        np.testing.assert_allclose(pre2, pre6d, atol=1e-8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
