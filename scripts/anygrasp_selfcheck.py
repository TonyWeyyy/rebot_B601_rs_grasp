#!/usr/bin/env python3
"""Offline AnyGrasp tests using a fake detector; no weights, camera, CAN, or arm."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from graspnetAPI import Grasp, GraspGroup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import anygrasp_utils
from utils.transforms import (
    graspnet_rotation_to_rebot_tcp_rotation,
    transform_grasp_pose_to_base_with_retreat,
)

R_DOWN = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
])  # GraspNet X=approach down, Y=jaw axis, Z=right-handed completion

SAFETY = {
    "min_base_z_m": 0.05,
    "max_gripper_width": 0.09,
    "width_margin_m": 0.003,
    "max_tilt_deg": 45.0,
    "base_workspace_min": [0.10, -0.20, 0.02],
    "base_workspace_max": [0.40, 0.20, 0.40],
}


def make_grasp(center=None, rotation=None, width=0.05, depth=0.03) -> Grasp:
    grasp = Grasp()
    grasp.score = 0.9
    grasp.width = width
    grasp.height = 0.03
    grasp.depth = depth
    grasp.rotation_matrix = R_DOWN.copy() if rotation is None else np.asarray(rotation)
    grasp.translation = np.array([0.20, 0.0, 0.12] if center is None else center, dtype=float)
    return grasp


def poses(grasp, T=None):
    T = np.eye(4) if T is None else T
    return transform_grasp_pose_to_base_with_retreat(
        np.asarray(grasp.translation),
        graspnet_rotation_to_rebot_tcp_rotation(grasp.rotation_matrix),
        T,
        0.08,
        0.08,
        0.02,
    )


def check(grasp):
    target, pre, retreat = poses(grasp)
    return anygrasp_utils.check_grasp_base_geometry(grasp, np.eye(4), pre, target, retreat, SAFETY)


class FakeDetector:
    def __init__(self, group):
        self.group = group
        self.options = None

    def get_grasp(self, points, options):
        assert points.dtype == np.float32
        self.options = options
        return GraspGroup(self.group.grasp_group_array.copy())


def main() -> None:
    good = make_grasp()
    reasons, details = check(good)
    assert not reasons, reasons
    assert abs(details["tilt_deg"]) < 1e-6
    np.testing.assert_allclose(details["tip_base"], [0.20, 0.0, 0.09], atol=1e-8)
    print("PASS 1: AnyGrasp X-axis approach and depth-based tip position")

    target, pre, retreat = poses(good)
    # TCP X is approach down; pre/retreat must be above the inserted target.
    assert pre[2] > target[2] and retreat[2] > target[2]
    print("PASS 2: existing reBot TCP conversion, insertion, pregrasp and retreat are preserved")

    cases = [
        ("low tip", make_grasp(center=[0.20, 0.0, 0.075])),
        ("horizontal approach", make_grasp(rotation=np.eye(3))),
        ("excess width", make_grasp(width=0.089)),
        ("outside workspace", make_grasp(center=[0.50, 0.0, 0.12])),
    ]
    for name, grasp in cases:
        reasons, _ = check(grasp)
        assert reasons, f"unsafe case not rejected: {name}"
        print(f"PASS: rejected {name}: {'; '.join(reasons)}")

    group = GraspGroup()
    group.add(make_grasp())
    fake = FakeDetector(group)
    points = np.array([
        [0.0, 0.0, 0.4], [0.1, 0.0, 0.4], [0.0, 0.1, 0.4], [1.0, 1.0, 0.4]
    ], dtype=np.float32)
    region = anygrasp_utils.workspace_mask(points, [-0.2, 0.2, -0.2, 0.2, 0.1, 0.8])
    result = anygrasp_utils.infer_grasps(
        fake, points, region,
        {"min_region_points": 3, "collision_detection": True, "score_threshold": 0.0},
        preferred_approach_cam=[0, 0, 1],
    )
    assert len(result) == 1
    assert fake.options["collision_detection"] is True
    assert fake.options["region_steering"].tolist() == [True, True, True, False]
    np.testing.assert_allclose(fake.options["approach_steering"], [0, 0, 1])
    print("PASS 3: current SDK steering options, collision detection and workspace mask")

    color = np.zeros((20, 20, 3), dtype=np.uint8)
    depth = np.full((20, 20), 500, dtype=np.uint16)
    K = np.array([[20.0, 0, 9.5], [0, 20.0, 9.5], [0, 0, 1.0]])
    _, cloud, _ = anygrasp_utils.build_point_cloud(
        color, depth, K, min_depth_m=0.05, max_depth_m=1.0, max_input_points=0
    )
    np.testing.assert_allclose(cloud[0], [-0.2375, -0.2375, 0.5], atol=1e-6)
    assert len(cloud) == 400
    print("PASS 4: Gemini depth-mm to metric camera-cloud projection")

    print("\nAll offline AnyGrasp tests passed. Hardware validation is still mandatory.")


if __name__ == "__main__":
    main()
