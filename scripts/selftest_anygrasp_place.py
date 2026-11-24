"""Offline tests for AnyGrasp pick-and-place. No camera, CAN, arm, or GPU required."""

from __future__ import annotations

import builtins
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


class FakeGrasp:
    pass


class FakeGraspGroup:
    def __init__(self, array=None):
        self.items = list(array or [])

    @property
    def grasp_group_array(self):
        return self.items

    @property
    def scores(self):
        return np.array([g.score for g in self.items])

    @property
    def widths(self):
        return np.array([g.width for g in self.items])

    def add(self, grasp):
        self.items.append(grasp)

    def nms(self):
        return self

    def sort_by_score(self):
        return FakeGraspGroup(sorted(self.items, key=lambda g: g.score, reverse=True))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            return self.items[item]
        return FakeGraspGroup(np.asarray(self.items, dtype=object)[item].tolist())


if "graspnetAPI" not in sys.modules:
    graspnet_api_stub = types.ModuleType("graspnetAPI")
    graspnet_api_stub.Grasp = FakeGrasp
    graspnet_api_stub.GraspGroup = FakeGraspGroup
    sys.modules["graspnetAPI"] = graspnet_api_stub

import anygrasp_place as app  # noqa: E402
import core.robot_session as robot_session  # noqa: E402
import utils.anygrasp_utils as ag  # noqa: E402
import utils.place_utils as pu  # noqa: E402
from utils.yolo_utils import YoloDetection  # noqa: E402


def detection(name, conf, bbox, shape=(40, 40)):
    mask = np.zeros(shape, dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    mask[max(0, y1):min(shape[0], y2), max(0, x1):min(shape[1], x2)] = 1
    return YoloDetection(0, x1 + y1 * 1000, name, conf, bbox, mask)


class PlaceGeometryTest(unittest.TestCase):
    def test_container_selection_excludes_target(self):
        toy = detection("toy", 0.9, (1, 1, 10, 10))
        box = detection("box", 0.8, (15, 15, 35, 35))
        self.assertIs(pu.select_container([toy, box], "box", exclude=toy), box)
        self.assertIsNone(pu.select_container([toy], "toy", exclude=toy))

    def test_container_projection_and_poses(self):
        box = detection("box", 0.8, (10, 10, 30, 30))
        depth = np.full((40, 40), 500, dtype=np.uint16)
        K = np.array([[100.0, 0, 20], [0, 100.0, 20], [0, 0, 1]])
        center = pu.detection_center_cam(box, depth, K, use_mask_centroid=False)
        np.testing.assert_allclose(center, [0, 0, 0.5])
        T = np.eye(4)
        place, pre, measured = pu.build_place_poses(
            center, T, {"height_m": 0.15, "pre_place_offset_m": 0.08}
        )
        self.assertEqual(tuple(place[:3]), (0.0, 0.0, 0.15))
        np.testing.assert_allclose(pre[:3], [0.0, 0.0, 0.23])
        np.testing.assert_allclose(measured, [0, 0, 0.5])
        np.testing.assert_allclose(place[3:], [0, 1.2, 0])

    def test_workspace_floor_and_separation(self):
        safety = {"base_workspace_min": [0.1, -0.25, 0.01], "base_workspace_max": [0.5, 0.25, 0.45]}
        pu.validate_base_pose((0.3, 0.0, 0.15, 0, 1.2, 0), safety, min_base_z_m=0.05, name="place")
        with self.assertRaisesRegex(ValueError, "below floor"):
            pu.validate_base_pose((0.3, 0.0, 0.04, 0, 1.2, 0), safety, min_base_z_m=0.05, name="place")
        with self.assertRaisesRegex(ValueError, "outside workspace"):
            pu.validate_base_pose((0.6, 0.0, 0.15, 0, 1.2, 0), safety, min_base_z_m=0.05, name="place")
        with self.assertRaisesRegex(ValueError, "Target and container XY"):
            pu.validate_target_container_separation((0.3, 0, 0.06), (0.31, 0, 0.15))

    def test_infer_frame_reuses_same_frame_detections(self):
        toy = detection("toy", 0.9, (5, 5, 25, 25))
        box = detection("box", 0.8, (25, 5, 39, 25))
        color = np.zeros((40, 40, 3), dtype=np.uint8)
        depth = np.full((40, 40), 500, dtype=np.uint16)
        K = np.array([[50.0, 0, 20], [0, 50.0, 20], [0, 0, 1]])
        group = FakeGraspGroup()
        grasp = FakeGrasp()
        grasp.score, grasp.width, grasp.height, grasp.depth = 0.8, 0.05, 0.03, 0.02
        grasp.rotation_matrix = np.eye(3)
        grasp.translation = np.array([0, 0, 0.5])
        group.add(grasp)

        class Detector:
            def get_grasp(self, points, options):
                return group

        points = np.tile(np.array([0.0, 0.0, 0.5], dtype=np.float32), (400, 1))
        with patch.object(ag, "detect_objects", side_effect=AssertionError("YOLO ran twice")), patch.object(
            ag, "build_point_cloud", return_value=(None, points, None)
        ):
            result = ag.infer_frame(
                Detector(), color, depth, K,
                cfg={"lims": [-1, 1, -1, 1, 0.1, 0.8], "max_input_points": 1000},
                min_depth=0.05, max_depth=1.0,
                yolo_model=object(), detections=[toy, box], target_class="toy",
                build_o3d=False,
            )
        self.assertIs(result.selected_target, toy)
        self.assertEqual(len(result.detections), 2)


class FakeDriver:
    def __init__(self, grasp_result=True, holding_seq=None):
        self.calls = []
        self.grasp_result = grasp_result
        self._holding_seq = list(holding_seq) if holding_seq is not None else [True, True]

    def open_gripper(self, **kwargs):
        self.calls.append(("open", kwargs))

    def grasp(self):
        self.calls.append(("grasp",))
        return self.grasp_result

    def is_holding(self):
        self.calls.append(("is_holding",))
        return self._holding_seq.pop(0) if self._holding_seq else True


class FakeController:
    """Matches real RebotArmEndPose.move_to_traj signature."""

    def __init__(self):
        self.calls = []

    def move_to_traj(self, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, duration=2.0):
        self.calls.append(("move_to_traj", (x, y, z, roll, pitch, yaw), duration))
        return True


class SequenceSafetyTest(unittest.TestCase):
    def setUp(self):
        self.controller = FakeController()
        self.grasp6d = (0.30, 0.00, 0.06, 0.0, 1.2, 0.0)
        self.pre6d = (0.30, 0.00, 0.14, 0.0, 1.2, 0.0)
        self.retreat6d = (0.30, 0.00, 0.14, 0.0, 1.2, 0.0)
        self.pre_place6d = (0.25, 0.15, 0.18, 0.0, 0.7, 0.0)
        self.place6d = (0.25, 0.15, 0.10, 0.0, 0.7, 0.0)
        self.ready_cfg = {"x": 0.30, "y": 0.0, "z": 0.30, "roll": 0.0, "pitch": 0.7, "yaw": 0.0, "duration": 2.0}

    def execute(self, dry=False, empty=False, confirm=False, holding_seq=None, retract_to_pre_place=True):
        driver = FakeDriver(not empty, holding_seq=holding_seq)
        with patch.object(robot_session, "wait_motion", lambda *args, **kwargs: True), patch.object(
            robot_session.time, "sleep", lambda *args: None
        ):
            ok = app._execute_place_sequence(
                self.controller,
                driver,
                self.grasp6d,
                self.pre6d,
                self.retreat6d,
                self.pre_place6d,
                self.place6d,
                ready_cfg=self.ready_cfg,
                place_cfg={"retract_to_pre_place": retract_to_pre_place},
                dry_run=dry,
                confirm_motion=confirm,
            )
        return ok, driver

    def test_dry_run_has_no_motion(self):
        ok, driver = self.execute(dry=True)
        self.assertTrue(ok)
        self.assertEqual(self.controller.calls, [])
        self.assertEqual(driver.calls, [])

    def test_empty_grasp_retreats_vertically_and_never_moves_to_place(self):
        ok, driver = self.execute(empty=True)
        self.assertFalse(ok)
        poses = [call[1] for call in self.controller.calls]
        # pregrasp -> grasp -> retreat -> ready (no pre-place or place)
        self.assertEqual(len(poses), 4)
        self.assertEqual(poses[2], self.retreat6d)
        self.assertEqual(poses[3], (0.30, 0.0, 0.30, 0.0, 0.7, 0.0))
        self.assertEqual([c[0] for c in driver.calls], ["open", "grasp"])

    def test_midair_drop_aborts_placement(self):
        ok, driver = self.execute(holding_seq=[True, False])
        self.assertFalse(ok)
        poses = [call[1] for call in self.controller.calls]
        # pregrasp -> grasp -> retreat -> ready; aborts before pre-place
        self.assertEqual(len(poses), 4)
        self.assertNotIn(self.pre_place6d, poses)
        self.assertNotIn(self.place6d, poses)

    def test_bad_confirmation_cancels_before_motion(self):
        with patch.object(builtins, "input", return_value="yes"):
            ok, driver = self.execute(confirm=True)
        self.assertFalse(ok)
        self.assertEqual(self.controller.calls, [])
        self.assertEqual(driver.calls, [])

    def test_place_cancellation_keeps_object_held(self):
        with patch.object(builtins, "input", side_effect=["YES", "place"]):
            ok, driver = self.execute(confirm=True)
        self.assertFalse(ok)
        poses = [call[1] for call in self.controller.calls]
        self.assertEqual(len(poses), 4)  # pregrasp, grasp, retreat, ready
        self.assertEqual([c[0] for c in driver.calls if c[0] != "is_holding"], ["open", "grasp"])

    def test_success_sequence_retracts_vertically_after_release(self):
        ok, driver = self.execute(retract_to_pre_place=True)
        self.assertTrue(ok)
        poses = [call[1] for call in self.controller.calls]
        # pregrasp, grasp, retreat, ready, pre_place, place, pre_place (retract), ready
        self.assertEqual(len(poses), 8)
        self.assertEqual(poses[4], self.pre_place6d)
        self.assertEqual(poses[5], self.place6d)
        self.assertEqual(poses[6], self.pre_place6d)
        self.assertEqual([c[0] for c in driver.calls if c[0] != "is_holding"], ["open", "grasp", "open", "open"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
