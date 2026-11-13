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

# The delivered source archive intentionally omits the large vendor SDK. Stub
# hardware modules so this test remains runnable in the source-only archive.
for subdir in ("", "models", "dataset", "utils", "pointnet2", "graspnetAPI"):
    path = str(ROOT / "sdk" / "graspnet-baseline" / subdir)
    if path not in sys.path:
        sys.path.insert(0, path)

sys.modules.setdefault("open3d", types.ModuleType("open3d"))
graspnet_api_stub = types.ModuleType("graspnetAPI")

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

graspnet_api_stub.Grasp = FakeGrasp
graspnet_api_stub.GraspGroup = FakeGraspGroup
sys.modules["graspnetAPI"] = graspnet_api_stub

vis_stub = types.ModuleType("utils.graspnet_utils")
vis_stub.Open3DGraspWindow = object
vis_stub.draw_detections_overlay = lambda color, *args: color
vis_stub.draw_best_grasp_projection = lambda *args: None
vis_stub.draw_status = lambda color, *args, **kwargs: color
sys.modules["utils.graspnet_utils"] = vis_stub

grasp_stub = types.ModuleType("grasp")
grasp_stub.IkChecker = object
grasp_stub._select_executable_grasp = lambda *args: None
grasp_stub._wait_motion = lambda *args: None
grasp_stub._move_ready = lambda controller, cfg: controller.move_to_pose(
    *cfg.get("position", [0.3, 0, 0.2, 0, 1.2, 0]), duration=cfg.get("duration", 4.0)
)
sys.modules["grasp"] = grasp_stub

camera_stub = types.ModuleType("drivers.camera")
camera_stub.CameraDriver = object
camera_stub.make_camera = lambda cfg: None
sys.modules["drivers.camera"] = camera_stub

driver_stub = types.ModuleType("drivers.robot.grasp_driver")
driver_stub.GRIPPER_MAX_DISTANCE_M = 0.09
driver_stub.GraspDriver = object
driver_stub.selected_arm_config = lambda root: None
sys.modules["drivers.robot.grasp_driver"] = driver_stub

sdk_parent = types.ModuleType("reBotArm_control_py")
sdk_parent.__path__ = []
sys.modules["reBotArm_control_py"] = sdk_parent
for name, attr in (("reBotArm_control_py.actuator", "RebotArm"), ("reBotArm_control_py.controllers", "RebotArmEndPose")):
    module = types.ModuleType(name)
    setattr(module, attr, object)
    sys.modules[name] = module

import anygrasp_place as app
import utils.anygrasp_utils as ag
import utils.place_utils as pu
from utils.yolo_utils import YoloDetection


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
        center = pu.detection_center_cam(box, depth, K)
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
        group = ag.GraspGroup()
        grasp = ag.Grasp()
        grasp.score, grasp.width, grasp.height, grasp.depth = 0.8, 0.05, 0.03, 0.02
        grasp.rotation_matrix = np.eye(3)
        grasp.translation = np.array([0, 0, 0.5])
        group.add(grasp)

        class Detector:
            def get_grasp(self, points, options):
                return group

        points = np.tile(np.array([0.0, 0.0, 0.5]), (400, 1))
        with patch.object(ag, "detect_objects", side_effect=AssertionError("YOLO ran twice")), patch.object(
            ag, "build_point_cloud", return_value=(None, points, None)
        ):
            result = ag.infer_frame(
                Detector(), color, depth, K,
                cfg={"lims": [-1, 1, -1, 1, 0.1, 0.8], "max_input_points": 1000},
                min_depth=0.05, max_depth=1.0,
                yolo_model=object(), detections=[toy, box], target_class="toy",
            )
        self.assertIs(result.selected_target, toy)
        self.assertEqual(len(result.detections), 2)


class FakeDriver:
    def __init__(self, grasp_result=True):
        self.calls = []
        self.grasp_result = grasp_result

    def open_gripper(self, **kwargs):
        self.calls.append(("open", kwargs))

    def grasp(self):
        self.calls.append(("grasp",))
        return self.grasp_result


class FakeController:
    def __init__(self):
        self.calls = []

    def move_to_traj(self, *args, duration=None):
        self.calls.append(("move", args, duration))
        return True

    def move_to_pose(self, *args, duration=None):
        self.calls.append(("ready", args, duration))
        return True


class SequenceSafetyTest(unittest.TestCase):
    def setUp(self):
        self.controller = FakeController()
        self.driver = FakeDriver()
        self.poses = ((0.3, 0, 0.1, 0, 1.2, 0),) * 5

    def execute(self, dry=False, empty=False, confirm=False):
        driver = FakeDriver(not empty)
        with patch.object(app, "_wait_motion", lambda *args: None), patch.object(app.time, "sleep", lambda *args: None):
            ok = app._execute_place_sequence(
                self.controller, driver, *self.poses,
                ready_cfg={"position": [0.3, 0, 0.2, 0, 1.2, 0]},
                place_cfg={}, dry_run=dry, confirm_motion=confirm,
            )
        return ok, driver

    def test_dry_run_has_no_motion(self):
        ok, driver = self.execute(dry=True)
        self.assertTrue(ok)
        self.assertEqual(self.controller.calls, [])
        self.assertEqual(driver.calls, [])

    def test_empty_grasp_never_moves_to_place(self):
        ok, driver = self.execute(empty=True)
        self.assertFalse(ok)
        moves = [call for call in self.controller.calls if call[0] == "move"]
        self.assertEqual(len(moves), 2)  # pregrasp, grasp only
        self.assertEqual(len(self.driver.calls), 0)

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
        moves = [call for call in self.controller.calls if call[0] == "move"]
        self.assertEqual(len(moves), 3)  # pregrasp, grasp, retreat; no pre-place/place
        self.assertEqual([call[0] for call in driver.calls], ["open", "grasp"])

    def test_success_sequence_releases_only_at_container(self):
        ok, driver = self.execute()
        self.assertTrue(ok)
        moves = [call for call in self.controller.calls if call[0] == "move"]
        self.assertEqual(len(moves), 5)
        self.assertEqual([call[0] for call in driver.calls], ["open", "grasp", "open", "open"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
