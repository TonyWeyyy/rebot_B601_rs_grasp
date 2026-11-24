"""Unit tests for camera factory/serial wiring and GraspDriver lifecycle & holding checks."""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers.camera import OrbbecGemini2, RealsenseCamera, make_camera
from drivers.robot.grasp_driver import GraspDriver, _DEFAULT_GRIPPER_BY_ARM


class _FakeArm:
    def __init__(self):
        self.stopped_loop = False
        self.disconnected = False

    def stop_control_loop(self):
        self.stopped_loop = True

    def disconnect(self):
        self.disconnected = True


class _FakeController:
    def __init__(self):
        self._running = True
        self._stop_send = threading.Event()
        self._send_thread = None
        self.end_called = False

    def end(self):
        self.end_called = True
        self._running = False


def _make_bare_driver(arm_type: str = "rs") -> GraspDriver:
    driver = GraspDriver.__new__(GraspDriver)
    driver._arm = _FakeArm()
    driver._controller = _FakeController()
    driver._gripper_motor = None
    driver._last_gripper_state = None
    driver._state_lock = threading.Lock()
    driver._state = GraspDriver._STATE_IDLE
    driver._target_pos = 0.0
    driver._start_pos = 4.9
    driver._contact_pos = 4.9
    driver._position_reached = True
    driver._grasp_result = None
    driver._debug_last_print = 0.0
    driver._configure_gripper_params(_DEFAULT_GRIPPER_BY_ARM[arm_type])
    driver._hold_torque = driver._default_force
    return driver


class CameraAndDriverTest(unittest.TestCase):
    def test_make_camera_forwards_serial(self):
        cam_ob = make_camera({"camera": {"type": "orbbec_gemini2", "serial": "AY12345"}})
        self.assertIsInstance(cam_ob, OrbbecGemini2)
        self.assertEqual(cam_ob._serial, "AY12345")

        cam_rs = make_camera({"camera": {"type": "realsense_d435i", "serial": "04912207"}})
        self.assertIsInstance(cam_rs, RealsenseCamera)
        self.assertEqual(cam_rs._serial, "04912207")

    def test_dm_and_rs_have_isolated_hard_stop_and_debug_false(self):
        d_dm = _make_bare_driver("dm")
        d_rs = _make_bare_driver("rs")
        self.assertFalse(d_dm._debug)
        self.assertFalse(d_rs._debug)
        self.assertNotAlmostEqual(d_dm._hard_stop_angle, d_rs._hard_stop_angle)

    def test_is_holding_detects_slip_to_hard_stop(self):
        driver = _make_bare_driver("rs")
        driver._state = GraspDriver._STATE_HOLDING
        driver._last_gripper_state = (0.50, 0.0, -0.8)
        self.assertTrue(driver.is_holding())

        # Collapsed to hard stop (-0.70 < -0.65 + 0.05) -> dropped!
        driver._last_gripper_state = (-0.70, 0.0, -0.8)
        self.assertFalse(driver.is_holding())

    def test_stop_without_safe_home_skips_controller_end(self):
        driver = _make_bare_driver("rs")
        driver.stop(safe_home=False, release=False)
        self.assertFalse(driver._controller.end_called)
        self.assertFalse(driver._controller._running)
        self.assertTrue(driver._arm.stopped_loop)
        self.assertTrue(driver._arm.disconnected)

    def test_close_gripper_async_sets_position_target_cleanly(self):
        driver = _make_bare_driver("rs")
        t = driver.close_gripper_async(delay_s=0.0)
        t.join(timeout=1.0)
        self.assertEqual(driver._state, GraspDriver._STATE_POSITION)
        self.assertEqual(driver._target_pos, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
