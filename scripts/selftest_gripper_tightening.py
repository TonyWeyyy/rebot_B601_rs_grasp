#!/usr/bin/env python3
"""Offline regression tests for RS gripper post-contact tightening.

No camera, CAN bus, arm, gripper, or SDK connection is used. Run from the
project root: python scripts/selftest_gripper_tightening.py
"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drivers.robot.grasp_driver import GraspDriver


class FakeGripperGroup:
    def __init__(self) -> None:
        self.commands: list[dict] = []

    def send_mit(self, pos, vel, kp, kd, tau) -> None:
        self.commands.append(
            {
                "pos": float(pos[0]),
                "vel": float(vel[0]),
                "kp": float(kp[0]),
                "kd": float(kd[0]),
                "tau": float(tau[0]),
            }
        )


def make_driver(extra_angle: float = 0.12) -> GraspDriver:
    """Build only the RS gripper state machine, without hardware or SDK."""
    driver = GraspDriver.__new__(GraspDriver)
    driver._gripper_group = FakeGripperGroup()
    driver._gripper_motor = None
    driver._last_gripper_state = None
    driver._state_lock = threading.Lock()
    driver._state = GraspDriver._STATE_IDLE
    driver._target_pos = 0.0
    driver._start_pos = 4.9
    driver._contact_pos = 4.9
    driver._position_reached = True
    driver._grasp_result = None

    # RS direction and measured flexible-gripper empty-stop threshold.
    driver._angle_open = 5.0
    driver._open_sign = 1.0
    driver._close_sign = -1.0
    driver._open_lo = 0.0
    driver._open_hi = 4.9
    driver._hard_stop_angle = -0.65
    driver._tau_max = 1.5
    driver._close_torque = -1.5
    driver._default_force = -0.8
    driver._hold_torque = -0.8

    driver._arrive_tol = 0.12
    driver._kp_move = 5.0
    driver._kd_move = 1.0
    driver._kd_close = 0.5
    driver._stall_vel = 0.20
    driver._startup_dist = 0.30
    driver._post_contact_close_angle = extra_angle
    driver._post_contact_arrive_tol = 0.03
    driver._post_contact_timeout_s = 0.30
    driver._tighten_deadline = 0.0
    driver._debug = False
    driver._debug_last_print = 0.0
    return driver


def begin_closing(driver: GraspDriver) -> None:
    driver._state = GraspDriver._STATE_CLOSING
    driver._start_pos = 4.9
    driver._contact_pos = 4.9
    driver._target_pos = 0.0
    driver._grasp_result = None


def feedback(driver: GraspDriver, pos: float, vel: float, torq: float) -> None:
    driver._last_gripper_state = (pos, vel, torq)


def last_command(driver: GraspDriver) -> dict:
    return driver._gripper_group.commands[-1]


class GripperTighteningTests(unittest.TestCase):
    def test_local_contact_tightens_before_reporting_success(self) -> None:
        driver = make_driver(extra_angle=0.12)
        begin_closing(driver)
        feedback(driver, pos=1.20, vel=-1.0, torq=-0.60)
        driver.gripper_tick()

        self.assertEqual(driver._state, GraspDriver._STATE_TIGHTENING)
        self.assertIsNone(driver._grasp_result)
        self.assertAlmostEqual(driver._target_pos, 1.08, places=6)
        self.assertAlmostEqual(last_command(driver)["pos"], 1.08, places=6)
        self.assertAlmostEqual(last_command(driver)["tau"], -0.8, places=6)

        feedback(driver, pos=1.075, vel=0.0, torq=-0.8)
        driver.gripper_tick()
        self.assertEqual(driver._state, GraspDriver._STATE_HOLDING)
        self.assertTrue(driver._grasp_result)
        self.assertAlmostEqual(driver._target_pos, 1.08, places=6)

    def test_tightening_short_timeout_ends_in_holding(self) -> None:
        driver = make_driver()
        begin_closing(driver)
        feedback(driver, pos=1.20, vel=-1.0, torq=-0.60)
        driver.gripper_tick()
        self.assertIsNone(driver._grasp_result)

        # Simulate a blocked target and expire only the 0.30 s tightening phase.
        driver._tighten_deadline = time.monotonic() - 0.01
        driver.gripper_tick()
        self.assertEqual(driver._state, GraspDriver._STATE_HOLDING)
        self.assertTrue(driver._grasp_result)
        self.assertAlmostEqual(driver._target_pos, 1.08, places=6)

    def test_true_empty_close_still_returns_false(self) -> None:
        driver = make_driver()
        begin_closing(driver)
        feedback(driver, pos=-0.70, vel=0.0, torq=-1.40)
        driver.gripper_tick()

        self.assertEqual(driver._state, GraspDriver._STATE_POSITION)
        self.assertFalse(driver._grasp_result)
        self.assertEqual(driver._target_pos, 0.0)

    def test_soft_object_stopped_above_stop_is_not_empty(self) -> None:
        driver = make_driver()
        begin_closing(driver)
        feedback(driver, pos=0.20, vel=0.0, torq=-0.10)
        driver.gripper_tick()

        self.assertEqual(driver._state, GraspDriver._STATE_TIGHTENING)
        self.assertIsNone(driver._grasp_result)
        self.assertAlmostEqual(driver._target_pos, 0.08, places=6)
        feedback(driver, pos=0.08, vel=0.0, torq=-0.5)
        driver.gripper_tick()
        self.assertTrue(driver._grasp_result)

    def test_extra_target_never_crosses_closed_or_hard_stop_limit(self) -> None:
        driver = make_driver(extra_angle=0.30)
        begin_closing(driver)
        feedback(driver, pos=0.05, vel=0.0, torq=-0.60)
        driver.gripper_tick()

        self.assertEqual(driver._target_pos, 0.0)
        self.assertGreaterEqual(driver._target_pos, driver._open_lo)
        self.assertGreater(driver._target_pos, driver._hard_stop_angle)
        self.assertEqual(last_command(driver)["pos"], 0.0)

    def test_zero_extra_angle_restores_legacy_immediate_hold(self) -> None:
        driver = make_driver(extra_angle=0.0)
        begin_closing(driver)
        feedback(driver, pos=1.20, vel=-1.0, torq=-0.60)
        driver.gripper_tick()

        self.assertEqual(driver._state, GraspDriver._STATE_HOLDING)
        self.assertTrue(driver._grasp_result)
        self.assertEqual(driver._target_pos, 1.20)

    def test_free_flight_torque_does_not_trigger_contact(self) -> None:
        driver = make_driver()
        begin_closing(driver)
        feedback(driver, pos=2.0, vel=-2.0, torq=-0.40)
        driver.gripper_tick()

        self.assertEqual(driver._state, GraspDriver._STATE_CLOSING)
        self.assertIsNone(driver._grasp_result)

    def test_default_force_and_explicit_force_are_limited(self) -> None:
        for force, expected in ((None, -0.8), (99.0, -1.5)):
            with self.subTest(force=force):
                driver = make_driver()
                driver._ensure_running = lambda: None
                driver._wait_gripper_state = lambda timeout=1.0: (4.9, 0.0, 0.0)
                driver._wait_until = lambda predicate, timeout: False
                driver.grasp(force=force)
                self.assertAlmostEqual(driver._hold_torque, expected, places=6)

    def test_mit_torque_command_is_limited_by_tau_max(self) -> None:
        driver = make_driver()
        driver._send_gripper_mit(1.0, tau=-99.0)
        self.assertEqual(last_command(driver)["tau"], -1.5)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GripperTighteningTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(f"Gripper tightening self-test passed: {result.testsRun} tests")
