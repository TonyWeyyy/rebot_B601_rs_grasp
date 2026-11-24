"""Unit tests for core/ik_planner.py, core/robot_session.py, and CLI parameter precedence."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from core.ik_planner import select_executable_grasp
from core.robot_session import RobotSession
import grasp as grasp_script


R_DOWN = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]], dtype=np.float64)


class _GraspStub:
    def __init__(self, score, translation, rotation=None, width=0.04, depth=0.02):
        self.score = float(score)
        self.width = float(width)
        self.height = 0.03
        self.depth = float(depth)
        self.translation = np.asarray(translation, dtype=np.float64)
        self.rotation_matrix = R_DOWN.copy() if rotation is None else np.asarray(rotation, dtype=np.float64)


class _GraspGroupStub:
    def __init__(self, items=None):
        self.items = list(items or [])
        self.nms_calls = 0

    @property
    def grasp_group_array(self):
        return self.items

    def nms(self):
        self.nms_calls += 1
        return self

    def sort_by_score(self):
        self.items.sort(key=lambda g: g.score, reverse=True)
        return self

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class _ChainedIkStub:
    def __init__(self, fail_step: int | None = None):
        self.fail_step = fail_step
        self.chain_calls = 0

    def current_q(self):
        return np.zeros(6, dtype=np.float64)

    def check_chain(self, poses, q_init=None):
        self.chain_calls += 1
        sols = []
        errs = []
        for idx, pose in enumerate(poses):
            if self.fail_step == idx:
                errs.append(0.5)
                return False, errs, sols
            errs.append(1e-5)
            sols.append(np.full(6, 0.1 * (idx + 1) + float(pose[0])))
        return True, errs, sols


class IkAndSessionTest(unittest.TestCase):
    def test_retreat_ik_failure_rejects_candidate(self):
        group = _GraspGroupStub([_GraspStub(0.9, [0.25, 0.0, 0.12])])
        ik_fail_retreat = _ChainedIkStub(fail_step=2)
        res = select_executable_grasp(
            ik_fail_retreat,
            group,
            np.eye(4),
            pregrasp_offset_m=0.08,
            retreat_offset_m=0.08,
            insertion_depth_m=0.02,
            min_base_z_m=0.02,
            already_nms=True,
        )
        self.assertIsNone(res)
        self.assertEqual(ik_fail_retreat.chain_calls, 1)

    def test_already_nms_skips_redundant_nms_and_caps_candidates(self):
        items = [_GraspStub(0.9 - 0.01 * i, [0.25, 0.0, 0.12]) for i in range(10)]
        group = _GraspGroupStub(items)
        ik_always_fail = _ChainedIkStub(fail_step=0)

        with patch("core.ik_planner.rank_grasps", wraps=lambda g, run_nms=True: g) as spy_rank:
            select_executable_grasp(
                ik_always_fail,
                group,
                np.eye(4),
                0.08,
                0.08,
                0.02,
                0.02,
                already_nms=True,
                max_ik_candidates=4,
            )
            spy_rank.assert_called_once()
            _, kwargs = spy_rank.call_args
            self.assertFalse(kwargs["run_nms"])
        self.assertEqual(ik_always_fail.chain_calls, 4)

    def test_dry_run_robot_session_never_homes_or_releases_on_exit(self):
        stop_calls = []

        class FakeDriver:
            def stop(self, *, safe_home=True, release=False):
                stop_calls.append((safe_home, release))

        session = RobotSession({}, dry_run=True)
        session._started = True
        session.grasp_driver = FakeDriver()
        session.close()
        self.assertEqual(stop_calls, [(False, False)])

        stop_calls.clear()
        real_session = RobotSession({}, dry_run=False)
        real_session._started = True
        real_session.grasp_driver = FakeDriver()
        real_session.close()
        self.assertEqual(stop_calls, [(True, True)])

    def test_cli_overrides_anygrasp_yaml_values(self):
        cfg = {
            "grasp_pipeline": {"grasp": {"estimator": "anygrasp", "depth_fusion_frames": 3}},
            "anygrasp": {
                "min_depth": 0.05,
                "max_depth": 1.0,
                "target_expand_ratio": 1.35,
                "target_margin_px": 12,
                "top_k": 20,
            },
            "graspnet": {"top_k": 50},
        }
        args = argparse.Namespace(
            estimator="anygrasp",
            min_depth=0.12,
            max_depth=0.75,
            target_margin_px=4,
            target_expand_ratio=1.10,
            top_k=None,
            fusion_frames=5,
            pregrasp_offset=None,
            retreat_offset=None,
            min_base_z=None,
        )
        params = grasp_script.resolve_estimator_params(args, cfg)
        self.assertAlmostEqual(params["min_depth"], 0.12)
        self.assertAlmostEqual(params["max_depth"], 0.75)
        self.assertAlmostEqual(params["target_expand_ratio"], 1.10)
        self.assertEqual(params["target_margin_px"], 4)
        # top_k uses anygrasp.top_k (20), NOT graspnet.top_k (50)
        self.assertEqual(params["top_k"], 20)
        self.assertEqual(params["fusion_frames"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
