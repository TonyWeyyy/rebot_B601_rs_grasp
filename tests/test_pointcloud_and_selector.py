"""Unit tests for utils/pointcloud.py, utils/target_selector.py, and utils/place_utils.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers.camera.base import fuse_depth_frames
from utils.place_utils import detection_center_cam
from utils.pointcloud import build_point_cloud, target_region_mask, workspace_mask
from utils.target_selector import select_by_keywords, select_container, select_target
from utils.yolo_utils import YoloDetection


def _make_det(name, conf, bbox, mask=None, shape=(40, 40), idx=0):
    if mask is None:
        mask = np.zeros(shape, dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 1
    return YoloDetection(0, idx, name, conf, bbox, mask)


class PointCloudAndSelectorTest(unittest.TestCase):
    def test_depth_fusion_suppresses_holes_and_spikes(self):
        f1 = np.full((10, 10), 500, dtype=np.uint16)
        f2 = np.full((10, 10), 502, dtype=np.uint16)
        f3 = np.full((10, 10), 504, dtype=np.uint16)
        f1[2, 2] = 0     # transient hole
        f3[5, 5] = 4000  # transient spike
        fused = fuse_depth_frames([f1, f2, f3])
        self.assertEqual(int(fused[2, 2]), 503)  # median of (502, 504)
        self.assertEqual(int(fused[5, 5]), 502)  # median of (500, 502, 4000)
        self.assertEqual(int(fused[0, 0]), 502)

    def test_mask_steering_prefers_mask_and_falls_back_to_bbox(self):
        color = np.zeros((40, 40, 3), dtype=np.uint8)
        depth = np.full((40, 40), 500, dtype=np.uint16)
        K = np.array([[40.0, 0.0, 20.0], [0.0, 40.0, 20.0], [0.0, 0.0, 1.0]])
        cloud_o3d, points, _ = build_point_cloud(
            color, depth, K, min_depth_m=0.1, max_depth_m=1.0, max_input_points=0, build_o3d=False
        )
        self.assertIsNone(cloud_o3d)
        self.assertEqual(len(points), 1600)

        # L-shaped mask inside bbox (5,5)-(35,35)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[5:35, 5:15] = 1
        target = _make_det("toy", 0.9, (5, 5, 35, 35), mask=mask)
        ws = workspace_mask(points, [-1, 1, -1, 1, 0.1, 1.0])

        region_mask, mode = target_region_mask(
            points, target, K, image_shape=(40, 40), base_mask=ws, use_mask=True, mask_dilate_px=0, min_points=100
        )
        self.assertEqual(mode, "mask")
        self.assertLess(int(region_mask.sum()), 500)

        # Tiny mask falls back to bbox
        tiny_mask = np.zeros((40, 40), dtype=np.uint8)
        tiny_mask[10:12, 10:12] = 1
        tiny_target = _make_det("toy", 0.9, (5, 5, 35, 35), mask=tiny_mask)
        region_fb, mode_fb = target_region_mask(
            points, tiny_target, K, image_shape=(40, 40), base_mask=ws, use_mask=True, mask_dilate_px=0, min_points=200
        )
        self.assertEqual(mode_fb, "bbox")
        self.assertGreater(int(region_fb.sum()), 200)

    def test_mask_centroid_tracks_asymmetric_container(self):
        # Asymmetric mask concentrated near left side of bbox (10..30)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:30, 10:14] = 1
        det = _make_det("box", 0.8, (10, 10, 30, 30), mask=mask)
        depth = np.full((40, 40), 500, dtype=np.uint16)
        K = np.array([[100.0, 0.0, 20.0], [0.0, 100.0, 20.0], [0.0, 0.0, 1.0]])

        c_bbox = detection_center_cam(det, depth, K, use_mask_centroid=False)
        c_mask = detection_center_cam(det, depth, K, use_mask_centroid=True)
        # Bbox center is at u=20 -> x=0; mask centroid is at u=11.5 -> x < 0
        self.assertAlmostEqual(float(c_bbox[0]), 0.0, places=6)
        self.assertLess(float(c_mask[0]), -0.03)

    def test_target_and_keyword_selection(self):
        d1 = _make_det("red toy", 0.85, (0, 0, 10, 10), idx=1)
        d2 = _make_det("toy", 0.70, (10, 10, 20, 20), idx=2)
        d3 = _make_det("cardboard box", 0.90, (20, 20, 35, 35), idx=3)
        # Exact match wins over higher-conf substring match
        self.assertIs(select_target([d1, d2, d3], "toy"), d2)
        self.assertIs(select_container([d1, d2, d3], "box", exclude=d2), d3)
        self.assertIs(select_by_keywords([d1, d2, d3], ("red",)), d1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
