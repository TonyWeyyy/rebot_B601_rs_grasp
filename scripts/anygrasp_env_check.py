#!/usr/bin/env python3
"""Read-only AnyGrasp environment check; does not open the camera or CAN."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_import(name):
    try:
        module = importlib.import_module(name)
        print(f"[OK] import {name}: {getattr(module, '__version__', 'unknown')}")
        return module
    except Exception as exc:
        print(f"[FAIL] import {name}: {type(exc).__name__}: {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--feature-id", action="store_true")
    parser.add_argument("--check-license", action="store_true")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f).get("anygrasp", {})

    sdk_root = Path(str(cfg.get("sdk_root", PROJECT_ROOT / "sdk" / "anygrasp_sdk"))).expanduser()
    if not sdk_root.is_absolute():
        sdk_root = PROJECT_ROOT / sdk_root
    detection = sdk_root / "grasp_detection"
    checkpoint = Path(str(cfg.get("checkpoint", "log/checkpoint_detection.tar"))).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = detection / checkpoint

    failures = 0
    print("Python:", sys.version)
    if sys.version_info[:2] != (3, 10):
        print("[WARN] Expected project Python 3.10; select the gsnet.so matching your actual Python")

    torch = check_import("torch")
    if torch is None:
        failures += 1
    else:
        print("  torch CUDA:", torch.version.cuda)
        print("  CUDA available:", torch.cuda.is_available())
        if not torch.cuda.is_available():
            failures += 1
        else:
            print("  GPU:", torch.cuda.get_device_name(0))

    for name in ("numpy", "cv2", "open3d", "graspnetAPI", "MinkowskiEngine"):
        failures += int(check_import(name) is None)
    failures += int(check_import("pointnet2.pointnet2_utils") is None)

    for path in (detection / "gsnet.so", detection / "license", checkpoint):
        exists = path.exists()
        print(f"[{'OK' if exists else 'FAIL'}] path: {path}")
        failures += int(not exists)

    if (detection / "gsnet.so").is_file():
        sys.path.insert(0, str(detection))
        gsnet = check_import("gsnet")
        if gsnet is None:
            failures += 1
        else:
            if args.feature_id:
                try:
                    print("feature_id =", gsnet.get_feature_id())
                except Exception as exc:
                    failures += 1
                    print("[FAIL] get_feature_id:", exc)
            if args.check_license:
                try:
                    gsnet.check_license(str(detection / "license"))
                    print("[OK] license check passed")
                except Exception as exc:
                    failures += 1
                    print("[FAIL] license check:", exc)

    print(f"\nAnyGrasp environment failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
