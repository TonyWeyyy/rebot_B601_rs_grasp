"""
Preset OBB pick-and-place wrapper (toy/red -> box/container/tray/bowl, height 0.15 m).

Delegates all execution and safety logic to scripts/set.py.

Usage:
    python scripts/set2.py
    python scripts/set2.py --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in (str(PROJECT_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from set import parse_args, run_obb_place  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(
        argv,
        default_target_keywords="toy,red",
        default_container_keywords="box,container,tray,bowl",
        default_place_height=0.15,
        window_title="Set2 - Place Target into Container",
    )
    return run_obb_place(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
