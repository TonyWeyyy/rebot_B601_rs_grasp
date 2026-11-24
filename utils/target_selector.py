"""Shared target and container selection helpers for YOLO detections."""

from __future__ import annotations

from typing import Optional, Sequence, TypeVar

from utils.yolo_utils import YoloDetection

T = TypeVar("T")


def same_detection(a: YoloDetection, b: YoloDetection) -> bool:
    return (a.result_index, a.detection_index) == (b.result_index, b.detection_index)


def select_target(
    detections: Sequence[YoloDetection],
    target_class: Optional[str],
    *,
    exclude: Optional[YoloDetection] = None,
) -> Optional[YoloDetection]:
    """Select the highest-confidence exact/contains class match."""
    candidates = [
        det for det in detections
        if exclude is None or not same_detection(det, exclude)
    ]
    if not candidates:
        return None
    if target_class:
        query = target_class.casefold()
        exact = [det for det in candidates if det.class_name.casefold() == query]
        contains = [det for det in candidates if query in det.class_name.casefold()]
        candidates = exact or contains
    if not candidates:
        return None
    return max(candidates, key=lambda det: det.conf)


def select_container(
    detections: Sequence[YoloDetection],
    container_class: str,
    *,
    exclude: Optional[YoloDetection] = None,
) -> Optional[YoloDetection]:
    """Select the highest-confidence container detection, excluding the grasp target."""
    return select_target(detections, container_class, exclude=exclude)


def select_by_keywords(
    items: Sequence[T],
    keywords: Sequence[str],
    *,
    exclude: Optional[T] = None,
) -> Optional[T]:
    """Select highest-confidence object (YoloDetection or GraspPose) matching any keyword."""
    lowered = [kw.casefold() for kw in keywords if kw]
    best = None
    for item in items:
        if exclude is not None and item is exclude:
            continue
        if hasattr(item, "is_valid") and not item.is_valid:
            continue
        name = str(getattr(item, "class_name", "")).casefold()
        if lowered and not any(kw in name for kw in lowered):
            continue
        if best is None or float(getattr(item, "conf", 0.0)) > float(getattr(best, "conf", 0.0)):
            best = item
    return best


def selected_target_text(selected: Optional[YoloDetection], target_class: Optional[str]) -> str:
    if selected is None:
        return f"target={target_class or 'best'} not found"
    return f"target={selected.class_name} {selected.conf:.2f}"


def target_status_text(
    selected: Optional[YoloDetection],
    detections: Sequence[YoloDetection],
    target_class: Optional[str],
) -> str:
    if selected is not None:
        return f"target={selected.class_name} {selected.conf:.2f} detections={len(detections)}"
    if target_class:
        return f"target={target_class} not found detections={len(detections)}"
    return f"target not found detections={len(detections)}"
