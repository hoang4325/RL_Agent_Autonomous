from __future__ import annotations

import numpy as np


def carla_bgra_to_rgb(image) -> np.ndarray:
    """Convert CARLA Image (BGRA raw_data) to RGB uint8 array (H,W,3)."""
    h = int(image.height)
    w = int(image.width)
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((h, w, 4))
    bgr = arr[:, :, :3]
    return bgr[:, :, ::-1].copy()


def carla_bgra_to_bgr(image) -> np.ndarray:
    """Convert CARLA Image (BGRA raw_data) to BGR uint8 array (H,W,3)."""
    h = int(image.height)
    w = int(image.width)
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((h, w, 4))
    return arr[:, :, :3].copy()


def decode_depth_meters(depth_image) -> np.ndarray:
    """Decode CARLA depth camera to meters.

    CARLA depth camera encodes depth in the RGB channels:
        normalized = (R + G*256 + B*256^2) / (256^3 - 1)
        depth_m = 1000 * normalized

    Returns a float32 array (H,W) in meters.
    """
    bgr = carla_bgra_to_bgr(depth_image).astype(np.float32)
    b = bgr[:, :, 0]
    g = bgr[:, :, 1]
    r = bgr[:, :, 2]
    normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 ** 3 - 1.0)
    return (1000.0 * normalized).astype(np.float32)


def semantic_drivable_ratio(seg_image_bgr: np.ndarray) -> float:
    """Approximate drivable ratio from semantic segmentation palette.

    If you do NOT convert to RAW labels, CARLA semantic camera is often in
    CityScapes-like palette. Road color is typically (128, 64, 128) in RGB.

    We treat road + roadline as drivable.
    """
    if seg_image_bgr.ndim != 3 or seg_image_bgr.shape[2] != 3:
        return 0.0

    # Convert to RGB for easier color matching
    rgb = seg_image_bgr[:, :, ::-1]

    # Cityscapes palette colors (common):
    road = np.array([128, 64, 128], dtype=np.uint8)
    road_line = np.array([157, 234, 50], dtype=np.uint8)

    m_road = np.all(rgb == road, axis=2)
    m_line = np.all(rgb == road_line, axis=2)

    drivable = (m_road | m_line)
    return float(drivable.mean())


def center_depth_min(depth_m: np.ndarray, box: int = 9) -> float:
    """Return min depth in a small box around the image center."""
    h, w = depth_m.shape[:2]
    cy, cx = h // 2, w // 2
    r = max(1, int(box // 2))
    patch = depth_m[max(0, cy - r): min(h, cy + r + 1), max(0, cx - r): min(w, cx + r + 1)]
    if patch.size == 0:
        return float('inf')
    return float(np.min(patch))
