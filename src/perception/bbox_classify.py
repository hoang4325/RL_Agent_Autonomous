from __future__ import annotations

import numpy as np


def aabb_from_points(pts: np.ndarray):
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    center = 0.5 * (mn + mx)
    size = (mx - mn)
    return center, size, mn, mx


def yaw_from_pca_xy(pts: np.ndarray) -> float:
    xy = pts[:, :2]
    xy = xy - xy.mean(axis=0, keepdims=True)
    cov = (xy.T @ xy) / max(len(xy) - 1, 1)
    w, v = np.linalg.eigh(cov)
    principal = v[:, np.argmax(w)]
    return float(np.arctan2(principal[1], principal[0]))


def heuristic_classify(size_xyz: np.ndarray, n_points: int) -> str:
    dx, dy, dz = float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])
    length = max(dx, dy)
    width = min(dx, dy)
    height = dz

    if 0.2 <= width <= 1.0 and 0.2 <= length <= 1.4 and 1.0 <= height <= 2.6:
        return "walker"
    if 1.0 <= width <= 3.5 and 2.0 <= length <= 8.5 and 1.0 <= height <= 4.0:
        return "vehicle"
    return "static"
