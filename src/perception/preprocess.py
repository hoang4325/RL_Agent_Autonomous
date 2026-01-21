from __future__ import annotations

import numpy as np


def roi_filter(pts: np.ndarray,
               x_min=0.0, x_max=60.0,
               y_min=-25.0, y_max=25.0,
               z_min=-2.5, z_max=3.0) -> np.ndarray:
    if len(pts) == 0:
        return pts
    x = pts[:, 0]; y = pts[:, 1]; z = pts[:, 2]
    m = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z >= z_min) & (z <= z_max)
    return pts[m]


def voxel_downsample(pts: np.ndarray, voxel_size: float = 0.15) -> np.ndarray:
    if len(pts) == 0:
        return pts
    idx = np.floor(pts / voxel_size).astype(np.int32)
    _, uniq = np.unique(idx, axis=0, return_index=True)
    return pts[uniq]


def ground_removal_height(pts: np.ndarray, z_threshold: float = -1.4) -> np.ndarray:
    """Remove ground points by simple z-threshold in LiDAR frame.

    In CARLA, LiDAR frame is attached to ego. If sensor is at z~2.2,
    ground points often have z around -2.0..-1.2 depending on terrain.
    Tune z_threshold per town.
    """
    if len(pts) == 0:
        return pts
    return pts[pts[:, 2] > z_threshold]
