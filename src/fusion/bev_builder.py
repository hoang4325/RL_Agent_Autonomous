from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class BEVConfig:
    size: int = 96
    meters_per_cell: float = 0.5
    # normalize velocities by this reference (m/s)
    v_ref_mps: float = 12.0  # ~43 km/h
    # make points thicker so CNN sees them
    point_radius: int = 1    # 1 -> 3x3


def world_to_grid(x: float, y: float, cfg: BEVConfig) -> Tuple[int, int]:
    half = cfg.size // 2
    ix = int(np.round(x / cfg.meters_per_cell)) + half
    iy = int(np.round(y / cfg.meters_per_cell)) + half
    return ix, iy


def _stamp(bev: np.ndarray, ch: int, iy: int, ix: int, value: float, r: int) -> None:
    """Stamp value into a (2r+1)x(2r+1) square."""
    H, W = bev.shape[1], bev.shape[2]
    y0, y1 = max(0, iy - r), min(H, iy + r + 1)
    x0, x1 = max(0, ix - r), min(W, ix + r + 1)
    # for occupancy/mask: take max
    if ch in (0, 1, 4):
        bev[ch, y0:y1, x0:x1] = np.maximum(bev[ch, y0:y1, x0:x1], value)
    else:
        # for velocity channels: overwrite center region (or max by magnitude)
        # here we take value as-is but keep the one with larger abs
        patch = bev[ch, y0:y1, x0:x1]
        sel = np.abs(value) >= np.abs(patch)
        patch[sel] = value
        bev[ch, y0:y1, x0:x1] = patch


def build_bev_from_tracks(
    tracks: Dict[int, any],
    drivable_mask: np.ndarray | None,
    route_points_xy: np.ndarray | None,
    cfg: BEVConfig,
) -> np.ndarray:
    """Returns BEV tensor (C,H,W) float32.

    Channels:
      0 drivable [0,1]
      1 obstacles occupancy [0,1]
      2 velocity_x scaled to [-1,1]
      3 velocity_y scaled to [-1,1]
      4 route [0,1]
    """
    H = W = cfg.size
    bev = np.zeros((5, H, W), dtype=np.float32)

    # drivable
    if drivable_mask is not None and drivable_mask.shape == (H, W):
        bev[0] = np.clip(drivable_mask.astype(np.float32), 0.0, 1.0)

    v_ref = max(1.0, float(cfg.v_ref_mps))
    r = int(cfg.point_radius)

    # obstacles + velocity (ego frame)
    for _tid, t in tracks.items():
        ix, iy = world_to_grid(float(t.x), float(t.y), cfg)
        if 0 <= ix < W and 0 <= iy < H:
            # occupancy
            _stamp(bev, 1, iy, ix, 1.0, r)

            # normalized velocities
            vx = float(t.vx) / v_ref
            vy = float(t.vy) / v_ref
            vx = float(np.clip(vx, -1.0, 1.0))
            vy = float(np.clip(vy, -1.0, 1.0))

            _stamp(bev, 2, iy, ix, vx, r)
            _stamp(bev, 3, iy, ix, vy, r)

    # route
    if route_points_xy is not None and len(route_points_xy) > 0:
        for p in route_points_xy:
            ix, iy = world_to_grid(float(p[0]), float(p[1]), cfg)
            if 0 <= ix < W and 0 <= iy < H:
                _stamp(bev, 4, iy, ix, 1.0, r)

    # ensure masks are in [0,1]
    bev[1] = np.clip(bev[1], 0.0, 1.0)
    bev[4] = np.clip(bev[4], 0.0, 1.0)

    return bev
