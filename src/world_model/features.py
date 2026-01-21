from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FeatureConfig:
    vec_dim: int = 32
    lane_width_m: float = 3.5
    tl_consider_dist_m: float = 35.0


def _wrap_pi(x: float) -> float:
    while x > math.pi:
        x -= 2.0 * math.pi
    while x < -math.pi:
        x += 2.0 * math.pi
    return x


def lane_metrics_from_map(world_map, vehicle_transform) -> Tuple[float, float]:
    """Return (lateral_deviation_m, heading_err_rad) using CARLA map waypoint.

    world_map: carla.Map
    vehicle_transform: carla.Transform
    """
    loc = vehicle_transform.location
    wp = world_map.get_waypoint(loc, project_to_road=True)

    # Compute lateral distance to lane center in 2D
    cx = wp.transform.location.x
    cy = wp.transform.location.y
    vx = loc.x
    vy = loc.y
    dx = vx - cx
    dy = vy - cy

    # lane forward (2D)
    yaw = math.radians(wp.transform.rotation.yaw)
    fx = math.cos(yaw)
    fy = math.sin(yaw)

    # perpendicular component magnitude
    proj = dx * fx + dy * fy
    px = dx - proj * fx
    py = dy - proj * fy
    lat = math.sqrt(px * px + py * py)

    veh_yaw = math.radians(vehicle_transform.rotation.yaw)
    heading_err = _wrap_pi(veh_yaw - yaw)

    return float(lat), float(heading_err)


def _gaps_in_lane(tracks: List[Any], lane_selector: str, lane_width_m: float) -> Tuple[float, float, float]:
    """Compute (front_gap, rear_gap, rear_closing) in ego frame.

    lane_selector: 'current' | 'left' | 'right'

    We classify lane by y (right positive). This is approximate but works as a template.
    """
    half = lane_width_m * 0.5

    def in_lane(y: float) -> bool:
        if lane_selector == 'current':
            return abs(y) <= half
        if lane_selector == 'left':
            return (-1.5 * lane_width_m) <= y < (-half)
        if lane_selector == 'right':
            return (half) < y <= (1.5 * lane_width_m)
        return False

    front_gap = float('inf')
    rear_gap = float('inf')
    rear_closing = 0.0

    # Ego frame: x forward, y right
    rear_x = -float('inf')
    rear_vx = 0.0

    for t in tracks:
        x, y, vx, vy = float(t.state[0]), float(t.state[1]), float(t.state[2]), float(t.state[3])
        if not in_lane(y):
            continue

        if x >= 0.0 and x < front_gap:
            front_gap = x
        if x < 0.0 and x > rear_x:
            rear_x = x
            rear_vx = vx

    if rear_x > -float('inf'):
        rear_gap = abs(rear_x)
        # closing speed from rear (positive means rear is approaching ego)
        rear_closing = max(0.0, rear_vx)  # ego forward axis; we don't subtract ego speed here

    if not math.isfinite(front_gap):
        front_gap = 999.0
    if not math.isfinite(rear_gap):
        rear_gap = 999.0

    return float(front_gap), float(rear_gap), float(rear_closing)


def compute_ttc(front_gap_m: float, ego_speed_mps: float, lead_speed_mps: float) -> float:
    closing = max(0.0, ego_speed_mps - lead_speed_mps)
    if closing <= 1e-3:
        return 99.0
    return float(np.clip(front_gap_m / closing, 0.0, 99.0))


def build_feature_vector(
    cfg: FeatureConfig,
    ego_speed_mps: float,
    lane_dev_m: float,
    heading_err_rad: float,
    desired_speed_mps: float,
    tl_red: float,
    tl_dist_m: float,
    tracks: List[Any],
    radar_min_range: float,
    radar_closing_speed: float,
    depth_min_ahead_m: float,
    seg_drivable_ratio: float,
    imu_accel_x: float,
    imu_gyro_z: float,
    gnss_lat: float,
    gnss_lon: float,
    lane_invasion: float,
    collision: float,
    route_turn_hint: float,
) -> np.ndarray:
    """Return a fixed-size feature vector for RL."""

    lane_w = cfg.lane_width_m

    cur_front, cur_rear, _ = _gaps_in_lane(tracks, 'current', lane_w)
    left_front, left_rear, left_rear_cl = _gaps_in_lane(tracks, 'left', lane_w)
    right_front, right_rear, right_rear_cl = _gaps_in_lane(tracks, 'right', lane_w)

    # estimate lead speed from nearest front track in current lane
    lead_speed = 0.0
    for t in tracks:
        x, y, vx, vy = float(t.state[0]), float(t.state[1]), float(t.state[2]), float(t.state[3])
        if abs(y) <= lane_w * 0.5 and 0.0 <= x <= cur_front + 0.5:
            lead_speed = vx
            break

    ttc = compute_ttc(cur_front, ego_speed_mps, lead_speed)

    # normalize & clip
    tl_dist_n = float(np.clip(tl_dist_m / cfg.tl_consider_dist_m, 0.0, 1.0))
    gaps = [
        np.clip(cur_front / 80.0, 0.0, 1.0),
        np.clip(lead_speed / 20.0, -1.0, 1.0),
        np.clip(ttc / 10.0, 0.0, 1.0),
        np.clip(left_front / 80.0, 0.0, 1.0),
        np.clip(left_rear / 80.0, 0.0, 1.0),
        np.clip(right_front / 80.0, 0.0, 1.0),
        np.clip(right_rear / 80.0, 0.0, 1.0),
        np.clip(left_rear_cl / 20.0, 0.0, 1.0),
        np.clip(right_rear_cl / 20.0, 0.0, 1.0),
    ]

    vec = np.zeros((cfg.vec_dim,), dtype=np.float32)

    base = [
        np.clip(ego_speed_mps / 20.0, 0.0, 1.0),
        np.clip(lane_dev_m / 3.0, 0.0, 1.0),
        np.clip(abs(heading_err_rad) / math.pi, 0.0, 1.0),
        np.clip(desired_speed_mps / 20.0, 0.0, 1.0),
        float(tl_red),
        tl_dist_n,
    ]

    sensor_misc = [
        np.clip(radar_min_range / 80.0, 0.0, 1.0),
        np.clip(radar_closing_speed / 30.0, -1.0, 1.0),
        np.clip(depth_min_ahead_m / 100.0, 0.0, 1.0),
        np.clip(seg_drivable_ratio, 0.0, 1.0),
        np.clip(imu_accel_x / 10.0, -1.0, 1.0),
        np.clip(imu_gyro_z / 5.0, -1.0, 1.0),
        np.clip(gnss_lat / 90.0, -1.0, 1.0),
        np.clip(gnss_lon / 180.0, -1.0, 1.0),
        float(lane_invasion),
        float(collision),
        np.clip(route_turn_hint, -1.0, 1.0),
    ]

    all_feats = base + gaps + sensor_misc
    n = min(len(all_feats), cfg.vec_dim)
    vec[:n] = np.array(all_feats[:n], dtype=np.float32)

    return vec
