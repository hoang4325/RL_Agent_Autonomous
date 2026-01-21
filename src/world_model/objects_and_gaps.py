from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    import carla
except Exception:  # pragma: no cover
    carla = None  # type: ignore

from ..utils.carla_utils import forward_vector_yaw_deg


@dataclass
class GapFeatures:
    front_gap: float
    front_rel_speed: float
    left_front_gap: float
    left_rear_gap: float
    right_front_gap: float
    right_rear_gap: float
    min_ttc: float


def ego_to_world(ego_tf: 'carla.Transform', x_ego: float, y_ego: float) -> Tuple[float, float]:
    # using ego frame where x forward, y right
    yaw = math.radians(ego_tf.rotation.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    wx = ego_tf.location.x + c * x_ego - s * y_ego
    wy = ego_tf.location.y + s * x_ego + c * y_ego
    return wx, wy


def compute_gap_features(map_obj: 'carla.Map', ego_vehicle: 'carla.Vehicle', tracks: Dict[int, any], max_consider_m: float = 60.0) -> GapFeatures:
    """Compute simple lane-relative gaps from tracked objects.

    Tracks are in ego frame: (x forward, y right).
    We assign objects to lanes by projecting their world location to a driving waypoint and using lane_id.
    """
    ego_tf = ego_vehicle.get_transform()
    ego_wp = map_obj.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    ego_lane_id = ego_wp.lane_id if ego_wp else 0
    road_id = ego_wp.road_id if ego_wp else 0

    # init large gaps
    INF = 1e9
    front_gap = INF
    front_rel_speed = 0.0
    left_front = INF
    left_rear = INF
    right_front = INF
    right_rear = INF

    min_ttc = INF

    ego_speed = float(math.sqrt(ego_vehicle.get_velocity().x**2 + ego_vehicle.get_velocity().y**2 + ego_vehicle.get_velocity().z**2))
    fwd = forward_vector_yaw_deg(ego_tf.rotation.yaw)

    for _, t in tracks.items():
        # only consider in front/rear region
        dx = float(t.x)
        dy = float(t.y)
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > max_consider_m:
            continue

        wx, wy = ego_to_world(ego_tf, dx, dy)
        wp = map_obj.get_waypoint(carla.Location(x=wx, y=wy, z=ego_tf.location.z), project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            continue
        if wp.road_id != road_id:
            # skip different road segments (simplify)
            continue

        # relative longitudinal along ego forward (world)
        # approximate: use ego-frame x as longitudinal
        longi = dx

        # relative speed along ego forward
        rel_v = float(t.vx * 1.0 + t.vy * 0.0)  # in ego frame x is forward
        closing = ego_speed - rel_v

        # TTC for objects in front
        if longi > 0.5 and closing > 0.1:
            ttc = longi / closing
            min_ttc = min(min_ttc, ttc)

        lane = wp.lane_id
        if lane == ego_lane_id:
            if longi > 0.5 and longi < front_gap:
                front_gap = longi
                front_rel_speed = rel_v - ego_speed
        elif lane == ego_lane_id - 1:
            # right lane in CARLA lane_id sign depends on direction; heuristic: use lane_id +/-1
            if longi > 0.5:
                left_front = min(left_front, longi)
            elif longi < -0.5:
                left_rear = min(left_rear, abs(longi))
        elif lane == ego_lane_id + 1:
            if longi > 0.5:
                right_front = min(right_front, longi)
            elif longi < -0.5:
                right_rear = min(right_rear, abs(longi))

    # replace INF with max_consider_m
    def cap(x):
        return float(max_consider_m if x >= INF/2 else x)

    return GapFeatures(
        front_gap=cap(front_gap),
        front_rel_speed=float(front_rel_speed),
        left_front_gap=cap(left_front),
        left_rear_gap=cap(left_rear),
        right_front_gap=cap(right_front),
        right_rear_gap=cap(right_rear),
        min_ttc=float(10.0 if min_ttc >= INF/2 else min_ttc),
    )
