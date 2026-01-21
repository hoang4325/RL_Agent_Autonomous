from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PurePursuitParams:
    lookahead_m: float = 6.0
    wheelbase_m: float = 2.8
    steer_limit: float = 1.0


def _normalize_angle(x: float) -> float:
    while x > math.pi:
        x -= 2 * math.pi
    while x < -math.pi:
        x += 2 * math.pi
    return x


class PurePursuitController:
    """Pure pursuit lateral controller using CARLA waypoints.

    Works in world frame.
    """

    def __init__(self, params: PurePursuitParams = PurePursuitParams()):
        self.p = params

    def compute_steer(self, vehicle_transform, route_waypoints: List) -> float:
        if not route_waypoints:
            return 0.0

        vx = vehicle_transform.location.x
        vy = vehicle_transform.location.y
        yaw = math.radians(vehicle_transform.rotation.yaw)

        # find target waypoint at or beyond lookahead
        target = route_waypoints[-1]
        for wp in route_waypoints:
            dx = wp.transform.location.x - vx
            dy = wp.transform.location.y - vy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist >= self.p.lookahead_m:
                target = wp
                break

        dx = target.transform.location.x - vx
        dy = target.transform.location.y - vy

        # transform to vehicle frame
        x_v = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        y_v = math.sin(-yaw) * dx + math.cos(-yaw) * dy

        if x_v <= 1e-3:
            return 0.0

        # curvature kappa = 2*y / Ld^2
        Ld = math.sqrt(x_v * x_v + y_v * y_v)
        kappa = 2.0 * y_v / max(1e-3, Ld * Ld)

        steer = math.atan(self.p.wheelbase_m * kappa)
        # map steer angle to [-1,1] approximately
        steer_norm = float(max(-1.0, min(1.0, steer / 0.5)))
        steer_norm = max(-self.p.steer_limit, min(self.p.steer_limit, steer_norm))
        return steer_norm
