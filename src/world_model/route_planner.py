from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class RoutePlannerConfig:
    lookahead_m: float = 25.0
    step_m: float = 2.0
    branch_random: bool = True


class LocalRoutePlanner:
    """A lightweight local route planner using CARLA waypoints.

    It follows the current lane forward and, at junctions, optionally picks
    a random branch. This is NOT a full global route planner, but is enough
    to learn urban behaviors (stop, yield, lane change, speed control).
    """

    def __init__(self, world_map, cfg: RoutePlannerConfig = RoutePlannerConfig()):
        self.map = world_map
        self.cfg = cfg
        self._rng = random.Random(0)
        self._cached: List = []
        self._last_branch_yaw: Optional[float] = None

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = random.Random(int(seed))
        self._cached = []
        self._last_branch_yaw = None

    def compute_route(self, vehicle_transform) -> List:
        """Return a list of waypoints ahead."""
        wp = self.map.get_waypoint(vehicle_transform.location, project_to_road=True)
        route = [wp]
        dist = 0.0
        step = float(self.cfg.step_m)
        while dist < float(self.cfg.lookahead_m):
            nxt = route[-1].next(step)
            if not nxt:
                break
            if len(nxt) == 1:
                chosen = nxt[0]
            else:
                # choose branch
                if self.cfg.branch_random:
                    chosen = self._rng.choice(nxt)
                else:
                    chosen = nxt[0]
                self._last_branch_yaw = float(chosen.transform.rotation.yaw)
            route.append(chosen)
            dist += step

        self._cached = route
        return route

    def turn_hint(self) -> float:
        """Return -1 left, 0 straight, +1 right based on chosen branch yaw.

        This is a heuristic: compares the chosen branch yaw with current route start.
        """
        if not self._cached or self._last_branch_yaw is None:
            return 0.0
        base_yaw = float(self._cached[0].transform.rotation.yaw)
        dyaw = (self._last_branch_yaw - base_yaw + 180.0) % 360.0 - 180.0
        if dyaw > 15.0:
            return 1.0
        if dyaw < -15.0:
            return -1.0
        return 0.0
