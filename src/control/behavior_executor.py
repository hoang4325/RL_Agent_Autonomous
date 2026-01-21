from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import carla

from src.control.pid import PIDController, PIDParams
from src.control.pure_pursuit import PurePursuitController, PurePursuitParams


@dataclass
class ExecutorConfig:
    max_throttle: float = 0.6
    max_brake: float = 1.0
    max_speed_mps: float = 12.0
    creep_speed_mps: float = 2.0
    lane_change_hold_steps: int = 30  # 3 seconds at 10Hz


class BehaviorExecutor:
    """Execute high-level actions via simple planning + control.

    High-level actions (strings):
      - KEEP_LANE
      - CHANGE_LEFT
      - CHANGE_RIGHT
      - STOP
      - GO
      - CREEP
      - YIELD

    This module converts the selected action into:
      - target lane (current/left/right)
      - target speed
      - low-level VehicleControl using pure pursuit + speed PID
    """

    def __init__(self, cfg: ExecutorConfig = ExecutorConfig()):
        self.cfg = cfg
        self.speed_pid = PIDController(PIDParams(kp=0.9, ki=0.05, kd=0.1, i_max=1.0))
        self.pp = PurePursuitController(PurePursuitParams(lookahead_m=6.0, wheelbase_m=2.8))

        self._lane_mode: int = 0  # -1 left, 0 current, +1 right
        self._lane_timer: int = 0

    def reset(self) -> None:
        self.speed_pid.reset()
        self._lane_mode = 0
        self._lane_timer = 0

    def _select_lane_route(self, base_route: List[carla.Waypoint], lane_mode: int) -> List[carla.Waypoint]:
        if lane_mode == 0 or not base_route:
            return base_route

        out: List[carla.Waypoint] = []
        for wp in base_route:
            adj = None
            if lane_mode < 0:
                adj = wp.get_left_lane()
            else:
                adj = wp.get_right_lane()
            if adj is None:
                out.append(wp)
            else:
                # keep driving lanes only
                if adj.lane_type == carla.LaneType.Driving:
                    out.append(adj)
                else:
                    out.append(wp)
        return out

    def step(
        self,
        action_name: str,
        world_map: carla.Map,
        vehicle: carla.Vehicle,
        base_route: List[carla.Waypoint],
        dt: float,
        safety_stop: bool,
        desired_speed_mps: float,
    ) -> tuple[carla.VehicleControl, float]:
        """Return (control, effective_desired_speed_mps)."""

        # lane mode persistence for lane changes
        if action_name == 'CHANGE_LEFT':
            self._lane_mode = -1
            self._lane_timer = self.cfg.lane_change_hold_steps
        elif action_name == 'CHANGE_RIGHT':
            self._lane_mode = 1
            self._lane_timer = self.cfg.lane_change_hold_steps
        elif action_name == 'KEEP_LANE':
            # allow cancel
            if self._lane_timer == 0:
                self._lane_mode = 0

        if self._lane_timer > 0:
            self._lane_timer -= 1
        else:
            # after hold, return to keep lane
            if action_name in ('CHANGE_LEFT', 'CHANGE_RIGHT'):
                self._lane_mode = 0

        route = self._select_lane_route(base_route, self._lane_mode)

        # speed target from action
        target_speed = float(desired_speed_mps)
        if action_name == 'STOP':
            target_speed = 0.0
        elif action_name == 'CREEP':
            target_speed = float(min(self.cfg.creep_speed_mps, desired_speed_mps))
        elif action_name == 'GO':
            target_speed = float(desired_speed_mps)
        elif action_name == 'YIELD':
            target_speed = float(min(self.cfg.creep_speed_mps, desired_speed_mps))

        if safety_stop:
            target_speed = 0.0

        steer = self.pp.compute_steer(vehicle.get_transform(), route)

        # speed PID
        v = vehicle.get_velocity()
        speed = float((v.x**2 + v.y**2 + v.z**2) ** 0.5)
        err = target_speed - speed
        u = self.speed_pid.step(err, dt)

        throttle = 0.0
        brake = 0.0
        if u >= 0:
            throttle = float(min(self.cfg.max_throttle, u))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(min(self.cfg.max_brake, -u))

        control = carla.VehicleControl(
            steer=float(steer),
            throttle=float(throttle),
            brake=float(brake),
            hand_brake=False,
            reverse=False,
        )

        return control, target_speed
