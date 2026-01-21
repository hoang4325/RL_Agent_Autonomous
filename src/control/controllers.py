from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import carla
except Exception:  # pragma: no cover
    carla = None  # type: ignore


@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    integ: float = 0.0
    prev: float = 0.0

    def step(self, err: float, dt: float) -> float:
        self.integ += err * dt
        der = (err - self.prev) / dt if dt > 1e-6 else 0.0
        self.prev = err
        return self.kp * err + self.ki * self.integ + self.kd * der


@dataclass
class ControlParams:
    wheel_base: float = 2.7
    lookahead_min: float = 6.0
    lookahead_gain: float = 0.5
    max_steer: float = 1.0


class PurePursuitController:
    def __init__(self, params: ControlParams | None = None):
        self.p = params or ControlParams()
        self.speed_pid = PID(kp=0.8, ki=0.0, kd=0.05)

    def compute_steer(self, target_loc: 'carla.Location', ego_tf: 'carla.Transform', speed_mps: float) -> float:
        # Transform target into ego coordinates
        dx = target_loc.x - ego_tf.location.x
        dy = target_loc.y - ego_tf.location.y
        yaw = math.radians(ego_tf.rotation.yaw)
        # ego frame: x forward, y right
        x = math.cos(-yaw)*dx - math.sin(-yaw)*dy
        y = math.sin(-yaw)*dx + math.cos(-yaw)*dy

        Ld = max(self.p.lookahead_min, self.p.lookahead_gain * speed_mps)
        if x < 1e-3:
            return 0.0
        # curvature
        curv = 2.0 * y / (Ld * Ld)
        steer = math.atan(self.p.wheel_base * curv)
        steer = max(-1.0, min(1.0, steer))
        return float(steer)

    def compute_throttle_brake(self, target_speed_mps: float, current_speed_mps: float, dt: float) -> tuple[float, float]:
        err = target_speed_mps - current_speed_mps
        u = self.speed_pid.step(err, dt)
        # map to throttle/brake
        if u >= 0:
            throttle = max(0.0, min(1.0, u))
            brake = 0.0
        else:
            throttle = 0.0
            brake = max(0.0, min(1.0, -u))
        return float(throttle), float(brake)
