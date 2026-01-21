from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PIDParams:
    kp: float = 0.8
    ki: float = 0.05
    kd: float = 0.1
    i_max: float = 1.0


class PIDController:
    def __init__(self, params: PIDParams = PIDParams()):
        self.p = params
        self._i = 0.0
        self._prev_e = 0.0

    def reset(self) -> None:
        self._i = 0.0
        self._prev_e = 0.0

    def step(self, error: float, dt: float) -> float:
        if dt <= 0:
            dt = 1e-3
        self._i += error * dt
        self._i = max(-self.p.i_max, min(self.p.i_max, self._i))
        de = (error - self._prev_e) / dt
        self._prev_e = error
        return self.p.kp * error + self.p.ki * self._i + self.p.kd * de
