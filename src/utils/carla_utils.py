from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

try:
    import carla
except Exception:  # pragma: no cover
    carla = None  # type: ignore


@dataclass
class TransformSpec:
    x: float
    y: float
    z: float
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0


def to_carla_transform(spec: Dict[str, Any]) -> 'carla.Transform':
    s = TransformSpec(**spec)
    return carla.Transform(
        carla.Location(x=s.x, y=s.y, z=s.z),
        carla.Rotation(pitch=s.pitch, yaw=s.yaw, roll=s.roll),
    )


def set_world_sync(world: 'carla.World', synchronous_mode: bool, fixed_delta_seconds: float, no_rendering: bool) -> None:
    settings = world.get_settings()
    settings.synchronous_mode = synchronous_mode
    settings.fixed_delta_seconds = fixed_delta_seconds
    settings.no_rendering_mode = no_rendering
    world.apply_settings(settings)


def get_speed_mps(vehicle: 'carla.Vehicle') -> float:
    v = vehicle.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def forward_vector_yaw_deg(yaw_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    return np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def destroy_actors(actors):
    """Best-effort actor destruction without spamming CARLA server errors.

    CARLA may already have removed actors (e.g., sensors/controllers) when the world resets.
    Calling destroy() on stale handles can produce server-side 'actor not found' logs.
    We reduce that by:
      - de-duplicating actors by id
      - skipping actors that are not alive
      - skipping ids that world no longer knows about
      - swallowing RuntimeError/Exception
    """
    if not actors:
        return

    uniq = []
    seen = set()
    for a in list(actors):
        if a is None:
            continue
        try:
            aid = int(getattr(a, 'id', -1))
        except Exception:
            aid = -1
        if aid != -1:
            if aid in seen:
                continue
            seen.add(aid)
        uniq.append(a)

    for a in uniq:
        try:
            if hasattr(a, 'is_alive') and (not a.is_alive):
                continue
            aid = getattr(a, 'id', None)
            if aid is not None:
                try:
                    w = a.get_world()
                    if w is not None and w.get_actor(aid) is None:
                        continue
                except Exception:
                    pass
            a.destroy()
        except RuntimeError:
            # Typically: actor already destroyed / not found
            pass
        except Exception:
            pass

    # Clear in-place if it's a list
    try:
        actors.clear()
    except Exception:
        pass
def try_get_traffic_light_state(vehicle: 'carla.Vehicle') -> Tuple[int, float]:
    """Returns (state_code, distance_m). state_code: 0=Unknown,1=Green,2=Yellow,3=Red."""
    state_code = 0
    dist = 999.0

    # state
    try:
        if hasattr(vehicle, 'get_traffic_light_state'):
            st = vehicle.get_traffic_light_state()
            # st is carla.TrafficLightState
            if str(st).endswith('Green'):
                state_code = 1
            elif str(st).endswith('Yellow'):
                state_code = 2
            elif str(st).endswith('Red'):
                state_code = 3
        elif hasattr(vehicle, 'get_traffic_light'):
            tl = vehicle.get_traffic_light()
            if tl is not None:
                st = tl.state
                if st == carla.TrafficLightState.Green:
                    state_code = 1
                elif st == carla.TrafficLightState.Yellow:
                    state_code = 2
                elif st == carla.TrafficLightState.Red:
                    state_code = 3
    except Exception:
        state_code = 0

    # distance
    try:
        if hasattr(vehicle, 'get_traffic_light'):
            tl = vehicle.get_traffic_light()
            if tl is not None:
                loc_v = vehicle.get_transform().location
                loc_tl = tl.get_transform().location
                dx = loc_tl.x - loc_v.x
                dy = loc_tl.y - loc_v.y
                dist = math.sqrt(dx*dx + dy*dy)
    except Exception:
        dist = 999.0

    return state_code, dist
