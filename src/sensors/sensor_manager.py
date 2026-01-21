from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..utils.sync_queue import SensorQueue
from ..utils.carla_utils import to_carla_transform

try:
    import carla
except Exception:  # pragma: no cover
    carla = None  # type: ignore


@dataclass
class SensorHandle:
    name: str
    actor: 'carla.Actor'
    queue: SensorQueue


class SensorManager:
    def __init__(self, world: 'carla.World', ego_vehicle: 'carla.Vehicle', sensors_cfg: Dict[str, Any]):
        self.world = world
        self.ego = ego_vehicle
        self.sensors_cfg = sensors_cfg
        self.bp_lib = world.get_blueprint_library()
        self.handles: List[SensorHandle] = []
        self.actors: List['carla.Actor'] = []

    def spawn_all(self) -> None:
        for name, cfg in self.sensors_cfg.get('sensors', {}).items():
            self._spawn_sensor(name, cfg)

    def _spawn_sensor(self, name: str, cfg: Dict[str, Any]) -> None:
        s_type = cfg['type']
        bp = self.bp_lib.find(s_type)
        attrs = cfg.get('attributes', {}) or {}
        for k, v in attrs.items():
            try:
                bp.set_attribute(str(k), str(v))
            except Exception:
                pass
        if 'sensor_tick' in cfg and cfg['sensor_tick'] is not None:
            try:
                bp.set_attribute('sensor_tick', str(cfg['sensor_tick']))
            except Exception:
                pass

        tf = to_carla_transform(cfg.get('transform', {'x': 0, 'y': 0, 'z': 0}))
        actor = self.world.spawn_actor(bp, tf, attach_to=self.ego)
        q = SensorQueue(maxsize=256)

        def _cb(data, _q=q):
            try:
                frame = int(getattr(data, 'frame', -1))
            except Exception:
                frame = -1
            _q.push(frame, data)

        actor.listen(_cb)

        self.actors.append(actor)
        self.handles.append(SensorHandle(name=name, actor=actor, queue=q))

    def get(self, name: str) -> Optional[SensorQueue]:
        for h in self.handles:
            if h.name == name:
                return h.queue
        return None

    def stop_destroy(self) -> None:
        for h in self.handles:
            try:
                h.actor.stop()
            except Exception:
                pass
        for h in self.handles:
            try:
                h.actor.destroy()
            except Exception:
                pass
        self.handles = []
        self.actors = []
