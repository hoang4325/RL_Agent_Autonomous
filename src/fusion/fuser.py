from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from src.fusion.bev_builder import BEVConfig, build_bev_from_tracks


@dataclass
class FusionConfig:
    bev_size: int = 64
    meters_per_cell: float = 0.5


class SensorFuser:
    """Fuse multi-sensor outputs into a compact world-model for RL."""

    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg
        self._bev_cfg = BEVConfig(size=cfg.bev_size, meters_per_cell=cfg.meters_per_cell)

    def build_observation(
        self,
        tracks: list[Any],
        vec_features: np.ndarray,
        drivable_ratio: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        bev = build_bev_from_tracks(tracks, self._bev_cfg, drivable_ratio=drivable_ratio)
        vec = vec_features.astype(np.float32)
        return {"bev": bev, "vec": vec}
