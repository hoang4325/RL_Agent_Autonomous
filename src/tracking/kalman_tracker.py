from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Track:
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    P: np.ndarray
    age: int
    misses: int
    label: str
    size_xyz: np.ndarray


class KalmanMultiTracker:
    """Minimal CV Kalman multi-object tracker.

    State: [x, y, vx, vy]
    Measurement: [x, y]
    """

    def __init__(self, dt: float, process_var: float = 1.0, meas_var: float = 0.25, max_misses: int = 5, assoc_dist: float = 4.0):
        self.dt = dt
        self.process_var = process_var
        self.meas_var = meas_var
        self.max_misses = max_misses
        self.assoc_dist = assoc_dist
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        q = process_var
        self.Q = q * np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2],
        ], dtype=np.float32)
        self.R = meas_var * np.eye(2, dtype=np.float32)

    def _init_track(self, meas_xy: np.ndarray, label: str, size_xyz: np.ndarray) -> None:
        x, y = float(meas_xy[0]), float(meas_xy[1])
        state = np.array([x, y, 0.0, 0.0], dtype=np.float32)
        P = np.eye(4, dtype=np.float32) * 10.0
        tid = self._next_id
        self._next_id += 1
        self.tracks[tid] = Track(
            track_id=tid, x=state[0], y=state[1], vx=state[2], vy=state[3],
            P=P, age=1, misses=0, label=label, size_xyz=size_xyz
        )

    def predict(self) -> None:
        for tid, t in list(self.tracks.items()):
            x = np.array([t.x, t.y, t.vx, t.vy], dtype=np.float32)
            x = self.F @ x
            P = self.F @ t.P @ self.F.T + self.Q
            t.x, t.y, t.vx, t.vy = float(x[0]), float(x[1]), float(x[2]), float(x[3])
            t.P = P
            t.age += 1

    def update(self, measurements: List[Tuple[np.ndarray, str, np.ndarray]]) -> None:
        """measurements: list of (xy, label, size_xyz) in ego frame."""
        # Predict step already done outside.
        if len(measurements) == 0:
            # mark misses
            for tid, t in list(self.tracks.items()):
                t.misses += 1
                if t.misses > self.max_misses:
                    del self.tracks[tid]
            return

        track_ids = list(self.tracks.keys())
        track_xy = np.array([[self.tracks[tid].x, self.tracks[tid].y] for tid in track_ids], dtype=np.float32)
        meas_xy = np.array([m[0] for m in measurements], dtype=np.float32)

        if len(track_ids) == 0:
            for mxy, lbl, sz in measurements:
                self._init_track(mxy, lbl, sz)
            return

        # Nearest-neighbor association
        dist = np.linalg.norm(track_xy[:, None, :] - meas_xy[None, :, :], axis=2)  # (T,M)
        assigned_tracks = set()
        assigned_meas = set()

        while True:
            t_idx, m_idx = np.unravel_index(np.argmin(dist), dist.shape)
            d = dist[t_idx, m_idx]
            if not np.isfinite(d) or d > self.assoc_dist:
                break
            tid = track_ids[t_idx]
            if tid in assigned_tracks or m_idx in assigned_meas:
                dist[t_idx, m_idx] = np.inf
                continue
            assigned_tracks.add(tid)
            assigned_meas.add(m_idx)

            # Kalman update for tid with measurement m_idx
            t = self.tracks[tid]
            z = meas_xy[m_idx].astype(np.float32)
            x = np.array([t.x, t.y, t.vx, t.vy], dtype=np.float32)
            P = t.P
            y = z - (self.H @ x)
            S = self.H @ P @ self.H.T + self.R
            K = P @ self.H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(4, dtype=np.float32) - K @ self.H) @ P

            t.x, t.y, t.vx, t.vy = float(x[0]), float(x[1]), float(x[2]), float(x[3])
            t.P = P
            t.misses = 0
            t.label = measurements[m_idx][1]
            t.size_xyz = measurements[m_idx][2]

            dist[t_idx, :] = np.inf
            dist[:, m_idx] = np.inf

        # Unassigned tracks -> miss
        for tid in list(self.tracks.keys()):
            if tid not in assigned_tracks:
                self.tracks[tid].misses += 1
                if self.tracks[tid].misses > self.max_misses:
                    del self.tracks[tid]

        # Unassigned measurements -> new tracks
        for i, (mxy, lbl, sz) in enumerate(measurements):
            if i not in assigned_meas:
                self._init_track(mxy, lbl, sz)

    def step(self, measurements: List[Tuple[np.ndarray, str, np.ndarray]]) -> Dict[int, Track]:
        self.predict()
        self.update(measurements)
        return self.tracks
