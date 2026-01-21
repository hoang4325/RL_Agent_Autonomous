from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import carla
except Exception:  # pragma: no cover
    carla = None  # type: ignore

from ..control.controllers import PurePursuitController
from ..fusion.bev_builder import BEVConfig, build_bev_from_tracks
from ..perception.bbox_classify import aabb_from_points, heuristic_classify
from ..perception.euclidean_cluster import euclidean_clustering_adaptive
from ..perception.lidar_io import ray_cast_to_numpy, semantic_lidar_to_numpy
from ..perception.preprocess import ground_removal_height, roi_filter, voxel_downsample
from ..perception.semantic_validate import validate_clusters_with_semantic
from ..sensors.sensor_manager import SensorManager
from ..tracking.kalman_tracker import KalmanMultiTracker
from ..utils.carla_utils import (
    destroy_actors,
    get_speed_mps,
    set_world_sync,
    try_get_traffic_light_state,
)
from ..utils.image_utils import (
    carla_bgra_to_bgr,
    carla_bgra_to_rgb,
    center_depth_min,
    decode_depth_meters,
    semantic_drivable_ratio,
)
from ..utils.yaml_io import load_yaml
from ..world_model.map_features import build_route_ahead, sample_drivable_points, world_to_ego_frame
from ..world_model.objects_and_gaps import compute_gap_features


@dataclass
class UrbanEnvConfig:
    carla_cfg: Dict[str, Any]
    sensors_cfg: Dict[str, Any]
    scenarios_cfg: Dict[str, Any]
    rl_cfg: Dict[str, Any]


class CarlaUrbanEnv(gym.Env):
    """Urban driving environment with multi-sensor fusion.

    High-level discrete actions; classical controller executes.

    Obs is Dict:
      - vec: float vector features (normalized to [-1, 1])
      - bev: (C,H,W) BEV tensor (optional), expected in [-1,1] (vel) and [0,1] masks
    """

    metadata = {"render_modes": []}

    ACTIONS = [
        "KEEP_LANE",
        "CHANGE_LEFT",
        "CHANGE_RIGHT",
        "STOP",
        "GO",
        "CREEP",
        "YIELD",
    ]

    def __init__(
        self,
        carla_config_path: str = "config/carla.yaml",
        sensors_config_path: str = "config/sensors.yaml",
        scenarios_config_path: str = "config/scenarios.yaml",
        rl_config_path: str = "config/rl.yaml",
        seed: int = 0,
    ):
        super().__init__()

        self.carla_cfg = load_yaml(carla_config_path)
        self.sensors_cfg = load_yaml(sensors_config_path)
        self.scenarios_cfg = load_yaml(scenarios_config_path)
        self.rl_cfg = load_yaml(rl_config_path)

        self.rng = random.Random(seed)
        np.random.seed(seed)

        # ---------------- RL / OBS settings ----------------
        self.use_bev: bool = bool(self.rl_cfg.get("obs", {}).get("use_bev", True))
        bev_cfg = self.rl_cfg.get("obs", {}).get("bev", {})

        size = int(bev_cfg.get("size", 96))
        mpc = float(bev_cfg.get("meters_per_cell", 0.5))

        # v_ref: from rl.yaml, else auto by target speed
        target_kmh = float(self.scenarios_cfg.get("scenario", {}).get("target_speed_kmh", 35))
        v_ref_default = max(5.0, target_kmh / 3.6)  # m/s
        v_ref_mps = float(bev_cfg.get("v_ref_mps", v_ref_default))

        point_radius = int(bev_cfg.get("point_radius", 1))

        # NOTE: must match your bev_builder.py
        self.bev_cfg = BEVConfig(
            size=size,
            meters_per_cell=mpc,
            v_ref_mps=v_ref_mps,
            point_radius=point_radius,
        )

        # vec: 14 base + 16 misc sensors
        self.vec_dim = 14 + 16

        self.observation_space = spaces.Dict(
            {"vec": spaces.Box(low=-1.0, high=1.0, shape=(self.vec_dim,), dtype=np.float32)}
        )

        # IMPORTANT: with your patched bev_builder.py, BEV stays within [-1,1]
        if self.use_bev:
            self.observation_space.spaces["bev"] = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(5, self.bev_cfg.size, self.bev_cfg.size),
                dtype=np.float32,
            )

        self.action_space = spaces.Discrete(len(self.ACTIONS))

        # ---------------- CARLA handles ----------------
        host = self.carla_cfg["carla"]["host"]
        port = int(self.carla_cfg["carla"]["port"])
        timeout_s = float(self.carla_cfg["carla"].get("timeout_s", 10.0))
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout_s)

        self.world: Optional["carla.World"] = None
        self.map: Optional["carla.Map"] = None
        self.tm: Optional["carla.TrafficManager"] = None

        self.ego: Optional["carla.Vehicle"] = None
        self.sensor_mgr: Optional[SensorManager] = None
        self.actors: List["carla.Actor"] = []
        self.npc_actors: List["carla.Actor"] = []

        self.collision_happened = False
        self.lane_invasion_happened = False

        # controller/tracker
        fixed_dt = float(self.carla_cfg["world"]["fixed_delta_seconds"])
        action_repeat = int(self.scenarios_cfg["scenario"].get("action_repeat", 2))
        self.dt = fixed_dt * action_repeat
        self.controller = PurePursuitController()
        self.tracker = KalmanMultiTracker(dt=self.dt)

        # episode params
        self.max_steps = int(self.scenarios_cfg["scenario"].get("max_steps", 900))
        self.episode_seconds = float(self.scenarios_cfg["scenario"].get("episode_seconds", 45))
        self.target_speed_kmh = float(self.scenarios_cfg["scenario"].get("target_speed_kmh", 35))
        self.action_repeat = action_repeat

        # episode state
        self._step_count = 0
        self._stuck_time = 0.0
        self._prev_loc: Optional["carla.Location"] = None
        self._prev_fwd: Optional[np.ndarray] = None

        self._lane_target_offset = 0  # -1 right, 0 same, +1 left
        self._action_cooldown = 0

        # GNSS origin (to avoid huge lat/lon magnitudes in obs)
        self._gnss_origin = None  # (lat, lon, alt)

        # keep last sensor features so obs doesn't jump to 999 when a frame is missed
        self._last_misc = np.zeros(16, dtype=np.float32)
        # cache last raw vec (before normalization) for shield/reward shaping
        self._last_vec_raw = np.zeros(30, dtype=np.float32)
        self._last_action = 0
        self._last_action_overridden = False
        self._last_misc[1] = 50.0  # depth_center_min_m
        self._last_misc[3] = 50.0  # radar_min_range_m

        # traffic light last state for robust reward
        self._last_tl_red = 0.0
        self._last_tl_dist = 999.0

        # curriculum bookkeeping
        self._episode_count = 0
        self._curr_stage: Optional[Dict[str, Any]] = None

        self._connect_world()

    # -------------------- world setup --------------------
    def _connect_world(self) -> None:
        town = self.carla_cfg["carla"]["town"]
        self.world = self.client.load_world(town)
        self.map = self.world.get_map()

        set_world_sync(
            self.world,
            synchronous_mode=bool(self.carla_cfg["world"]["synchronous_mode"]),
            fixed_delta_seconds=float(self.carla_cfg["world"]["fixed_delta_seconds"]),
            no_rendering=bool(self.carla_cfg["world"]["no_rendering_mode"]),
        )

        # Traffic manager
        tm_port = int(self.carla_cfg.get("traffic", {}).get("tm_port", 8000))
        self.tm = self.client.get_trafficmanager(tm_port)
        self.tm.set_synchronous_mode(True)
        if bool(self.carla_cfg.get("traffic", {}).get("hybrid_physics", True)):
            self.tm.set_hybrid_physics_mode(True)

        # warmup
        warm = int(self.carla_cfg.get("runtime", {}).get("warmup_ticks", 10))
        for _ in range(warm):
            self.world.tick()

    def _spawn_ego(self) -> None:
        assert self.world is not None
        bp_lib = self.world.get_blueprint_library()
        bp_name = self.carla_cfg["ego"].get("vehicle_blueprint", "vehicle.tesla.model3")
        bp = bp_lib.filter(bp_name)[0] if "*" in bp_name else bp_lib.find(bp_name)

        role_name = self.carla_cfg["ego"].get("role_name", "hero")
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", role_name)

        spawns = self.map.get_spawn_points() if self.map else self.world.get_map().get_spawn_points()
        spawn_random = bool(self.carla_cfg["ego"].get("spawn_random", True))
        spawn_tries = int(self.carla_cfg["ego"].get("spawn_tries", 80))

        ego = None
        for _ in range(spawn_tries):
            spawn = self.rng.choice(spawns) if spawn_random else spawns[0]
            ego = self.world.try_spawn_actor(bp, spawn)
            if ego is not None:
                break
            try:
                self.world.tick()
            except Exception:
                time.sleep(0.05)

        if ego is None:
            raise RuntimeError(f"Spawn failed: could not spawn ego after {spawn_tries} tries (all colliding).")

        self.ego = ego
        self.actors.append(self.ego)

    def _select_curriculum_stage(self) -> Optional[Dict[str, Any]]:
        cur = self.scenarios_cfg.get("curriculum", {})
        if not bool(cur.get("enabled", False)):
            return None
        stages = cur.get("stages", [])
        if not stages:
            return None

        eps_per_stage = int(cur.get("episodes_per_stage", 100))
        idx = min(len(stages) - 1, self._episode_count // max(1, eps_per_stage))
        return stages[idx]

    def _spawn_traffic(self) -> None:
        if not bool(self.carla_cfg.get("traffic", {}).get("enabled", True)):
            return
        assert self.world is not None and self.tm is not None

        t_cfg = self.carla_cfg.get("traffic", {})
        stage = self._curr_stage
        if stage is not None:
            num_veh = int(stage.get("num_vehicles", t_cfg.get("num_vehicles", t_cfg.get("vehicles", 25))))
            num_walk = int(stage.get("num_walkers", t_cfg.get("num_walkers", t_cfg.get("walkers", 15))))
        else:
            num_veh = int(t_cfg.get("num_vehicles", t_cfg.get("vehicles", 25)))
            num_walk = int(t_cfg.get("num_walkers", t_cfg.get("walkers", 15)))

        bp_lib = self.world.get_blueprint_library()
        veh_bps = bp_lib.filter("vehicle.*")
        spawn_points = self.world.get_map().get_spawn_points()
        self.rng.shuffle(spawn_points)

        # vehicles
        for i in range(min(num_veh, len(spawn_points))):
            try:
                bp = self.rng.choice(veh_bps)
                v = self.world.try_spawn_actor(bp, spawn_points[i])
                if v is None:
                    continue
                v.set_autopilot(True, self.tm.get_port())
                self.npc_actors.append(v)
            except Exception:
                continue

        # walkers
        try:
            walker_bps = bp_lib.filter("walker.pedestrian.*")
            controller_bp = bp_lib.find("controller.ai.walker")
            walker_spawn = []
            for _ in range(num_walk):
                loc = self.world.get_random_location_from_navigation()
                if loc is None:
                    continue
                walker_spawn.append(carla.Transform(loc))

            walkers = []
            for sp in walker_spawn:
                bp = self.rng.choice(walker_bps)
                w = self.world.try_spawn_actor(bp, sp)
                if w is None:
                    continue
                walkers.append(w)

            for w in walkers:
                c = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=w)
                c.start()
                c.go_to_location(self.world.get_random_location_from_navigation())
                c.set_max_speed(1.0 + self.rng.random() * 1.5)
                self.npc_actors.append(w)
                self.npc_actors.append(c)
        except Exception:
            pass

    def _spawn_sensors(self) -> None:
        assert self.world is not None and self.ego is not None
        self.sensor_mgr = SensorManager(self.world, self.ego, self.sensors_cfg)
        self.sensor_mgr.spawn_all()

    # -------------------- episode lifecycle --------------------
    def close(self):
        self._cleanup()
        super().close()

    def _cleanup(self):
        if self.sensor_mgr is not None:
            self.sensor_mgr.stop_destroy()
            self.sensor_mgr = None

        destroy_actors(self.npc_actors)
        self.npc_actors = []
        destroy_actors(self.actors)
        self.actors = []
        self.ego = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._cleanup()

        self.collision_happened = False
        self.lane_invasion_happened = False

        # curriculum: select stage based on current episode_count, then increment
        self._curr_stage = self._select_curriculum_stage()
        self._episode_count += 1

        # Curriculum can optionally randomize target speed per episode (stability -> diversity)
        if self._curr_stage is not None:
            mn = float(self._curr_stage.get("min_speed_kmh", self.target_speed_kmh))
            mx = float(self._curr_stage.get("max_speed_kmh", self.target_speed_kmh))
            if mx >= mn and (mx - mn) > 1e-6:
                self.target_speed_kmh = float(self.rng.uniform(mn, mx))
            else:
                self.target_speed_kmh = float(mn)

        self._spawn_ego()
        self._spawn_sensors()
        self._spawn_traffic()

        assert self.world is not None
        for _ in range(10):
            self.world.tick()

        # Initialize GNSS origin
        if self.sensor_mgr is not None:
            frame = self.world.get_snapshot().frame
            gnss_q = self.sensor_mgr.get("gnss")
            gnss = gnss_q.pop_for_frame(frame, timeout_s=0.05) if gnss_q else None
            if gnss is not None:
                self._gnss_origin = (float(gnss.latitude), float(gnss.longitude), float(gnss.altitude))

        self._step_count = 0
        self._stuck_time = 0.0
        self._lane_target_offset = 0
        self._action_cooldown = 0

        assert self.ego is not None
        self._prev_loc = self.ego.get_transform().location
        fwd = self.ego.get_transform().get_forward_vector()
        self._prev_fwd = np.array([fwd.x, fwd.y], dtype=np.float32)

        self._last_tl_red = 0.0
        self._last_tl_dist = 999.0

        obs = self._get_observation()
        info: Dict[str, Any] = {}
        return obs, info

    # -------------------- sensor misc features (robust) --------------------
    def _misc_sensor_features(self, frame: int) -> np.ndarray:
        """Returns length 16:
          rgb_brightness,
          depth_center_min_m,
          semseg_drivable_ratio,
          radar_min_range_m, radar_min_rel_vel, radar_count,
          imu_ax, imu_ay, imu_az, imu_gx, imu_gy, imu_gz, imu_compass,
          gnss_dlat, gnss_dlon, gnss_dalt
        """
        assert self.sensor_mgr is not None
        last = self._last_misc

        def pop(name: str, timeout_s: float = 0.05):
            q = self.sensor_mgr.get(name)
            return q.pop_for_frame(frame, timeout_s=timeout_s) if q else None

        # RGB brightness
        rgb_brightness = float(last[0])
        rgb = pop("rgb_front", 0.05)
        if rgb is not None:
            img = carla_bgra_to_rgb(rgb)
            rgb_brightness = float(img.mean() / 255.0)
        rgb_brightness = float(np.clip(rgb_brightness, 0.0, 1.0))

        # Depth min center
        depth_min = float(last[1])
        depth = pop("depth_front", 0.05)
        if depth is not None:
            depth_m = decode_depth_meters(depth)
            depth_min = float(center_depth_min(depth_m, box=9))
        depth_min = float(np.clip(depth_min, 0.0, 50.0))

        # Semantic drivable ratio
        sem_ratio = float(last[2])
        sem = pop("semseg_front", 0.05)
        if sem is not None:
            sem_bgr = carla_bgra_to_bgr(sem)
            sem_ratio = float(semantic_drivable_ratio(sem_bgr))
        sem_ratio = float(np.clip(sem_ratio, 0.0, 1.0))

        # Radar
        radar_min_r = float(last[3])
        radar_min_v = float(last[4])
        radar_count = float(last[5])
        radar = pop("radar_front", 0.05)
        if radar is not None:
            try:
                det = list(radar)
                radar_count = float(len(det))
                if det:
                    depths = np.array([float(d.depth) for d in det], dtype=np.float32)
                    i = int(np.argmin(depths))
                    radar_min_r = float(depths[i])
                    radar_min_v = float(det[i].velocity)
            except Exception:
                pass
        radar_min_r = float(np.clip(radar_min_r, 0.0, 50.0))
        radar_min_v = float(np.clip(radar_min_v, -20.0, 20.0))
        radar_count = float(np.clip(radar_count, 0.0, 50.0))

        # IMU
        imu_ax, imu_ay, imu_az = float(last[6]), float(last[7]), float(last[8])
        imu_gx, imu_gy, imu_gz = float(last[9]), float(last[10]), float(last[11])
        imu_compass = float(last[12])
        imu = pop("imu", 0.05)
        if imu is not None:
            try:
                imu_ax, imu_ay, imu_az = float(imu.accelerometer.x), float(imu.accelerometer.y), float(imu.accelerometer.z)
                imu_gx, imu_gy, imu_gz = float(imu.gyroscope.x), float(imu.gyroscope.y), float(imu.gyroscope.z)
                imu_compass = float(getattr(imu, "compass", imu_compass))
            except Exception:
                pass

        imu_ax = float(np.clip(imu_ax, -10.0, 10.0))
        imu_ay = float(np.clip(imu_ay, -10.0, 10.0))
        imu_az = float(np.clip(imu_az, -10.0, 10.0))
        imu_gx = float(np.clip(imu_gx, -5.0, 5.0))
        imu_gy = float(np.clip(imu_gy, -5.0, 5.0))
        imu_gz = float(np.clip(imu_gz, -5.0, 5.0))
        imu_compass = float(((imu_compass + math.pi) % (2 * math.pi)) - math.pi)  # wrap [-pi, pi]

        # GNSS deltas
        dlat, dlon, dalt = float(last[13]), float(last[14]), float(last[15])
        gnss = pop("gnss", 0.05)
        if gnss is not None:
            try:
                lat, lon, alt = float(gnss.latitude), float(gnss.longitude), float(gnss.altitude)
                if self._gnss_origin is None:
                    self._gnss_origin = (lat, lon, alt)
                olat, olon, oalt = self._gnss_origin
                dlat, dlon, dalt = lat - olat, lon - olon, alt - oalt
            except Exception:
                pass
        dlat = float(np.clip(dlat, -1e-3, 1e-3))
        dlon = float(np.clip(dlon, -1e-3, 1e-3))
        dalt = float(np.clip(dalt, -50.0, 50.0))

        out = np.array(
            [
                rgb_brightness,
                depth_min,
                sem_ratio,
                radar_min_r,
                radar_min_v,
                radar_count,
                imu_ax,
                imu_ay,
                imu_az,
                imu_gx,
                imu_gy,
                imu_gz,
                imu_compass,
                dlat,
                dlon,
                dalt,
            ],
            dtype=np.float32,
        )
        self._last_misc = out
        return out

    # -------------------- observation normalization --------------------
    def _normalize_vec(self, vec: np.ndarray) -> np.ndarray:
        """Map raw vec to roughly [-1, 1] to stabilize PPO critic."""
        v = vec.astype(np.float32).copy()

        # base_vec indices
        v[0] = np.clip(v[0] / 20.0, 0.0, 1.5)          # speed m/s
        v[1] = np.clip(v[1] / 3.0, 0.0, 3.0)           # lane_dist m
        v[2] = np.clip(v[2] / math.pi, -1.0, 1.0)      # heading_err rad/pi

        # 3-5 tl onehot stay 0/1

        v[6] = np.clip(v[6] / 50.0, 0.0, 1.0)          # tl_dist
        v[7] = np.clip(v[7] / 50.0, 0.0, 1.0)          # front_gap
        v[8] = np.clip(v[8] / 20.0, -1.0, 1.0)         # front_rel_speed

        for i in (9, 10, 11, 12):                      # gaps
            v[i] = np.clip(v[i] / 50.0, 0.0, 1.0)

        v[13] = np.clip(v[13] / 10.0, 0.0, 1.0)        # min_ttc

        off = 14
        v[off + 0] = np.clip(v[off + 0], 0.0, 1.0)              # brightness
        v[off + 1] = np.clip(v[off + 1] / 50.0, 0.0, 1.0)       # depth
        v[off + 2] = np.clip(v[off + 2], 0.0, 1.0)              # sem_ratio
        v[off + 3] = np.clip(v[off + 3] / 50.0, 0.0, 1.0)       # radar_min_r
        v[off + 4] = np.clip(v[off + 4] / 20.0, -1.0, 1.0)      # radar_min_v
        v[off + 5] = np.clip(v[off + 5] / 50.0, 0.0, 1.0)       # radar_count

        for j in range(off + 6, off + 9):                       # imu accel
            v[j] = np.clip(v[j] / 10.0, -1.0, 1.0)
        for j in range(off + 9, off + 12):                      # imu gyro
            v[j] = np.clip(v[j] / 5.0, -1.0, 1.0)

        v[off + 12] = np.clip(v[off + 12] / math.pi, -1.0, 1.0) # compass
        v[off + 13] = np.clip(v[off + 13] / 1e-3, -1.0, 1.0)    # dlat
        v[off + 14] = np.clip(v[off + 14] / 1e-3, -1.0, 1.0)    # dlon
        v[off + 15] = np.clip(v[off + 15] / 50.0, -1.0, 1.0)    # dalt

        v = np.clip(v, -1.0, 1.0)
        return v.astype(np.float32)

    # -------------------- lane metrics --------------------
    def _lane_metrics(self) -> Tuple[float, float]:
        """Return (lane_dist_m, heading_err_rad)."""
        assert self.map is not None and self.ego is not None
        tf = self.ego.get_transform()
        wp = self.map.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            return 10.0, math.pi

        loc = tf.location
        c = wp.transform.location
        dx = loc.x - c.x
        dy = loc.y - c.y
        lyaw = math.radians(wp.transform.rotation.yaw)
        fx, fy = math.cos(lyaw), math.sin(lyaw)
        proj = dx * fx + dy * fy
        perp_x = dx - proj * fx
        perp_y = dy - proj * fy
        lane_dist = math.sqrt(perp_x**2 + perp_y**2)

        heading_err = math.radians(tf.rotation.yaw) - math.radians(wp.transform.rotation.yaw)
        while heading_err > math.pi:
            heading_err -= 2 * math.pi
        while heading_err < -math.pi:
            heading_err += 2 * math.pi
        return float(lane_dist), float(heading_err)

    def _rasterize_points_to_mask(self, pts_ego_xy: np.ndarray) -> np.ndarray:
        """Rasterize ego-frame points (x,y) into BEV drivable mask."""
        H = W = self.bev_cfg.size
        mask = np.zeros((H, W), dtype=np.float32)
        half = self.bev_cfg.size // 2
        if len(pts_ego_xy) == 0:
            return mask
        ix = np.round(pts_ego_xy[:, 0] / self.bev_cfg.meters_per_cell).astype(np.int32) + half
        iy = np.round(pts_ego_xy[:, 1] / self.bev_cfg.meters_per_cell).astype(np.int32) + half
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        mask[iy[ok], ix[ok]] = 1.0

        # simple dilation
        for _ in range(1):
            mask = np.maximum(mask, np.roll(mask, 1, axis=0))
            mask = np.maximum(mask, np.roll(mask, -1, axis=0))
            mask = np.maximum(mask, np.roll(mask, 1, axis=1))
            mask = np.maximum(mask, np.roll(mask, -1, axis=1))
        return mask

    def _get_tracks_from_sensors(self) -> Tuple[Dict[int, Any], Dict[str, Any]]:
        """Run LiDAR clustering + tracking. Returns (tracks, debug dict)."""
        assert self.sensor_mgr is not None and self.world is not None
        frame = self.world.get_snapshot().frame
        dbg: Dict[str, Any] = {}

        lidar_q = self.sensor_mgr.get("lidar")
        sem_q = self.sensor_mgr.get("lidar_semantic")
        lidar = lidar_q.pop_for_frame(frame, timeout_s=0.2) if lidar_q else None
        sem = sem_q.pop_for_frame(frame, timeout_s=0.2) if sem_q else None

        if lidar is None:
            return self.tracker.tracks, dbg

        pts, _inten = ray_cast_to_numpy(lidar)
        pts = roi_filter(pts)
        pts = voxel_downsample(pts, voxel_size=0.15)
        pts = ground_removal_height(pts, z_threshold=-1.4)

        clusters = euclidean_clustering_adaptive(pts, min_points=10)

        measurements = []
        cluster_reports = []

        sem_pts = sem_tags = sem_ids = None
        if sem is not None:
            sem_pts, sem_tags, sem_ids = semantic_lidar_to_numpy(sem)

        if sem is not None and sem_pts is not None and len(sem_pts) > 0 and len(clusters) > 0:
            cluster_reports = validate_clusters_with_semantic(clusters, pts, sem_pts, sem_tags, sem_ids)
            dbg["semantic_purity_mean"] = float(np.mean([r.purity for r in cluster_reports])) if cluster_reports else 0.0
            dbg["semantic_underseg_mean_instances"] = float(
                np.mean([r.instances_in_cluster for r in cluster_reports])
            ) if cluster_reports else 0.0

        for idx in clusters:
            cpts = pts[idx]
            center, size, _mn, _mx = aabb_from_points(cpts)
            label = heuristic_classify(size, n_points=len(cpts))
            measurements.append((center[:2].astype(np.float32), label, size.astype(np.float32)))

        tracks = self.tracker.step(measurements)
        dbg["num_clusters"] = int(len(clusters))
        dbg["num_tracks"] = int(len(tracks))
        return tracks, dbg

    def _tl_onehot(self, tl_state: Any) -> Tuple[np.ndarray, float]:
        """Return (onehot[G,Y,R], red_flag_float)."""
        onehot = np.zeros(3, dtype=np.float32)
        red = 0.0

        # CARLA enum
        try:
            if hasattr(carla, "TrafficLightState") and isinstance(tl_state, carla.TrafficLightState):
                if tl_state == carla.TrafficLightState.Green:
                    onehot[0] = 1.0
                elif tl_state == carla.TrafficLightState.Yellow:
                    onehot[1] = 1.0
                elif tl_state == carla.TrafficLightState.Red:
                    onehot[2] = 1.0
                    red = 1.0
                return onehot, red
        except Exception:
            pass

        # int convention: 1=G,2=Y,3=R
        if isinstance(tl_state, (int, np.integer)):
            if int(tl_state) == 1:
                onehot[0] = 1.0
            elif int(tl_state) == 2:
                onehot[1] = 1.0
            elif int(tl_state) == 3:
                onehot[2] = 1.0
                red = 1.0
            return onehot, red

        # string
        if isinstance(tl_state, str):
            s = tl_state.lower()
            if "green" in s:
                onehot[0] = 1.0
            elif "yellow" in s:
                onehot[1] = 1.0
            elif "red" in s:
                onehot[2] = 1.0
                red = 1.0

        return onehot, red

    # -------------------- observation --------------------
    def _get_observation(self) -> Dict[str, np.ndarray]:
        assert self.world is not None and self.map is not None and self.ego is not None

        speed = get_speed_mps(self.ego)
        lane_dist, heading_err = self._lane_metrics()

        tl_state, tl_dist = try_get_traffic_light_state(self.ego)
        tl_onehot, tl_red = self._tl_onehot(tl_state)

        self._last_tl_red = float(tl_red)
        self._last_tl_dist = float(tl_dist)

        tracks, dbg = self._get_tracks_from_sensors()
        gaps = compute_gap_features(self.map, self.ego, tracks)

        base_vec = np.array(
            [
                speed,
                lane_dist,
                heading_err,
                tl_onehot[0],
                tl_onehot[1],
                tl_onehot[2],  # red at index 5 by design
                float(tl_dist),
                gaps.front_gap,
                gaps.front_rel_speed,
                gaps.left_front_gap,
                gaps.left_rear_gap,
                gaps.right_front_gap,
                gaps.right_rear_gap,
                gaps.min_ttc,
            ],
            dtype=np.float32,
        )

        frame = self.world.get_snapshot().frame
        misc = self._misc_sensor_features(frame)

        vec_raw = np.concatenate([base_vec, misc], axis=0).astype(np.float32)
        self._last_vec_raw = vec_raw.copy()
        vec = self._normalize_vec(vec_raw)

        obs: Dict[str, np.ndarray] = {"vec": vec}

        if self.use_bev:
            ego_tf = self.ego.get_transform()
            drivable_world = sample_drivable_points(self.map, ego_tf.location, radius_m=30.0, step_m=3.0)
            route_world = build_route_ahead(self.map, ego_tf.location, length_m=50.0, step_m=2.0)
            drivable_ego = world_to_ego_frame(ego_tf, drivable_world)
            route_ego = world_to_ego_frame(ego_tf, route_world)
            drivable_mask = self._rasterize_points_to_mask(drivable_ego)

            # IMPORTANT: your bev_builder.py already normalizes velocities to [-1,1]
            bev = build_bev_from_tracks(tracks, drivable_mask, route_ego, self.bev_cfg)
            obs["bev"] = bev

        self._last_dbg = dbg
        return obs

    # -------------------- action -> target --------------------
    def _apply_high_level_action(self, action: int) -> Tuple[float, int]:
        """Returns (target_speed_mps, lane_offset_delta)."""
        target_speed = self.target_speed_kmh / 3.6

        if action == 0:  # keep
            return target_speed, 0
        if action == 1:  # change left
            return target_speed, +1
        if action == 2:  # change right
            return target_speed, -1
        if action == 3:  # stop
            return 0.0, 0
        if action == 4:  # go
            return max(target_speed, 12.0 / 3.6), 0
        if action == 5:  # creep
            return 5.0 / 3.6, 0
        if action == 6:  # yield
            return 0.0, 0
        return target_speed, 0


    def _shield_action(self, proposed_action: int) -> int:
        """Lightweight safety shield for training stability (option C).

        It may override obviously unsafe / illegal actions, but we still penalize the
        *decision* so the policy learns to pick safe actions itself.
        """
        s_cfg = self.scenarios_cfg.get("shield", {}) or {}
        if not bool(s_cfg.get("enabled", False)):
            self._last_action_overridden = False
            return proposed_action

        # Use cached raw vec from previous observation (seconds/meters, not normalized)
        v = self._last_vec_raw
        tl_red = float(v[5])
        tl_dist = float(v[6])
        front_gap = float(v[7])
        left_front = float(v[9])
        left_rear = float(v[10])
        right_front = float(v[11])
        right_rear = float(v[12])
        min_ttc = float(v[13])

        red_light_dist = float(s_cfg.get("red_light_dist_m", 20.0))
        ttc_stop = float(s_cfg.get("ttc_stop_s", 1.5))
        front_gap_stop = float(s_cfg.get("front_gap_stop_m", 6.0))

        lc_front_min = float(s_cfg.get("lane_change_min_front_gap_m", 12.0))
        lc_rear_min = float(s_cfg.get("lane_change_min_rear_gap_m", 8.0))
        lc_ttc_min = float(s_cfg.get("lane_change_min_ttc_s", 3.0))

        override_to = str(s_cfg.get("override_to", "STOP")).upper()
        override_action = 3 if override_to == "STOP" else 6  # STOP=3, YIELD=6

        overridden = False
        a = int(proposed_action)

        # Rule 1: red light close -> must STOP/YIELD/CREEP
        if tl_red > 0.5 and tl_dist < red_light_dist:
            if a not in (3, 5, 6):
                a = override_action
                overridden = True

        # Rule 2: imminent collision risk -> STOP
        if (min_ttc > 0.0 and min_ttc < ttc_stop) or (front_gap > 0.0 and front_gap < front_gap_stop):
            if a not in (3, 5, 6):
                a = 3
                overridden = True

        # Rule 3: lane change only if adjacent gaps are safe
        if a == 1:  # change left
            if (left_front < lc_front_min) or (left_rear < lc_rear_min) or (min_ttc < lc_ttc_min):
                a = 0
                overridden = True
        if a == 2:  # change right
            if (right_front < lc_front_min) or (right_rear < lc_rear_min) or (min_ttc < lc_ttc_min):
                a = 0
                overridden = True

        self._last_action_overridden = overridden
        return a


    def _choose_target_waypoint(self) -> "carla.Waypoint":
        assert self.map is not None and self.ego is not None
        tf = self.ego.get_transform()
        wp = self.map.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            return self.map.get_waypoint(tf.location, project_to_road=True)

        target_wp = wp
        if self._lane_target_offset > 0:
            lw = wp.get_left_lane()
            if lw is not None and lw.lane_type == carla.LaneType.Driving:
                target_wp = lw
        elif self._lane_target_offset < 0:
            rw = wp.get_right_lane()
            if rw is not None and rw.lane_type == carla.LaneType.Driving:
                target_wp = rw

        speed = get_speed_mps(self.ego)
        look = max(6.0, 0.6 * speed)
        nxt = target_wp.next(look)
        if nxt:
            return nxt[0]
        return target_wp

    # -------------------- step --------------------
    def step(self, action: int):
        assert self.world is not None and self.ego is not None
        self._step_count += 1

        # cooldown to prevent oscillating lane changes
        if self._action_cooldown > 0:
            self._action_cooldown -= 1
            if action in (1, 2):
                action = 0

        proposed_action = int(action)
        self._last_action = proposed_action
        action = self._shield_action(proposed_action)
        target_speed, lane_delta = self._apply_high_level_action(int(action))
        if lane_delta != 0:
            self._lane_target_offset = int(np.clip(self._lane_target_offset + lane_delta, -1, 1))
            self._action_cooldown = 10

        collision_penalty = 0.0
        lane_penalty = 0.0

        for _ in range(self.action_repeat):
            wp = self._choose_target_waypoint()
            speed = get_speed_mps(self.ego)
            steer = self.controller.compute_steer(wp.transform.location, self.ego.get_transform(), speed)
            throttle, brake = self.controller.compute_throttle_brake(target_speed, speed, self.dt / self.action_repeat)

            control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))
            self.ego.apply_control(control)

            self.world.tick()

            self._update_event_sensors()
            if self.collision_happened:
                collision_penalty = float(self.scenarios_cfg["reward"]["collision_penalty"])
                break
            if self.lane_invasion_happened:
                lane_penalty = float(self.scenarios_cfg["reward"]["lane_invasion_penalty"])

        obs = self._get_observation()

        # reward
        reward = self._compute_reward(obs)
        reward -= collision_penalty
        reward -= lane_penalty

        # optional reward stabilization
        r_cfg = self.scenarios_cfg.get("reward", {})
        reward_scale = float(r_cfg.get("reward_scale", 1.0))
        reward_clip_abs = float(r_cfg.get("reward_clip_abs", 10.0))
        reward = float(np.clip(reward * reward_scale, -reward_clip_abs, reward_clip_abs))

        terminated = False
        truncated = False

        # done conditions use raw lane deviation
        max_dev = float(self.scenarios_cfg["done"]["max_lane_dev_m"])
        lane_dist_m, _ = self._lane_metrics()
        if lane_dist_m > max_dev:
            terminated = True

        if self.collision_happened:
            terminated = True

        # stuck
        speed_mps = get_speed_mps(self.ego)
        if speed_mps < 0.5:
            self._stuck_time += self.dt
        else:
            self._stuck_time = 0.0
        if self._stuck_time > float(self.scenarios_cfg["done"]["stuck_seconds"]):
            truncated = True

        if self._step_count >= self.max_steps:
            truncated = True

        info: Dict[str, Any] = {
            "action_name": self.ACTIONS[int(action)],
            "proposed_action_name": self.ACTIONS[int(getattr(self, "_last_action", int(action)))],
            "action_overridden": bool(getattr(self, "_last_action_overridden", False)),
            "collision": self.collision_happened,
            "lane_invasion": self.lane_invasion_happened,
        }
        if hasattr(self, "_last_dbg"):
            info.update(self._last_dbg)

        return obs, float(reward), terminated, truncated, info

    def _update_event_sensors(self):
        assert self.sensor_mgr is not None
        if self.world is None:
            return
        frame = self.world.get_snapshot().frame

        col_q = self.sensor_mgr.get("collision")
        if col_q:
            data = col_q.pop_for_frame(frame, timeout_s=0.0)
            if data is not None:
                self.collision_happened = True

        li_q = self.sensor_mgr.get("lane_invasion")
        if li_q:
            data = li_q.pop_for_frame(frame, timeout_s=0.0)
            if data is not None:
                self.lane_invasion_happened = True

    def _compute_reward(self, obs: Dict[str, np.ndarray]) -> float:
        """Reward uses raw world values to reduce dependence on normalized obs."""
        assert self.ego is not None
        r_cfg = self.scenarios_cfg["reward"]

        # compute raw metrics
        speed = get_speed_mps(self.ego)
        lane_dist, heading_err = self._lane_metrics()

        # progress: forward displacement
        tf = self.ego.get_transform()
        if self._prev_loc is None:
            self._prev_loc = tf.location

        dx = tf.location.x - self._prev_loc.x
        dy = tf.location.y - self._prev_loc.y
        fwd_now = tf.get_forward_vector()
        fwd_vec = np.array([fwd_now.x, fwd_now.y], dtype=np.float32)
        progress = float(dx * fwd_vec[0] + dy * fwd_vec[1])
        self._prev_loc = tf.location
        self._prev_fwd = fwd_vec

        # TTC penalty: obs vec[13] is normalized [0,1] -> undo to seconds approx
        min_ttc = float(np.clip(obs["vec"][13], 0.0, 1.0)) * 10.0
        ttc_pen = 0.0
        if min_ttc < 2.0:
            ttc_pen = (2.0 - min_ttc)

        # speed target
        target_speed = self.target_speed_kmh / 3.6
        speed_err = abs(speed - target_speed)

        reward = 0.0
        reward += float(r_cfg["progress_w"]) * progress
        reward -= float(r_cfg["lane_w"]) * lane_dist
        reward -= float(r_cfg["heading_w"]) * abs(heading_err)
        reward -= float(r_cfg["speed_w"]) * speed_err
        reward -= float(r_cfg["ttc_w"]) * ttc_pen

        # idle penalty (per step)
        reward -= float(r_cfg.get("idle_penalty", 0.0))

        # red light penalty: robust using cached tl state
        tl_red = float(self._last_tl_red)
        tl_dist = float(self._last_tl_dist)
        if tl_red > 0.5 and tl_dist < 15.0 and speed > 3.0:
            reward -= float(r_cfg.get("red_light_penalty", 0.0))

        # -------- decision penalties (teach policy, even if shield overrides) --------
        # Note: self._last_action is the *proposed* action before shield.
        a = int(getattr(self, "_last_action", 0))
        if bool(getattr(self, "_last_action_overridden", False)):
            reward -= float(r_cfg.get("unsafe_action_penalty", 0.0))

        # Red light: penalize choosing GO/KEEP/LC when red is close
        s_cfg = self.scenarios_cfg.get("shield", {}) or {}
        red_light_dist = float(s_cfg.get("red_light_dist_m", 15.0))
        if tl_red > 0.5 and tl_dist < red_light_dist:
            if a not in (3, 5, 6):  # STOP/CREEP/YIELD
                reward -= float(r_cfg.get("red_light_action_penalty", 0.0))

        # TTC: penalize not choosing defensive action when TTC is low
        min_ttc_raw = float(getattr(self, "_last_vec_raw", np.zeros(30, dtype=np.float32))[13])
        if min_ttc_raw > 0.0 and min_ttc_raw < 2.0:
            if a not in (3, 5, 6):
                reward -= float(r_cfg.get("ttc_action_penalty", 0.0))

        # small cost for lane change to avoid oscillation
        if a in (1, 2):
            reward -= float(r_cfg.get("lane_change_penalty", 0.0))

        return float(reward)
