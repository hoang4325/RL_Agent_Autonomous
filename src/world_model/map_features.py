from __future__ import annotations

import numpy as np

try:
    import carla
except Exception:  # pragma: no cover
    carla = None  # type: ignore


def sample_drivable_points(map_obj: 'carla.Map', ego_loc: 'carla.Location', radius_m: float = 35.0, step_m: float = 2.0) -> np.ndarray:
    """Sample driving waypoints around ego in world coordinates."""
    # CARLA doesn't provide direct drivable mask, so we sample waypoints near ego.
    # Approach: sample a grid and project to road waypoints.
    pts = []
    for dx in np.arange(-radius_m, radius_m + 1e-3, step_m):
        for dy in np.arange(-radius_m, radius_m + 1e-3, step_m):
            loc = carla.Location(x=ego_loc.x + float(dx), y=ego_loc.y + float(dy), z=ego_loc.z)
            wp = map_obj.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is None:
                continue
            wloc = wp.transform.location
            pts.append([wloc.x, wloc.y])
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.unique(np.array(pts, dtype=np.float32), axis=0)


def build_route_ahead(map_obj: 'carla.Map', ego_loc: 'carla.Location', length_m: float = 50.0, step_m: float = 2.0) -> np.ndarray:
    """Generate route points along current lane ahead in world coordinates."""
    wp0 = map_obj.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp0 is None:
        return np.zeros((0, 2), dtype=np.float32)

    pts = []
    dist = 0.0
    wp = wp0
    while dist < length_m:
        loc = wp.transform.location
        pts.append([loc.x, loc.y])
        nxt = wp.next(step_m)
        if not nxt:
            break
        wp = nxt[0]
        dist += step_m

    return np.array(pts, dtype=np.float32)


def world_to_ego_frame(ego_tf: 'carla.Transform', world_xy: np.ndarray) -> np.ndarray:
    """Convert world (x,y) points to ego frame (x forward, y right?)

    CARLA coordinate: x forward, y right. We'll use ego frame: x forward, y left?
    To keep consistent with LiDAR (often x forward, y right), we use CARLA convention: x forward, y right.
    """
    if len(world_xy) == 0:
        return world_xy
    loc = ego_tf.location
    yaw = np.deg2rad(ego_tf.rotation.yaw)
    c, s = np.cos(-yaw), np.sin(-yaw)
    dx = world_xy[:, 0] - loc.x
    dy = world_xy[:, 1] - loc.y
    x_ego = c * dx - s * dy
    y_ego = s * dx + c * dy
    return np.stack([x_ego, y_ego], axis=1).astype(np.float32)
