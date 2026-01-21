from __future__ import annotations

import numpy as np

try:
    import carla
except Exception:  # pragma: no cover
    carla = None  # type: ignore


def ray_cast_to_numpy(lidar_data: 'carla.LidarMeasurement'):
    """Return (points_xyz Nx3, intensity N)."""
    arr = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
    arr = arr.reshape((-1, 4))
    pts = arr[:, :3].astype(np.float32, copy=False)
    inten = arr[:, 3].astype(np.float32, copy=False)
    return pts, inten


def semantic_lidar_to_numpy(sem_data: 'carla.SemanticLidarMeasurement'):
    """Return (points_xyz Nx3, obj_tag N, obj_idx N).

    CARLA semantic LiDAR point layout: x,y,z,cos_angle,ObjIdx,ObjTag.
    raw buffer is float32 + uint32? In PythonAPI it is packed as floats.

    We parse using structured dtype that matches CARLA docs.
    """
    dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('cos_angle', np.float32),
        ('obj_idx', np.uint32),
        ('obj_tag', np.uint32),
    ])
    arr = np.frombuffer(sem_data.raw_data, dtype=dtype)
    pts = np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float32, copy=False)
    obj_tag = arr['obj_tag'].astype(np.int32, copy=False)
    obj_idx = arr['obj_idx'].astype(np.int32, copy=False)
    return pts, obj_tag, obj_idx
