from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def adaptive_tol(r: float, tol_min: float = 0.35, tol_max: float = 0.9) -> float:
    return float(np.clip(0.35 + 0.01 * r, tol_min, tol_max))


def euclidean_clustering_adaptive(points_xyz: np.ndarray,
                                 min_points: int = 10,
                                 max_points: int = 5000) -> list[np.ndarray]:
    """Adaptive Euclidean clustering using cKDTree over XY.

    Returns a list of index arrays (one per cluster).
    """
    if points_xyz is None or len(points_xyz) == 0:
        return []

    xy = points_xyz[:, :2]
    tree = cKDTree(xy)
    r = np.linalg.norm(xy, axis=1)
    N = xy.shape[0]
    visited = np.zeros(N, dtype=bool)
    clusters: list[np.ndarray] = []

    for i in range(N):
        if visited[i]:
            continue

        tol_i = adaptive_tol(r[i])
        neigh = tree.query_ball_point(xy[i], tol_i)
        if len(neigh) < min_points:
            visited[i] = True
            continue

        queue = list(neigh)
        cluster: list[int] = []

        while queue:
            j = queue.pop()
            if visited[j]:
                continue
            visited[j] = True
            cluster.append(j)
            if len(cluster) >= max_points:
                break

            tol_j = adaptive_tol(r[j])
            neigh_j = tree.query_ball_point(xy[j], tol_j)
            if len(neigh_j) >= min_points:
                queue.extend(neigh_j)

        if len(cluster) >= min_points:
            clusters.append(np.array(cluster, dtype=np.int32))

    return clusters
