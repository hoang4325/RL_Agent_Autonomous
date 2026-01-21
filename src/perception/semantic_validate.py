from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class ClusterSemanticReport:
    purity: float
    dominant_instance: int
    dominant_tag: int
    instances_in_cluster: int


def validate_clusters_with_semantic(
    cluster_indices: List[np.ndarray],
    ray_points: np.ndarray,
    semantic_points: np.ndarray,
    semantic_tags: np.ndarray,
    semantic_instance_ids: np.ndarray,
    max_match_distance: float = 0.6,
) -> List[ClusterSemanticReport]:
    """For each cluster (from ray_cast), estimate semantic purity using nearest semantic points.

    Returns per-cluster report.
    """
    if len(cluster_indices) == 0:
        return []
    if semantic_points is None or len(semantic_points) == 0:
        return [ClusterSemanticReport(purity=0.0, dominant_instance=-1, dominant_tag=-1, instances_in_cluster=0) for _ in cluster_indices]

    tree = cKDTree(semantic_points[:, :2])
    reports: List[ClusterSemanticReport] = []

    for idx in cluster_indices:
        pts = ray_points[idx]
        if len(pts) == 0:
            reports.append(ClusterSemanticReport(0.0, -1, -1, 0))
            continue

        # nearest semantic for each ray point
        d, nn = tree.query(pts[:, :2], k=1)
        mask = d <= max_match_distance
        if not np.any(mask):
            reports.append(ClusterSemanticReport(0.0, -1, -1, 0))
            continue

        inst = semantic_instance_ids[nn[mask]]
        tags = semantic_tags[nn[mask]]

        # majority instance
        uniq, counts = np.unique(inst, return_counts=True)
        dom_i = int(uniq[np.argmax(counts)])
        purity = float(np.max(counts) / np.sum(counts))

        # dominant tag among points of dominant instance
        dom_mask = inst == dom_i
        if np.any(dom_mask):
            uniq_t, counts_t = np.unique(tags[dom_mask], return_counts=True)
            dom_t = int(uniq_t[np.argmax(counts_t)])
        else:
            dom_t = -1

        reports.append(ClusterSemanticReport(
            purity=purity,
            dominant_instance=dom_i,
            dominant_tag=dom_t,
            instances_in_cluster=int(len(uniq)),
        ))

    return reports
