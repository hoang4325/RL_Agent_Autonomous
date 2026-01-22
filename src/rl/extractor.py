from __future__ import annotations

from typing import Dict

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MultiModalExtractor(BaseFeaturesExtractor):
    """
    Extractor for Dict obs:
      - vec: (B, vec_dim)
      - bev: (B, C, H, W) optional, float in [-1, 1]
    Output features: vec_feat (128) + bev_feat (128 if bev enabled).
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        # We'll compute real features_dim below; initialize with a dummy first,
        # then set self._features_dim properly.
        super().__init__(observation_space, features_dim=1)

        vec_space = observation_space.spaces["vec"]
        self.vec_dim = int(vec_space.shape[0])

        bev_space = observation_space.spaces.get("bev", None)
        self.use_bev = bev_space is not None

        # -------- vec MLP --------
        self.vec_net = nn.Sequential(
            nn.Linear(self.vec_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

        # -------- bev CNN (optional) --------
        cnn_out = 0
        if self.use_bev:
            c, h, w = bev_space.shape

            # GroupNorm is more stable than BatchNorm for RL (small/variable batch sizes)
            self.cnn = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.GroupNorm(num_groups=8, num_channels=32),

                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.GroupNorm(num_groups=8, num_channels=64),

                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.GroupNorm(num_groups=8, num_channels=64),

                nn.Flatten(),
            )

            with torch.no_grad():
                n_flat = int(self.cnn(torch.zeros((1, c, h, w))).shape[1])

            self.cnn_fc = nn.Sequential(
                nn.Linear(n_flat, 128),
                nn.ReLU(),
                nn.LayerNorm(128),
            )
            cnn_out = 128

        # real output dim
        self._features_dim = 128 + cnn_out

        # Dropout: keep small; you can set p=0.0 for max stability
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # vec always exists
        v = obs["vec"]
        # safety: ensure float32
        if v.dtype != torch.float32:
            v = v.float()

        v_feat = self.dropout(self.vec_net(v))

        if self.use_bev:
            b = obs["bev"]
            if b.dtype != torch.float32:
                b = b.float()

            # safety clamp in case something upstream normalizes/perturbs BEV
            b = torch.clamp(b, -1.0, 1.0)

            b_feat = self.dropout(self.cnn_fc(self.cnn(b)))
            return torch.cat([v_feat, b_feat], dim=1)

        return v_feat
