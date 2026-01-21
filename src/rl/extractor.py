from __future__ import annotations

from typing import Dict

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MultiModalExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim=1)

        vec_space = observation_space.spaces["vec"]
        self.vec_dim = int(vec_space.shape[0])

        bev_space = observation_space.spaces.get("bev", None)
        self.use_bev = bev_space is not None

        self.vec_net = nn.Sequential(
            nn.Linear(self.vec_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

        cnn_out = 0
        if self.use_bev:
            c, h, w = bev_space.shape
            self.cnn = nn.Sequential(
                nn.Conv2d(c, 32, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 2, 1),
                nn.ReLU(),
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

        self._features_dim = 128 + cnn_out
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        v = self.dropout(self.vec_net(obs["vec"]))
        if self.use_bev:
            b = self.dropout(self.cnn_fc(self.cnn(obs["bev"])))
            return torch.cat([v, b], dim=1)
        return v
