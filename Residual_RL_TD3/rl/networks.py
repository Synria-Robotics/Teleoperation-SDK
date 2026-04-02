from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(slots=True)
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray


class TensorNormalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray, device: torch.device):
        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
        self.std = torch.clamp(torch.as_tensor(std, dtype=torch.float32, device=device), min=1e-6)

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean) / self.std

    def unnormalize(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.std + self.mean


class ScalarNormalizer:
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = max(float(std), 1e-6)

    def normalize_tensor(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean) / self.std

    def unnormalize_tensor(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.std + self.mean


class MLPActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, *, hidden_dim: int = 256, residual_limit: float = 0.1):
        super().__init__()
        self.residual_limit = float(residual_limit)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        final = self.net[-1]
        assert isinstance(final, nn.Linear)
        nn.init.uniform_(final.weight, -1e-3, 1e-3)
        nn.init.uniform_(final.bias, -1e-3, 1e-3)

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs_flat)) * self.residual_limit


class MLPCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, *, hidden_dim: int = 256):
        super().__init__()
        input_dim = int(obs_dim) + int(action_dim)
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _fuse_inputs(self, obs_flat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.cat([obs_flat, action], dim=-1)

    def forward(self, obs_flat: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._fuse_inputs(obs_flat, action)
        return self.q1(x), self.q2(x)

    def q1_only(self, obs_flat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(self._fuse_inputs(obs_flat, action))

    def min_q(self, obs_flat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(obs_flat, action)
        return torch.minimum(q1, q2)
