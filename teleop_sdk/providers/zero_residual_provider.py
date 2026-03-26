from __future__ import annotations

import numpy as np

from teleop_sdk.providers.base import BaseResidualPolicy


class ZeroResidualPolicy(BaseResidualPolicy):
    """Residual policy used for human-only baselines."""

    def get_action(self, obs: dict[str, np.ndarray], base_action: np.ndarray) -> np.ndarray:
        del obs, base_action
        return np.zeros(self.action_dim, dtype=np.float32)
