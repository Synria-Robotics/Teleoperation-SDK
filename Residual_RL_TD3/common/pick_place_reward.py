from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PickPlaceReward:
    """Reward model for shared-control pick-place experiments."""

    step_penalty: float = 1.0
    success_reward: float = 0.0
    failure_penalty: float = 100.0

    def compute_env_reward(
        self,
        obs: dict[str, np.ndarray],
        success: bool,
        *,
        failed: bool = False,
    ) -> tuple[float, dict[str, float]]:
        del obs
        if success:
            reward_env = self.success_reward
        elif failed:
            reward_env = -self.failure_penalty
        else:
            reward_env = -self.step_penalty
        components = {
            "step_penalty": float(self.step_penalty),
            "success_reward": float(self.success_reward if success else 0.0),
            "failure_penalty": float(self.failure_penalty if failed else 0.0),
        }
        return float(reward_env), components
