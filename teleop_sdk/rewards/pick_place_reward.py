from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PickPlaceReward:
    """Reward model for shared-control pick-place experiments."""

    reach_weight: float = 1.5
    lift_weight: float = 2.0
    place_weight: float = 4.0
    success_bonus: float = 10.0
    time_penalty: float = 0.01
    alignment_bonus: float = 0.05
    conflict_penalty: float = 0.25
    residual_l2_penalty: float = 0.02
    smoothness_penalty: float = 0.02
    min_lift_height: float = 0.085
    success_xy_threshold: float = 0.03
    success_z_threshold: float = 0.04

    def compute_env_reward(self, obs: dict[str, np.ndarray], success: bool) -> tuple[float, dict[str, float]]:
        box_to_ee = np.asarray(obs["box_to_ee"], dtype=np.float32)
        box_to_basket = np.asarray(obs["box_to_basket"], dtype=np.float32)
        box_pos = np.asarray(obs["box_pos"], dtype=np.float32)

        reach = float(np.exp(-8.0 * np.linalg.norm(box_to_ee)))
        lift = float(np.clip((box_pos[2] - self.min_lift_height) / 0.08, 0.0, 1.0))
        place = float(np.exp(-10.0 * np.linalg.norm(box_to_basket[:2]))) * max(lift, 0.1)
        success_term = self.success_bonus if success else 0.0
        reward_env = (
            self.reach_weight * reach
            + self.lift_weight * lift
            + self.place_weight * place
            + success_term
            - self.time_penalty
        )
        components = {
            "reach": reach,
            "lift": lift,
            "place": place,
            "success_bonus": success_term,
            "time_penalty": self.time_penalty,
        }
        return float(reward_env), components

    def compute_total_reward(
        self,
        reward_env: float,
        base_action: np.ndarray,
        residual_action: np.ndarray,
        prev_residual_action: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        base_action = np.asarray(base_action, dtype=np.float32)
        residual_action = np.asarray(residual_action, dtype=np.float32)
        prev_residual_action = np.asarray(prev_residual_action, dtype=np.float32)

        h_norm = float(np.linalg.norm(base_action))
        r_norm = float(np.linalg.norm(residual_action))
        cosine = 0.0
        if h_norm > 1e-6 and r_norm > 1e-6:
            cosine = float(np.dot(base_action, residual_action) / (h_norm * r_norm))

        correction_score = max(cosine, 0.0) * r_norm
        conflict_score = max(-cosine, 0.0) * r_norm
        smoothness = float(np.linalg.norm(residual_action - prev_residual_action))
        reward_total = (
            reward_env
            + self.alignment_bonus * correction_score
            - self.conflict_penalty * conflict_score
            - self.residual_l2_penalty * (r_norm**2)
            - self.smoothness_penalty * smoothness
        )
        components = {
            "correction_score": correction_score,
            "conflict_score": conflict_score,
            "residual_norm": r_norm,
            "smoothness": smoothness,
        }
        return float(reward_total), components
