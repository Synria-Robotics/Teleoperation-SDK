from __future__ import annotations

import numpy as np

from teleop_sdk.providers.base import BaseResidualPolicy


class HeuristicResidualPolicy(BaseResidualPolicy):
    """A conservative geometric helper used for collecting assist demonstrations.

    This policy only nudges translational motion. Rotation and gripper residuals
    are kept at zero to avoid fighting the operator.
    """

    def __init__(
        self,
        *,
        residual_limit: float = 0.08,
        reach_gain: float = 0.12,
        place_gain: float = 0.14,
        near_box_threshold: float = 0.12,
        lifted_height: float = 0.10,
        idle_deadband: float = 0.02,
    ):
        self.residual_limit = float(residual_limit)
        self.reach_gain = float(reach_gain)
        self.place_gain = float(place_gain)
        self.near_box_threshold = float(near_box_threshold)
        self.lifted_height = float(lifted_height)
        self.idle_deadband = float(idle_deadband)

    def reset(self) -> None:
        pass

    def get_action(self, obs: dict[str, np.ndarray], base_action: np.ndarray) -> np.ndarray:
        base_action = np.asarray(base_action, dtype=np.float32)
        if np.linalg.norm(base_action) < self.idle_deadband:
            return np.zeros(self.action_dim, dtype=np.float32)

        box_to_ee = np.asarray(obs["box_to_ee"], dtype=np.float32)
        box_to_basket = np.asarray(obs["box_to_basket"], dtype=np.float32)
        box_pos = np.asarray(obs["box_pos"], dtype=np.float32)

        residual = np.zeros(self.action_dim, dtype=np.float32)

        # Before lifting, gently help the hand move toward the box.
        if float(box_pos[2]) < self.lifted_height:
            dist_to_box = float(np.linalg.norm(box_to_ee))
            if dist_to_box > self.near_box_threshold:
                residual[:3] = np.clip(box_to_ee, -1.0, 1.0) * self.reach_gain
            else:
                residual[:3] = np.clip(box_to_ee, -1.0, 1.0) * (0.5 * self.reach_gain)
        else:
            # Once lifted, guide toward basket center.
            residual[:3] = np.clip(box_to_basket, -1.0, 1.0) * self.place_gain

        return np.clip(residual, -self.residual_limit, self.residual_limit).astype(np.float32)
