from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Transition:
    """One shared-control transition."""

    observation_state: np.ndarray
    next_observation_state: np.ndarray
    base_action: np.ndarray
    next_base_action: np.ndarray
    residual_action: np.ndarray
    prev_residual_action: np.ndarray
    commanded_action: np.ndarray
    realized_action: np.ndarray
    reward_env: float
    reward_total: float
    terminated: bool
    truncated: bool
    base_joint_target: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    base_gripper_target: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    teleop_active: bool = True
    success: bool = False
    alpha: float = 1.0
    copilot_enabled: bool = False
    conflict_score: float = 0.0
    correction_score: float = 0.0
    residual_norm: float = 0.0
    episode_time_s: float = 0.0
    control_dt: float = 0.0
    policy_latency_ms: float = 0.0

    @property
    def obs_flat(self) -> np.ndarray:
        return self.observation_state

    @property
    def next_obs_flat(self) -> np.ndarray:
        return self.next_observation_state

    @property
    def human_action(self) -> np.ndarray:
        return self.base_action

    @property
    def next_human_action(self) -> np.ndarray:
        return self.next_base_action

    @property
    def exec_action(self) -> np.ndarray:
        return self.realized_action


@dataclass(slots=True)
class EpisodeRollout:
    """Rollout container used for offline and online teleop datasets."""

    transitions: list[Transition] = field(default_factory=list)
    obs_keys: tuple[str, ...] = field(default_factory=tuple)
    success: bool = False
    episode_return_env: float = 0.0
    episode_return_total: float = 0.0
    meta: dict[str, float | int | str | bool] = field(default_factory=dict)

    @property
    def steps(self) -> int:
        return len(self.transitions)
