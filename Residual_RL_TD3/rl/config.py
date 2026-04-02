from __future__ import annotations

from dataclasses import dataclass

from Residual_RL_TD3.rl.observation import TELEOP_RESIDUAL_OBS_MODE


@dataclass(slots=True)
class ResidualTD3Config:
    obs_dim: int
    action_dim: int = 7
    hidden_dim: int = 256
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.1
    noise_clip: float = 0.25
    policy_delay: int = 2
    residual_limit: float = 0.1
    history_len: int = 4
    obs_mode: str = TELEOP_RESIDUAL_OBS_MODE
