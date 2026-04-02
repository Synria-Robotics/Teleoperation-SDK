from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from Residual_RL_TD3.rl.config import ResidualTD3Config
from Residual_RL_TD3.rl.networks import MLPActor, MLPCritic, TensorNormalizer
from Residual_RL_TD3.rl.policy import TorchResidualPolicy


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


class ResidualTD3Agent:
    """Minimal TD3 agent container for residual policy experiments."""

    def __init__(self, cfg: ResidualTD3Config, *, device: torch.device | str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)

        self.actor = MLPActor(
            cfg.obs_dim,
            cfg.action_dim,
            hidden_dim=cfg.hidden_dim,
            residual_limit=cfg.residual_limit,
        ).to(self.device)
        self.assist_critic = MLPCritic(cfg.obs_dim, cfg.action_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        self.base_critic = MLPCritic(cfg.obs_dim, cfg.action_dim, hidden_dim=cfg.hidden_dim).to(self.device)

        self.actor_target = deepcopy(self.actor).to(self.device)
        self.assist_critic_target = deepcopy(self.assist_critic).to(self.device)
        self.base_critic_target = deepcopy(self.base_critic).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        critic_params = list(self.assist_critic.parameters()) + list(self.base_critic.parameters())
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=cfg.critic_lr)

    def reset_targets(self) -> None:
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.assist_critic_target.load_state_dict(self.assist_critic.state_dict())
        self.base_critic_target.load_state_dict(self.base_critic.state_dict())

    def act(
        self,
        obs_flat: np.ndarray,
        *,
        noise_scale: float = 0.0,
        state_normalizer: TensorNormalizer | None = None,
    ) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs_flat, dtype=torch.float32, device=self.device).unsqueeze(0)
        if state_normalizer is not None:
            obs_tensor = state_normalizer.normalize(obs_tensor)
        with torch.no_grad():
            residual = self.actor(obs_tensor).squeeze(0).cpu().numpy()
        if noise_scale > 0.0:
            residual = residual + np.random.normal(scale=noise_scale, size=residual.shape).astype(np.float32)
        return np.clip(residual, -self.cfg.residual_limit, self.cfg.residual_limit).astype(np.float32)

    def build_policy(
        self,
        *,
        noise_scale: float = 0.0,
        state_normalizer: TensorNormalizer | None = None,
        translation_step: float = 0.012,
        rotation_step: float = 0.20,
        gripper_step: float = 0.004,
    ) -> TorchResidualPolicy:
        return TorchResidualPolicy(
            self.actor,
            self.device,
            noise_scale=noise_scale,
            residual_limit=self.cfg.residual_limit,
            state_normalizer=state_normalizer,
            history_len=self.cfg.history_len,
            translation_step=translation_step,
            rotation_step=rotation_step,
            gripper_step=gripper_step,
            obs_mode=self.cfg.obs_mode,
        )

    def update_targets(self) -> None:
        soft_update(self.actor_target, self.actor, self.cfg.tau)
        soft_update(self.assist_critic_target, self.assist_critic, self.cfg.tau)
        soft_update(self.base_critic_target, self.base_critic, self.cfg.tau)

    def state_dict(self) -> dict[str, object]:
        return {
            "config": self.cfg,
            "actor_state_dict": self.actor.state_dict(),
            "assist_critic_state_dict": self.assist_critic.state_dict(),
            "base_critic_state_dict": self.base_critic.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "assist_critic_target_state_dict": self.assist_critic_target.state_dict(),
            "base_critic_target_state_dict": self.base_critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.actor.load_state_dict(state["actor_state_dict"])  # type: ignore[arg-type]
        self.assist_critic.load_state_dict(state["assist_critic_state_dict"])  # type: ignore[arg-type]
        self.base_critic.load_state_dict(state["base_critic_state_dict"])  # type: ignore[arg-type]
        self.actor_target.load_state_dict(state["actor_target_state_dict"])  # type: ignore[arg-type]
        self.assist_critic_target.load_state_dict(state["assist_critic_target_state_dict"])  # type: ignore[arg-type]
        self.base_critic_target.load_state_dict(state["base_critic_target_state_dict"])  # type: ignore[arg-type]
        self.actor_optimizer.load_state_dict(state["actor_optimizer_state_dict"])  # type: ignore[arg-type]
        self.critic_optimizer.load_state_dict(state["critic_optimizer_state_dict"])  # type: ignore[arg-type]
