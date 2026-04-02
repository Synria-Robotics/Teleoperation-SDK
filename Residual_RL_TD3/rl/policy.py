from __future__ import annotations

import numpy as np
import torch

from Residual_RL_TD3.rl.networks import MLPActor, TensorNormalizer
from Residual_RL_TD3.rl.observation import TELEOP_RESIDUAL_OBS_MODE, build_policy_observation_from_dict


class TorchResidualPolicy:
    def __init__(
        self,
        actor: MLPActor,
        device: torch.device,
        *,
        noise_scale: float = 0.0,
        residual_limit: float = 0.1,
        state_normalizer: TensorNormalizer | None = None,
        history_len: int = 4,
        translation_step: float = 0.012,
        rotation_step: float = 0.20,
        gripper_step: float = 0.004,
        obs_mode: str = TELEOP_RESIDUAL_OBS_MODE,
    ):
        self.actor = actor
        self.device = device
        self.noise_scale = float(noise_scale)
        self.residual_limit = float(residual_limit)
        self.state_normalizer = state_normalizer
        self.history_len = int(history_len)
        self.translation_step = float(translation_step)
        self.rotation_step = float(rotation_step)
        self.gripper_step = float(gripper_step)
        self.obs_mode = str(obs_mode)
        self.residual_history: list[np.ndarray] = []

    def reset(self) -> None:
        self.residual_history = []

    def get_action(self, obs: dict[str, np.ndarray], base_action: np.ndarray) -> np.ndarray:
        obs_flat_np = build_policy_observation_from_dict(
            obs,
            base_action=np.asarray(base_action, dtype=np.float32),
            residual_history=self.residual_history,
            history_len=self.history_len,
            translation_step=self.translation_step,
            rotation_step=self.rotation_step,
            gripper_step=self.gripper_step,
            obs_mode=self.obs_mode,
        )
        obs_flat = torch.as_tensor(obs_flat_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.state_normalizer is not None:
            obs_flat = self.state_normalizer.normalize(obs_flat)

        with torch.no_grad():
            residual = self.actor(obs_flat).squeeze(0).cpu().numpy()
        if self.noise_scale > 0.0:
            residual = residual + np.random.normal(scale=self.noise_scale, size=residual.shape).astype(np.float32)
        residual = np.clip(residual, -self.residual_limit, self.residual_limit).astype(np.float32)
        self.residual_history.append(residual.copy())
        if len(self.residual_history) > self.history_len:
            self.residual_history = self.residual_history[-self.history_len :]
        return residual
