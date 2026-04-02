from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from Residual_RL_TD3.common import PickPlaceReward, flatten_observation, unflatten_observation
from Residual_RL_TD3.data import EpisodeRollout, Transition, load_episode_npz
from Residual_RL_TD3.env.mujoco_pick_place_env import (
    MujocoPickPlaceTeleopEnv,
    PickPlaceTaskConfig,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_fixed_box_xy(fixed_box_xy: Sequence[float] | None) -> tuple[float, float] | None:
    if fixed_box_xy is None:
        return None
    values = tuple(float(v) for v in fixed_box_xy)
    if not values:
        return None
    if len(values) != 2:
        raise ValueError(f"fixed_box_xy must contain exactly 2 values, got {values}")
    return values[0], values[1]


def build_pick_place_task_config(
    xml_path: str,
    *,
    seed: int,
    fixed_scene: bool = False,
    fixed_box_xy: Sequence[float] | None = None,
) -> PickPlaceTaskConfig:
    resolved_fixed_box_xy = normalize_fixed_box_xy(fixed_box_xy)
    return PickPlaceTaskConfig(
        xml_path=xml_path,
        seed=seed,
        randomize_box_position=(not fixed_scene) and resolved_fixed_box_xy is None,
        randomize_basket_position=not fixed_scene,
        fixed_box_xy=resolved_fixed_box_xy,
    )


class ZeroResidualPolicy:
    def reset(self) -> None:
        pass

    def get_action(self, obs: dict[str, np.ndarray], base_action: np.ndarray) -> np.ndarray:
        del obs, base_action
        return np.zeros(7, dtype=np.float32)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.storage: list[dict[str, np.ndarray | float | bool]] = []
        self.pos = 0

    def add(self, item: dict[str, np.ndarray | float | bool]) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def __len__(self) -> int:
        return len(self.storage)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, len(self.storage), size=batch_size)
        batch = [self.storage[i] for i in idx]
        keys = batch[0].keys()
        out: dict[str, torch.Tensor] = {}
        for key in keys:
            values = [sample[key] for sample in batch]
            arr = np.asarray(values, dtype=np.float32)
            out[key] = torch.as_tensor(arr, dtype=torch.float32)
        return out


@dataclass(slots=True)
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray


class TensorNormalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray, device: torch.device):
        mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
        std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
        self.mean = mean_t
        self.std = torch.clamp(std_t, min=1e-6)

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


class MetricTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, **metrics: float) -> None:
        for key, value in metrics.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value)
            self.counts[key] = self.counts.get(key, 0) + 1

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, total in self.sums.items():
            count = max(self.counts.get(key, 0), 1)
            out[key] = total / count
        return out


class MLPActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 256,
        residual_limit: float = 0.1,
    ):
        super().__init__()
        self.residual_limit = residual_limit
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
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 256,
    ):
        super().__init__()
        input_dim = obs_dim + action_dim
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
        x = self._fuse_inputs(obs_flat, action)
        return self.q1(x)

    def min_q(self, obs_flat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(obs_flat, action)
        return torch.minimum(q1, q2)


TELEOP_RESIDUAL_OBS_MODE = "follower_pose_hdelta_residual"


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(-1)
    norm = max(float(np.linalg.norm(quat)), 1e-6)
    return quat / norm


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    quat = _quat_normalize(quat)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float32)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = _quat_normalize(q1)
    w2, x2, y2, z2 = _quat_normalize(q2)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def _quat_align_sign(quat: np.ndarray, reference: np.ndarray) -> np.ndarray:
    quat = _quat_normalize(quat)
    reference = _quat_normalize(reference)
    if float(np.dot(quat, reference)) < 0.0:
        return -quat
    return quat


def _axis_angle_to_quat(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(3)
    angle = float(np.linalg.norm(vec))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    axis = vec / angle
    half = 0.5 * angle
    return np.array([np.cos(half), *(np.sin(half) * axis)], dtype=np.float32)


def _quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    quat = _quat_normalize(quat)
    qw = float(np.clip(quat[0], -1.0, 1.0))
    angle = 2.0 * np.arccos(qw)
    if angle < 1e-9:
        return np.zeros(3, dtype=np.float32)
    s = np.sqrt(max(1.0 - qw * qw, 0.0))
    axis = quat[1:] / max(s, 1e-6)
    return (axis * angle).astype(np.float32)


def _flatten_residual_history(
    residual_history: Sequence[np.ndarray],
    *,
    history_len: int,
    action_dim: int,
) -> np.ndarray:
    history = np.zeros((history_len, action_dim), dtype=np.float32)
    if history_len <= 0:
        return history.reshape(-1)
    recent = list(residual_history[-history_len:])
    start = history_len - len(recent)
    for idx, residual in enumerate(recent):
        history[start + idx] = np.asarray(residual, dtype=np.float32).reshape(action_dim)
    return history.reshape(-1)


def _compose_base_target_feature(
    follower_ee_pos: np.ndarray,
    follower_ee_quat: np.ndarray,
    follower_gripper: np.ndarray | float,
    base_action: np.ndarray,
    *,
    translation_step: float,
    rotation_step: float,
    gripper_step: float,
) -> np.ndarray:
    follower_pos = np.asarray(follower_ee_pos, dtype=np.float32).reshape(3)
    follower_quat = _quat_normalize(np.asarray(follower_ee_quat, dtype=np.float32).reshape(4))
    follower_grip = float(np.asarray(follower_gripper, dtype=np.float32).reshape(-1)[0])
    base = np.asarray(base_action, dtype=np.float32).reshape(7)

    target_pos = follower_pos + base[:3] * float(translation_step)
    dq = _axis_angle_to_quat(base[3:6].astype(np.float32) * float(rotation_step))
    target_quat = _quat_normalize(_quat_mul(dq, follower_quat))
    target_rotvec = _quat_to_rotvec(target_quat)
    target_grip = np.asarray([follower_grip + float(base[6]) * float(gripper_step)], dtype=np.float32)
    return np.concatenate([target_pos, target_rotvec, target_grip], axis=0).astype(np.float32)


def build_teleop_residual_observation(
    follower_ee_pos: np.ndarray,
    follower_ee_quat: np.ndarray,
    follower_gripper: np.ndarray | float,
    base_action: np.ndarray,
    residual_history: Sequence[np.ndarray],
    *,
    history_len: int,
    translation_step: float,
    rotation_step: float,
    gripper_step: float,
) -> np.ndarray:
    follower_pos = np.asarray(follower_ee_pos, dtype=np.float32).reshape(-1)
    follower_quat = _quat_normalize(np.asarray(follower_ee_quat, dtype=np.float32).reshape(-1))
    follower_grip = np.asarray(follower_gripper, dtype=np.float32).reshape(1)
    base = _compose_base_target_feature(
        follower_ee_pos,
        follower_ee_quat,
        follower_gripper,
        base_action,
        translation_step=translation_step,
        rotation_step=rotation_step,
        gripper_step=gripper_step,
    )
    residual_hist = _flatten_residual_history(
        residual_history,
        history_len=history_len,
        action_dim=int(base.shape[0]),
    )

    return np.concatenate(
        [
            follower_pos,
            follower_quat,
            follower_grip,
            base,
            residual_hist,
        ],
        axis=0,
    ).astype(np.float32)


def build_policy_observation_from_dict(
    obs: dict[str, np.ndarray],
    *,
    base_action: np.ndarray,
    residual_history: Sequence[np.ndarray],
    history_len: int,
    translation_step: float,
    rotation_step: float,
    gripper_step: float,
    obs_mode: str,
) -> np.ndarray:
    if obs_mode == TELEOP_RESIDUAL_OBS_MODE:
        required = ("ee_pos", "ee_quat", "gripper_pos")
        missing = [key for key in required if key not in obs]
        if missing:
            raise KeyError(f"Teleop residual observation requires keys {missing}, got {sorted(obs)}")
        return build_teleop_residual_observation(
            obs["ee_pos"],
            obs["ee_quat"],
            obs["gripper_pos"],
            base_action,
            residual_history,
            history_len=history_len,
            translation_step=translation_step,
            rotation_step=rotation_step,
            gripper_step=gripper_step,
        )
    raise ValueError(f"Unsupported obs_mode={obs_mode}. Only {TELEOP_RESIDUAL_OBS_MODE!r} is supported.")


def build_policy_observation_from_flat_state(
    obs_flat: np.ndarray,
    *,
    base_action: np.ndarray,
    residual_history: Sequence[np.ndarray],
    history_len: int,
    translation_step: float,
    rotation_step: float,
    gripper_step: float,
    obs_mode: str,
) -> np.ndarray:
    if obs_mode == TELEOP_RESIDUAL_OBS_MODE:
        parsed = unflatten_observation(obs_flat)
        return build_teleop_residual_observation(
            parsed["ee_pos"],
            parsed["ee_quat"],
            parsed["gripper_pos"],
            base_action,
            residual_history,
            history_len=history_len,
            translation_step=translation_step,
            rotation_step=rotation_step,
            gripper_step=gripper_step,
        )
    raise ValueError(f"Unsupported obs_mode={obs_mode}. Only {TELEOP_RESIDUAL_OBS_MODE!r} is supported.")


def infer_policy_obs_dim(
    obs: dict[str, np.ndarray],
    *,
    action_dim: int,
    obs_mode: str,
    history_len: int,
    translation_step: float,
    rotation_step: float,
    gripper_step: float,
) -> int:
    if obs_mode != TELEOP_RESIDUAL_OBS_MODE:
        raise ValueError(f"Unsupported obs_mode={obs_mode}. Only {TELEOP_RESIDUAL_OBS_MODE!r} is supported.")
    policy_obs = build_policy_observation_from_dict(
        obs,
        base_action=np.zeros(action_dim, dtype=np.float32),
        residual_history=[],
        history_len=history_len,
        translation_step=translation_step,
        rotation_step=rotation_step,
        gripper_step=gripper_step,
        obs_mode=obs_mode,
    )
    return int(policy_obs.shape[0])


def clip_residual_tensor(residual: torch.Tensor, limit: float) -> torch.Tensor:
    return torch.clamp(residual, -limit, limit)


def clip_commanded_action_tensor(commanded_action: torch.Tensor) -> torch.Tensor:
    return torch.clamp(commanded_action, -1.0, 1.0)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def build_rollout_policy_observations(
    rollout: EpisodeRollout,
    *,
    obs_mode: str,
    history_len: int,
    translation_step: float,
    rotation_step: float,
    gripper_step: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    transitions = rollout.transitions
    if not transitions:
        return [], []

    policy_obs: list[np.ndarray] = []
    next_policy_obs: list[np.ndarray] = []
    residual_history: list[np.ndarray] = []
    for idx, transition in enumerate(transitions):
        policy_obs.append(
            build_policy_observation_from_flat_state(
                transition.observation_state,
                base_action=transition.base_action,
                residual_history=residual_history,
                history_len=history_len,
                translation_step=translation_step,
                rotation_step=rotation_step,
                gripper_step=gripper_step,
                obs_mode=obs_mode,
            )
        )
        next_residual_history = [*residual_history, np.asarray(transition.residual_action, dtype=np.float32).copy()]
        next_policy_obs.append(
            build_policy_observation_from_flat_state(
                transition.next_observation_state,
                base_action=transition.next_base_action,
                residual_history=next_residual_history,
                history_len=history_len,
                translation_step=translation_step,
                rotation_step=rotation_step,
                gripper_step=gripper_step,
                obs_mode=obs_mode,
            )
        )
        residual_history = next_residual_history[-history_len:]
    return policy_obs, next_policy_obs


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
        return np.clip(residual, -self.residual_limit, self.residual_limit).astype(np.float32)


class RandomResidualPolicy:
    def __init__(self, residual_limit: float = 0.1, noise_scale: float = 0.05):
        self.residual_limit = float(residual_limit)
        self.noise_scale = float(noise_scale)

    def reset(self) -> None:
        pass

    def get_action(self, obs: dict[str, np.ndarray], base_action: np.ndarray) -> np.ndarray:
        del obs, base_action
        residual = np.random.normal(scale=self.noise_scale, size=7).astype(np.float32)
        return np.clip(residual, -self.residual_limit, self.residual_limit).astype(np.float32)


class HeuristicBootstrapResidualPolicy:
    def __init__(
        self,
        *,
        residual_limit: float = 0.1,
        pos_gain: float = 0.35,
        hover_height: float = 0.08,
        grasp_height: float = 0.015,
        carry_height: float = 0.12,
        lifted_height: float = 0.09,
        place_xy_radius: float = 0.05,
    ):
        self.residual_limit = float(residual_limit)
        self.pos_gain = float(pos_gain)
        self.hover_height = float(hover_height)
        self.grasp_height = float(grasp_height)
        self.carry_height = float(carry_height)
        self.lifted_height = float(lifted_height)
        self.place_xy_radius = float(place_xy_radius)

    def reset(self) -> None:
        pass

    def get_action(self, obs: dict[str, np.ndarray], base_action: np.ndarray) -> np.ndarray:
        del base_action
        ee_pos = np.asarray(obs["ee_pos"], dtype=np.float32).reshape(3)
        box_pos = np.asarray(obs["box_pos"], dtype=np.float32).reshape(3)
        basket_pos = np.asarray(obs["basket_pos"], dtype=np.float32).reshape(3)
        box_to_basket = np.asarray(obs.get("box_to_basket", basket_pos - box_pos), dtype=np.float32).reshape(3)
        box_lifted = float(box_pos[2]) > self.lifted_height
        box_near_basket = float(np.linalg.norm(box_to_basket[:2])) < self.place_xy_radius

        if box_lifted and not box_near_basket:
            target_pos = basket_pos.copy()
            target_pos[2] = max(float(target_pos[2]), float(box_pos[2]), float(ee_pos[2])) + self.carry_height
        elif box_lifted and box_near_basket:
            target_pos = basket_pos.copy()
            target_pos[2] = float(basket_pos[2]) + self.grasp_height
        else:
            target_pos = box_pos.copy()
            xy_close = float(np.linalg.norm((box_pos - ee_pos)[:2])) < 0.03
            target_pos[2] = float(box_pos[2]) + (self.grasp_height if xy_close else self.hover_height)

        pos_err = target_pos - ee_pos
        residual = np.zeros(7, dtype=np.float32)
        residual[:3] = np.clip(self.pos_gain * pos_err, -self.residual_limit, self.residual_limit)
        return residual


@dataclass(slots=True)
class TrainConfig:
    dataset_dirs: tuple[Path, ...]
    save_path: Path
    xml_path: str
    init_checkpoint: Path | None = None
    seed: int = 0
    fixed_scene: bool = False
    fixed_box_xy: tuple[float, float] | None = None
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    offline_fraction: float = 0.5
    learning_starts: int = 1000
    critic_warmup_steps: int = 1000
    total_timesteps: int = 10_000
    update_every_n_steps: int = 200
    num_updates_per_iteration: int = 200
    actor_updates_per_iteration: int = 100
    eval_episodes: int = 5
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    buffer_capacity: int = 200_000
    residual_limit: float = 0.1
    residual_mag_reg_weight: float = 1.0
    offline_zero_bc_weight: float = 5.0
    teacher_bc_weight: float = 2.0
    teacher_rollout_steps: int = 4000
    target_policy_noise: float = 0.01
    target_noise_clip: float = 0.02
    exploration_noise: float = 0.005
    warmup_policy: str = "heuristic"
    warmup_residual_noise: float = 0.0
    hidden_dim: int = 256
    obs_mode: str = TELEOP_RESIDUAL_OBS_MODE
    history_len: int = 4
    translation_step: float = 0.012
    rotation_step: float = 0.20
    gripper_step: float = 0.004
    normalize_state: bool = True
    normalize_reward: bool = True
    log_interval: int = 200

def compute_state_stats(buffer: ReplayBuffer) -> NormalizationStats:
    return compute_state_stats_from_buffers(buffer)


def compute_state_stats_from_items(items: Sequence[dict[str, np.ndarray | float | bool]]) -> NormalizationStats:
    if not items:
        raise RuntimeError("Cannot compute state normalization stats from an empty dataset")
    stacked = np.stack([np.asarray(item["observation_state"], dtype=np.float32) for item in items], axis=0)
    return NormalizationStats(
        mean=stacked.mean(axis=0).astype(np.float32),
        std=np.clip(stacked.std(axis=0), 1e-3, None).astype(np.float32),
    )


def compute_state_stats_from_buffers(*buffers: ReplayBuffer) -> NormalizationStats:
    items: list[dict[str, np.ndarray | float | bool]] = []
    for buffer in buffers:
        items.extend(buffer.storage)
    return compute_state_stats_from_items(items)


def compute_reward_stats(episodes: Sequence[EpisodeRollout]) -> tuple[float, float]:
    rewards = np.asarray(
        [float(t.reward_env) for rollout in episodes for t in rollout.transitions],
        dtype=np.float32,
    )
    if rewards.size == 0:
        raise RuntimeError("Cannot compute reward normalization stats from an empty dataset")
    return float(rewards.mean()), max(float(rewards.std()), 1e-3)


def transition_to_buffer_item(
    transition: Transition,
    *,
    policy_obs: np.ndarray,
    next_policy_obs: np.ndarray,
    is_teacher: bool = False,
) -> dict[str, np.ndarray | float | bool]:
    return {
        "observation_state": np.asarray(policy_obs, dtype=np.float32),
        "next_observation_state": np.asarray(next_policy_obs, dtype=np.float32),
        "base_action": transition.base_action,
        "next_base_action": transition.next_base_action,
        "behavior_residual": transition.residual_action,
        "realized_action": transition.realized_action,
        "reward_env": np.float32(transition.reward_env),
        "done": np.float32(float(transition.terminated or transition.truncated)),
        "is_teacher": np.float32(float(is_teacher)),
    }


def load_dataset_into_buffer(
    dataset_dirs: Sequence[Path],
    offline_rb: ReplayBuffer,
    cfg: TrainConfig,
    *,
    allow_empty: bool = False,
) -> list[EpisodeRollout]:
    if not dataset_dirs:
        if allow_empty:
            return []
        raise FileNotFoundError("No dataset directories were provided")

    playback_episodes: list[EpisodeRollout] = []
    total_explicit_base_target_episodes = 0
    total_transitions = 0
    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            if allow_empty:
                continue
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
        episode_paths = sorted(dataset_dir.glob("episode_*.npz"))
        if not episode_paths:
            if allow_empty:
                print(f"Loaded 0 episodes from {dataset_dir} (directory is currently empty)")
                continue
            raise FileNotFoundError(f"No episode_*.npz files found under {dataset_dir}")

        explicit_base_target_episodes = 0
        dir_transitions = 0
        for path in episode_paths:
            with np.load(path, allow_pickle=True) as data:
                has_explicit_base_targets = "base_joint_target" in data and "base_gripper_target" in data
            rollout = load_episode_npz(path)
            playback_episodes.append(rollout)
            dir_transitions += len(rollout.transitions)
            if has_explicit_base_targets:
                explicit_base_target_episodes += 1

            policy_obs, next_policy_obs = build_rollout_policy_observations(
                rollout,
                obs_mode=cfg.obs_mode,
                history_len=cfg.history_len,
                translation_step=cfg.translation_step,
                rotation_step=cfg.rotation_step,
                gripper_step=cfg.gripper_step,
            )
            for transition, obs_t, next_obs_t in zip(
                rollout.transitions,
                policy_obs,
                next_policy_obs,
                strict=True,
            ):
                offline_rb.add(
                    transition_to_buffer_item(
                        transition,
                        policy_obs=obs_t,
                        next_policy_obs=next_obs_t,
                        is_teacher=False,
                    )
                )
        total_explicit_base_target_episodes += explicit_base_target_episodes
        total_transitions += dir_transitions
        print(
            f"Loaded {len(episode_paths)} episodes from {dataset_dir}"
            f" ({explicit_base_target_episodes} with explicit absolute base targets,"
            f" {dir_transitions} transitions)"
        )
    print(
        f"Loaded {len(playback_episodes)} total episodes from {len(dataset_dirs)} dataset directories"
        f" ({total_explicit_base_target_episodes} with explicit absolute base targets,"
        f" {total_transitions} transitions)"
    )
    return playback_episodes


def load_matching_state_dict(module: nn.Module, state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    current = module.state_dict()
    matched: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key in current and current[key].shape == value.shape:
            matched[key] = value
    current.update(matched)
    module.load_state_dict(current)
    return len(matched), len(state_dict) - len(matched)


def maybe_load_init_checkpoint(
    checkpoint_path: Path | None,
    *,
    actor: MLPActor,
    assist_critic: MLPCritic,
    base_critic: MLPCritic,
    actor_target: MLPActor,
    assist_critic_target: MLPCritic,
    base_critic_target: MLPCritic,
) -> dict[str, Any] | None:
    if checkpoint_path is None:
        return None

    payload = torch.load(checkpoint_path.resolve(), map_location="cpu", weights_only=False)
    assist_state = payload.get("assist_critic_state_dict", payload.get("critic_state_dict"))
    if assist_state is None:
        raise KeyError(f"Checkpoint does not contain assist_critic_state_dict or critic_state_dict: {checkpoint_path}")
    base_state = payload.get("base_critic_state_dict", assist_state)

    assist_loaded, assist_skipped = load_matching_state_dict(assist_critic, assist_state)
    base_loaded, base_skipped = load_matching_state_dict(base_critic, base_state)
    actor_loaded = 0
    actor_skipped = 0
    if "actor_state_dict" in payload:
        actor_loaded, actor_skipped = load_matching_state_dict(actor, payload["actor_state_dict"])

    actor_target.load_state_dict(actor.state_dict())
    assist_critic_target.load_state_dict(assist_critic.state_dict())
    base_critic_target.load_state_dict(base_critic.state_dict())
    print(
        "Initialized networks from checkpoint:",
        checkpoint_path,
        f"(actor matched={actor_loaded} skipped={actor_skipped},",
        f"assist matched={assist_loaded} skipped={assist_skipped},",
        f"base matched={base_loaded} skipped={base_skipped})",
    )
    return payload


def push_records_to_buffer(
    buffer: ReplayBuffer,
    rollout: EpisodeRollout,
    history_len: int,
    obs_mode: str,
    translation_step: float,
    rotation_step: float,
    gripper_step: float,
    *,
    is_teacher: bool = False,
) -> None:
    policy_obs, next_policy_obs = build_rollout_policy_observations(
        rollout,
        obs_mode=obs_mode,
        history_len=history_len,
        translation_step=translation_step,
        rotation_step=rotation_step,
        gripper_step=gripper_step,
    )
    for transition, obs_t, next_obs_t in zip(
        rollout.transitions,
        policy_obs,
        next_policy_obs,
        strict=True,
    ):
        buffer.add(
            transition_to_buffer_item(
                transition,
                policy_obs=obs_t,
                next_policy_obs=next_obs_t,
                is_teacher=is_teacher,
            )
        )


def sample_mixed_batch(
    offline_rb: ReplayBuffer,
    online_rb: ReplayBuffer,
    batch_size: int,
    offline_fraction: float,
) -> tuple[dict[str, torch.Tensor], float]:
    if len(offline_rb) == 0 and len(online_rb) == 0:
        raise RuntimeError("Replay buffers are empty")
    if len(offline_rb) == 0:
        online_batch = online_rb.sample(batch_size)
        online_batch["is_offline"] = torch.zeros(batch_size, dtype=torch.float32)
        return online_batch, 1.0

    online_bs = min(len(online_rb), int(round(batch_size * (1.0 - offline_fraction))))
    offline_bs = batch_size - online_bs
    if online_bs <= 0:
        offline_batch = offline_rb.sample(batch_size)
        offline_batch["is_offline"] = torch.ones(batch_size, dtype=torch.float32)
        return offline_batch, 0.0

    offline_batch = offline_rb.sample(offline_bs)
    online_batch = online_rb.sample(online_bs)
    merged: dict[str, torch.Tensor] = {}
    for key in offline_batch:
        merged[key] = torch.cat([offline_batch[key], online_batch[key]], dim=0)
    merged["is_offline"] = torch.cat(
        [
            torch.ones(offline_bs, dtype=torch.float32),
            torch.zeros(online_bs, dtype=torch.float32),
        ],
        dim=0,
    )
    return merged, float(online_bs) / max(batch_size, 1)


def prepare_batch(
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
    state_normalizer: TensorNormalizer | None,
    reward_normalizer: ScalarNormalizer | None,
) -> dict[str, torch.Tensor]:
    obs_flat = batch["observation_state"].to(device)
    next_obs_flat = batch["next_observation_state"].to(device)
    if state_normalizer is not None:
        obs_flat = state_normalizer.normalize(obs_flat)
        next_obs_flat = state_normalizer.normalize(next_obs_flat)

    reward_value = batch["reward_env"].to(device).unsqueeze(-1)
    if reward_normalizer is not None:
        reward_value = reward_normalizer.normalize_tensor(reward_value)

    return {
        "obs_flat": obs_flat,
        "next_obs_flat": next_obs_flat,
        "base_action": batch["base_action"].to(device),
        "next_base_action": batch["next_base_action"].to(device),
        "behavior_residual": batch["behavior_residual"].to(device),
        "assist_action": batch["realized_action"].to(device),
        "reward_env": reward_value,
        "done": batch["done"].to(device).unsqueeze(-1),
        "is_offline": batch["is_offline"].to(device).unsqueeze(-1),
        "is_teacher": batch["is_teacher"].to(device).unsqueeze(-1),
    }


def collect_episode(
    env: MujocoPickPlaceTeleopEnv,
    demo_rollout: EpisodeRollout,
    residual_policy,
    *,
    seed: int | None = None,
) -> EpisodeRollout:
    del seed
    if not demo_rollout.transitions:
        return EpisodeRollout(
            transitions=[],
            obs_keys=demo_rollout.obs_keys,
            success=False,
            episode_return_env=0.0,
            meta={"source": "empty_demo"},
        )

    obs, _ = env.reset_from_flat_observation(demo_rollout.transitions[0].obs_flat)
    residual_policy.reset()
    obs_keys = tuple(obs.keys())
    transitions: list[Transition] = []
    reward_env_sum = 0.0

    demo_actions = [np.asarray(t.base_action, dtype=np.float32) for t in demo_rollout.transitions]
    for idx, demo_transition in enumerate(demo_rollout.transitions):
        base_action = demo_actions[idx]
        obs_copy = {k: np.asarray(v, dtype=np.float32).copy() for k, v in obs.items()}
        residual_action = np.clip(residual_policy.get_action(obs_copy, base_action), -1.0, 1.0).astype(np.float32)
        base_joint_target = np.asarray(demo_transition.base_joint_target, dtype=np.float32)
        base_gripper_target = float(np.asarray(demo_transition.base_gripper_target, dtype=np.float32).reshape(-1)[0])
        next_obs, reward_env, terminated, truncated, info = env.step_base_joint_target(
            base_joint_target,
            base_gripper_target,
            residual_action,
            base_action=base_action,
        )
        next_base_action = demo_actions[idx + 1].copy() if idx + 1 < len(demo_actions) else np.zeros_like(base_action)
        transitions.append(
            Transition(
                observation_state=flatten_observation(obs_copy),
                next_observation_state=flatten_observation(next_obs),
                base_action=np.asarray(info.get("base_action", base_action), dtype=np.float32).copy(),
                next_base_action=next_base_action.copy(),
                residual_action=residual_action.copy(),
                realized_action=np.asarray(info["realized_action"], dtype=np.float32).copy(),
                reward_env=float(reward_env),
                terminated=bool(terminated),
                truncated=bool(truncated),
                base_joint_target=base_joint_target.copy(),
                base_gripper_target=np.asarray([base_gripper_target], dtype=np.float32),
                success=bool(info.get("success", False)),
                control_dt=float(env.cfg.control_dt),
            )
        )
        reward_env_sum += float(reward_env)
        obs = next_obs
        if terminated or truncated:
            break

    success = any(t.success for t in transitions)
    return EpisodeRollout(
        transitions=transitions,
        obs_keys=obs_keys,
        success=success,
        episode_return_env=reward_env_sum,
        meta={"source": "playback_rollout"},
    )


def evaluate_policy(
    env: MujocoPickPlaceTeleopEnv,
    playback_episodes: list[EpisodeRollout],
    residual_policy,
    *,
    eval_episodes: int,
) -> dict[str, float]:
    returns_env = []
    successes = []
    episode_pool = playback_episodes if eval_episodes <= 0 else playback_episodes[: min(eval_episodes, len(playback_episodes))]
    for idx, demo_rollout in enumerate(episode_pool):
        rollout = collect_episode(env, demo_rollout, residual_policy, seed=idx)
        returns_env.append(rollout.episode_return_env)
        successes.append(float(rollout.success))
    return {
        "avg_return_env": float(np.mean(returns_env)) if returns_env else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
    }


def compute_actor_loss(
    assist_critic: MLPCritic,
    base_critic: MLPCritic,
    obs_flat: torch.Tensor,
    base_action: torch.Tensor,
    pred_residual: torch.Tensor,
    pred_commanded_action: torch.Tensor,
    behavior_residual: torch.Tensor,
    is_offline: torch.Tensor,
    is_teacher: torch.Tensor,
    residual_limit: float,
    residual_mag_reg_weight: float,
    offline_zero_bc_weight: float,
    teacher_bc_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    policy_help_adv = assist_critic.min_q(obs_flat, pred_commanded_action) - base_critic.min_q(obs_flat, base_action)
    norm_scale = max(float(residual_limit), 1e-6)
    pred_residual_normed = pred_residual / norm_scale
    residual_mag_l2 = pred_residual_normed.pow(2).mean()
    offline_mask = is_offline.squeeze(-1) > 0.5
    if bool(offline_mask.any().item()):
        offline_zero_bc_l2 = pred_residual_normed[offline_mask].pow(2).mean()
    else:
        offline_zero_bc_l2 = pred_residual_normed.new_zeros(())
    teacher_mask = is_teacher.squeeze(-1) > 0.5
    if bool(teacher_mask.any().item()):
        teacher_residual_bc_l2 = ((pred_residual - behavior_residual)[teacher_mask] / norm_scale).pow(2).mean()
    else:
        teacher_residual_bc_l2 = pred_residual_normed.new_zeros(())
    actor_loss = (
        -policy_help_adv.mean()
        + float(residual_mag_reg_weight) * residual_mag_l2
        + float(offline_zero_bc_weight) * offline_zero_bc_l2
        + float(teacher_bc_weight) * teacher_residual_bc_l2
    )
    residual_abs = pred_residual.abs()
    residual_sat_ratio = (residual_abs >= (0.98 * norm_scale)).float().mean()
    commanded_action_sat_ratio = (pred_commanded_action.abs() >= 0.98).float().mean()
    metrics = {
        "policy_help_adv": float(policy_help_adv.mean().item()),
        "residual_mag_l2": float(residual_mag_l2.item()),
        "offline_zero_bc_l2": float(offline_zero_bc_l2.item()),
        "teacher_residual_bc_l2": float(teacher_residual_bc_l2.item()),
        "residual_mag_reg_weight": float(residual_mag_reg_weight),
        "offline_zero_bc_weight": float(offline_zero_bc_weight),
        "teacher_bc_weight": float(teacher_bc_weight),
        "pred_residual_abs_mean": float(residual_abs.mean().item()),
        "pred_residual_sat_ratio": float(residual_sat_ratio.item()),
        "pred_commanded_action_sat_ratio": float(commanded_action_sat_ratio.item()),
    }
    return actor_loss, metrics


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    ordered = " ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))
    return f"{prefix} {ordered}".strip()


def build_config_payload(cfg: TrainConfig) -> dict[str, Any]:
    config_payload: dict[str, Any] = {}
    for key, value in asdict(cfg).items():
        if isinstance(value, Path):
            config_payload[key] = str(value)
        elif isinstance(value, (list, tuple)) and value and all(isinstance(item, Path) for item in value):
            config_payload[key] = [str(item) for item in value]
        else:
            config_payload[key] = value
    return config_payload


def build_checkpoint_payload(
    *,
    cfg: TrainConfig,
    actor: MLPActor,
    assist_critic: MLPCritic,
    base_critic: MLPCritic,
    obs_dim: int,
    action_dim: int,
    state_stats: NormalizationStats,
    reward_mean: float,
    reward_std: float,
    eval_metrics: dict[str, float] | None = None,
    baseline_metrics: dict[str, float] | None = None,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "actor_state_dict": actor.state_dict(),
        "assist_critic_state_dict": assist_critic.state_dict(),
        "base_critic_state_dict": base_critic.state_dict(),
        "critic_state_dict": assist_critic.state_dict(),
        "config": build_config_payload(cfg),
        "eval_metrics": {} if eval_metrics is None else eval_metrics,
        "baseline_metrics": {} if baseline_metrics is None else baseline_metrics,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "state_mean": state_stats.mean,
        "state_std": state_stats.std,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
    }
    if extra_payload:
        payload.update(extra_payload)
    return payload


def compute_critic_loss(
    q1: torch.Tensor,
    q2: torch.Tensor,
    target_q: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)


def perform_update_step(
    *,
    offline_rb: ReplayBuffer,
    online_rb: ReplayBuffer,
    cfg: TrainConfig,
    device: torch.device,
    state_normalizer: TensorNormalizer | None,
    reward_normalizer: ScalarNormalizer | None,
    actor: MLPActor,
    assist_critic: MLPCritic,
    base_critic: MLPCritic,
    actor_target: MLPActor,
    assist_critic_target: MLPCritic,
    base_critic_target: MLPCritic,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    tracker: MetricTracker,
    update_actor: bool,
) -> int:
    batch, online_fraction = sample_mixed_batch(
        offline_rb,
        online_rb,
        cfg.batch_size,
        cfg.offline_fraction,
    )
    prepared = prepare_batch(
        batch,
        device=device,
        state_normalizer=state_normalizer,
        reward_normalizer=reward_normalizer,
    )
    obs_flat = prepared["obs_flat"]
    next_obs_flat = prepared["next_obs_flat"]
    base_action = prepared["base_action"]
    next_base_action = prepared["next_base_action"]
    behavior_residual = prepared["behavior_residual"]
    assist_action = prepared["assist_action"]
    reward_env = prepared["reward_env"]
    done = prepared["done"]
    is_offline = prepared["is_offline"]
    is_teacher = prepared["is_teacher"]

    with torch.no_grad():
        next_residual = actor_target(next_obs_flat)
        if cfg.target_policy_noise > 0.0:
            smoothing_noise = torch.randn_like(next_residual) * cfg.target_policy_noise
            if cfg.target_noise_clip > 0.0:
                smoothing_noise = torch.clamp(smoothing_noise, -cfg.target_noise_clip, cfg.target_noise_clip)
            next_residual = next_residual + smoothing_noise
        next_residual = clip_residual_tensor(next_residual, cfg.residual_limit)
        next_commanded_action = clip_commanded_action_tensor(next_base_action + next_residual)
        assist_target_q = reward_env + cfg.gamma * (1.0 - done) * assist_critic_target.min_q(next_obs_flat, next_commanded_action)
        base_target_q = reward_env + cfg.gamma * (1.0 - done) * base_critic_target.min_q(next_obs_flat, next_base_action)

    assist_q1, assist_q2 = assist_critic(obs_flat, assist_action)
    assist_loss = compute_critic_loss(assist_q1, assist_q2, assist_target_q)

    base_q1, base_q2 = base_critic(obs_flat, base_action)
    base_loss = compute_critic_loss(base_q1, base_q2, base_target_q)

    critic_loss = assist_loss + base_loss
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    actor_metrics: dict[str, float] = {}
    actor_loss_value = float("nan")
    if update_actor:
        pred_residual = clip_residual_tensor(actor(obs_flat), cfg.residual_limit)
        pred_commanded_action = clip_commanded_action_tensor(base_action + pred_residual)
        actor_loss, actor_metrics = compute_actor_loss(
            assist_critic,
            base_critic,
            obs_flat,
            base_action,
            pred_residual,
            pred_commanded_action,
            behavior_residual,
            is_offline,
            is_teacher,
            cfg.residual_limit,
            cfg.residual_mag_reg_weight,
            cfg.offline_zero_bc_weight,
            cfg.teacher_bc_weight,
        )
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()
        soft_update(actor_target, actor, cfg.tau)
        actor_loss_value = float(actor_loss.item())

    soft_update(assist_critic_target, assist_critic, cfg.tau)
    soft_update(base_critic_target, base_critic, cfg.tau)

    with torch.no_grad():
        logged_assist_q = assist_critic.min_q(obs_flat, assist_action)
        logged_base_q = base_critic.min_q(obs_flat, base_action)
        logged_help_adv = logged_assist_q - logged_base_q
        td1 = assist_q1 - assist_target_q
        td2 = assist_q2 - assist_target_q

    tracker.update(
        critic_loss=float(critic_loss.item()),
        assist_critic_loss=float(assist_loss.item()),
        base_critic_loss=float(base_loss.item()),
        assist_q1=float(assist_q1.mean().item()),
        assist_q2=float(assist_q2.mean().item()),
        assist_target_q=float(assist_target_q.mean().item()),
        base_q=float(logged_base_q.mean().item()),
        base_target_q=float(base_target_q.mean().item()),
        logged_help_adv=float(logged_help_adv.mean().item()),
        td_abs_mean=float(0.5 * (td1.abs().mean().item() + td2.abs().mean().item())),
        td_abs_max=float(torch.maximum(td1.abs().max(), td2.abs().max()).item()),
        q_gap=float((assist_q1 - assist_q2).abs().mean().item()),
        reward=float(reward_env.mean().item()),
        online_fraction=online_fraction,
        offline_fraction=float(is_offline.mean().item()),
        teacher_fraction=float(is_teacher.mean().item()),
    )
    if not np.isnan(actor_loss_value):
        tracker.update(actor_loss=actor_loss_value)
    if actor_metrics:
        tracker.update(**actor_metrics)
    return 1


class RealtimeResidualTrainer:
    """Incremental trainer used by live teleoperation collection."""

    def __init__(
        self,
        *,
        cfg: TrainConfig,
        obs_example: dict[str, np.ndarray],
        action_dim: int,
        device: torch.device | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = int(action_dim)
        self.obs_dim = infer_policy_obs_dim(
            obs_example,
            action_dim=self.action_dim,
            obs_mode=cfg.obs_mode,
            history_len=cfg.history_len,
            translation_step=cfg.translation_step,
            rotation_step=cfg.rotation_step,
            gripper_step=cfg.gripper_step,
        )

        self.offline_rb = ReplayBuffer(cfg.buffer_capacity)
        self.online_rb = ReplayBuffer(cfg.buffer_capacity)
        self.offline_episodes: list[EpisodeRollout] = []
        self.online_episodes: list[EpisodeRollout] = []
        self.state_stats = NormalizationStats(
            mean=np.zeros(self.obs_dim, dtype=np.float32),
            std=np.ones(self.obs_dim, dtype=np.float32),
        )
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.state_normalizer: TensorNormalizer | None = None
        self.reward_normalizer: ScalarNormalizer | None = None
        self.update_steps = 0
        self.last_train_metrics: dict[str, float] = {}

        set_seed(cfg.seed)

        self.actor = MLPActor(
            self.obs_dim,
            self.action_dim,
            hidden_dim=cfg.hidden_dim,
            residual_limit=cfg.residual_limit,
        ).to(self.device)
        self.assist_critic = MLPCritic(
            self.obs_dim,
            self.action_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)
        self.base_critic = MLPCritic(
            self.obs_dim,
            self.action_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)
        self.actor_target = MLPActor(
            self.obs_dim,
            self.action_dim,
            hidden_dim=cfg.hidden_dim,
            residual_limit=cfg.residual_limit,
        ).to(self.device)
        self.assist_critic_target = MLPCritic(
            self.obs_dim,
            self.action_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)
        self.base_critic_target = MLPCritic(
            self.obs_dim,
            self.action_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.assist_critic_target.load_state_dict(self.assist_critic.state_dict())
        self.base_critic_target.load_state_dict(self.base_critic.state_dict())

        self.init_checkpoint_payload = maybe_load_init_checkpoint(
            cfg.init_checkpoint,
            actor=self.actor,
            assist_critic=self.assist_critic,
            base_critic=self.base_critic,
            actor_target=self.actor_target,
            assist_critic_target=self.assist_critic_target,
            base_critic_target=self.base_critic_target,
        )

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(
            chain(self.assist_critic.parameters(), self.base_critic.parameters()),
            lr=cfg.critic_lr,
        )

        self.offline_episodes = load_dataset_into_buffer(
            cfg.dataset_dirs,
            self.offline_rb,
            cfg,
            allow_empty=True,
        )
        self._refresh_normalizers()

    @property
    def total_transitions(self) -> int:
        return len(self.offline_rb) + len(self.online_rb)

    def _refresh_normalizers(self) -> None:
        if self.total_transitions > 0:
            self.state_stats = compute_state_stats_from_buffers(self.offline_rb, self.online_rb)
            reward_episodes = [*self.offline_episodes, *self.online_episodes]
            self.reward_mean, self.reward_std = compute_reward_stats(reward_episodes)
        elif self.init_checkpoint_payload is not None:
            payload = self.init_checkpoint_payload
            state_mean = np.asarray(payload.get("state_mean", np.zeros(self.obs_dim, dtype=np.float32)), dtype=np.float32).reshape(-1)
            state_std = np.asarray(payload.get("state_std", np.ones(self.obs_dim, dtype=np.float32)), dtype=np.float32).reshape(-1)
            if state_mean.shape[0] != self.obs_dim or state_std.shape[0] != self.obs_dim:
                state_mean = np.zeros(self.obs_dim, dtype=np.float32)
                state_std = np.ones(self.obs_dim, dtype=np.float32)
            self.state_stats = NormalizationStats(
                mean=state_mean.astype(np.float32),
                std=np.clip(state_std.astype(np.float32), 1e-3, None),
            )
            self.reward_mean = float(payload.get("reward_mean", 0.0))
            self.reward_std = max(float(payload.get("reward_std", 1.0)), 1e-3)
        else:
            self.state_stats = NormalizationStats(
                mean=np.zeros(self.obs_dim, dtype=np.float32),
                std=np.ones(self.obs_dim, dtype=np.float32),
            )
            self.reward_mean = 0.0
            self.reward_std = 1.0

        self.state_normalizer = (
            TensorNormalizer(self.state_stats.mean, self.state_stats.std, self.device)
            if self.cfg.normalize_state
            else None
        )
        self.reward_normalizer = (
            ScalarNormalizer(self.reward_mean, self.reward_std)
            if self.cfg.normalize_reward
            else None
        )

    def build_policy(self, *, noise_scale: float = 0.0) -> TorchResidualPolicy:
        actor_snapshot = MLPActor(
            self.obs_dim,
            self.action_dim,
            hidden_dim=self.cfg.hidden_dim,
            residual_limit=self.cfg.residual_limit,
        ).to(self.device)
        actor_snapshot.load_state_dict(self.actor.state_dict())
        actor_snapshot.eval()
        return TorchResidualPolicy(
            actor_snapshot,
            self.device,
            noise_scale=noise_scale,
            residual_limit=self.cfg.residual_limit,
            state_normalizer=self.state_normalizer,
            history_len=self.cfg.history_len,
            translation_step=self.cfg.translation_step,
            rotation_step=self.cfg.rotation_step,
            gripper_step=self.cfg.gripper_step,
            obs_mode=self.cfg.obs_mode,
        )

    def ingest_rollout(self, rollout: EpisodeRollout, *, is_teacher: bool = False, as_online: bool = True) -> None:
        target_buffer = self.online_rb if as_online else self.offline_rb
        target_episodes = self.online_episodes if as_online else self.offline_episodes
        push_records_to_buffer(
            target_buffer,
            rollout,
            history_len=self.cfg.history_len,
            obs_mode=self.cfg.obs_mode,
            translation_step=self.cfg.translation_step,
            rotation_step=self.cfg.rotation_step,
            gripper_step=self.cfg.gripper_step,
            is_teacher=is_teacher,
        )
        target_episodes.append(rollout)
        self._refresh_normalizers()

    @property
    def critic_warmup_done(self) -> bool:
        """Whether the critic-only warmup phase has been completed."""
        return self.update_steps >= self.cfg.critic_warmup_steps

    def train(
        self,
        *,
        num_updates: int,
        actor_updates: int | None = None,
    ) -> dict[str, float]:
        if num_updates <= 0 or self.total_transitions <= 0:
            return {}

        actor_updates = num_updates if actor_updates is None else max(0, min(int(actor_updates), int(num_updates)))

        # --- Critic Warmup Gate ---
        # If the critic has not been warmed up yet, force actor_updates=0
        # so that only the critic is trained until Q-values are reliable.
        if not self.critic_warmup_done:
            remaining_warmup = self.cfg.critic_warmup_steps - self.update_steps
            if remaining_warmup > 0:
                print(
                    f"[realtime] Critic warmup in progress: "
                    f"{self.update_steps}/{self.cfg.critic_warmup_steps} "
                    f"(forcing actor_updates=0)"
                )
            actor_updates = 0

        tracker = MetricTracker()
        last_summary: dict[str, float] = {}
        warmup_crossed = False
        for update_idx in range(int(num_updates)):
            # Check if we just crossed the warmup boundary mid-batch
            if not warmup_crossed and not self.critic_warmup_done:
                do_actor = False
            elif not warmup_crossed and self.critic_warmup_done:
                # Already past warmup from a previous call; use normal schedule
                do_actor = update_idx < actor_updates
            else:
                # Warmup was crossed during this batch; enable actor for remaining
                do_actor = True

            self.update_steps += perform_update_step(
                offline_rb=self.offline_rb,
                online_rb=self.online_rb,
                cfg=self.cfg,
                device=self.device,
                state_normalizer=self.state_normalizer,
                reward_normalizer=self.reward_normalizer,
                actor=self.actor,
                assist_critic=self.assist_critic,
                base_critic=self.base_critic,
                actor_target=self.actor_target,
                assist_critic_target=self.assist_critic_target,
                base_critic_target=self.base_critic_target,
                actor_opt=self.actor_opt,
                critic_opt=self.critic_opt,
                tracker=tracker,
                update_actor=do_actor,
            )

            # Detect the warmup→main transition within this batch
            if not warmup_crossed and self.critic_warmup_done:
                warmup_crossed = True
                print(
                    f"[realtime] Critic warmup completed at update_step={self.update_steps}. "
                    f"Actor updates now enabled."
                )

            if self.cfg.log_interval > 0 and self.update_steps % self.cfg.log_interval == 0:
                last_summary = tracker.summary()
                phase = "warmup" if not self.critic_warmup_done else "train"
                print(format_metrics(f"[realtime {phase} step {self.update_steps}]", last_summary))
                tracker.reset()

        tail_summary = tracker.summary()
        self.last_train_metrics = tail_summary or last_summary
        return self.last_train_metrics

    def save_checkpoint(
        self,
        *,
        extra_payload: dict[str, Any] | None = None,
        path: Path | None = None,
    ) -> Path:
        target = (path or self.cfg.save_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = build_checkpoint_payload(
            cfg=self.cfg,
            actor=self.actor,
            assist_critic=self.assist_critic,
            base_critic=self.base_critic,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            state_stats=self.state_stats,
            reward_mean=self.reward_mean,
            reward_std=self.reward_std,
            eval_metrics={},
            baseline_metrics={},
            extra_payload=extra_payload,
        )
        torch.save(payload, target)
        return target


def main(args: argparse.Namespace) -> None:
    if args.dataset_dirs is None and args.dataset_dir is None:
        raise FileNotFoundError("Provide --dataset_dir or --dataset_dirs")
    dataset_dirs = tuple(path.resolve() for path in (args.dataset_dirs or [args.dataset_dir]))
    if not (0.0 <= args.offline_fraction <= 1.0):
        raise ValueError(f"offline_fraction must be in [0, 1], got {args.offline_fraction}")
    if args.learning_starts < 0:
        raise ValueError(f"learning_starts must be >= 0, got {args.learning_starts}")
    if args.critic_warmup_steps < 0:
        raise ValueError(f"critic_warmup_steps must be >= 0, got {args.critic_warmup_steps}")
    if args.total_timesteps < 0:
        raise ValueError(f"total_timesteps must be >= 0, got {args.total_timesteps}")
    if args.update_every_n_steps <= 0:
        raise ValueError(f"update_every_n_steps must be > 0, got {args.update_every_n_steps}")
    if args.num_updates_per_iteration < 0:
        raise ValueError(f"num_updates_per_iteration must be >= 0, got {args.num_updates_per_iteration}")
    if args.actor_updates_per_iteration < 0:
        raise ValueError(f"actor_updates_per_iteration must be >= 0, got {args.actor_updates_per_iteration}")
    if args.actor_updates_per_iteration > args.num_updates_per_iteration:
        raise ValueError(
            "actor_updates_per_iteration must be <= num_updates_per_iteration, "
            f"got {args.actor_updates_per_iteration} > {args.num_updates_per_iteration}"
        )

    cfg = TrainConfig(
        dataset_dirs=dataset_dirs,
        save_path=args.save_path.resolve(),
        xml_path=str(args.xml.resolve()),
        init_checkpoint=args.init_checkpoint.resolve() if args.init_checkpoint is not None else None,
        seed=args.seed,
        fixed_scene=getattr(args, "fixed_scene", False),
        fixed_box_xy=normalize_fixed_box_xy(getattr(args, "fixed_box_xy", None)),
        batch_size=args.batch_size,
        offline_fraction=args.offline_fraction,
        learning_starts=args.learning_starts,
        critic_warmup_steps=args.critic_warmup_steps,
        total_timesteps=args.total_timesteps,
        update_every_n_steps=args.update_every_n_steps,
        num_updates_per_iteration=args.num_updates_per_iteration,
        actor_updates_per_iteration=args.actor_updates_per_iteration,
        eval_episodes=args.eval_episodes,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        residual_limit=args.residual_limit,
        residual_mag_reg_weight=args.residual_mag_reg_weight,
        offline_zero_bc_weight=args.offline_zero_bc_weight,
        teacher_bc_weight=args.teacher_bc_weight,
        teacher_rollout_steps=args.teacher_rollout_steps,
        target_policy_noise=args.target_policy_noise,
        target_noise_clip=args.target_noise_clip,
        exploration_noise=args.exploration_noise,
        warmup_policy=args.warmup_policy,
        warmup_residual_noise=args.warmup_residual_noise,
        normalize_state=not args.disable_state_norm,
        normalize_reward=not args.disable_reward_norm,
        obs_mode=getattr(args, "obs_mode", TELEOP_RESIDUAL_OBS_MODE),
        history_len=getattr(args, "history_len", 4),
        log_interval=args.log_interval,
    )
    set_seed(cfg.seed)

    reward_model = PickPlaceReward()
    env = MujocoPickPlaceTeleopEnv(
        build_pick_place_task_config(
            cfg.xml_path,
            seed=cfg.seed,
            fixed_scene=cfg.fixed_scene,
            fixed_box_xy=cfg.fixed_box_xy,
        ),
        reward=reward_model,
    )
    obs, _ = env.reset(seed=cfg.seed)
    action_dim = env.action_dim
    obs_dim = infer_policy_obs_dim(
        obs,
        action_dim=action_dim,
        obs_mode=cfg.obs_mode,
        history_len=cfg.history_len,
        translation_step=env.cfg.translation_step,
        rotation_step=env.cfg.rotation_step,
        gripper_step=env.cfg.gripper_step,
    )
    cfg.translation_step = float(env.cfg.translation_step)
    cfg.rotation_step = float(env.cfg.rotation_step)
    cfg.gripper_step = float(env.cfg.gripper_step)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    offline_rb = ReplayBuffer(cfg.buffer_capacity)
    online_rb = ReplayBuffer(cfg.buffer_capacity)
    playback_episodes = load_dataset_into_buffer(cfg.dataset_dirs, offline_rb, cfg)
    state_stats = compute_state_stats(offline_rb)
    reward_mean, reward_std = compute_reward_stats(playback_episodes)
    state_normalizer = TensorNormalizer(state_stats.mean, state_stats.std, device) if cfg.normalize_state else None
    reward_normalizer = ScalarNormalizer(reward_mean, reward_std) if cfg.normalize_reward else None

    print(
        format_metrics(
            "Dataset stats",
            {
                "episodes": float(len(playback_episodes)),
                "offline_transitions": float(len(offline_rb)),
                "state_std_mean": float(np.mean(state_stats.std)),
                "reward_mean": reward_mean,
                "reward_std": reward_std,
            },
        )
    )
    print(
        f"offline_fraction={cfg.offline_fraction:.2f} learning_starts={cfg.learning_starts} "
        f"critic_warmup_steps={cfg.critic_warmup_steps} total_timesteps={cfg.total_timesteps} "
        f"update_every_n_steps={cfg.update_every_n_steps} num_updates_per_iteration={cfg.num_updates_per_iteration} "
        f"actor_updates_per_iteration={cfg.actor_updates_per_iteration} "
        f"residual_mag_reg_weight={cfg.residual_mag_reg_weight} "
        f"offline_zero_bc_weight={cfg.offline_zero_bc_weight} "
        f"teacher_bc_weight={cfg.teacher_bc_weight} "
        f"teacher_rollout_steps={cfg.teacher_rollout_steps} "
        f"target_policy_noise={cfg.target_policy_noise} "
        f"target_noise_clip={cfg.target_noise_clip} "
        f"exploration_noise={cfg.exploration_noise} "
        f"warmup_policy={cfg.warmup_policy} "
        f"warmup_residual_noise={cfg.warmup_residual_noise} "
        f"state_norm={cfg.normalize_state} reward_norm={cfg.normalize_reward} "
        f"obs_mode={cfg.obs_mode} residual_history_len={cfg.history_len}"
    )

    actor = MLPActor(
        obs_dim,
        action_dim,
        hidden_dim=cfg.hidden_dim,
        residual_limit=cfg.residual_limit,
    ).to(device)
    assist_critic = MLPCritic(
        obs_dim,
        action_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(device)
    base_critic = MLPCritic(
        obs_dim,
        action_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(device)
    actor_target = MLPActor(
        obs_dim,
        action_dim,
        hidden_dim=cfg.hidden_dim,
        residual_limit=cfg.residual_limit,
    ).to(device)
    assist_critic_target = MLPCritic(
        obs_dim,
        action_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(device)
    base_critic_target = MLPCritic(
        obs_dim,
        action_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(device)
    actor_target.load_state_dict(actor.state_dict())
    assist_critic_target.load_state_dict(assist_critic.state_dict())
    base_critic_target.load_state_dict(base_critic.state_dict())
    maybe_load_init_checkpoint(
        cfg.init_checkpoint,
        actor=actor,
        assist_critic=assist_critic,
        base_critic=base_critic,
        actor_target=actor_target,
        assist_critic_target=assist_critic_target,
        base_critic_target=base_critic_target,
    )

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(chain(assist_critic.parameters(), base_critic.parameters()), lr=cfg.critic_lr)

    tracker = MetricTracker()
    update_steps = 0
    online_steps = 0
    rollout_steps_since_update = 0

    if cfg.warmup_policy == "heuristic":
        warmup_policy = HeuristicBootstrapResidualPolicy(
            residual_limit=cfg.residual_limit,
        )
        warmup_is_teacher = True
    elif cfg.warmup_policy == "noise":
        warmup_policy = RandomResidualPolicy(
            residual_limit=cfg.residual_limit,
            noise_scale=cfg.warmup_residual_noise,
        )
        warmup_is_teacher = False
    else:
        warmup_policy = ZeroResidualPolicy()
        warmup_is_teacher = False
    warmup_episode_idx = 0
    while len(online_rb) < cfg.learning_starts:
        demo_rollout = playback_episodes[warmup_episode_idx % len(playback_episodes)]
        rollout = collect_episode(
            env,
            demo_rollout,
            warmup_policy,
            seed=cfg.seed + warmup_episode_idx,
        )
        push_records_to_buffer(
            online_rb,
            rollout,
            cfg.history_len,
            cfg.obs_mode,
            cfg.translation_step,
            cfg.rotation_step,
            cfg.gripper_step,
            is_teacher=warmup_is_teacher,
        )
        online_steps += rollout.steps
        warmup_episode_idx += 1
        print(
            format_metrics(
                f"[online warmup {warmup_episode_idx}]",
                {
                    "steps": float(rollout.steps),
                    "success": float(rollout.success),
                    "online_size": float(len(online_rb)),
                    "return_env": rollout.episode_return_env,
                    "teacher": float(warmup_is_teacher),
                },
            )
        )

    for warmup_step in range(cfg.critic_warmup_steps):
        update_steps += perform_update_step(
            offline_rb=offline_rb,
            online_rb=online_rb,
            cfg=cfg,
            device=device,
            state_normalizer=state_normalizer,
            reward_normalizer=reward_normalizer,
            actor=actor,
            assist_critic=assist_critic,
            base_critic=base_critic,
            actor_target=actor_target,
            assist_critic_target=assist_critic_target,
            base_critic_target=base_critic_target,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            tracker=tracker,
            update_actor=False,
        )
        if update_steps % cfg.log_interval == 0:
            print(format_metrics(f"[critic warmup step {warmup_step + 1}]", tracker.summary()))
            tracker.reset()

    rollout_episode_idx = 0
    while online_steps < cfg.total_timesteps:
        demo_rollout = playback_episodes[rollout_episode_idx % len(playback_episodes)]
        use_teacher_rollout = cfg.warmup_policy == "heuristic" and online_steps < cfg.teacher_rollout_steps
        if use_teacher_rollout:
            rollout_policy = HeuristicBootstrapResidualPolicy(
                residual_limit=cfg.residual_limit,
            )
        else:
            rollout_policy = TorchResidualPolicy(
                actor,
                device,
                noise_scale=cfg.exploration_noise,
                residual_limit=cfg.residual_limit,
                state_normalizer=state_normalizer,
                history_len=cfg.history_len,
                translation_step=cfg.translation_step,
                rotation_step=cfg.rotation_step,
                gripper_step=cfg.gripper_step,
                obs_mode=cfg.obs_mode,
            )
        rollout = collect_episode(
            env,
            demo_rollout,
            rollout_policy,
            seed=cfg.seed + 10_000 + rollout_episode_idx,
        )
        push_records_to_buffer(
            online_rb,
            rollout,
            history_len=cfg.history_len,
            obs_mode=cfg.obs_mode,
            translation_step=cfg.translation_step,
            rotation_step=cfg.rotation_step,
            gripper_step=cfg.gripper_step,
            is_teacher=use_teacher_rollout,
        )
        rollout_episode_idx += 1
        online_steps += rollout.steps
        rollout_steps_since_update += rollout.steps

        print(
            format_metrics(
                f"[rollout {rollout_episode_idx}]",
                {
                    "steps": float(rollout.steps),
                    "success": float(rollout.success),
                    "online_size": float(len(online_rb)),
                    "online_steps": float(online_steps),
                    "return_env": rollout.episode_return_env,
                    "teacher": float(use_teacher_rollout),
                },
            )
        )

        while rollout_steps_since_update >= cfg.update_every_n_steps:
            rollout_steps_since_update -= cfg.update_every_n_steps
            for update_idx in range(cfg.num_updates_per_iteration):
                update_steps += perform_update_step(
                    offline_rb=offline_rb,
                    online_rb=online_rb,
                    cfg=cfg,
                    device=device,
                    state_normalizer=state_normalizer,
                    reward_normalizer=reward_normalizer,
                    actor=actor,
                    assist_critic=assist_critic,
                    base_critic=base_critic,
                    actor_target=actor_target,
                    assist_critic_target=assist_critic_target,
                    base_critic_target=base_critic_target,
                    actor_opt=actor_opt,
                    critic_opt=critic_opt,
                    tracker=tracker,
                    update_actor=update_idx < cfg.actor_updates_per_iteration,
                )
                if update_steps % cfg.log_interval == 0:
                    print(format_metrics(f"[update step {update_steps}]", tracker.summary()))
                    tracker.reset()

    eval_metrics = evaluate_policy(
        env,
        playback_episodes,
        TorchResidualPolicy(
            actor,
            device,
            residual_limit=cfg.residual_limit,
            state_normalizer=state_normalizer,
            history_len=cfg.history_len,
            translation_step=cfg.translation_step,
            rotation_step=cfg.rotation_step,
            gripper_step=cfg.gripper_step,
            obs_mode=cfg.obs_mode,
        ),
        eval_episodes=cfg.eval_episodes,
    )
    baseline_metrics = evaluate_policy(
        env,
        playback_episodes,
        ZeroResidualPolicy(),
        eval_episodes=cfg.eval_episodes,
    )

    cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_checkpoint_payload(
        cfg=cfg,
        actor=actor,
        assist_critic=assist_critic,
        base_critic=base_critic,
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_stats=state_stats,
        reward_mean=reward_mean,
        reward_std=reward_std,
        eval_metrics=eval_metrics,
        baseline_metrics=baseline_metrics,
    )
    torch.save(payload, cfg.save_path)
    print("Saved checkpoint to", cfg.save_path)
    print("Baseline metrics:", baseline_metrics)
    print("Assisted metrics:", eval_metrics)


if __name__ == "__main__":
    default_xml = PACKAGE_ROOT / "assets" / "mujoco" / "Alicia_D_v5_6" / "gripper_50mm" / "alicia_d_follower.xml"
    default_save = PROJECT_ROOT / "logs" / "residual_td3_teleop.pt"
    parser = argparse.ArgumentParser(description="Train a teleoperation residual TD3 copilot in MuJoCo.")
    parser.add_argument("--dataset_dir", type=Path, default=None, help="Single dataset directory containing episode_*.npz rollouts")
    parser.add_argument("--dataset_dirs", type=Path, nargs="+", default=None, help="One or more dataset directories to load jointly")
    parser.add_argument("--xml", type=Path, default=default_xml, help="MuJoCo XML path")
    parser.add_argument("--save_path", type=Path, default=default_save, help="Where to save the checkpoint")
    parser.add_argument("--init_checkpoint", type=Path, default=None, help="Optional checkpoint used to initialize critic/actor weights")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed_scene", action="store_true", help="Disable box/basket randomization for rollout and evaluation environments")
    parser.add_argument(
        "--fixed_box_xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Optional fixed box XY used by rollout and evaluation environments",
    )
    parser.add_argument(
        "--obs_mode",
        type=str,
        default=TELEOP_RESIDUAL_OBS_MODE,
        choices=(TELEOP_RESIDUAL_OBS_MODE,),
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--offline_fraction", type=float, default=0.5)
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--critic_warmup_steps", type=int, default=1000)
    parser.add_argument("--total_timesteps", type=int, default=10_000)
    parser.add_argument("--update_every_n_steps", type=int, default=200)
    parser.add_argument("--num_updates_per_iteration", type=int, default=200)
    parser.add_argument("--actor_updates_per_iteration", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=0, help="How many playback episodes to evaluate; <=0 means all")
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--residual_limit", type=float, default=0.1)
    parser.add_argument("--residual_mag_reg_weight", type=float, default=1.0)
    parser.add_argument("--offline_zero_bc_weight", type=float, default=5.0)
    parser.add_argument("--teacher_bc_weight", type=float, default=2.0)
    parser.add_argument("--teacher_rollout_steps", type=int, default=4000)
    parser.add_argument("--target_policy_noise", type=float, default=0.01)
    parser.add_argument("--target_noise_clip", type=float, default=0.02)
    parser.add_argument("--exploration_noise", type=float, default=0.005)
    parser.add_argument("--warmup_policy", type=str, default="heuristic", choices=("heuristic", "zero", "noise"))
    parser.add_argument("--warmup_residual_noise", type=float, default=0.0)
    parser.add_argument("--history_len", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--disable_state_norm", action="store_true", help="Disable state standardization during training and policy rollout")
    parser.add_argument("--disable_reward_norm", action="store_true", help="Disable reward normalization in TD targets")
    parser.add_argument("--log_interval", type=int, default=200)
    main(parser.parse_args())
