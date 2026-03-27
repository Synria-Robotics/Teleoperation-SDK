from __future__ import annotations

import argparse
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from teleop_sdk.data import EpisodeRollout, Transition, load_episode_npz
from teleop_sdk.envs import (
    OBS_BASE_ACTION_KEY,
    OBS_STATE_KEY,
    MujocoPickPlaceTeleopEnv,
    PickPlaceTaskConfig,
    flatten_observation,
)
from teleop_sdk.providers import ZeroResidualPolicy
from teleop_sdk.rewards import PickPlaceReward


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, residual_limit: float = 0.35):
        super().__init__()
        self.residual_limit = residual_limit
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        final = self.net[-1]
        assert isinstance(final, nn.Linear)
        nn.init.uniform_(final.weight, -1e-3, 1e-3)
        nn.init.uniform_(final.bias, -1e-3, 1e-3)

    def forward(self, obs_flat: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs_flat, base_action], dim=-1)
        return torch.tanh(self.net(x)) * self.residual_limit


class MLPCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
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

    def forward(self, obs_flat: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs_flat, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, obs_flat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs_flat, action], dim=-1)
        return self.q1(x)


class TorchResidualPolicy:
    def __init__(
        self,
        actor: MLPActor,
        device: torch.device,
        *,
        noise_scale: float = 0.0,
        residual_limit: float = 0.35,
        idle_deadband: float = 0.05,
        idle_scale_span: float = 0.20,
        state_normalizer: TensorNormalizer | None = None,
    ):
        self.actor = actor
        self.device = device
        self.noise_scale = float(noise_scale)
        self.residual_limit = float(residual_limit)
        self.idle_deadband = float(idle_deadband)
        self.idle_scale_span = float(idle_scale_span)
        self.state_normalizer = state_normalizer

    def reset(self) -> None:
        pass

    def get_action(self, obs: dict[str, np.ndarray], base_action: np.ndarray) -> np.ndarray:
        obs_flat = torch.as_tensor(flatten_observation(obs), device=self.device).unsqueeze(0)
        if self.state_normalizer is not None:
            obs_flat = self.state_normalizer.normalize(obs_flat)
        base = torch.as_tensor(base_action.astype(np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            residual = self.actor(obs_flat, base).squeeze(0).cpu().numpy()
        if self.noise_scale > 0.0:
            residual = residual + np.random.normal(scale=self.noise_scale, size=residual.shape).astype(np.float32)
        residual = np.clip(residual, -self.residual_limit, self.residual_limit).astype(np.float32)
        return apply_idle_gate_np(residual, base_action, self.idle_deadband, self.idle_scale_span)


class RandomResidualPolicy:
    def __init__(self, residual_limit: float = 0.35):
        self.residual_limit = float(residual_limit)

    def reset(self) -> None:
        pass

    def get_action(self, obs: dict[str, np.ndarray], base_action: np.ndarray) -> np.ndarray:
        del obs, base_action
        return np.random.uniform(-self.residual_limit, self.residual_limit, size=7).astype(np.float32)


@dataclass(slots=True)
class TrainConfig:
    dataset_dirs: tuple[Path, ...]
    save_path: Path
    xml_path: str
    init_checkpoint: Path | None = None
    seed: int = 0
    alpha: float = 0.2
    gamma: float = 0.99
    reward_mode: str = "logged_total"
    tau: float = 0.005
    batch_size: int = 256
    offline_fraction: float = 0.5
    replay_mix_mode: str = "fixed_offline_fraction"
    offline_updates: int = 2000
    updates_per_episode: int = 300
    critic_warmup_steps: int = 1000
    baseline_episodes: int = 5
    assisted_episodes: int = 20
    eval_episodes: int = 5
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    critic_loss: str = "huber"
    buffer_capacity: int = 200_000
    residual_limit: float = 0.35
    exploration_noise: float = 0.05
    policy_delay: int = 2
    hidden_dim: int = 256
    zero_reg: float = 0.05
    conflict_reg: float = 0.10
    teleop_zero_reg: float = 0.50
    assisted_bc_reg: float = 0.10
    idle_zero_reg: float = 0.50
    idle_deadband: float = 0.05
    idle_scale_span: float = 0.20
    actor_bc_warmup_steps: int = 1000
    normalize_state: bool = True
    normalize_reward: bool = True
    filter_assist_by_success: bool = False
    filter_assist_min_cmd_delta: float = 0.0
    filter_assist_alpha: float | None = None
    filter_assist_reach_gain: float | None = None
    filter_assist_place_gain: float | None = None
    log_interval: int = 200


def clip_residual_tensor(residual: torch.Tensor, limit: float) -> torch.Tensor:
    return torch.clamp(residual, -limit, limit)


def clip_commanded_action_tensor(commanded_action: torch.Tensor) -> torch.Tensor:
    return torch.clamp(commanded_action, -1.0, 1.0)


def compute_idle_gate_tensor(base_action: torch.Tensor, deadband: float, scale_span: float) -> torch.Tensor:
    if deadband <= 0.0 and scale_span <= 0.0:
        return torch.ones(base_action.shape[0], 1, device=base_action.device, dtype=base_action.dtype)
    h_norm = base_action.norm(dim=-1, keepdim=True)
    if scale_span <= 1e-6:
        return (h_norm > deadband).to(dtype=base_action.dtype)
    return torch.clamp((h_norm - deadband) / max(scale_span, 1e-6), 0.0, 1.0)


def apply_idle_gate_tensor(residual: torch.Tensor, base_action: torch.Tensor, deadband: float, scale_span: float) -> torch.Tensor:
    return residual * compute_idle_gate_tensor(base_action, deadband, scale_span)


def apply_idle_gate_np(residual: np.ndarray, base_action: np.ndarray, deadband: float, scale_span: float) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float32)
    base_action = np.asarray(base_action, dtype=np.float32)
    if deadband <= 0.0 and scale_span <= 0.0:
        return residual.copy()
    h_norm = float(np.linalg.norm(base_action))
    if scale_span <= 1e-6:
        gate = 1.0 if h_norm > deadband else 0.0
    else:
        gate = float(np.clip((h_norm - deadband) / max(scale_span, 1e-6), 0.0, 1.0))
    return (gate * residual).astype(np.float32)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def conflict_metric_tensor(base_action: torch.Tensor, residual_action: torch.Tensor) -> torch.Tensor:
    h_norm = base_action.norm(dim=-1, keepdim=True)
    r_norm = residual_action.norm(dim=-1, keepdim=True)
    dot = (base_action * residual_action).sum(dim=-1, keepdim=True)
    denom = (h_norm * r_norm).clamp_min(1e-6)
    valid = (h_norm > 1e-6) & (r_norm > 1e-6)
    cosine = torch.where(valid, dot / denom, torch.zeros_like(dot))
    return torch.clamp(-cosine, min=0.0) * r_norm


def make_total_reward_fn(reward_model: PickPlaceReward):
    return reward_model.compute_total_reward


def infer_dataset_source(dataset_dir: Path) -> float:
    name = dataset_dir.name.lower()
    if "assist" in name or "copilot" in name:
        return 1.0
    return 0.0


def infer_transition_assisted_label(transition: Transition, rollout: EpisodeRollout) -> float:
    if transition.copilot_enabled:
        return 1.0
    collection_mode = str(rollout.meta.get("collection_mode", "")).lower()
    if "assist" in collection_mode or "copilot" in collection_mode:
        return 1.0
    return 0.0


def compute_state_stats(episodes: Sequence[EpisodeRollout]) -> NormalizationStats:
    states = [np.asarray(t.observation_state, dtype=np.float32) for rollout in episodes for t in rollout.transitions]
    if not states:
        raise RuntimeError("Cannot compute state normalization stats from an empty dataset")
    stacked = np.stack(states, axis=0)
    return NormalizationStats(
        mean=stacked.mean(axis=0).astype(np.float32),
        std=np.clip(stacked.std(axis=0), 1e-3, None).astype(np.float32),
    )


def compute_reward_stats(episodes: Sequence[EpisodeRollout], reward_mode: str) -> tuple[float, float]:
    rewards = np.asarray(
        [transition_training_reward(t, reward_mode) for rollout in episodes for t in rollout.transitions],
        dtype=np.float32,
    )
    if rewards.size == 0:
        raise RuntimeError("Cannot compute reward normalization stats from an empty dataset")
    return float(rewards.mean()), max(float(rewards.std()), 1e-3)


def transition_training_reward(transition: Transition, reward_mode: str) -> float:
    if reward_mode == "logged_total":
        return float(transition.reward_total)
    if reward_mode == "env_dense":
        return float(transition.reward_env)
    if reward_mode == "success_only":
        return float(transition.success)
    raise ValueError(f"Unknown reward mode: {reward_mode}")


def rollout_mean_command_delta(rollout: EpisodeRollout) -> float:
    if not rollout.transitions:
        return 0.0
    deltas = [
        float(np.linalg.norm(np.asarray(t.commanded_action, dtype=np.float32) - np.asarray(t.base_action, dtype=np.float32)))
        for t in rollout.transitions
    ]
    return float(np.mean(deltas)) if deltas else 0.0


def rollout_passes_filters(rollout: EpisodeRollout, cfg: TrainConfig) -> tuple[bool, str]:
    is_assisted_rollout = any(infer_transition_assisted_label(t, rollout) > 0.5 for t in rollout.transitions)
    if not is_assisted_rollout:
        return True, ""

    if cfg.filter_assist_by_success and not rollout.success:
        return False, "assist_not_success"

    if cfg.filter_assist_min_cmd_delta > 0.0:
        cmd_delta_mean = rollout_mean_command_delta(rollout)
        if cmd_delta_mean < cfg.filter_assist_min_cmd_delta:
            return False, f"assist_cmd_delta<{cfg.filter_assist_min_cmd_delta:.4f}"

    meta = rollout.meta
    if cfg.filter_assist_alpha is not None:
        if abs(float(meta.get("assist_alpha", -999.0)) - cfg.filter_assist_alpha) > 1e-6:
            return False, "assist_alpha_mismatch"
    if cfg.filter_assist_reach_gain is not None:
        if abs(float(meta.get("heuristic_reach_gain", -999.0)) - cfg.filter_assist_reach_gain) > 1e-6:
            return False, "assist_reach_mismatch"
    if cfg.filter_assist_place_gain is not None:
        if abs(float(meta.get("heuristic_place_gain", -999.0)) - cfg.filter_assist_place_gain) > 1e-6:
            return False, "assist_place_mismatch"
    return True, ""


def load_dataset_into_buffer(dataset_dirs: Sequence[Path], offline_rb: ReplayBuffer, cfg: TrainConfig) -> list[EpisodeRollout]:
    if not dataset_dirs:
        raise FileNotFoundError("No dataset directories were provided")

    playback_episodes: list[EpisodeRollout] = []
    total_explicit_base_target_episodes = 0
    total_transitions = 0
    skipped_rollouts: dict[str, int] = {}
    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
        episode_paths = sorted(dataset_dir.glob("episode_*.npz"))
        if not episode_paths:
            raise FileNotFoundError(f"No episode_*.npz files found under {dataset_dir}")

        explicit_base_target_episodes = 0
        dir_transitions = 0
        for path in episode_paths:
            rollout = load_episode_npz(path)
            keep_rollout, skip_reason = rollout_passes_filters(rollout, cfg)
            if not keep_rollout:
                skipped_rollouts[skip_reason] = skipped_rollouts.get(skip_reason, 0) + 1
                continue
            playback_episodes.append(rollout)
            dir_transitions += len(rollout.transitions)
            with np.load(path, allow_pickle=True) as data:
                if "base_joint_target" in data and "base_gripper_target" in data:
                    explicit_base_target_episodes += 1
            for transition in rollout.transitions:
                source_is_assisted = infer_transition_assisted_label(transition, rollout)
                offline_rb.add(
                    {
                        "observation_state": transition.observation_state,
                        "next_observation_state": transition.next_observation_state,
                        "base_action": transition.base_action,
                        "next_base_action": transition.next_base_action,
                        "residual_action": transition.residual_action,
                        "prev_residual_action": transition.prev_residual_action,
                        "commanded_action": transition.commanded_action,
                        "realized_action": transition.realized_action,
                        "reward_env": np.float32(transition.reward_env),
                        "reward_total": np.float32(transition.reward_total),
                        "reward_success": np.float32(float(transition.success)),
                        "done": np.float32(float(transition.terminated or transition.truncated)),
                        "is_assisted": np.float32(source_is_assisted),
                    }
                )
        total_explicit_base_target_episodes += explicit_base_target_episodes
        total_transitions += dir_transitions
        print(
            f"Loaded {len(episode_paths)} episodes from {dataset_dir}"
            f" ({explicit_base_target_episodes} with explicit absolute base targets,"
            f" {dir_transitions} transitions)"
        )
    if skipped_rollouts:
        print("Skipped rollouts:", ", ".join(f"{key}={value}" for key, value in sorted(skipped_rollouts.items())))
    print(
        f"Loaded {len(playback_episodes)} total episodes from {len(dataset_dirs)} dataset directories"
        f" ({total_explicit_base_target_episodes} with explicit absolute base targets,"
        f" {total_transitions} transitions)"
    )
    return playback_episodes


def maybe_load_init_checkpoint(
    checkpoint_path: Path | None,
    *,
    actor: MLPActor,
    critic: MLPCritic,
    actor_target: MLPActor,
    critic_target: MLPCritic,
) -> None:
    if checkpoint_path is None:
        return
    payload = torch.load(checkpoint_path.resolve(), map_location="cpu", weights_only=False)
    if "critic_state_dict" in payload:
        critic.load_state_dict(payload["critic_state_dict"])
        critic_target.load_state_dict(payload["critic_state_dict"])
    else:
        raise KeyError(f"Checkpoint does not contain critic_state_dict: {checkpoint_path}")
    if "actor_state_dict" in payload:
        actor.load_state_dict(payload["actor_state_dict"], strict=False)
        actor_target.load_state_dict(actor.state_dict())
    print(f"Initialized networks from checkpoint: {checkpoint_path}")


def push_records_to_buffer(buffer: ReplayBuffer, rollout) -> None:
    for transition in rollout.transitions:
        buffer.add(
            {
                "observation_state": transition.observation_state,
                "next_observation_state": transition.next_observation_state,
                "base_action": transition.base_action,
                "next_base_action": transition.next_base_action,
                "residual_action": transition.residual_action,
                "prev_residual_action": transition.prev_residual_action,
                "commanded_action": transition.commanded_action,
                "realized_action": transition.realized_action,
                "reward_env": np.float32(transition.reward_env),
                "reward_total": np.float32(transition.reward_total),
                "reward_success": np.float32(float(transition.success)),
                "done": np.float32(float(transition.terminated or transition.truncated)),
                "is_assisted": np.float32(float(transition.copilot_enabled)),
            }
        )


def sample_mixed_batch(
    offline_rb: ReplayBuffer,
    online_rb: ReplayBuffer,
    batch_size: int,
    offline_fraction: float,
    mixing_mode: str,
) -> dict[str, torch.Tensor]:
    if mixing_mode == "offline_only":
        if len(offline_rb) == 0:
            raise RuntimeError("Offline replay buffer is empty")
        return offline_rb.sample(batch_size)
    if mixing_mode == "online_only":
        if len(online_rb) == 0:
            raise RuntimeError("Online replay buffer is empty")
        return online_rb.sample(batch_size)
    if mixing_mode == "available_balanced":
        if len(online_rb) == 0:
            return offline_rb.sample(batch_size)
        offline_bs = batch_size // 2
        online_bs = batch_size - offline_bs
    elif mixing_mode == "fixed_offline_fraction":
        online_bs = min(len(online_rb), int(round(batch_size * (1.0 - offline_fraction))))
        offline_bs = batch_size - online_bs
    else:
        raise ValueError(f"Unknown replay mixing mode: {mixing_mode}")

    if offline_bs > 0 and len(offline_rb) == 0:
        raise RuntimeError("Offline replay buffer is empty")
    if online_bs == 0:
        return offline_rb.sample(batch_size)
    if offline_bs == 0:
        return online_rb.sample(batch_size)

    offline_batch = offline_rb.sample(offline_bs)
    online_batch = online_rb.sample(online_bs)
    merged: dict[str, torch.Tensor] = {}
    for key in offline_batch:
        merged[key] = torch.cat([offline_batch[key], online_batch[key]], dim=0)
    return merged


def prepare_batch(
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
    state_normalizer: TensorNormalizer | None,
    reward_normalizer: ScalarNormalizer | None,
    reward_mode: str,
) -> dict[str, torch.Tensor]:
    obs_flat = batch["observation_state"].to(device)
    next_obs_flat = batch["next_observation_state"].to(device)
    if state_normalizer is not None:
        obs_flat = state_normalizer.normalize(obs_flat)
        next_obs_flat = state_normalizer.normalize(next_obs_flat)

    reward_key = {
        "logged_total": "reward_total",
        "env_dense": "reward_env",
        "success_only": "reward_success",
    }.get(reward_mode)
    if reward_key is None:
        raise ValueError(f"Unknown reward mode: {reward_mode}")

    reward_value = batch[reward_key].to(device).unsqueeze(-1)
    if reward_normalizer is not None:
        reward_value = reward_normalizer.normalize_tensor(reward_value)

    return {
        "obs_flat": obs_flat,
        "next_obs_flat": next_obs_flat,
        "base_action": batch["base_action"].to(device),
        "next_base_action": batch["next_base_action"].to(device),
        "commanded_action": batch["commanded_action"].to(device),
        "logged_residual": batch["residual_action"].to(device),
        "reward_total": reward_value,
        "done": batch["done"].to(device).unsqueeze(-1),
        "is_assisted": batch["is_assisted"].to(device).unsqueeze(-1),
    }


def collect_episode(
    env: MujocoPickPlaceTeleopEnv,
    demo_rollout: EpisodeRollout,
    residual_policy,
    reward_model: PickPlaceReward,
    *,
    alpha: float,
    seed: int | None = None,
):
    del seed
    if not demo_rollout.transitions:
        return EpisodeRollout(
            transitions=[],
            obs_keys=demo_rollout.obs_keys,
            success=False,
            episode_return_env=0.0,
            episode_return_total=0.0,
            meta={"source": "empty_demo"},
        )

    obs, _ = env.reset_from_flat_observation(demo_rollout.transitions[0].obs_flat)
    residual_policy.reset()
    obs_keys = tuple(obs.keys())
    transitions: list[Transition] = []
    prev_residual = np.zeros(env.action_dim, dtype=np.float32)
    reward_env_sum = 0.0
    reward_total_sum = 0.0

    demo_actions = [np.asarray(t.base_action, dtype=np.float32) for t in demo_rollout.transitions]
    copilot_enabled = not isinstance(residual_policy, ZeroResidualPolicy)
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
            alpha=alpha,
            base_action=base_action,
        )
        next_base_action = demo_actions[idx + 1].copy() if idx + 1 < len(demo_actions) else np.zeros_like(base_action)
        reward_total, extra_terms = reward_model.compute_total_reward(
            reward_env=reward_env,
            base_action=base_action,
            residual_action=residual_action,
            prev_residual_action=prev_residual,
        )
        transitions.append(
            Transition(
                observation_state=flatten_observation(obs_copy),
                next_observation_state=flatten_observation(next_obs),
                base_action=np.asarray(info.get("base_action", base_action), dtype=np.float32).copy(),
                next_base_action=next_base_action.copy(),
                residual_action=residual_action.copy(),
                prev_residual_action=prev_residual.copy(),
                commanded_action=np.asarray(info.get("commanded_action", info["realized_action"]), dtype=np.float32).copy(),
                realized_action=np.asarray(info["realized_action"], dtype=np.float32).copy(),
                reward_env=float(reward_env),
                reward_total=float(reward_total),
                terminated=bool(terminated),
                truncated=bool(truncated),
                base_joint_target=base_joint_target.copy(),
                base_gripper_target=np.asarray([base_gripper_target], dtype=np.float32),
                teleop_active=True,
                success=bool(info.get("success", False)),
                alpha=float(info.get("alpha", alpha)),
                copilot_enabled=copilot_enabled,
                conflict_score=float(extra_terms.get("conflict_score", 0.0)),
                correction_score=float(extra_terms.get("correction_score", 0.0)),
                residual_norm=float(extra_terms.get("residual_norm", np.linalg.norm(residual_action))),
                control_dt=float(env.cfg.control_dt),
            )
        )
        prev_residual = residual_action.copy()
        reward_env_sum += float(reward_env)
        reward_total_sum += float(reward_total)
        obs = next_obs
        if terminated or truncated:
            break

    success = any(t.success for t in transitions)
    return EpisodeRollout(
        transitions=transitions,
        obs_keys=obs_keys,
        success=success,
        episode_return_env=reward_env_sum,
        episode_return_total=reward_total_sum,
        meta={"source": "playback_rollout"},
    )


def evaluate_policy(
    env: MujocoPickPlaceTeleopEnv,
    playback_episodes: list[EpisodeRollout],
    residual_policy,
    reward_model: PickPlaceReward,
    *,
    alpha: float,
    eval_episodes: int,
) -> dict[str, float]:
    returns_env = []
    returns_total = []
    successes = []
    conflicts = []
    episode_pool = playback_episodes if eval_episodes <= 0 else playback_episodes[: min(eval_episodes, len(playback_episodes))]
    for idx, demo_rollout in enumerate(episode_pool):
        rollout = collect_episode(env, demo_rollout, residual_policy, reward_model, alpha=alpha, seed=idx)
        returns_env.append(rollout.episode_return_env)
        returns_total.append(rollout.episode_return_total)
        successes.append(float(rollout.success))
        if rollout.transitions:
            conflicts.append(float(np.mean([t.conflict_score for t in rollout.transitions])))
    return {
        "avg_return_env": float(np.mean(returns_env)) if returns_env else 0.0,
        "avg_return_total": float(np.mean(returns_total)) if returns_total else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "avg_conflict": float(np.mean(conflicts)) if conflicts else 0.0,
    }


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def compute_actor_loss(
    actor: MLPActor,
    critic: MLPCritic,
    obs_flat: torch.Tensor,
    base_action: torch.Tensor,
    logged_residual: torch.Tensor,
    is_assisted: torch.Tensor,
    cfg: TrainConfig,
    *,
    use_q_loss: bool,
) -> torch.Tensor:
    raw_pred_residual = clip_residual_tensor(actor(obs_flat, base_action), cfg.residual_limit)
    pred_residual = apply_idle_gate_tensor(raw_pred_residual, base_action, cfg.idle_deadband, cfg.idle_scale_span)
    pred_commanded_action = clip_commanded_action_tensor(base_action + cfg.alpha * pred_residual)

    actor_loss = torch.zeros((), device=obs_flat.device)
    if use_q_loss:
        actor_loss = actor_loss - critic.q1_only(obs_flat, pred_commanded_action).mean()

    if cfg.zero_reg > 0.0:
        actor_loss = actor_loss + cfg.zero_reg * (pred_residual**2).mean()
    if cfg.conflict_reg > 0.0:
        actor_loss = actor_loss + cfg.conflict_reg * conflict_metric_tensor(base_action, pred_residual).mean()

    teleop_mask = 1.0 - is_assisted
    if cfg.teleop_zero_reg > 0.0:
        actor_loss = actor_loss + cfg.teleop_zero_reg * masked_mean((pred_residual**2).sum(dim=-1, keepdim=True), teleop_mask)

    if cfg.assisted_bc_reg > 0.0:
        gated_logged = apply_idle_gate_tensor(logged_residual, base_action, cfg.idle_deadband, cfg.idle_scale_span)
        actor_loss = actor_loss + cfg.assisted_bc_reg * masked_mean(
            ((pred_residual - gated_logged) ** 2).sum(dim=-1, keepdim=True),
            is_assisted,
        )

    idle_mask = (base_action.norm(dim=-1, keepdim=True) < cfg.idle_deadband).to(dtype=base_action.dtype)
    if cfg.idle_zero_reg > 0.0:
        actor_loss = actor_loss + cfg.idle_zero_reg * masked_mean((raw_pred_residual**2).sum(dim=-1, keepdim=True), idle_mask)

    return actor_loss


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    ordered = " ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))
    return f"{prefix} {ordered}".strip()


def compute_critic_loss(
    q1: torch.Tensor,
    q2: torch.Tensor,
    target_q: torch.Tensor,
    *,
    loss_type: str,
) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
    if loss_type == "huber":
        return F.smooth_l1_loss(q1, target_q) + F.smooth_l1_loss(q2, target_q)
    raise ValueError(f"Unknown critic loss type: {loss_type}")


def main(args: argparse.Namespace) -> None:
    dataset_dirs = tuple(path.resolve() for path in (args.dataset_dirs or [args.dataset_dir]))
    if not (0.0 <= args.offline_fraction <= 1.0):
        raise ValueError(f"offline_fraction must be in [0, 1], got {args.offline_fraction}")
    cfg = TrainConfig(
        dataset_dirs=dataset_dirs,
        save_path=args.save_path.resolve(),
        xml_path=str(args.xml.resolve()),
        init_checkpoint=args.init_checkpoint.resolve() if args.init_checkpoint is not None else None,
        seed=args.seed,
        alpha=args.alpha,
        reward_mode=args.reward_mode,
        batch_size=args.batch_size,
        offline_fraction=args.offline_fraction,
        replay_mix_mode=args.replay_mix_mode,
        offline_updates=args.offline_updates,
        updates_per_episode=args.updates_per_episode,
        critic_warmup_steps=args.critic_warmup_steps,
        baseline_episodes=args.baseline_episodes,
        assisted_episodes=args.assisted_episodes,
        eval_episodes=args.eval_episodes,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        critic_loss=args.critic_loss,
        residual_limit=args.residual_limit,
        exploration_noise=args.exploration_noise,
        zero_reg=args.zero_reg,
        conflict_reg=args.conflict_reg,
        teleop_zero_reg=args.teleop_zero_reg,
        assisted_bc_reg=args.assisted_bc_reg,
        idle_zero_reg=args.idle_zero_reg,
        idle_deadband=args.idle_deadband,
        idle_scale_span=args.idle_scale_span,
        actor_bc_warmup_steps=args.actor_bc_warmup_steps,
        normalize_state=not args.disable_state_norm,
        normalize_reward=not args.disable_reward_norm,
        filter_assist_by_success=args.filter_assist_by_success,
        filter_assist_min_cmd_delta=args.filter_assist_min_cmd_delta,
        filter_assist_alpha=args.filter_assist_alpha,
        filter_assist_reach_gain=args.filter_assist_reach_gain,
        filter_assist_place_gain=args.filter_assist_place_gain,
        log_interval=args.log_interval,
    )
    set_seed(cfg.seed)

    reward_model = PickPlaceReward()
    env = MujocoPickPlaceTeleopEnv(PickPlaceTaskConfig(xml_path=cfg.xml_path, seed=cfg.seed), reward=reward_model)
    obs, _ = env.reset(seed=cfg.seed)
    obs_dim = int(flatten_observation(obs).shape[0])
    action_dim = env.action_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    offline_rb = ReplayBuffer(cfg.buffer_capacity)
    online_rb = ReplayBuffer(cfg.buffer_capacity)
    playback_episodes = load_dataset_into_buffer(cfg.dataset_dirs, offline_rb, cfg)
    state_stats = compute_state_stats(playback_episodes)
    reward_mean, reward_std = compute_reward_stats(playback_episodes, cfg.reward_mode)
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
        f"Replay mixing mode={cfg.replay_mix_mode} offline_fraction={cfg.offline_fraction:.2f} "
        f"state_norm={cfg.normalize_state} reward_norm={cfg.normalize_reward} reward_mode={cfg.reward_mode}"
    )

    actor = MLPActor(obs_dim, action_dim, hidden_dim=cfg.hidden_dim, residual_limit=cfg.residual_limit).to(device)
    critic = MLPCritic(obs_dim, action_dim, hidden_dim=cfg.hidden_dim).to(device)
    actor_target = MLPActor(obs_dim, action_dim, hidden_dim=cfg.hidden_dim, residual_limit=cfg.residual_limit).to(device)
    critic_target = MLPCritic(obs_dim, action_dim, hidden_dim=cfg.hidden_dim).to(device)
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())
    maybe_load_init_checkpoint(
        cfg.init_checkpoint,
        actor=actor,
        critic=critic,
        actor_target=actor_target,
        critic_target=critic_target,
    )
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    tracker = MetricTracker()
    train_steps = 0
    for _ in range(cfg.offline_updates):
        batch = sample_mixed_batch(
            offline_rb,
            online_rb,
            cfg.batch_size,
            cfg.offline_fraction,
            cfg.replay_mix_mode,
        )
        prepared = prepare_batch(
            batch,
            device=device,
            state_normalizer=state_normalizer,
            reward_normalizer=reward_normalizer,
            reward_mode=cfg.reward_mode,
        )
        obs_flat = prepared["obs_flat"]
        next_obs_flat = prepared["next_obs_flat"]
        base_action = prepared["base_action"]
        next_base_action = prepared["next_base_action"]
        commanded_action = prepared["commanded_action"]
        reward_total = prepared["reward_total"]
        done = prepared["done"]

        with torch.no_grad():
            next_residual = clip_residual_tensor(actor_target(next_obs_flat, next_base_action), cfg.residual_limit)
            next_residual = apply_idle_gate_tensor(next_residual, next_base_action, cfg.idle_deadband, cfg.idle_scale_span)
            next_commanded_action = clip_commanded_action_tensor(next_base_action + cfg.alpha * next_residual)
            target_q1, target_q2 = critic_target(next_obs_flat, next_commanded_action)
            target_q = reward_total + cfg.gamma * (1.0 - done) * torch.minimum(target_q1, target_q2)

        q1, q2 = critic(obs_flat, commanded_action)
        critic_loss = compute_critic_loss(q1, q2, target_q, loss_type=cfg.critic_loss)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        if train_steps >= cfg.critic_warmup_steps and train_steps % cfg.policy_delay == 0:
            use_q_loss = train_steps >= (cfg.critic_warmup_steps + cfg.actor_bc_warmup_steps)
            actor_loss = compute_actor_loss(
                actor,
                critic,
                obs_flat,
                base_action,
                prepared["logged_residual"],
                prepared["is_assisted"],
                cfg,
                use_q_loss=use_q_loss,
            )
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            soft_update(actor_target, actor, cfg.tau)
            soft_update(critic_target, critic, cfg.tau)
            actor_loss_value = float(actor_loss.item())
        else:
            soft_update(critic_target, critic, cfg.tau)
            actor_loss_value = float("nan")

        td1 = q1 - target_q
        td2 = q2 - target_q
        tracker.update(
            critic_loss=float(critic_loss.item()),
            q1=float(q1.mean().item()),
            q2=float(q2.mean().item()),
            target_q=float(target_q.mean().item()),
            target_std=float(target_q.std().item()),
            td_abs_mean=float(0.5 * (td1.abs().mean().item() + td2.abs().mean().item())),
            td_abs_max=float(torch.maximum(td1.abs().max(), td2.abs().max()).item()),
            q_gap=float((q1 - q2).abs().mean().item()),
            reward=float(reward_total.mean().item()),
            assisted_fraction=float(prepared["is_assisted"].mean().item()),
            online_fraction=float(0.0),
        )
        if not np.isnan(actor_loss_value):
            tracker.update(actor_loss=actor_loss_value)

        train_steps += 1
        if train_steps % cfg.log_interval == 0:
            print(format_metrics(f"[offline step {train_steps}]", tracker.summary()))
            tracker.reset()

    for idx in range(cfg.baseline_episodes):
        demo_rollout = playback_episodes[idx % len(playback_episodes)]
        rollout = collect_episode(env, demo_rollout, ZeroResidualPolicy(), reward_model, alpha=cfg.alpha, seed=cfg.seed + idx)
        push_records_to_buffer(online_rb, rollout)

    for episode_idx in range(cfg.assisted_episodes):
        demo_rollout = playback_episodes[(cfg.baseline_episodes + episode_idx) % len(playback_episodes)]
        policy = TorchResidualPolicy(
            actor,
            device,
            noise_scale=cfg.exploration_noise,
            residual_limit=cfg.residual_limit,
            idle_deadband=cfg.idle_deadband,
            idle_scale_span=cfg.idle_scale_span,
            state_normalizer=state_normalizer,
        )
        rollout = collect_episode(env, demo_rollout, policy, reward_model, alpha=cfg.alpha, seed=cfg.seed + 100 + episode_idx)
        push_records_to_buffer(online_rb, rollout)
        episode_assisted_fraction = (
            float(np.mean([float(t.copilot_enabled) for t in rollout.transitions])) if rollout.transitions else 0.0
        )
        print(
            format_metrics(
                f"[episode {episode_idx + 1}/{cfg.assisted_episodes}]",
                {
                    "return_total": rollout.episode_return_total,
                    "return_env": rollout.episode_return_env,
                    "steps": float(rollout.steps),
                    "success": float(rollout.success),
                    "assisted_fraction": episode_assisted_fraction,
                    "online_size": float(len(online_rb)),
                },
            )
        )

        for _ in range(cfg.updates_per_episode):
            batch = sample_mixed_batch(
                offline_rb,
                online_rb,
                cfg.batch_size,
                cfg.offline_fraction,
                cfg.replay_mix_mode,
            )
            prepared = prepare_batch(
                batch,
                device=device,
                state_normalizer=state_normalizer,
                reward_normalizer=reward_normalizer,
                reward_mode=cfg.reward_mode,
            )
            obs_flat = prepared["obs_flat"]
            next_obs_flat = prepared["next_obs_flat"]
            base_action = prepared["base_action"]
            next_base_action = prepared["next_base_action"]
            commanded_action = prepared["commanded_action"]
            reward_total = prepared["reward_total"]
            done = prepared["done"]

            with torch.no_grad():
                next_residual = clip_residual_tensor(actor_target(next_obs_flat, next_base_action), cfg.residual_limit)
                next_residual = apply_idle_gate_tensor(next_residual, next_base_action, cfg.idle_deadband, cfg.idle_scale_span)
                next_commanded_action = clip_commanded_action_tensor(next_base_action + cfg.alpha * next_residual)
                target_q1, target_q2 = critic_target(next_obs_flat, next_commanded_action)
                target_q = reward_total + cfg.gamma * (1.0 - done) * torch.minimum(target_q1, target_q2)

            q1, q2 = critic(obs_flat, commanded_action)
            critic_loss = compute_critic_loss(q1, q2, target_q, loss_type=cfg.critic_loss)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            if train_steps >= cfg.critic_warmup_steps and train_steps % cfg.policy_delay == 0:
                use_q_loss = train_steps >= (cfg.critic_warmup_steps + cfg.actor_bc_warmup_steps)
                actor_loss = compute_actor_loss(
                    actor,
                    critic,
                    obs_flat,
                    base_action,
                    prepared["logged_residual"],
                    prepared["is_assisted"],
                    cfg,
                    use_q_loss=use_q_loss,
                )
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()
                soft_update(actor_target, actor, cfg.tau)
                soft_update(critic_target, critic, cfg.tau)
                actor_loss_value = float(actor_loss.item())
            else:
                soft_update(critic_target, critic, cfg.tau)
                actor_loss_value = float("nan")

            td1 = q1 - target_q
            td2 = q2 - target_q
            tracker.update(
                critic_loss=float(critic_loss.item()),
                q1=float(q1.mean().item()),
                q2=float(q2.mean().item()),
                target_q=float(target_q.mean().item()),
                target_std=float(target_q.std().item()),
                td_abs_mean=float(0.5 * (td1.abs().mean().item() + td2.abs().mean().item())),
                td_abs_max=float(torch.maximum(td1.abs().max(), td2.abs().max()).item()),
                q_gap=float((q1 - q2).abs().mean().item()),
                reward=float(reward_total.mean().item()),
                assisted_fraction=float(prepared["is_assisted"].mean().item()),
                online_fraction=float(min(len(online_rb), cfg.batch_size) / max(cfg.batch_size, 1)),
            )
            if not np.isnan(actor_loss_value):
                tracker.update(actor_loss=actor_loss_value)

            train_steps += 1
            if train_steps % cfg.log_interval == 0:
                print(format_metrics(f"[train step {train_steps}]", tracker.summary()))
                tracker.reset()

    eval_metrics = evaluate_policy(
        env,
        playback_episodes,
        TorchResidualPolicy(
            actor,
            device,
            residual_limit=cfg.residual_limit,
            idle_deadband=cfg.idle_deadband,
            idle_scale_span=cfg.idle_scale_span,
            state_normalizer=state_normalizer,
        ),
        reward_model,
        alpha=cfg.alpha,
        eval_episodes=cfg.eval_episodes,
    )
    baseline_metrics = evaluate_policy(
        env,
        playback_episodes,
        ZeroResidualPolicy(),
        reward_model,
        alpha=cfg.alpha,
        eval_episodes=cfg.eval_episodes,
    )

    cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
    config_payload = {}
    for key, value in asdict(cfg).items():
        if isinstance(value, Path):
            config_payload[key] = str(value)
        elif isinstance(value, (list, tuple)) and value and all(isinstance(item, Path) for item in value):
            config_payload[key] = [str(item) for item in value]
        else:
            config_payload[key] = value
    payload = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "config": config_payload,
        "eval_metrics": eval_metrics,
        "baseline_metrics": baseline_metrics,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "state_mean": state_stats.mean,
        "state_std": state_stats.std,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
    }
    torch.save(payload, cfg.save_path)
    print("Saved checkpoint to", cfg.save_path)
    print("Baseline metrics:", baseline_metrics)
    print("Assisted metrics:", eval_metrics)


if __name__ == "__main__":
    default_xml = PROJECT_ROOT / "assets" / "mujoco" / "Alicia_D_v5_6" / "gripper_50mm" / "alicia_d_follower.xml"
    default_save = PROJECT_ROOT / "logs" / "residual_td3_teleop.pt"
    parser = argparse.ArgumentParser(description="Train a teleoperation residual TD3 copilot in MuJoCo.")
    parser.add_argument("--dataset_dir", type=Path, default=None, help="Single dataset directory containing episode_*.npz rollouts")
    parser.add_argument("--dataset_dirs", type=Path, nargs="+", default=None, help="One or more dataset directories to load jointly")
    parser.add_argument("--xml", type=Path, default=default_xml, help="MuJoCo XML path")
    parser.add_argument("--save_path", type=Path, default=default_save, help="Where to save the checkpoint")
    parser.add_argument("--init_checkpoint", type=Path, default=None, help="Optional checkpoint used to initialize critic/actor weights")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--reward_mode", type=str, default="logged_total", choices=("logged_total", "env_dense", "success_only"))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--offline_fraction", type=float, default=0.5, help="Offline share used by fixed_offline_fraction replay mixing")
    parser.add_argument(
        "--replay_mix_mode",
        type=str,
        default="fixed_offline_fraction",
        choices=("fixed_offline_fraction", "available_balanced", "offline_only", "online_only"),
        help="How to mix offline and online replay samples inside each training batch",
    )
    parser.add_argument("--offline_updates", type=int, default=2000)
    parser.add_argument("--critic_warmup_steps", type=int, default=1000)
    parser.add_argument("--baseline_episodes", type=int, default=5)
    parser.add_argument("--assisted_episodes", type=int, default=20)
    parser.add_argument("--updates_per_episode", type=int, default=300)
    parser.add_argument("--eval_episodes", type=int, default=0, help="How many playback episodes to evaluate; <=0 means all")
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--critic_loss", type=str, default="huber", choices=("huber", "mse"))
    parser.add_argument("--residual_limit", type=float, default=0.35)
    parser.add_argument("--exploration_noise", type=float, default=0.05)
    parser.add_argument("--zero_reg", type=float, default=0.05)
    parser.add_argument("--conflict_reg", type=float, default=0.10)
    parser.add_argument("--teleop_zero_reg", type=float, default=0.50)
    parser.add_argument("--assisted_bc_reg", type=float, default=0.10)
    parser.add_argument("--idle_zero_reg", type=float, default=0.50)
    parser.add_argument("--idle_deadband", type=float, default=0.05)
    parser.add_argument("--idle_scale_span", type=float, default=0.20)
    parser.add_argument("--actor_bc_warmup_steps", type=int, default=1000)
    parser.add_argument("--disable_state_norm", action="store_true", help="Disable state standardization during training and policy rollout")
    parser.add_argument("--disable_reward_norm", action="store_true", help="Disable reward normalization in TD targets")
    parser.add_argument("--filter_assist_by_success", action="store_true", help="Keep only successful assist rollouts from assisted datasets")
    parser.add_argument("--filter_assist_min_cmd_delta", type=float, default=0.0, help="Drop assist rollouts whose mean ||commanded-base|| is below this threshold")
    parser.add_argument("--filter_assist_alpha", type=float, default=None, help="Keep only assist rollouts whose meta assist_alpha matches this value")
    parser.add_argument("--filter_assist_reach_gain", type=float, default=None, help="Keep only assist rollouts whose meta heuristic_reach_gain matches this value")
    parser.add_argument("--filter_assist_place_gain", type=float, default=None, help="Keep only assist rollouts whose meta heuristic_place_gain matches this value")
    parser.add_argument("--log_interval", type=int, default=200, help="How often to print aggregated training metrics")
    parsed = parser.parse_args()
    if parsed.dataset_dirs is None and parsed.dataset_dir is None:
        parser.error("Provide --dataset_dir or --dataset_dirs")
    main(parsed)
