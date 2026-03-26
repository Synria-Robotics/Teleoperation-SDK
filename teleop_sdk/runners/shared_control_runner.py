from __future__ import annotations

from typing import Any, Callable

import numpy as np

from teleop_sdk.data.episode_types import EpisodeRollout, Transition
from teleop_sdk.envs.mujoco_pick_place_env import flatten_observation


def _clone_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=np.float32).copy() for k, v in obs.items()}


class SharedControlRunner:
    """Run one shared-control episode and record transitions."""

    def __init__(
        self,
        env,
        teleop_provider,
        residual_policy,
        total_reward_fn: Callable[..., tuple[float, dict[str, float]]],
    ):
        self.env = env
        self.teleop_provider = teleop_provider
        self.residual_policy = residual_policy
        self.total_reward_fn = total_reward_fn

    def run_episode(
        self,
        *,
        alpha: float = 1.0,
        max_steps: int | None = None,
        seed: int | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> EpisodeRollout:
        obs, _ = self.env.reset(seed=seed)
        self.teleop_provider.reset()
        self.residual_policy.reset()

        prev_residual = np.zeros(7, dtype=np.float32)
        obs_keys = tuple(obs.keys())
        transitions: list[Transition] = []
        reward_env_sum = 0.0
        reward_total_sum = 0.0
        steps_limit = self.env.cfg.max_steps if max_steps is None else min(max_steps, self.env.cfg.max_steps)

        for _ in range(steps_limit):
            if should_stop is not None and should_stop():
                break
            obs_copy = _clone_obs(obs)
            base_action = np.clip(self.teleop_provider.get_action(obs_copy), -1.0, 1.0).astype(np.float32)
            if not self.teleop_provider.is_active():
                base_action = np.zeros_like(base_action)
            residual_action = np.clip(self.residual_policy.get_action(obs_copy, base_action), -1.0, 1.0).astype(np.float32)

            next_obs, reward_env, terminated, truncated, info = self.env.step(base_action, residual_action, alpha=alpha)
            next_base_action = np.clip(self.teleop_provider.peek_next_action(), -1.0, 1.0).astype(np.float32)

            reward_total, extra_terms = self.total_reward_fn(
                reward_env=reward_env,
                base_action=base_action,
                residual_action=residual_action,
                prev_residual_action=prev_residual,
            )
            success = bool(info.get("success", False))
            copilot_enabled = bool(float(info.get("alpha", alpha)) > 0.0 and not np.allclose(residual_action, 0.0))
            transition = Transition(
                observation_state=flatten_observation(obs_copy),
                next_observation_state=flatten_observation(next_obs),
                base_action=np.asarray(info.get("base_action", base_action), dtype=np.float32).copy(),
                next_base_action=next_base_action.copy(),
                residual_action=residual_action,
                prev_residual_action=prev_residual.copy(),
                commanded_action=np.asarray(info.get("commanded_action", info["realized_action"]), dtype=np.float32).copy(),
                realized_action=np.asarray(info["realized_action"], dtype=np.float32).copy(),
                reward_env=float(reward_env),
                reward_total=float(reward_total),
                terminated=bool(terminated),
                truncated=bool(truncated),
                base_joint_target=np.asarray(info.get("target_joint_pos", next_obs["joint_pos"]), dtype=np.float32).copy(),
                base_gripper_target=np.asarray(
                    [info.get("target_gripper", float(next_obs["gripper_pos"][0]))],
                    dtype=np.float32,
                ),
                teleop_active=bool(self.teleop_provider.is_active()),
                success=success,
                alpha=float(info.get("alpha", alpha)),
                copilot_enabled=bool(info.get("copilot_enabled", copilot_enabled)),
                conflict_score=float(extra_terms.get("conflict_score", 0.0)),
                correction_score=float(extra_terms.get("correction_score", 0.0)),
                residual_norm=float(extra_terms.get("residual_norm", np.linalg.norm(residual_action))),
                control_dt=float(getattr(self.env.cfg, "control_dt", 0.0)),
            )
            transitions.append(transition)
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
        )
