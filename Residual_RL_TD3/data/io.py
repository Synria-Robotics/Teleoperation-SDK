from __future__ import annotations

from pathlib import Path

import numpy as np

from Residual_RL_TD3.data.episode_types import EpisodeRollout, Transition


JOINT_POS_DIM = 6
GRIPPER_POS_INDEX = 12


def save_episode_npz(path: str | Path, rollout: EpisodeRollout) -> Path:
    """Persist one rollout to a compact ``.npz`` archive."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not rollout.transitions:
        raise ValueError("Cannot save an empty rollout")

    arrays = {
        "observation_state": np.stack([t.observation_state for t in rollout.transitions]).astype(np.float32),
        "next_observation_state": np.stack([t.next_observation_state for t in rollout.transitions]).astype(np.float32),
        "base_action": np.stack([t.base_action for t in rollout.transitions]).astype(np.float32),
        "next_base_action": np.stack([t.next_base_action for t in rollout.transitions]).astype(np.float32),
        "residual_action": np.stack([t.residual_action for t in rollout.transitions]).astype(np.float32),
        "realized_action": np.stack([t.realized_action for t in rollout.transitions]).astype(np.float32),
        "base_joint_target": np.stack([t.base_joint_target for t in rollout.transitions]).astype(np.float32),
        "base_gripper_target": np.stack([t.base_gripper_target for t in rollout.transitions]).astype(np.float32),
        "reward_env": np.asarray([t.reward_env for t in rollout.transitions], dtype=np.float32),
        "terminated": np.asarray([t.terminated for t in rollout.transitions], dtype=np.bool_),
        "truncated": np.asarray([t.truncated for t in rollout.transitions], dtype=np.bool_),
        "success_flags": np.asarray([t.success for t in rollout.transitions], dtype=np.bool_),
        "control_dt": np.asarray([t.control_dt for t in rollout.transitions], dtype=np.float32),
        "success": np.asarray(rollout.success, dtype=np.bool_),
        "episode_return_env": np.asarray(rollout.episode_return_env, dtype=np.float32),
        "steps": np.asarray(rollout.steps, dtype=np.int32),
        "obs_keys": np.asarray(rollout.obs_keys, dtype=object),
        "meta_keys": np.asarray(list(rollout.meta.keys()), dtype=object),
        "meta_values": np.asarray(list(rollout.meta.values()), dtype=object),
    }
    np.savez_compressed(target, **arrays)
    return target


def load_episode_npz(path: str | Path) -> EpisodeRollout:
    """Load one rollout produced by :func:`save_episode_npz`."""
    src = Path(path)
    data = np.load(src, allow_pickle=True)

    obs_keys = tuple(str(x) for x in data.get("obs_keys", np.asarray([], dtype=object)).tolist())
    meta_keys = tuple(str(x) for x in data.get("meta_keys", np.asarray([], dtype=object)).tolist())
    meta_values = data.get("meta_values", np.asarray([], dtype=object)).tolist()
    meta = dict(zip(meta_keys, meta_values, strict=False))

    observation_state = (
        np.asarray(data["observation_state"], dtype=np.float32)
        if "observation_state" in data
        else np.asarray(data["obs_flat"], dtype=np.float32)
    )
    next_observation_state = (
        np.asarray(data["next_observation_state"], dtype=np.float32)
        if "next_observation_state" in data
        else np.asarray(data["next_obs_flat"], dtype=np.float32)
    )
    steps = int(data["steps"]) if "steps" in data else int(len(observation_state))

    if "base_action" in data:
        base_action = np.asarray(data["base_action"], dtype=np.float32)
    elif "human_action" in data:
        base_action = np.asarray(data["human_action"], dtype=np.float32)
    else:
        base_action = np.zeros((steps, 7), dtype=np.float32)

    if "next_base_action" in data:
        next_base_action = np.asarray(data["next_base_action"], dtype=np.float32)
    elif "next_human_action" in data:
        next_base_action = np.asarray(data["next_human_action"], dtype=np.float32)
    else:
        next_base_action = np.zeros_like(base_action)
        if len(base_action) > 1:
            next_base_action[:-1] = base_action[1:]

    residual_action = (
        np.asarray(data["residual_action"], dtype=np.float32)
        if "residual_action" in data
        else np.zeros_like(base_action)
    )
    if "realized_action" in data:
        realized_action = np.asarray(data["realized_action"], dtype=np.float32)
    elif "exec_action" in data:
        realized_action = np.asarray(data["exec_action"], dtype=np.float32)
    elif "commanded_action" in data:
        realized_action = np.asarray(data["commanded_action"], dtype=np.float32)
    else:
        realized_action = np.clip(base_action + residual_action, -1.0, 1.0).astype(np.float32)

    reward_env = (
        np.asarray(data["reward_env"], dtype=np.float32)
        if "reward_env" in data
        else np.asarray(data.get("reward_total", np.zeros(steps, dtype=np.float32)), dtype=np.float32)
    )
    terminated = np.asarray(data.get("terminated", np.zeros(steps, dtype=np.bool_)), dtype=np.bool_)
    truncated = np.asarray(data.get("truncated", np.zeros(steps, dtype=np.bool_)), dtype=np.bool_)
    if "done" in data:
        done = np.asarray(data["done"], dtype=np.bool_)
        terminated = np.logical_or(terminated, done)
    success_flags = np.asarray(data.get("success_flags", np.zeros(steps, dtype=np.bool_)), dtype=np.bool_)
    control_dt = np.asarray(data.get("control_dt", np.zeros(steps, dtype=np.float32)), dtype=np.float32)

    if "base_joint_target" in data:
        base_joint_target = np.asarray(data["base_joint_target"], dtype=np.float32)
    else:
        base_joint_target = np.asarray(next_observation_state[:, :JOINT_POS_DIM], dtype=np.float32)

    if "base_gripper_target" in data:
        base_gripper_target = np.asarray(data["base_gripper_target"], dtype=np.float32)
    else:
        base_gripper_target = np.asarray(
            next_observation_state[:, GRIPPER_POS_INDEX : GRIPPER_POS_INDEX + 1],
            dtype=np.float32,
        )

    transitions: list[Transition] = []
    for idx in range(steps):
        transitions.append(
            Transition(
                observation_state=np.asarray(observation_state[idx], dtype=np.float32),
                next_observation_state=np.asarray(next_observation_state[idx], dtype=np.float32),
                base_action=np.asarray(base_action[idx], dtype=np.float32),
                next_base_action=np.asarray(next_base_action[idx], dtype=np.float32),
                residual_action=np.asarray(residual_action[idx], dtype=np.float32),
                realized_action=np.asarray(realized_action[idx], dtype=np.float32),
                reward_env=float(reward_env[idx]),
                terminated=bool(terminated[idx]),
                truncated=bool(truncated[idx]),
                base_joint_target=np.asarray(base_joint_target[idx], dtype=np.float32).reshape(JOINT_POS_DIM),
                base_gripper_target=np.asarray(base_gripper_target[idx], dtype=np.float32).reshape(1),
                success=bool(success_flags[idx]),
                control_dt=float(control_dt[idx]),
            )
        )

    return EpisodeRollout(
        transitions=transitions,
        obs_keys=obs_keys,
        success=bool(data.get("success", np.asarray(False))),
        episode_return_env=float(data.get("episode_return_env", np.asarray(reward_env.sum(), dtype=np.float32))),
        meta=meta,
    )
