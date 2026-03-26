from __future__ import annotations

from pathlib import Path

import numpy as np

from teleop_sdk.data.episode_types import EpisodeRollout, Transition


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
        "prev_residual_action": np.stack([t.prev_residual_action for t in rollout.transitions]).astype(np.float32),
        "commanded_action": np.stack([t.commanded_action for t in rollout.transitions]).astype(np.float32),
        "realized_action": np.stack([t.realized_action for t in rollout.transitions]).astype(np.float32),
        # Backward-compatibility aliases for older readers.
        "obs_flat": np.stack([t.observation_state for t in rollout.transitions]).astype(np.float32),
        "next_obs_flat": np.stack([t.next_observation_state for t in rollout.transitions]).astype(np.float32),
        "human_action": np.stack([t.base_action for t in rollout.transitions]).astype(np.float32),
        "next_human_action": np.stack([t.next_base_action for t in rollout.transitions]).astype(np.float32),
        "exec_action": np.stack([t.realized_action for t in rollout.transitions]).astype(np.float32),
        "base_joint_target": np.stack([t.base_joint_target for t in rollout.transitions]).astype(np.float32),
        "base_gripper_target": np.stack([t.base_gripper_target for t in rollout.transitions]).astype(np.float32),
        "reward_env": np.asarray([t.reward_env for t in rollout.transitions], dtype=np.float32),
        "reward_total": np.asarray([t.reward_total for t in rollout.transitions], dtype=np.float32),
        "terminated": np.asarray([t.terminated for t in rollout.transitions], dtype=np.bool_),
        "truncated": np.asarray([t.truncated for t in rollout.transitions], dtype=np.bool_),
        "teleop_active": np.asarray([t.teleop_active for t in rollout.transitions], dtype=np.bool_),
        "success_flags": np.asarray([t.success for t in rollout.transitions], dtype=np.bool_),
        "alpha": np.asarray([t.alpha for t in rollout.transitions], dtype=np.float32),
        "copilot_enabled": np.asarray([t.copilot_enabled for t in rollout.transitions], dtype=np.bool_),
        "conflict_score": np.asarray([t.conflict_score for t in rollout.transitions], dtype=np.float32),
        "correction_score": np.asarray([t.correction_score for t in rollout.transitions], dtype=np.float32),
        "residual_norm": np.asarray([t.residual_norm for t in rollout.transitions], dtype=np.float32),
        "episode_time_s": np.asarray([t.episode_time_s for t in rollout.transitions], dtype=np.float32),
        "control_dt": np.asarray([t.control_dt for t in rollout.transitions], dtype=np.float32),
        "policy_latency_ms": np.asarray([t.policy_latency_ms for t in rollout.transitions], dtype=np.float32),
        "success": np.asarray(rollout.success, dtype=np.bool_),
        "episode_return_env": np.asarray(rollout.episode_return_env, dtype=np.float32),
        "episode_return_total": np.asarray(rollout.episode_return_total, dtype=np.float32),
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

    obs_keys = tuple(str(x) for x in data["obs_keys"].tolist())
    meta_keys = tuple(str(x) for x in data.get("meta_keys", np.asarray([], dtype=object)).tolist())
    meta_values = data.get("meta_values", np.asarray([], dtype=object)).tolist()
    meta = dict(zip(meta_keys, meta_values, strict=False))

    prev_residual = data["prev_residual_action"] if "prev_residual_action" in data else np.zeros_like(data["residual_action"])
    teleop_active = data["teleop_active"] if "teleop_active" in data else np.ones_like(data["terminated"], dtype=np.bool_)
    success_flags = data["success_flags"] if "success_flags" in data else np.zeros_like(data["terminated"], dtype=np.bool_)
    alpha = data["alpha"] if "alpha" in data else np.ones_like(data["reward_total"], dtype=np.float32)
    copilot_enabled = data["copilot_enabled"] if "copilot_enabled" in data else (alpha > 0.0)
    conflict = data["conflict_score"] if "conflict_score" in data else np.zeros_like(data["reward_total"], dtype=np.float32)
    correction = data["correction_score"] if "correction_score" in data else np.zeros_like(data["reward_total"], dtype=np.float32)
    residual_norm = data["residual_norm"] if "residual_norm" in data else np.linalg.norm(data["residual_action"], axis=1).astype(np.float32)
    episode_time_s = data["episode_time_s"] if "episode_time_s" in data else np.zeros_like(data["reward_total"], dtype=np.float32)
    control_dt = data["control_dt"] if "control_dt" in data else np.zeros_like(data["reward_total"], dtype=np.float32)
    policy_latency_ms = (
        data["policy_latency_ms"] if "policy_latency_ms" in data else np.zeros_like(data["reward_total"], dtype=np.float32)
    )
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

    legacy_base_action = np.asarray(data["human_action"], dtype=np.float32) if "human_action" in data else None
    legacy_next_base_action = (
        np.asarray(data["next_human_action"], dtype=np.float32) if "next_human_action" in data else None
    )
    if legacy_base_action is not None and legacy_next_base_action is None:
        legacy_next_base_action = np.zeros_like(legacy_base_action)
    if legacy_base_action is not None and legacy_next_base_action is not None:
        if legacy_next_base_action.shape == legacy_base_action.shape and len(legacy_next_base_action) > 1:
            if np.max(np.abs(legacy_next_base_action[:-1])) < 1e-8 and np.max(np.abs(legacy_base_action[:-1])) > 1e-8:
                legacy_next_base_action = np.concatenate(
                    [np.asarray(legacy_base_action[1:], dtype=np.float32), np.zeros_like(legacy_base_action[:1])],
                    axis=0,
                )

    if "base_action" in data:
        base_action = np.asarray(data["base_action"], dtype=np.float32)
    elif legacy_base_action is not None:
        base_action = legacy_base_action
    else:
        base_action = np.zeros((len(observation_state), 7), dtype=np.float32)

    if "next_base_action" in data:
        next_base_action = np.asarray(data["next_base_action"], dtype=np.float32)
    elif legacy_next_base_action is not None:
        next_base_action = legacy_next_base_action
    else:
        next_base_action = np.zeros_like(base_action)

    commanded_action = (
        np.asarray(data["commanded_action"], dtype=np.float32)
        if "commanded_action" in data
        else np.asarray(data["exec_action"] if "exec_action" in data else data["realized_action"], dtype=np.float32)
    )
    realized_action = (
        np.asarray(data["realized_action"], dtype=np.float32)
        if "realized_action" in data
        else np.asarray(data["exec_action"], dtype=np.float32)
    )
    if "base_joint_target" in data:
        base_joint_target = data["base_joint_target"]
    else:
        base_joint_target = np.asarray(data["next_obs_flat"][:, :JOINT_POS_DIM], dtype=np.float32)
    if "base_gripper_target" in data:
        base_gripper_target = data["base_gripper_target"]
    else:
        base_gripper_target = np.asarray(data["next_obs_flat"][:, GRIPPER_POS_INDEX : GRIPPER_POS_INDEX + 1], dtype=np.float32)

    transitions: list[Transition] = []
    for idx in range(int(data["steps"])):
        transitions.append(
            Transition(
                observation_state=np.asarray(observation_state[idx], dtype=np.float32),
                next_observation_state=np.asarray(next_observation_state[idx], dtype=np.float32),
                base_action=np.asarray(base_action[idx], dtype=np.float32),
                next_base_action=np.asarray(next_base_action[idx], dtype=np.float32),
                residual_action=np.asarray(data["residual_action"][idx], dtype=np.float32),
                prev_residual_action=np.asarray(prev_residual[idx], dtype=np.float32),
                commanded_action=np.asarray(commanded_action[idx], dtype=np.float32),
                realized_action=np.asarray(realized_action[idx], dtype=np.float32),
                reward_env=float(data["reward_env"][idx]),
                reward_total=float(data["reward_total"][idx]),
                terminated=bool(data["terminated"][idx]),
                truncated=bool(data["truncated"][idx]),
                base_joint_target=np.asarray(base_joint_target[idx], dtype=np.float32),
                base_gripper_target=np.asarray(base_gripper_target[idx], dtype=np.float32).reshape(1),
                teleop_active=bool(teleop_active[idx]),
                success=bool(success_flags[idx]),
                alpha=float(alpha[idx]),
                copilot_enabled=bool(copilot_enabled[idx]),
                conflict_score=float(conflict[idx]),
                correction_score=float(correction[idx]),
                residual_norm=float(residual_norm[idx]),
                episode_time_s=float(episode_time_s[idx]),
                control_dt=float(control_dt[idx]),
                policy_latency_ms=float(policy_latency_ms[idx]),
            )
        )

    return EpisodeRollout(
        transitions=transitions,
        obs_keys=obs_keys,
        success=bool(data["success"]),
        episode_return_env=float(data["episode_return_env"]),
        episode_return_total=float(data["episode_return_total"]),
        meta=meta,
    )
