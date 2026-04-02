from __future__ import annotations

from typing import Sequence

import numpy as np

from Residual_RL_TD3.common.observation_utils import unflatten_observation


TELEOP_RESIDUAL_OBS_MODE = "follower_pose_hdelta_residual"


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(-1)
    norm = max(float(np.linalg.norm(quat)), 1e-6)
    return quat / norm


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


def flatten_residual_history(
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


def compose_base_target_feature(
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
    base = compose_base_target_feature(
        follower_ee_pos,
        follower_ee_quat,
        follower_gripper,
        base_action,
        translation_step=translation_step,
        rotation_step=rotation_step,
        gripper_step=gripper_step,
    )
    residual_hist = flatten_residual_history(
        residual_history,
        history_len=history_len,
        action_dim=int(base.shape[0]),
    )
    return np.concatenate([follower_pos, follower_quat, follower_grip, base, residual_hist], axis=0).astype(np.float32)


def build_policy_observation_from_dict(
    obs: dict[str, np.ndarray],
    *,
    base_action: np.ndarray,
    residual_history: Sequence[np.ndarray],
    history_len: int,
    translation_step: float,
    rotation_step: float,
    gripper_step: float,
    obs_mode: str = TELEOP_RESIDUAL_OBS_MODE,
) -> np.ndarray:
    if obs_mode != TELEOP_RESIDUAL_OBS_MODE:
        raise ValueError(f"Unsupported obs_mode={obs_mode}. Only {TELEOP_RESIDUAL_OBS_MODE!r} is supported.")
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


def build_policy_observation_from_flat_state(
    obs_flat: np.ndarray,
    *,
    base_action: np.ndarray,
    residual_history: Sequence[np.ndarray],
    history_len: int,
    translation_step: float,
    rotation_step: float,
    gripper_step: float,
    obs_mode: str = TELEOP_RESIDUAL_OBS_MODE,
) -> np.ndarray:
    return build_policy_observation_from_dict(
        unflatten_observation(obs_flat),
        base_action=base_action,
        residual_history=residual_history,
        history_len=history_len,
        translation_step=translation_step,
        rotation_step=rotation_step,
        gripper_step=gripper_step,
        obs_mode=obs_mode,
    )


def infer_policy_obs_dim(
    obs: dict[str, np.ndarray],
    *,
    action_dim: int,
    history_len: int,
    translation_step: float,
    rotation_step: float,
    gripper_step: float,
    obs_mode: str = TELEOP_RESIDUAL_OBS_MODE,
) -> int:
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
