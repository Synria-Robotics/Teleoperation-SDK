from __future__ import annotations

import numpy as np


OBS_FLAT_KEYS = (
    "joint_pos",
    "joint_vel",
    "gripper_pos",
    "ee_pos",
    "ee_quat",
    "box_pos",
    "box_quat",
    "box_linvel",
    "box_angvel",
    "basket_pos",
    "box_to_ee",
    "box_to_basket",
    "ee_to_basket",
)

OBS_STATE_KEY = "observation.state"

OBS_KEY_SHAPES = {
    "joint_pos": 6,
    "joint_vel": 6,
    "gripper_pos": 1,
    "ee_pos": 3,
    "ee_quat": 4,
    "box_pos": 3,
    "box_quat": 4,
    "box_linvel": 3,
    "box_angvel": 3,
    "basket_pos": 3,
    "box_to_ee": 3,
    "box_to_basket": 3,
    "ee_to_basket": 3,
}


def flatten_observation(obs: dict[str, np.ndarray], keys: tuple[str, ...] = OBS_FLAT_KEYS) -> np.ndarray:
    return np.concatenate([np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in keys], axis=0).astype(np.float32)


def unflatten_observation(obs_flat: np.ndarray, keys: tuple[str, ...] = OBS_FLAT_KEYS) -> dict[str, np.ndarray]:
    obs_flat = np.asarray(obs_flat, dtype=np.float32).reshape(-1)
    out: dict[str, np.ndarray] = {}
    offset = 0
    for key in keys:
        size = OBS_KEY_SHAPES[key]
        out[key] = obs_flat[offset : offset + size].copy()
        offset += size
    return out
