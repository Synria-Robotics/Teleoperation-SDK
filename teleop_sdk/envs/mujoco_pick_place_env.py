from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from teleop_sdk.rewards.pick_place_reward import PickPlaceReward


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
OBS_BASE_ACTION_KEY = "observation.base_action"

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


def flatten_state_observation(obs: dict[str, np.ndarray], keys: tuple[str, ...] = OBS_FLAT_KEYS) -> np.ndarray:
    if OBS_STATE_KEY in obs:
        return np.asarray(obs[OBS_STATE_KEY], dtype=np.float32).reshape(-1).copy()
    return flatten_observation(obs, keys=keys)


def unflatten_observation(obs_flat: np.ndarray, keys: tuple[str, ...] = OBS_FLAT_KEYS) -> dict[str, np.ndarray]:
    obs_flat = np.asarray(obs_flat, dtype=np.float32).reshape(-1)
    out: dict[str, np.ndarray] = {}
    offset = 0
    for key in keys:
        size = OBS_KEY_SHAPES[key]
        out[key] = obs_flat[offset : offset + size].copy()
        offset += size
    return out


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array([w, -x, -y, -z], dtype=np.float64)


def _axis_angle_to_quat(vec: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(vec)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = vec / angle
    half = 0.5 * angle
    return np.array([np.cos(half), *(np.sin(half) * axis)], dtype=np.float64)


def _quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    quat = _quat_normalize(quat)
    qw = float(np.clip(quat[0], -1.0, 1.0))
    angle = 2.0 * np.arccos(qw)
    if angle < 1e-9:
        return np.zeros(3, dtype=np.float64)
    s = np.sqrt(max(1.0 - qw * qw, 0.0))
    axis = quat[1:] / max(s, 1e-9)
    return axis * angle


def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(mat))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        return _quat_normalize(
            np.array(
                [
                    0.25 * s,
                    (mat[2, 1] - mat[1, 2]) / s,
                    (mat[0, 2] - mat[2, 0]) / s,
                    (mat[1, 0] - mat[0, 1]) / s,
                ],
                dtype=np.float64,
            )
        )
    diag = np.diag(mat)
    idx = int(np.argmax(diag))
    if idx == 0:
        s = np.sqrt(max(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2], 1e-9)) * 2.0
        quat = np.array(
            [
                (mat[2, 1] - mat[1, 2]) / s,
                0.25 * s,
                (mat[0, 1] + mat[1, 0]) / s,
                (mat[0, 2] + mat[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    elif idx == 1:
        s = np.sqrt(max(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2], 1e-9)) * 2.0
        quat = np.array(
            [
                (mat[0, 2] - mat[2, 0]) / s,
                (mat[0, 1] + mat[1, 0]) / s,
                0.25 * s,
                (mat[1, 2] + mat[2, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        s = np.sqrt(max(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1], 1e-9)) * 2.0
        quat = np.array(
            [
                (mat[1, 0] - mat[0, 1]) / s,
                (mat[0, 2] + mat[2, 0]) / s,
                (mat[1, 2] + mat[2, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )
    return _quat_normalize(quat)


@dataclass(slots=True)
class PickPlaceTaskConfig:
    xml_path: str = str(
        Path(__file__).resolve().parents[2]
        / "assets"
        / "mujoco"
        / "Alicia_D_v5_6"
        / "gripper_50mm"
        / "alicia_d_follower.xml"
    )
    max_steps: int = 300
    control_dt: float = 0.05
    frame_skip: int = 10
    translation_step: float = 0.012
    rotation_step: float = 0.20
    gripper_step: float = 0.004
    workspace_min: tuple[float, float, float] = (-0.38, -0.24, 0.05)
    workspace_max: tuple[float, float, float] = (-0.08, 0.24, 0.30)
    hover_offset: tuple[float, float, float] = (0.0, 0.0, 0.13)
    reset_arm_mode: str = "home"
    home_joint_pos: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    home_gripper_target: float = 0.0
    box_spawn_xy_low: tuple[float, float] = (-0.31, -0.07)
    box_spawn_xy_high: tuple[float, float] = (-0.19, 0.07)
    basket_jitter_xy: tuple[float, float] = (0.015, 0.015)
    success_xy_radius: float = 0.04
    success_height_margin: float = 0.05
    ik_damping: float = 1e-4
    ik_max_iters: int = 64
    ik_pos_gain: float = 1.0
    ik_rot_gain: float = 0.4
    seed: int = 0


class MujocoPickPlaceTeleopEnv:
    """State-based MuJoCo pick-place environment for teleoperation copilot work."""

    action_dim = 7

    def __init__(self, cfg: PickPlaceTaskConfig | None = None, reward: PickPlaceReward | None = None):
        self.cfg = cfg or PickPlaceTaskConfig()
        self.reward = reward or PickPlaceReward()
        self.rng = np.random.default_rng(self.cfg.seed)

        self.model = mujoco.MjModel.from_xml_path(self.cfg.xml_path)
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.metadata: dict[str, Any] = {"render_fps": int(round(1.0 / self.cfg.control_dt))}

        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        self.tool_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tool0_site")
        self.box_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_free")
        self.box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        self.basket_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "basket")
        self.table_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "table")
        self.arm_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ("Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6")
        ]
        self.gripper_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ("left_finger", "right_finger")
        ]
        self.arm_qpos_adr = np.asarray([self.model.jnt_qposadr[j] for j in self.arm_joint_ids], dtype=np.int32)
        self.arm_dof_adr = np.asarray([self.model.jnt_dofadr[j] for j in self.arm_joint_ids], dtype=np.int32)
        self.box_qpos_adr = int(self.model.jnt_qposadr[self.box_joint_id])
        self.box_dof_adr = int(self.model.jnt_dofadr[self.box_joint_id])
        self.arm_ctrl_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ("pos1", "pos2", "pos3", "pos4", "pos5", "pos6")
        ]
        self.grip_ctrl_l = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_grip_l")
        self.grip_ctrl_r = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_grip_r")
        self.arm_ranges = np.asarray([self.model.jnt_range[j] for j in self.arm_joint_ids], dtype=np.float64)
        self.default_basket_pos = np.asarray(self.model.body_pos[self.basket_body_id], dtype=np.float64)

        self.model.opt.timestep = self.cfg.control_dt / float(self.cfg.frame_skip)

        mujoco.mj_forward(self.model, self.data)
        self.default_ee_quat = self._site_quat(self.data, self.site_id)
        self.default_gripper_target = 0.0
        self.step_count = 0
        self.last_obs: dict[str, np.ndarray] | None = None
        self.last_exec_action = np.zeros(self.action_dim, dtype=np.float32)
        self.desired_pos = np.zeros(3, dtype=np.float64)
        self.desired_quat = self.default_ee_quat.copy()
        self.desired_gripper = self.default_gripper_target

    @property
    def default_xml(self) -> str:
        return self.cfg.xml_path

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.last_exec_action = np.zeros(self.action_dim, dtype=np.float32)

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        box_xy = self.rng.uniform(self.cfg.box_spawn_xy_low, self.cfg.box_spawn_xy_high)
        self.data.qpos[self.box_qpos_adr : self.box_qpos_adr + 7] = np.array(
            [box_xy[0], box_xy[1], 0.064, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self.data.qvel[self.box_dof_adr : self.box_dof_adr + 6] = 0.0

        jitter = self.rng.uniform(
            low=[-self.cfg.basket_jitter_xy[0], -self.cfg.basket_jitter_xy[1], 0.0],
            high=[self.cfg.basket_jitter_xy[0], self.cfg.basket_jitter_xy[1], 0.0],
        )
        self.model.body_pos[self.basket_body_id] = self.default_basket_pos + jitter

        mujoco.mj_forward(self.model, self.data)
        if self.cfg.reset_arm_mode == "hover":
            box_pos = np.asarray(self.data.xpos[self.box_body_id], dtype=np.float64)
            self.desired_pos = np.clip(
                box_pos + np.asarray(self.cfg.hover_offset, dtype=np.float64),
                np.asarray(self.cfg.workspace_min, dtype=np.float64),
                np.asarray(self.cfg.workspace_max, dtype=np.float64),
            )
            self.desired_quat = self.default_ee_quat.copy()
            self.desired_gripper = self.default_gripper_target
            q_target = self._solve_ik(self.desired_pos, self.desired_quat)
        elif self.cfg.reset_arm_mode == "home":
            q_target = np.clip(np.asarray(self.cfg.home_joint_pos, dtype=np.float64), self.arm_ranges[:, 0], self.arm_ranges[:, 1])
            self.desired_gripper = float(np.clip(self.cfg.home_gripper_target, 0.0, 0.025))
            self.desired_pos = np.zeros(3, dtype=np.float64)
            self.desired_quat = self.default_ee_quat.copy()
        else:
            raise ValueError(f"Unsupported reset_arm_mode: {self.cfg.reset_arm_mode}")

        self.data.qpos[self.arm_qpos_adr] = q_target
        self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_ids[0]]] = self.desired_gripper
        self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_ids[1]]] = -self.desired_gripper
        self._apply_ctrl(q_target, self.desired_gripper)
        mujoco.mj_forward(self.model, self.data)
        self.desired_pos = np.asarray(self.data.site_xpos[self.site_id], dtype=np.float64).copy()
        self.desired_quat = self._site_quat(self.data, self.site_id)

        obs = self._get_obs()
        self.last_obs = obs
        info = {
            "success": self._is_success(obs),
            "box_pos": obs["box_pos"].copy(),
            "basket_pos": obs["basket_pos"].copy(),
        }
        return obs, info

    def step(
        self,
        base_action: np.ndarray,
        residual_action: np.ndarray,
        *,
        alpha: float = 1.0,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        base_action = np.clip(np.asarray(base_action, dtype=np.float32), -1.0, 1.0)
        residual_action = np.clip(np.asarray(residual_action, dtype=np.float32), -1.0, 1.0)
        commanded_action = np.clip(base_action + float(alpha) * residual_action, -1.0, 1.0)
        realized_action = commanded_action.copy()
        self.last_exec_action = realized_action
        self.step_count += 1

        self.desired_pos = np.clip(
            self.desired_pos + commanded_action[:3] * self.cfg.translation_step,
            np.asarray(self.cfg.workspace_min, dtype=np.float64),
            np.asarray(self.cfg.workspace_max, dtype=np.float64),
        )
        dq = _axis_angle_to_quat(commanded_action[3:6].astype(np.float64) * self.cfg.rotation_step)
        self.desired_quat = _quat_normalize(_quat_mul(dq, self.desired_quat))
        self.desired_gripper = float(
            np.clip(self.desired_gripper + commanded_action[6] * self.cfg.gripper_step, 0.0, 0.025)
        )

        q_target = self._solve_ik(self.desired_pos, self.desired_quat)
        self._apply_ctrl(q_target, self.desired_gripper)
        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        success = self._is_success(obs)
        reward_env, env_terms = self.reward.compute_env_reward(obs, success)
        terminated = bool(success)
        truncated = bool(self.step_count >= self.cfg.max_steps)
        info = {
            **env_terms,
            "success": success,
            "base_action": base_action.copy(),
            "residual_action": residual_action.copy(),
            "commanded_action": commanded_action.copy(),
            "realized_action": realized_action.copy(),
            "exec_action": realized_action.copy(),
            "alpha": float(alpha),
            "copilot_enabled": bool(float(alpha) > 0.0 and np.linalg.norm(residual_action) > 1e-8),
            "desired_pos": self.desired_pos.copy(),
            "desired_quat": self.desired_quat.copy(),
        }
        self.last_obs = obs
        return obs, reward_env, terminated, truncated, info

    def close(self) -> None:
        pass

    def get_teleop_target(self) -> tuple[np.ndarray, np.ndarray, float]:
        return (
            self.desired_pos.astype(np.float64).copy(),
            self.desired_quat.astype(np.float64).copy(),
            float(self.desired_gripper),
        )

    def reset_from_flat_observation(self, obs_flat: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        parsed = unflatten_observation(obs_flat)
        self.step_count = 0
        self.last_exec_action = np.zeros(self.action_dim, dtype=np.float32)

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.model.body_pos[self.basket_body_id] = self.default_basket_pos.copy()
        mujoco.mj_forward(self.model, self.data)

        parent_id = int(self.model.body_parentid[self.basket_body_id])
        parent_pos = np.asarray(self.data.xpos[parent_id], dtype=np.float64).copy()
        parent_rot = np.asarray(self.data.xmat[parent_id], dtype=np.float64).reshape(3, 3)
        basket_world = np.asarray(parsed["basket_pos"], dtype=np.float64)
        basket_local = parent_rot.T @ (basket_world - parent_pos)
        self.model.body_pos[self.basket_body_id] = basket_local

        self.data.qpos[self.arm_qpos_adr] = np.asarray(parsed["joint_pos"], dtype=np.float64)
        gripper_target = float(parsed["gripper_pos"][0])
        self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_ids[0]]] = gripper_target
        self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_ids[1]]] = -gripper_target
        self.data.qvel[self.arm_dof_adr] = np.asarray(parsed["joint_vel"], dtype=np.float64)

        box_quat = np.asarray(parsed["box_quat"], dtype=np.float64)
        box_quat = _quat_normalize(box_quat)
        self.data.qpos[self.box_qpos_adr : self.box_qpos_adr + 7] = np.concatenate(
            [np.asarray(parsed["box_pos"], dtype=np.float64), box_quat],
            axis=0,
        )
        self.data.qvel[self.box_dof_adr : self.box_dof_adr + 6] = 0.0

        self._apply_ctrl(np.asarray(parsed["joint_pos"], dtype=np.float64), gripper_target)
        mujoco.mj_forward(self.model, self.data)
        self.desired_pos = np.asarray(self.data.site_xpos[self.site_id], dtype=np.float64).copy()
        self.desired_quat = self._site_quat(self.data, self.site_id)
        self.desired_gripper = gripper_target

        obs = self._get_obs()
        self.last_obs = obs
        info = {
            "success": self._is_success(obs),
            "box_pos": obs["box_pos"].copy(),
            "basket_pos": obs["basket_pos"].copy(),
        }
        return obs, info

    def get_joint_target(self) -> tuple[np.ndarray, float]:
        return (
            np.asarray(self.data.qpos[self.arm_qpos_adr], dtype=np.float64).copy(),
            float(self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_ids[0]]]),
        )

    def action_from_joint_target(
        self,
        joint_target: np.ndarray,
        gripper_target: float,
    ) -> np.ndarray:
        prev_obs = self._get_obs()
        joint_target = np.clip(np.asarray(joint_target, dtype=np.float64), self.arm_ranges[:, 0], self.arm_ranges[:, 1])
        gripper_target = float(np.clip(gripper_target, 0.0, 0.025))
        target_pos, target_quat = self._forward_pose_from_joint_target(joint_target, gripper_target)
        pos_delta = (target_pos - np.asarray(prev_obs["ee_pos"], dtype=np.float64)) / max(self.cfg.translation_step, 1e-6)
        quat_delta = _quat_mul(target_quat, _quat_conjugate(np.asarray(prev_obs["ee_quat"], dtype=np.float64)))
        rot_delta = _quat_to_rotvec(quat_delta) / max(self.cfg.rotation_step, 1e-6)
        grip_delta = (gripper_target - float(prev_obs["gripper_pos"][0])) / max(self.cfg.gripper_step, 1e-6)
        return np.clip(
            np.concatenate([pos_delta, rot_delta, np.array([grip_delta], dtype=np.float64)], axis=0),
            -1.0,
            1.0,
        ).astype(np.float32)

    def get_home_target(self) -> tuple[np.ndarray, float]:
        return (
            np.asarray(self.cfg.home_joint_pos, dtype=np.float64).copy(),
            float(self.cfg.home_gripper_target),
        )

    def sync_absolute_joint_target(
        self,
        joint_target: np.ndarray,
        gripper_target: float,
        *,
        n_substeps: int | None = None,
    ) -> dict[str, np.ndarray]:
        joint_target = np.clip(np.asarray(joint_target, dtype=np.float64), self.arm_ranges[:, 0], self.arm_ranges[:, 1])
        gripper_target = float(np.clip(gripper_target, 0.0, 0.025))
        self._apply_ctrl(joint_target, gripper_target)
        for _ in range(self.cfg.frame_skip if n_substeps is None else max(1, int(n_substeps))):
            mujoco.mj_step(self.model, self.data)
        self.desired_pos = np.asarray(self.data.site_xpos[self.site_id], dtype=np.float64).copy()
        self.desired_quat = self._site_quat(self.data, self.site_id)
        self.desired_gripper = float(self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_ids[0]]])
        obs = self._get_obs()
        self.last_obs = obs
        return obs

    def step_absolute_joint_target(
        self,
        joint_target: np.ndarray,
        gripper_target: float,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        prev_obs = self._get_obs()
        joint_target = np.clip(np.asarray(joint_target, dtype=np.float64), self.arm_ranges[:, 0], self.arm_ranges[:, 1])
        gripper_target = float(np.clip(gripper_target, 0.0, 0.025))

        self.step_count += 1
        self._apply_ctrl(joint_target, gripper_target)
        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.desired_pos = np.asarray(self.data.site_xpos[self.site_id], dtype=np.float64).copy()
        self.desired_quat = self._site_quat(self.data, self.site_id)
        self.desired_gripper = float(self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_ids[0]]])

        obs = self._get_obs()
        pos_delta = (np.asarray(obs["ee_pos"], dtype=np.float64) - np.asarray(prev_obs["ee_pos"], dtype=np.float64)) / max(self.cfg.translation_step, 1e-6)
        quat_delta = _quat_mul(
            np.asarray(obs["ee_quat"], dtype=np.float64),
            _quat_conjugate(np.asarray(prev_obs["ee_quat"], dtype=np.float64)),
        )
        rot_delta = _quat_to_rotvec(quat_delta) / max(self.cfg.rotation_step, 1e-6)
        grip_delta = (float(obs["gripper_pos"][0]) - float(prev_obs["gripper_pos"][0])) / max(self.cfg.gripper_step, 1e-6)
        realized_action = np.clip(
            np.concatenate([pos_delta, rot_delta, np.array([grip_delta], dtype=np.float64)], axis=0),
            -1.0,
            1.0,
        ).astype(np.float32)
        commanded_action = realized_action.copy()
        self.last_exec_action = realized_action

        success = self._is_success(obs)
        reward_env, env_terms = self.reward.compute_env_reward(obs, success)
        terminated = bool(success)
        truncated = bool(self.step_count >= self.cfg.max_steps)
        info = {
            **env_terms,
            "success": success,
            "base_action": realized_action.copy(),
            "residual_action": np.zeros(self.action_dim, dtype=np.float32),
            "commanded_action": commanded_action.copy(),
            "realized_action": realized_action.copy(),
            "exec_action": realized_action.copy(),
            "alpha": 0.0,
            "copilot_enabled": False,
            "desired_pos": self.desired_pos.copy(),
            "desired_quat": self.desired_quat.copy(),
            "target_joint_pos": joint_target.astype(np.float64).copy(),
            "target_gripper": float(gripper_target),
        }
        self.last_obs = obs
        return obs, reward_env, terminated, truncated, info

    def step_base_joint_target(
        self,
        base_joint_target: np.ndarray,
        base_gripper_target: float,
        residual_action: np.ndarray,
        *,
        alpha: float = 1.0,
        base_action: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        prev_obs = self._get_obs()
        base_joint_target = np.clip(np.asarray(base_joint_target, dtype=np.float64), self.arm_ranges[:, 0], self.arm_ranges[:, 1])
        base_gripper_target = float(np.clip(base_gripper_target, 0.0, 0.025))
        residual_action = np.clip(np.asarray(residual_action, dtype=np.float32), -1.0, 1.0)
        inferred_base_action = self.action_from_joint_target(base_joint_target, base_gripper_target)
        base_action = inferred_base_action.copy() if base_action is None else np.asarray(base_action, dtype=np.float32)
        commanded_action = np.clip(inferred_base_action + float(alpha) * residual_action, -1.0, 1.0).astype(np.float32)
        self.step_count += 1
        base_pos, base_quat = self._forward_pose_from_joint_target(base_joint_target, base_gripper_target)

        if float(alpha) == 0.0 or np.linalg.norm(residual_action) < 1e-8:
            q_target = base_joint_target
            target_gripper = base_gripper_target
            self.desired_pos = base_pos.copy()
            self.desired_quat = base_quat.copy()
            self.desired_gripper = target_gripper
        else:
            target_pos = np.clip(
                base_pos + float(alpha) * residual_action[:3].astype(np.float64) * self.cfg.translation_step,
                np.asarray(self.cfg.workspace_min, dtype=np.float64),
                np.asarray(self.cfg.workspace_max, dtype=np.float64),
            )
            dq = _axis_angle_to_quat(float(alpha) * residual_action[3:6].astype(np.float64) * self.cfg.rotation_step)
            target_quat = _quat_normalize(_quat_mul(dq, base_quat))
            target_gripper = float(
                np.clip(base_gripper_target + float(alpha) * float(residual_action[6]) * self.cfg.gripper_step, 0.0, 0.025)
            )
            q_target = self._solve_ik(target_pos, target_quat, initial_q_arm=base_joint_target)
            self.desired_pos = target_pos.copy()
            self.desired_quat = target_quat.copy()
            self.desired_gripper = target_gripper

        self._apply_ctrl(q_target, target_gripper)
        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        pos_delta = (np.asarray(obs["ee_pos"], dtype=np.float64) - np.asarray(prev_obs["ee_pos"], dtype=np.float64)) / max(self.cfg.translation_step, 1e-6)
        quat_delta = _quat_mul(
            np.asarray(obs["ee_quat"], dtype=np.float64),
            _quat_conjugate(np.asarray(prev_obs["ee_quat"], dtype=np.float64)),
        )
        rot_delta = _quat_to_rotvec(quat_delta) / max(self.cfg.rotation_step, 1e-6)
        grip_delta = (float(obs["gripper_pos"][0]) - float(prev_obs["gripper_pos"][0])) / max(self.cfg.gripper_step, 1e-6)
        realized_action = np.clip(
            np.concatenate([pos_delta, rot_delta, np.array([grip_delta], dtype=np.float64)], axis=0),
            -1.0,
            1.0,
        ).astype(np.float32)
        self.last_exec_action = realized_action

        success = self._is_success(obs)
        reward_env, env_terms = self.reward.compute_env_reward(obs, success)
        terminated = bool(success)
        truncated = bool(self.step_count >= self.cfg.max_steps)
        info = {
            **env_terms,
            "success": success,
            "base_action": base_action.copy(),
            "residual_action": residual_action.copy(),
            "commanded_action": commanded_action.copy(),
            "realized_action": realized_action.copy(),
            "exec_action": realized_action.copy(),
            "alpha": float(alpha),
            "copilot_enabled": bool(float(alpha) > 0.0 and np.linalg.norm(residual_action) > 1e-8),
            "desired_pos": self.desired_pos.copy(),
            "desired_quat": self.desired_quat.copy(),
            "base_joint_target": base_joint_target.astype(np.float64).copy(),
            "base_gripper_target": float(base_gripper_target),
            "target_joint_pos": q_target.astype(np.float64).copy(),
            "target_gripper": float(target_gripper),
        }
        self.last_obs = obs
        return obs, reward_env, terminated, truncated, info

    def render_rgb(self, width: int = 640, height: int = 480) -> np.ndarray:
        renderer = mujoco.Renderer(self.model, width=width, height=height)
        renderer.update_scene(self.data)
        rgb = renderer.render()
        close_fn = getattr(renderer, "close", None)
        if callable(close_fn):
            close_fn()
        return rgb

    def _apply_ctrl(self, q_target: np.ndarray, gripper_target: float) -> None:
        for ctrl_id, value in zip(self.arm_ctrl_ids, q_target, strict=False):
            self.data.ctrl[ctrl_id] = float(value)
        self.data.ctrl[self.grip_ctrl_l] = gripper_target
        self.data.ctrl[self.grip_ctrl_r] = -gripper_target

    def _solve_ik(self, target_pos: np.ndarray, target_quat: np.ndarray, initial_q_arm: np.ndarray | None = None) -> np.ndarray:
        self.ik_data.qpos[:] = self.data.qpos
        self.ik_data.qvel[:] = self.data.qvel
        if initial_q_arm is not None:
            self.ik_data.qpos[self.arm_qpos_adr] = np.asarray(initial_q_arm, dtype=np.float64)
        mujoco.mj_forward(self.model, self.ik_data)
        q_arm = self.ik_data.qpos[self.arm_qpos_adr].copy()

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        eye6 = np.eye(6, dtype=np.float64)
        for _ in range(self.cfg.ik_max_iters):
            current_pos = np.asarray(self.ik_data.site_xpos[self.site_id], dtype=np.float64)
            current_quat = self._site_quat(self.ik_data, self.site_id)
            pos_err = (target_pos - current_pos) * self.cfg.ik_pos_gain
            quat_err = _quat_mul(target_quat, _quat_conjugate(current_quat))
            rot_err = _quat_to_rotvec(quat_err) * self.cfg.ik_rot_gain
            err = np.concatenate([pos_err, rot_err], axis=0)
            if np.linalg.norm(err) < 1e-4:
                break

            mujoco.mj_jacSite(self.model, self.ik_data, jacp, jacr, self.site_id)
            jac = np.vstack([jacp[:, self.arm_dof_adr], jacr[:, self.arm_dof_adr]])
            lhs = jac @ jac.T + self.cfg.ik_damping * eye6
            dq = jac.T @ np.linalg.solve(lhs, err)
            q_arm = np.clip(q_arm + dq, self.arm_ranges[:, 0], self.arm_ranges[:, 1])
            self.ik_data.qpos[self.arm_qpos_adr] = q_arm
            mujoco.mj_forward(self.model, self.ik_data)
        return q_arm.astype(np.float64)

    def _forward_pose_from_joint_target(self, joint_target: np.ndarray, gripper_target: float) -> tuple[np.ndarray, np.ndarray]:
        self.ik_data.qpos[:] = self.data.qpos
        self.ik_data.qvel[:] = self.data.qvel
        self.ik_data.qpos[self.arm_qpos_adr] = np.asarray(joint_target, dtype=np.float64)
        self.ik_data.qpos[self.model.jnt_qposadr[self.gripper_joint_ids[0]]] = float(gripper_target)
        self.ik_data.qpos[self.model.jnt_qposadr[self.gripper_joint_ids[1]]] = -float(gripper_target)
        mujoco.mj_forward(self.model, self.ik_data)
        pos = np.asarray(self.ik_data.site_xpos[self.site_id], dtype=np.float64).copy()
        quat = self._site_quat(self.ik_data, self.site_id)
        return pos, quat

    def _get_obs(self) -> dict[str, np.ndarray]:
        joint_pos = self.data.qpos[self.arm_qpos_adr].astype(np.float32).copy()
        joint_vel = self.data.qvel[self.arm_dof_adr].astype(np.float32).copy()
        gripper_pos = np.array([self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_ids[0]]]], dtype=np.float32)
        ee_pos = np.asarray(self.data.site_xpos[self.site_id], dtype=np.float32).copy()
        ee_quat = self._site_quat(self.data, self.site_id).astype(np.float32)
        box_pos = np.asarray(self.data.xpos[self.box_body_id], dtype=np.float32).copy()
        box_quat = np.asarray(self.data.xquat[self.box_body_id], dtype=np.float32).copy()
        box_linvel = np.asarray(self.data.cvel[self.box_body_id][3:], dtype=np.float32).copy()
        box_angvel = np.asarray(self.data.cvel[self.box_body_id][:3], dtype=np.float32).copy()
        basket_pos = np.asarray(self.data.xpos[self.basket_body_id], dtype=np.float32).copy()
        box_to_ee = box_pos - ee_pos
        box_to_basket = basket_pos - box_pos
        ee_to_basket = basket_pos - ee_pos
        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "gripper_pos": gripper_pos,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "box_pos": box_pos,
            "box_quat": box_quat,
            "box_linvel": box_linvel,
            "box_angvel": box_angvel,
            "basket_pos": basket_pos,
            "box_to_ee": box_to_ee.astype(np.float32),
            "box_to_basket": box_to_basket.astype(np.float32),
            "ee_to_basket": ee_to_basket.astype(np.float32),
        }

    def _is_success(self, obs: dict[str, np.ndarray]) -> bool:
        box_pos = np.asarray(obs["box_pos"], dtype=np.float32)
        basket_pos = np.asarray(obs["basket_pos"], dtype=np.float32)
        xy_ok = np.linalg.norm((box_pos - basket_pos)[:2]) < self.cfg.success_xy_radius
        z_ok = abs(float(box_pos[2] - basket_pos[2])) < self.cfg.success_height_margin
        return bool(xy_ok and z_ok)

    def _site_quat(self, data: mujoco.MjData, site_id: int) -> np.ndarray:
        return _mat_to_quat(np.asarray(data.site_xmat[site_id], dtype=np.float64))
