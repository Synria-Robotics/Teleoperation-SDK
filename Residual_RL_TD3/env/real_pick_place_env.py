from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from Residual_RL_TD3.common.observation_utils import OBS_STATE_KEY, flatten_observation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALICIA_D_SDK_ROOT = PROJECT_ROOT / "Alicia-D-SDK"
if str(ALICIA_D_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(ALICIA_D_SDK_ROOT))

try:  # pragma: no cover - hardware-only dependency path
    import alicia_d_sdk
except ImportError:  # pragma: no cover - hardware-only dependency path
    alicia_d_sdk = None

try:  # pragma: no cover - hardware-only dependency path
    from robocore.kinematics import forward_kinematics
    from robocore.transform import matrix_to_quaternion
except ImportError:  # pragma: no cover - hardware-only dependency path
    forward_kinematics = None
    matrix_to_quaternion = None


GRIPPER_SDK_MAX = 1000.0
GRIPPER_MJ_MAX = 0.025


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
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
        dtype=np.float64,
    )


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = _quat_normalize(quat)
    return np.array([w, -x, -y, -z], dtype=np.float64)


def _quat_align_sign(quat: np.ndarray, reference: np.ndarray) -> np.ndarray:
    quat = _quat_normalize(quat)
    reference = _quat_normalize(reference)
    if float(np.dot(quat, reference)) < 0.0:
        return -quat
    return quat


def _axis_angle_to_quat(vec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vec))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = np.asarray(vec, dtype=np.float64) / angle
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


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def _wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)


def gripper_sdk_to_mujoco(sdk_val: float) -> float:
    clamped = float(np.clip(sdk_val, 0.0, GRIPPER_SDK_MAX))
    return float((1.0 - clamped / GRIPPER_SDK_MAX) * GRIPPER_MJ_MAX)


def gripper_mujoco_to_sdk(mj_val: float) -> float:
    clamped = float(np.clip(mj_val, 0.0, GRIPPER_MJ_MAX))
    return float((1.0 - clamped / GRIPPER_MJ_MAX) * GRIPPER_SDK_MAX)


def _apply_action_delta_to_obs(
    obs: dict[str, np.ndarray],
    commanded_action: np.ndarray,
    *,
    translation_step: float,
    rotation_step: float,
    gripper_step: float,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
    gripper_max: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    current_pos = np.asarray(obs["ee_pos"], dtype=np.float64).reshape(3)
    current_quat = _quat_normalize(np.asarray(obs["ee_quat"], dtype=np.float64).reshape(4))
    current_gripper = float(np.asarray(obs["gripper_pos"], dtype=np.float64).reshape(-1)[0])
    commanded_action = np.asarray(commanded_action, dtype=np.float64).reshape(7)

    target_pos = np.clip(
        current_pos + commanded_action[:3] * translation_step,
        workspace_min,
        workspace_max,
    )
    dq = _axis_angle_to_quat(commanded_action[3:6] * rotation_step)
    target_quat = _quat_normalize(_quat_mul(dq, current_quat))
    target_gripper = float(np.clip(current_gripper + commanded_action[6] * gripper_step, 0.0, gripper_max))
    return target_pos, target_quat, target_gripper


def _build_robot_only_observation(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    gripper_pos: float,
    ee_pos: np.ndarray,
    ee_quat: np.ndarray,
) -> dict[str, np.ndarray]:
    box_pos = np.zeros(3, dtype=np.float32)
    basket_pos = np.zeros(3, dtype=np.float32)
    obs = {
        "joint_pos": np.asarray(joint_pos, dtype=np.float32).reshape(6),
        "joint_vel": np.asarray(joint_vel, dtype=np.float32).reshape(6),
        "gripper_pos": np.asarray([gripper_pos], dtype=np.float32),
        "ee_pos": np.asarray(ee_pos, dtype=np.float32).reshape(3),
        "ee_quat": np.asarray(ee_quat, dtype=np.float32).reshape(4),
        "box_pos": box_pos.copy(),
        "box_quat": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "box_linvel": np.zeros(3, dtype=np.float32),
        "box_angvel": np.zeros(3, dtype=np.float32),
        "basket_pos": basket_pos.copy(),
        "box_to_ee": box_pos - np.asarray(ee_pos, dtype=np.float32).reshape(3),
        "box_to_basket": basket_pos - box_pos,
        "ee_to_basket": basket_pos - np.asarray(ee_pos, dtype=np.float32).reshape(3),
    }
    obs[OBS_STATE_KEY] = flatten_observation(obs)
    return obs


def _compute_realized_action(
    prev_obs: dict[str, np.ndarray],
    obs: dict[str, np.ndarray],
    cfg: TeleopBackendConfig,
) -> np.ndarray:
    pos_delta = (
        np.asarray(obs["ee_pos"], dtype=np.float64) - np.asarray(prev_obs["ee_pos"], dtype=np.float64)
    ) / max(cfg.translation_step, 1e-6)
    prev_quat = _quat_normalize(np.asarray(prev_obs["ee_quat"], dtype=np.float64))
    obs_quat = _quat_align_sign(np.asarray(obs["ee_quat"], dtype=np.float64), prev_quat)
    quat_delta = _quat_mul(obs_quat, _quat_conjugate(prev_quat))
    rot_delta = _quat_to_rotvec(quat_delta) / max(cfg.rotation_step, 1e-6)
    grip_delta = (
        float(np.asarray(obs["gripper_pos"], dtype=np.float64)[0]) - float(np.asarray(prev_obs["gripper_pos"], dtype=np.float64)[0])
    ) / max(cfg.gripper_step, 1e-6)
    return np.clip(
        np.concatenate([pos_delta, rot_delta, np.array([grip_delta], dtype=np.float64)], axis=0),
        -1.0,
        1.0,
    ).astype(np.float32)


@dataclass(slots=True)
class RealPickPlaceEnvConfig:
    max_steps: int = 1000
    control_dt: float = 0.05
    translation_step: float = 0.012
    rotation_step: float = 0.20
    gripper_step: float = 0.004
    workspace_min: tuple[float, float, float] | None = (-0.38, -0.24, 0.05)
    workspace_max: tuple[float, float, float] | None = (-0.08, 0.24, 0.30)
    port: str = ""
    version: str = "v5_6"
    variant: str | None = None
    gripper_type: str | None = "50mm"
    model_format: str = "urdf"
    debug_mode: bool = False
    auto_connect: bool = True
    ik_backend: str | None = None
    device: str = "cpu"
    speed_deg_s: float = 30.0
    gripper_speed_deg_s: float | None = 483.4
    read_timeout: float = 1.0
    home_joint_pos: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    home_gripper_target: float = 0.0
    home_on_reset: bool = False
    wait_for_completion: bool = False
    force_execute_ik: bool = True
    pose_method: str = "dls"
    pose_pos_tol: float = 1e-3
    pose_ori_tol: float = 1e-3
    pose_max_iters: int = 500


class RealPickPlaceEnv:
    """Real follower pick-place env used directly by Residual_RL_TD3 training."""

    action_dim = 7

    def __init__(self, cfg: RealPickPlaceEnvConfig | None = None):
        if alicia_d_sdk is None:  # pragma: no cover - hardware-only dependency path
            raise ImportError("alicia_d_sdk is required for RealPickPlaceEnv")
        if forward_kinematics is None or matrix_to_quaternion is None:  # pragma: no cover
            raise ImportError("robocore is required for RealPickPlaceEnv")

        self.cfg = cfg or RealPickPlaceEnvConfig()
        self.robot = alicia_d_sdk.create_robot(
            port=self.cfg.port,
            version=self.cfg.version,
            variant=self.cfg.variant,
            model_format=self.cfg.model_format,
            debug_mode=self.cfg.debug_mode,
            auto_connect=self.cfg.auto_connect,
            backend=self.cfg.ik_backend,
            device=self.cfg.device,
            gripper_type=self.cfg.gripper_type,
        )
        self.robot_api = self.robot
        self.sdk_api = self.robot
        self.metadata: dict[str, Any] = {"render_fps": int(round(1.0 / max(self.cfg.control_dt, 1e-6)))}
        self.step_count = 0
        self.last_obs: dict[str, np.ndarray] | None = None
        self.last_exec_action = np.zeros(self.action_dim, dtype=np.float32)
        self.desired_pos = np.zeros(3, dtype=np.float64)
        self.desired_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.desired_gripper = 0.0
        self.desired_joint_target = np.zeros(6, dtype=np.float64)
        self._last_joint_pos: np.ndarray | None = None
        self._last_state_timestamp: float | None = None

    def get_sdk_api(self):
        return self.sdk_api

    def get_robot_api(self):
        return self.robot_api

    def _clip_workspace(self, pos: np.ndarray) -> np.ndarray:
        target = np.asarray(pos, dtype=np.float64).reshape(3)
        if self.cfg.workspace_min is None or self.cfg.workspace_max is None:
            return target
        return np.clip(
            target,
            np.asarray(self.cfg.workspace_min, dtype=np.float64),
            np.asarray(self.cfg.workspace_max, dtype=np.float64),
        )

    def _read_robot_observation(self) -> dict[str, np.ndarray]:
        joint_state = self.robot.get_robot_state("joint_gripper", timeout=self.cfg.read_timeout)
        if joint_state is None:
            raise RuntimeError("Failed to read joint/gripper state from follower robot")

        pose = self.robot.get_pose()
        if pose is None:
            raise RuntimeError("Failed to read end-effector pose from follower robot")

        joint_pos = np.asarray(list(joint_state.angles)[:6], dtype=np.float64)
        if joint_pos.shape[0] < 6:
            joint_pos = np.pad(joint_pos, (0, 6 - joint_pos.shape[0]))

        raw_timestamp = getattr(joint_state, "timestamp", None)
        state_timestamp = float(raw_timestamp) if raw_timestamp is not None else time.perf_counter()
        if self._last_joint_pos is None or self._last_state_timestamp is None:
            joint_vel = np.zeros_like(joint_pos, dtype=np.float64)
        else:
            dt = max(state_timestamp - self._last_state_timestamp, 1e-6)
            joint_vel = (joint_pos - self._last_joint_pos) / dt
        self._last_joint_pos = joint_pos.copy()
        self._last_state_timestamp = state_timestamp

        gripper_pos = gripper_sdk_to_mujoco(float(getattr(joint_state, "gripper", GRIPPER_SDK_MAX)))
        ee_pos = np.asarray(pose["position"], dtype=np.float64).reshape(3)
        ee_quat = _xyzw_to_wxyz(np.asarray(pose["quaternion_xyzw"], dtype=np.float64).reshape(4))
        if self.last_obs is not None:
            ee_quat = _quat_align_sign(ee_quat, np.asarray(self.last_obs["ee_quat"], dtype=np.float64))

        return _build_robot_only_observation(joint_pos, joint_vel, gripper_pos, ee_pos, ee_quat)

    def _read_and_cache_observation(self) -> dict[str, np.ndarray]:
        obs = self._read_robot_observation()
        self.last_obs = obs
        return obs

    def _sync_targets_from_current_state(self) -> None:
        obs = self._read_and_cache_observation()
        self.desired_joint_target = np.asarray(obs["joint_pos"], dtype=np.float64).copy()
        self.desired_pos = np.asarray(obs["ee_pos"], dtype=np.float64).copy()
        self.desired_quat = _quat_normalize(np.asarray(obs["ee_quat"], dtype=np.float64))
        self.desired_gripper = float(np.asarray(obs["gripper_pos"], dtype=np.float64)[0])

    def _forward_pose_from_joint_target(
        self,
        joint_target: np.ndarray,
        gripper_target: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        del gripper_target
        transform = np.asarray(
            forward_kinematics(
                self.robot.robot_model,
                np.asarray(joint_target, dtype=np.float64).reshape(6),
                return_end=True,
            ),
            dtype=np.float64,
        )
        pos = np.asarray(transform[:3, 3], dtype=np.float64).copy()
        quat_xyzw = np.asarray(matrix_to_quaternion(transform[:3, :3]), dtype=np.float64).reshape(4)
        quat_wxyz = _xyzw_to_wxyz(quat_xyzw)
        return pos, _quat_normalize(quat_wxyz)

    def _command_joint_target(self, joint_target: np.ndarray, gripper_target: float) -> np.ndarray:
        joint_target = np.asarray(joint_target, dtype=np.float64).reshape(6)
        gripper_sdk = int(round(gripper_mujoco_to_sdk(gripper_target)))
        success = self.robot.set_robot_state(
            target_joints=joint_target.tolist(),
            gripper_value=gripper_sdk,
            joint_format="rad",
            speed_deg_s=self.cfg.speed_deg_s,
            gripper_speed_deg_s=self.cfg.gripper_speed_deg_s,
            wait_for_completion=self.cfg.wait_for_completion,
        )
        if not success:
            raise RuntimeError("Failed to command follower joint target")
        self.desired_joint_target = joint_target.astype(np.float64).copy()
        return joint_target.astype(np.float64)

    def _command_pose_target(
        self,
        ee_pos: np.ndarray,
        ee_quat: np.ndarray,
        gripper_target: float,
    ) -> np.ndarray:
        target_pose = np.concatenate(
            [
                np.asarray(ee_pos, dtype=np.float64).reshape(3),
                _wxyz_to_xyzw(np.asarray(ee_quat, dtype=np.float64).reshape(4)),
            ],
            axis=0,
        )
        ik_result = self.robot.set_pose(
            target_pose.tolist(),
            backend=self.cfg.ik_backend,
            method=self.cfg.pose_method,
            pos_tol=self.cfg.pose_pos_tol,
            ori_tol=self.cfg.pose_ori_tol,
            max_iters=self.cfg.pose_max_iters,
            speed_deg_s=self.cfg.speed_deg_s,
            gripper_speed_deg_s=self.cfg.gripper_speed_deg_s,
            execute=False,
            force_execute=False,
        )
        q_target = ik_result.get("q")
        if q_target is None:
            raise RuntimeError(f"IK failed for follower target pose: {ik_result.get('message', 'no solution')}")
        if not bool(ik_result.get("success", False)) and not self.cfg.force_execute_ik:
            raise RuntimeError(f"IK did not converge for follower target pose: {ik_result.get('message', 'unknown')}")
        return self._command_joint_target(np.asarray(q_target, dtype=np.float64), gripper_target)

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del seed
        self.step_count = 0
        self.last_obs = None
        self.last_exec_action = np.zeros(self.action_dim, dtype=np.float32)
        self._last_joint_pos = None
        self._last_state_timestamp = None
        if self.cfg.home_on_reset:
            self._command_joint_target(
                np.asarray(self.cfg.home_joint_pos, dtype=np.float64),
                float(self.cfg.home_gripper_target),
            )
        self._sync_targets_from_current_state()
        obs = self.last_obs if self.last_obs is not None else self._read_and_cache_observation()
        info = {
            "success": False,
            "task_state_available": False,
            "real_robot": True,
        }
        return obs, info

    def step(
        self,
        base_action: np.ndarray,
        residual_action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        prev_obs = self.last_obs if self.last_obs is not None else self._read_and_cache_observation()
        base_action = np.clip(np.asarray(base_action, dtype=np.float32), -1.0, 1.0)
        residual_action = np.clip(np.asarray(residual_action, dtype=np.float32), -1.0, 1.0)
        commanded_action = np.clip(base_action + residual_action, -1.0, 1.0).astype(np.float32)

        self.step_count += 1
        self.desired_pos = self._clip_workspace(
            self.desired_pos + commanded_action[:3].astype(np.float64) * self.cfg.translation_step
        )
        dq = _axis_angle_to_quat(commanded_action[3:6].astype(np.float64) * self.cfg.rotation_step)
        self.desired_quat = _quat_normalize(_quat_mul(dq, self.desired_quat))
        self.desired_gripper = float(
            np.clip(self.desired_gripper + float(commanded_action[6]) * self.cfg.gripper_step, 0.0, GRIPPER_MJ_MAX)
        )

        q_target = self._command_pose_target(self.desired_pos, self.desired_quat, self.desired_gripper)
        obs = self._read_and_cache_observation()
        realized_action = _compute_realized_action(prev_obs, obs, self.cfg)
        self.last_exec_action = realized_action

        truncated = bool(self.step_count >= self.cfg.max_steps)
        info = {
            "success": False,
            "task_state_available": False,
            "real_robot": True,
            "base_action": base_action.copy(),
            "residual_action": residual_action.copy(),
            "commanded_action": commanded_action.copy(),
            "realized_action": realized_action.copy(),
            "exec_action": realized_action.copy(),
            "desired_pos": self.desired_pos.copy(),
            "desired_quat": self.desired_quat.copy(),
            "target_ee_pos": self.desired_pos.copy(),
            "target_ee_quat": self.desired_quat.copy(),
            "target_joint_pos": q_target.astype(np.float64).copy(),
            "target_gripper": float(self.desired_gripper),
        }
        return obs, 0.0, False, truncated, info

    def step_absolute_joint_target(
        self,
        joint_target: np.ndarray,
        gripper_target: float,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        prev_obs = self.last_obs if self.last_obs is not None else self._read_and_cache_observation()
        joint_target = np.asarray(joint_target, dtype=np.float64).reshape(6)
        gripper_target = float(np.clip(gripper_target, 0.0, GRIPPER_MJ_MAX))
        target_ee_pos, target_ee_quat = self._forward_pose_from_joint_target(joint_target, gripper_target)
        base_action = self.action_from_joint_target(joint_target, gripper_target)

        self.step_count += 1
        q_target = self._command_joint_target(joint_target, gripper_target)
        self.desired_pos = target_ee_pos.copy()
        self.desired_quat = target_ee_quat.copy()
        self.desired_gripper = gripper_target

        obs = self._read_and_cache_observation()
        realized_action = _compute_realized_action(prev_obs, obs, self.cfg)
        self.last_exec_action = realized_action

        truncated = bool(self.step_count >= self.cfg.max_steps)
        info = {
            "success": False,
            "task_state_available": False,
            "real_robot": True,
            "base_action": base_action.copy(),
            "residual_action": np.zeros(self.action_dim, dtype=np.float32),
            "commanded_action": base_action.copy(),
            "realized_action": realized_action.copy(),
            "exec_action": realized_action.copy(),
            "desired_pos": self.desired_pos.copy(),
            "desired_quat": self.desired_quat.copy(),
            "target_ee_pos": target_ee_pos.copy(),
            "target_ee_quat": target_ee_quat.copy(),
            "target_joint_pos": q_target.astype(np.float64).copy(),
            "target_gripper": float(gripper_target),
        }
        return obs, 0.0, False, truncated, info

    def step_absolute_ee_target(
        self,
        ee_pos: np.ndarray,
        ee_quat: np.ndarray,
        gripper_target: float,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        prev_obs = self.last_obs if self.last_obs is not None else self._read_and_cache_observation()
        target_pos = self._clip_workspace(np.asarray(ee_pos, dtype=np.float64).reshape(3))
        target_quat = _quat_normalize(np.asarray(ee_quat, dtype=np.float64).reshape(4))
        gripper_target = float(np.clip(gripper_target, 0.0, GRIPPER_MJ_MAX))
        base_action = self.action_from_ee_target(target_pos, target_quat, gripper_target)

        self.step_count += 1
        q_target = self._command_pose_target(target_pos, target_quat, gripper_target)
        self.desired_pos = target_pos.copy()
        self.desired_quat = target_quat.copy()
        self.desired_gripper = gripper_target

        obs = self._read_and_cache_observation()
        realized_action = _compute_realized_action(prev_obs, obs, self.cfg)
        self.last_exec_action = realized_action

        truncated = bool(self.step_count >= self.cfg.max_steps)
        info = {
            "success": False,
            "task_state_available": False,
            "real_robot": True,
            "base_action": base_action.copy(),
            "residual_action": np.zeros(self.action_dim, dtype=np.float32),
            "commanded_action": base_action.copy(),
            "realized_action": realized_action.copy(),
            "exec_action": realized_action.copy(),
            "desired_pos": self.desired_pos.copy(),
            "desired_quat": self.desired_quat.copy(),
            "target_ee_pos": target_pos.copy(),
            "target_ee_quat": target_quat.copy(),
            "target_joint_pos": q_target.astype(np.float64).copy(),
            "target_gripper": float(gripper_target),
        }
        return obs, 0.0, False, truncated, info

    def step_base_joint_target(
        self,
        base_joint_target: np.ndarray,
        base_gripper_target: float,
        residual_action: np.ndarray,
        *,
        base_action: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        prev_obs = self.last_obs if self.last_obs is not None else self._read_and_cache_observation()
        base_joint_target = np.asarray(base_joint_target, dtype=np.float64).reshape(6)
        base_gripper_target = float(np.clip(base_gripper_target, 0.0, GRIPPER_MJ_MAX))
        residual_action = np.clip(np.asarray(residual_action, dtype=np.float32), -1.0, 1.0)
        inferred_base_action = self.action_from_joint_target(base_joint_target, base_gripper_target)
        base_action = inferred_base_action.copy() if base_action is None else np.asarray(base_action, dtype=np.float32)
        commanded_action = np.clip(base_action + residual_action, -1.0, 1.0).astype(np.float32)
        base_pos, base_quat = self._forward_pose_from_joint_target(base_joint_target, base_gripper_target)

        self.step_count += 1
        target_pos, target_quat, target_gripper = _apply_action_delta_to_obs(
            prev_obs,
            commanded_action,
            translation_step=self.cfg.translation_step,
            rotation_step=self.cfg.rotation_step,
            gripper_step=self.cfg.gripper_step,
            workspace_min=np.asarray(self.cfg.workspace_min, dtype=np.float64),
            workspace_max=np.asarray(self.cfg.workspace_max, dtype=np.float64),
            gripper_max=GRIPPER_MJ_MAX,
        )
        q_target = self._command_pose_target(target_pos, target_quat, target_gripper)

        self.desired_pos = target_pos.copy()
        self.desired_quat = target_quat.copy()
        self.desired_gripper = target_gripper

        obs = self._read_and_cache_observation()
        realized_action = _compute_realized_action(prev_obs, obs, self.cfg)
        self.last_exec_action = realized_action

        truncated = bool(self.step_count >= self.cfg.max_steps)
        info = {
            "success": False,
            "task_state_available": False,
            "real_robot": True,
            "base_action": base_action.copy(),
            "residual_action": residual_action.copy(),
            "commanded_action": commanded_action.copy(),
            "realized_action": realized_action.copy(),
            "exec_action": realized_action.copy(),
            "desired_pos": self.desired_pos.copy(),
            "desired_quat": self.desired_quat.copy(),
            "base_ee_pos": base_pos.copy(),
            "base_ee_quat": base_quat.copy(),
            "target_ee_pos": self.desired_pos.copy(),
            "target_ee_quat": self.desired_quat.copy(),
            "base_joint_target": base_joint_target.astype(np.float64).copy(),
            "base_gripper_target": float(base_gripper_target),
            "target_joint_pos": q_target.astype(np.float64).copy(),
            "target_gripper": float(target_gripper),
        }
        return obs, 0.0, False, truncated, info

    def sync_base_joint_target(
        self,
        base_joint_target: np.ndarray,
        base_gripper_target: float,
        *,
        base_action: np.ndarray | None = None,
        residual_action: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        prev_step_count = int(self.step_count)
        obs, _, _, _, info = self.step_base_joint_target(
            base_joint_target,
            base_gripper_target,
            np.zeros(7, dtype=np.float32) if residual_action is None else residual_action,
            base_action=base_action,
        )
        self.step_count = prev_step_count
        return obs, info

    def action_from_joint_target(
        self,
        joint_target: np.ndarray,
        gripper_target: float,
    ) -> np.ndarray:
        prev_obs = self.last_obs if self.last_obs is not None else self._read_and_cache_observation()
        joint_target = np.asarray(joint_target, dtype=np.float64).reshape(6)
        gripper_target = float(np.clip(gripper_target, 0.0, GRIPPER_MJ_MAX))
        target_pos, target_quat = self._forward_pose_from_joint_target(joint_target, gripper_target)
        prev_quat = _quat_normalize(np.asarray(prev_obs["ee_quat"], dtype=np.float64))
        target_quat = _quat_align_sign(target_quat, prev_quat)
        pos_delta = (target_pos - np.asarray(prev_obs["ee_pos"], dtype=np.float64)) / max(self.cfg.translation_step, 1e-6)
        quat_delta = _quat_mul(target_quat, _quat_conjugate(prev_quat))
        rot_delta = _quat_to_rotvec(quat_delta) / max(self.cfg.rotation_step, 1e-6)
        grip_delta = (gripper_target - float(np.asarray(prev_obs["gripper_pos"], dtype=np.float64)[0])) / max(
            self.cfg.gripper_step,
            1e-6,
        )
        return np.clip(
            np.concatenate([pos_delta, rot_delta, np.array([grip_delta], dtype=np.float64)], axis=0),
            -1.0,
            1.0,
        ).astype(np.float32)

    def action_from_ee_target(
        self,
        ee_pos: np.ndarray,
        ee_quat: np.ndarray,
        gripper_target: float,
    ) -> np.ndarray:
        prev_obs = self.last_obs if self.last_obs is not None else self._read_and_cache_observation()
        target_pos = self._clip_workspace(np.asarray(ee_pos, dtype=np.float64).reshape(3))
        prev_quat = _quat_normalize(np.asarray(prev_obs["ee_quat"], dtype=np.float64))
        target_quat = _quat_align_sign(np.asarray(ee_quat, dtype=np.float64).reshape(4), prev_quat)
        gripper_target = float(np.clip(gripper_target, 0.0, GRIPPER_MJ_MAX))
        pos_delta = (target_pos - np.asarray(prev_obs["ee_pos"], dtype=np.float64)) / max(self.cfg.translation_step, 1e-6)
        quat_delta = _quat_mul(target_quat, _quat_conjugate(prev_quat))
        rot_delta = _quat_to_rotvec(quat_delta) / max(self.cfg.rotation_step, 1e-6)
        grip_delta = (gripper_target - float(np.asarray(prev_obs["gripper_pos"], dtype=np.float64)[0])) / max(
            self.cfg.gripper_step,
            1e-6,
        )
        return np.clip(
            np.concatenate([pos_delta, rot_delta, np.array([grip_delta], dtype=np.float64)], axis=0),
            -1.0,
            1.0,
        ).astype(np.float32)

    def get_joint_target(self) -> tuple[np.ndarray, float]:
        if self.last_obs is None:
            self._sync_targets_from_current_state()
        return (
            self.desired_joint_target.astype(np.float64).copy(),
            float(self.desired_gripper),
        )

    def get_end_effector_target(self) -> tuple[np.ndarray, np.ndarray, float]:
        if self.last_obs is None:
            self._sync_targets_from_current_state()
        return (
            self.desired_pos.astype(np.float64).copy(),
            self.desired_quat.astype(np.float64).copy(),
            float(self.desired_gripper),
        )

    def close(self) -> None:  # pragma: no cover - hardware-only path
        disconnect = getattr(self.robot, "disconnect", None)
        if callable(disconnect):
            disconnect()


RealRobotTeleopConfig = RealPickPlaceEnvConfig
RealRobotTeleopBackend = RealPickPlaceEnv
