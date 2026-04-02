from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALICIA_D_SDK_ROOT = PROJECT_ROOT / "Alicia-D-SDK"
if str(ALICIA_D_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(ALICIA_D_SDK_ROOT))

try:
    import alicia_d_sdk
except ImportError:
    alicia_d_sdk = None

try:
    from robocore.kinematics import forward_kinematics
    from robocore.transform import matrix_to_quaternion
except ImportError:
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


def gripper_sdk_to_mujoco(sdk_val: float) -> float:
    clamped = float(np.clip(sdk_val, 0.0, GRIPPER_SDK_MAX))
    return float((1.0 - clamped / GRIPPER_SDK_MAX) * GRIPPER_MJ_MAX)


class LeaderTeleopProvider:
    _ACTIVE_STATUSES = frozenset({"sync", "sync_locked"})

    def __init__(
        self,
        *,
        port: str,
        variant: str = "leader",
        gripper_type: str = "50mm",
        xml_path: str | None = None,
        translation_step: float = 0.0,
        rotation_step: float = 0.0,
        gripper_step: float = 0.0,
        delta_action_sign: tuple[float, float, float, float, float, float, float] | None = None,
        deadman_required: bool = True,
        debug: bool = False,
        read_timeout: float = 0.1,
    ) -> None:
        del xml_path
        if alicia_d_sdk is None:
            raise ImportError("Alicia-D-SDK is required for LeaderTeleopProvider")

        self.deadman_required = bool(deadman_required)
        self.debug = bool(debug)
        self.read_timeout = float(read_timeout)
        self.translation_step = max(float(translation_step), 1e-6)
        self.rotation_step = max(float(rotation_step), 1e-6)
        self.gripper_step = max(float(gripper_step), 1e-6)
        if delta_action_sign is None:
            delta_action_sign = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0)
        self.delta_action_sign = np.asarray(delta_action_sign, dtype=np.float32).reshape(7)
        self.robot = alicia_d_sdk.create_robot(
            port=port,
            variant=variant,
            gripper_type=gripper_type,
            debug_mode=debug,
        )
        self._last_active = False
        self._pending_start = False
        self._pending_stop = False
        self._last_status = "unknown"
        self._last_read_error_ts = 0.0
        self._consecutive_read_failures = 0
        self._prev_active_task_state: np.ndarray | None = None

    def _log_read_error(self, exc: Exception) -> None:
        now = time.monotonic()
        self._consecutive_read_failures += 1
        if now - self._last_read_error_ts >= 1.0:
            print(
                "[leader] read failed"
                f" failures={self._consecutive_read_failures}"
                f" error={type(exc).__name__}: {exc}"
            )
            self._last_read_error_ts = now

    def _maybe_reconnect(self) -> None:
        is_connected = getattr(self.robot, "is_connected", None)
        connect = getattr(self.robot, "connect", None)
        if not callable(is_connected) or not callable(connect):
            return
        try:
            if not is_connected():
                connect()
        except Exception as exc:
            now = time.monotonic()
            if now - self._last_read_error_ts >= 1.0:
                print(f"[leader] reconnect failed error={type(exc).__name__}: {exc}")
                self._last_read_error_ts = now

    def _update_edge_flags(self, active: bool) -> None:
        if active and not self._last_active:
            self._pending_start = True
        elif (not active) and self._last_active:
            self._pending_stop = True
        self._last_active = bool(active)

    def _read_joint_target(self) -> tuple[np.ndarray, float, bool, str] | None:
        try:
            joint_state = self.robot.get_robot_state("joint_gripper", timeout=self.read_timeout)
        except Exception as exc:
            self._log_read_error(exc)
            if self._last_active:
                self._pending_stop = True
            self._last_active = False
            self._prev_active_task_state = None
            self._maybe_reconnect()
            return None
        if joint_state is None:
            if self._last_active:
                self._pending_stop = True
            self._last_active = False
            self._prev_active_task_state = None
            return None

        joint_pos = np.asarray(list(getattr(joint_state, "angles", ()))[:6], dtype=np.float32)
        if joint_pos.size < 6:
            joint_pos = np.pad(joint_pos, (0, 6 - joint_pos.size))
        joint_pos = joint_pos.reshape(6)

        sdk_gripper = float(getattr(joint_state, "gripper", GRIPPER_SDK_MAX))
        gripper_target = float(gripper_sdk_to_mujoco(sdk_gripper))

        status = str(getattr(joint_state, "run_status_text", "unknown") or "unknown").lower()
        active = (not self.deadman_required) or (status in self._ACTIVE_STATUSES)
        self._update_edge_flags(active)

        if self.debug and status != self._last_status:
            print(f"[leader] status={status} active={active}")
        self._last_status = status
        self._consecutive_read_failures = 0
        return joint_pos, gripper_target, active, status

    def _task_state_from_joint_sample(self, joint_pos: np.ndarray, gripper_target: float) -> np.ndarray | None:
        robot_model = getattr(self.robot, "robot_model", None)
        if forward_kinematics is not None and matrix_to_quaternion is not None and robot_model is not None:
            transform = np.asarray(
                forward_kinematics(
                    robot_model,
                    np.asarray(joint_pos, dtype=np.float64).reshape(6),
                    return_end=True,
                ),
                dtype=np.float64,
            )
            ee_pos = np.asarray(transform[:3, 3], dtype=np.float64).copy()
            ee_quat_xyzw = np.asarray(matrix_to_quaternion(transform[:3, :3]), dtype=np.float64).reshape(4)
            ee_quat_wxyz = _quat_normalize(_xyzw_to_wxyz(ee_quat_xyzw))
            return np.concatenate(
                [
                    ee_pos,
                    ee_quat_wxyz,
                    np.array([float(gripper_target)], dtype=np.float64),
                ],
                axis=0,
            )

        pose = self.robot.get_pose()
        if pose is None:
            return None
        ee_pos = np.asarray(pose["position"], dtype=np.float64).reshape(3)
        ee_quat_wxyz = _quat_normalize(_xyzw_to_wxyz(np.asarray(pose["quaternion_xyzw"], dtype=np.float64)))
        return np.concatenate(
            [
                ee_pos,
                ee_quat_wxyz,
                np.array([float(gripper_target)], dtype=np.float64),
            ],
            axis=0,
        )

    def _base_action_from_task_states(self, prev_task_state: np.ndarray, curr_task_state: np.ndarray) -> np.ndarray:
        prev_task_state = np.asarray(prev_task_state, dtype=np.float64).reshape(8)
        curr_task_state = np.asarray(curr_task_state, dtype=np.float64).reshape(8)
        prev_pos = prev_task_state[:3]
        prev_quat = _quat_normalize(prev_task_state[3:7])
        prev_gripper = float(prev_task_state[7])
        curr_pos = curr_task_state[:3]
        curr_quat = _quat_align_sign(curr_task_state[3:7], prev_quat)
        curr_gripper = float(curr_task_state[7])
        pos_delta = (curr_pos - prev_pos) / self.translation_step
        quat_delta = _quat_mul(curr_quat, _quat_conjugate(prev_quat))
        rot_delta = _quat_to_rotvec(quat_delta) / self.rotation_step
        gripper_delta = (curr_gripper - prev_gripper) / self.gripper_step
        base_action = np.concatenate(
            [
                pos_delta,
                rot_delta,
                np.array([gripper_delta], dtype=np.float64),
            ],
            axis=0,
        ).astype(np.float32)
        return (base_action * self.delta_action_sign).astype(np.float32)

    def get_joint_target(self) -> tuple[np.ndarray, float, bool] | None:
        sample = self._read_joint_target()
        if sample is None:
            return None
        joint_pos, gripper_target, active, _ = sample
        if self.deadman_required and not active:
            return None
        return joint_pos, gripper_target, active

    def get_delta_joint_target(self) -> tuple[np.ndarray, float, bool, np.ndarray] | None:
        sample = self._read_joint_target()
        if sample is None:
            return None
        joint_pos, gripper_target, active, _ = sample
        task_state = self._task_state_from_joint_sample(joint_pos, gripper_target)
        if task_state is None:
            self._prev_active_task_state = None
            return None
        if self.deadman_required and not active:
            self._prev_active_task_state = task_state.copy()
            return None
        if self._prev_active_task_state is None:
            base_action = np.zeros(7, dtype=np.float32)
        else:
            base_action = self._base_action_from_task_states(self._prev_active_task_state, task_state)
        self._prev_active_task_state = task_state.copy()
        return joint_pos, gripper_target, active, base_action

    def reset(self) -> None:
        self._prev_active_task_state = None
        self._pending_start = False
        self._pending_stop = False
        sample = self._read_joint_target()
        self._pending_start = False
        self._pending_stop = False
        self._last_active = bool(sample[2]) if sample is not None else False
        if sample is not None:
            joint_pos, gripper_target, active, _ = sample
            if active or not self.deadman_required:
                task_state = self._task_state_from_joint_sample(joint_pos, gripper_target)
                if task_state is not None:
                    self._prev_active_task_state = task_state.copy()

    def wait_for_start_signal(self) -> None:
        if not self.deadman_required:
            return
        while True:
            sample = self._read_joint_target()
            if sample is not None and (self._pending_start or sample[2]):
                self._pending_start = False
                self._pending_stop = False
                return
            time.sleep(0.01)

    def clear_pending_start_stop(self) -> None:
        self._pending_start = False
        self._pending_stop = False

    def consume_stop_signal(self) -> bool:
        if not self.deadman_required:
            return False
        self._read_joint_target()
        if self._pending_stop:
            self._pending_stop = False
            return True
        return False

    def close(self) -> None:
        disconnect = getattr(self.robot, "disconnect", None)
        if callable(disconnect):
            disconnect()
