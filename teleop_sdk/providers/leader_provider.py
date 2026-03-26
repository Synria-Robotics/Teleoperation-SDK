from __future__ import annotations

from pathlib import Path
import time

import mujoco
import numpy as np

from teleop_sdk.providers.base import BaseTeleopProvider

try:
    import alicia_d_sdk
except ImportError:  # pragma: no cover - hardware-only path
    alicia_d_sdk = None


TRIGGER_THRESHOLD = 900
GRIPPER_SDK_MAX = 1000.0
GRIPPER_MJ_MAX = 0.025


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
        quat = np.array(
            [
                0.25 * s,
                (mat[2, 1] - mat[1, 2]) / s,
                (mat[0, 2] - mat[2, 0]) / s,
                (mat[1, 0] - mat[0, 1]) / s,
            ],
            dtype=np.float64,
        )
        return _quat_normalize(quat)

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


def gripper_sdk_to_mujoco(sdk_val: float) -> float:
    clamped = max(0.0, min(GRIPPER_SDK_MAX, sdk_val))
    return (1.0 - clamped / GRIPPER_SDK_MAX) * GRIPPER_MJ_MAX


def _decode_handle_inputs(raw_status: int | None, run_status_text: str | None, gripper_value: float | None) -> dict[str, bool]:
    left_button = False
    right_button = False
    trigger = gripper_value is not None and gripper_value < TRIGGER_THRESHOLD

    if isinstance(run_status_text, str):
        if run_status_text == "sync":
            left_button = True
        elif run_status_text == "locked":
            right_button = True
        elif run_status_text == "sync_locked":
            left_button = True
            right_button = True

    if isinstance(raw_status, int):
        left_button = left_button or bool(raw_status & 0x10)
        right_button = right_button or bool(raw_status & 0x01)

    return {"button1": left_button, "button2": right_button, "trigger": trigger}


class LeaderTeleopProvider(BaseTeleopProvider):
    """Heuristic Alicia-D leader adapter for sim-side data collection.

    The provider converts leader joint deltas into a normalized 7D command.
    It is sufficient for collecting teleop-style sim data, but it is not meant
    to be a calibrated task-space controller.
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        variant: str = "leader",
        gripper_type: str = "50mm",
        *,
        xml_path: str | None = None,
        translation_step: float = 0.012,
        rotation_step: float = 0.20,
        gripper_step: float = 0.004,
        deadman_required: bool = True,
        debug: bool = False,
    ):
        if alicia_d_sdk is None:
            raise ImportError("alicia_d_sdk is required for LeaderTeleopProvider")
        self.robot = alicia_d_sdk.create_robot(
            port=port,
            variant=variant,
            gripper_type=gripper_type,
            debug_mode=debug,
        )
        default_xml = (
            Path(__file__).resolve().parents[2]
            / "assets"
            / "mujoco"
            / "Alicia_D_v5_6"
            / "gripper_50mm"
            / "alicia_d_follower.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(str(xml_path or default_xml))
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        self.arm_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ("Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6")
        ]
        self.arm_qpos_adr = np.asarray([self.model.jnt_qposadr[j] for j in self.arm_joint_ids], dtype=np.int32)
        self.gripper_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ("left_finger", "right_finger")
        ]
        self.left_gripper_qpos_adr = int(self.model.jnt_qposadr[self.gripper_joint_ids[0]])
        self.right_gripper_qpos_adr = int(self.model.jnt_qposadr[self.gripper_joint_ids[1]])

        self.translation_step = float(translation_step)
        self.rotation_step = float(rotation_step)
        self.gripper_step = float(gripper_step)
        self.deadman_required = deadman_required
        self.debug = debug
        self._last_pose_pos: np.ndarray | None = None
        self._last_pose_quat: np.ndarray | None = None
        self._last_gripper_target: float | None = None
        self._anchor_pose_pos: np.ndarray | None = None
        self._anchor_pose_quat: np.ndarray | None = None
        self._anchor_gripper_target: float | None = None
        self._anchor_sim_pos: np.ndarray | None = None
        self._anchor_sim_quat: np.ndarray | None = None
        self._anchor_sim_gripper: float | None = None
        self._target_pos: np.ndarray | None = None
        self._target_quat: np.ndarray | None = None
        self._target_gripper: float | None = None
        self._joint_angles = np.zeros(6, dtype=np.float64)
        self._gripper_target = float(GRIPPER_MJ_MAX)
        self._active = True
        self._button1 = False
        self._button2 = False
        self._trigger = False
        self._button2_prev = False
        self._button2_edge = False

    def reset(self) -> None:
        self._last_pose_pos = None
        self._last_pose_quat = None
        self._last_gripper_target = None
        self._anchor_pose_pos = None
        self._anchor_pose_quat = None
        self._anchor_gripper_target = None
        self._anchor_sim_pos = None
        self._anchor_sim_quat = None
        self._anchor_sim_gripper = None
        self._target_pos = None
        self._target_quat = None
        self._target_gripper = None
        self._joint_angles = np.zeros(6, dtype=np.float64)
        self._gripper_target = float(GRIPPER_MJ_MAX)
        self._active = True
        self._button2_prev = False
        self._button2_edge = False

    def align_to_sim_target(
        self,
        sim_pos: np.ndarray,
        sim_quat: np.ndarray,
        sim_gripper: float,
    ) -> bool:
        sample = self._poll_robot_state()
        if sample is None:
            self._active = False
            return False

        _, pose_pos, pose_quat, gripper_target, current_active = sample
        self._anchor_pose_pos = pose_pos.copy()
        self._anchor_pose_quat = pose_quat.copy()
        self._anchor_gripper_target = float(gripper_target)
        self._anchor_sim_pos = np.asarray(sim_pos, dtype=np.float64).copy()
        self._anchor_sim_quat = _quat_normalize(np.asarray(sim_quat, dtype=np.float64)).copy()
        self._anchor_sim_gripper = float(sim_gripper)
        self._target_pos = self._anchor_sim_pos.copy()
        self._target_quat = self._anchor_sim_quat.copy()
        self._target_gripper = self._anchor_sim_gripper
        self._last_pose_pos = pose_pos.copy()
        self._last_pose_quat = pose_quat.copy()
        self._last_gripper_target = float(gripper_target)
        self._active = current_active
        return True

    def _poll_robot_state(self):
        raw = self.robot.get_robot_state("joint_gripper")
        if raw is None:
            self._active = False
            return None

        angles = np.asarray(list(raw.angles)[:6], dtype=np.float64)
        if angles.shape[0] < 6:
            angles = np.pad(angles, (0, 6 - angles.shape[0]))
        gripper = float(getattr(raw, "gripper", GRIPPER_SDK_MAX))
        run_text = getattr(raw, "run_status_text", "unknown")
        run_raw = getattr(getattr(self.robot, "servo_driver", None), "data_parser", None)
        run_raw = getattr(run_raw, "_run_status", None)
        decoded = _decode_handle_inputs(run_raw, run_text, gripper)

        self._button1 = bool(decoded["button1"])
        self._button2 = bool(decoded["button2"])
        self._trigger = bool(decoded["trigger"])
        self._button2_edge = self._button2 and not self._button2_prev
        self._button2_prev = self._button2
        self._joint_angles = angles.copy()

        self.data.qpos[self.arm_qpos_adr] = angles
        gripper_target = gripper_sdk_to_mujoco(gripper)
        self._gripper_target = float(gripper_target)
        self.data.qpos[self.left_gripper_qpos_adr] = gripper_target
        self.data.qpos[self.right_gripper_qpos_adr] = -gripper_target
        mujoco.mj_forward(self.model, self.data)

        pose_pos = np.asarray(self.data.site_xpos[self.site_id], dtype=np.float64).copy()
        pose_quat = _mat_to_quat(np.asarray(self.data.site_xmat[self.site_id], dtype=np.float64))
        current_active = not self.deadman_required or self._button1
        return angles.copy(), pose_pos, pose_quat, gripper_target, current_active

    def poll_status(self) -> dict[str, bool]:
        sample = self._poll_robot_state()
        connected = sample is not None
        return {
            "connected": connected,
            "button1": self._button1,
            "button2": self._button2,
            "trigger": self._trigger,
            "active": self._active if connected else False,
            "start_edge": self._button2_edge,
        }

    def get_joint_target(self) -> tuple[np.ndarray, float, bool] | None:
        sample = self._poll_robot_state()
        if sample is None:
            return None
        joint_angles, _, _, gripper_target, current_active = sample
        self._active = current_active
        return joint_angles.astype(np.float64).copy(), float(gripper_target), bool(current_active)

    def check_home_alignment(
        self,
        home_joint_pos: np.ndarray,
        home_gripper_target: float,
        *,
        joint_tol: float = 0.12,
        gripper_tol: float = 0.006,
    ) -> tuple[bool, dict[str, float]]:
        sample = self._poll_robot_state()
        if sample is None:
            return False, {"joint_error_max": float("inf"), "gripper_error": float("inf")}

        joint_angles, _, _, gripper_target, _ = sample
        joint_error = np.abs(joint_angles - np.asarray(home_joint_pos, dtype=np.float64))
        gripper_error = abs(float(gripper_target) - float(home_gripper_target))
        aligned = bool(np.max(joint_error) <= joint_tol and gripper_error <= gripper_tol)
        return aligned, {
            "joint_error_max": float(np.max(joint_error)),
            "gripper_error": float(gripper_error),
        }

    def wait_for_start_signal(self, poll_hz: float = 20.0) -> None:
        interval = 1.0 / max(poll_hz, 1.0)
        print("[INFO] Waiting for BUTTON2 to start the next round...")
        while True:
            status = self.poll_status()
            if status["start_edge"]:
                print("[INFO] BUTTON2 detected – starting round")
                self._button2_edge = False
                return
            time.sleep(interval)

    def clear_pending_start_stop(self) -> None:
        """Drop a latched BUTTON2 edge after using it to start a round."""
        self._button2_edge = False

    def consume_stop_signal(self) -> bool:
        status = self.poll_status()
        if status["start_edge"]:
            self._button2_edge = False
            return True
        return False

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        sample = self._poll_robot_state()
        if sample is None:
            return np.zeros(self.action_dim, dtype=np.float32)
        _, pose_pos, pose_quat, gripper_target, current_active = sample

        if self._anchor_pose_pos is None or self._anchor_pose_quat is None or self._anchor_gripper_target is None:
            sim_pos = np.asarray(obs.get("ee_pos", np.zeros(3, dtype=np.float32)), dtype=np.float64)
            sim_quat = np.asarray(obs.get("ee_quat", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)), dtype=np.float64)
            sim_gripper = float(np.asarray(obs.get("gripper_pos", np.zeros(1, dtype=np.float32)), dtype=np.float64).reshape(-1)[0])
            self.align_to_sim_target(sim_pos, sim_quat, sim_gripper)
            return np.zeros(self.action_dim, dtype=np.float32)

        if not current_active:
            if self._target_pos is not None and self._target_quat is not None and self._target_gripper is not None:
                self._anchor_pose_pos = pose_pos.copy()
                self._anchor_pose_quat = pose_quat.copy()
                self._anchor_gripper_target = float(gripper_target)
                self._anchor_sim_pos = self._target_pos.copy()
                self._anchor_sim_quat = self._target_quat.copy()
                self._anchor_sim_gripper = float(self._target_gripper)
            self._last_pose_pos = pose_pos.copy()
            self._last_pose_quat = pose_quat.copy()
            self._last_gripper_target = float(gripper_target)
            self._active = False
            return np.zeros(self.action_dim, dtype=np.float32)

        rel_pos = pose_pos - self._anchor_pose_pos
        rel_quat = _quat_mul(pose_quat, _quat_conjugate(self._anchor_pose_quat))
        rel_gripper = float(gripper_target - self._anchor_gripper_target)

        target_pos = self._anchor_sim_pos + rel_pos
        target_quat = _quat_normalize(_quat_mul(rel_quat, self._anchor_sim_quat))
        target_gripper = float(self._anchor_sim_gripper + rel_gripper)

        if self._target_pos is None or self._target_quat is None or self._target_gripper is None:
            self._target_pos = self._anchor_sim_pos.copy()
            self._target_quat = self._anchor_sim_quat.copy()
            self._target_gripper = float(self._anchor_sim_gripper)

        pos_delta = (target_pos - self._target_pos) / max(self.translation_step, 1e-6)
        quat_delta = _quat_mul(target_quat, _quat_conjugate(self._target_quat))
        rot_delta = _quat_to_rotvec(quat_delta) / max(self.rotation_step, 1e-6)
        grip_delta = (target_gripper - self._target_gripper) / max(self.gripper_step, 1e-6)

        self._target_pos = target_pos.copy()
        self._target_quat = target_quat.copy()
        self._target_gripper = float(target_gripper)
        self._last_pose_pos = pose_pos.copy()
        self._last_pose_quat = pose_quat.copy()
        self._last_gripper_target = float(gripper_target)
        self._active = True

        action = np.concatenate([pos_delta, rot_delta, np.array([grip_delta], dtype=np.float64)], axis=0)
        return np.clip(action.astype(np.float32), -1.0, 1.0)

    def peek_next_action(self) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=np.float32)

    def is_active(self) -> bool:
        return self._active

    def close(self) -> None:  # pragma: no cover - hardware-only path
        disconnect = getattr(self.robot, "disconnect", None)
        if callable(disconnect):
            disconnect()
