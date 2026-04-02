from __future__ import annotations

import argparse
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import mujoco
import mujoco.viewer
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Residual_RL_TD3.common import PickPlaceReward, flatten_observation, unflatten_observation
from Residual_RL_TD3.data import EpisodeRollout, Transition, save_episode_npz
from Residual_RL_TD3.env.mujoco_pick_place_env import MujocoPickPlaceTeleopEnv, PickPlaceTaskConfig
from Residual_RL_TD3.providers import LeaderTeleopProvider
from Residual_RL_TD3.scripts.train_residual_td3_teleop import (
    TELEOP_RESIDUAL_OBS_MODE,
    MLPActor,
    RealtimeResidualTrainer,
    TensorNormalizer,
    TorchResidualPolicy,
    TrainConfig,
    format_metrics,
    infer_policy_obs_dim,
)
from utils.fps_utils import precise_sleep

MJ_REEXEC_ENV = "RECORD_TELEOP_MJPYTHON_REEXEC"


def _ensure_mjpython_on_macos() -> None:
    if sys.platform != "darwin" or os.environ.get(MJ_REEXEC_ENV) == "1":
        return

    mjpython = shutil.which("mjpython")
    if mjpython is None:
        candidate = Path(sys.executable).with_name("mjpython")
        if candidate.exists():
            mjpython = str(candidate)
    if mjpython is None:
        raise RuntimeError("record_teleop.py requires mjpython on macOS when viewer mode is enabled")

    env = os.environ.copy()
    env[MJ_REEXEC_ENV] = "1"
    os.execvpe(mjpython, [mjpython, *sys.argv], env)


def _clone_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value, dtype=np.float32).copy() for key, value in obs.items()}


def _build_rollout(transitions: list[Transition], obs_keys: tuple[str, ...]) -> EpisodeRollout:
    for idx in range(len(transitions) - 1):
        transitions[idx].next_base_action = transitions[idx + 1].base_action.copy()
    if transitions:
        transitions[-1].next_base_action = np.zeros_like(transitions[-1].base_action)
    return EpisodeRollout(
        transitions=transitions,
        obs_keys=obs_keys,
        success=any(t.success for t in transitions),
        episode_return_env=float(sum(t.reward_env for t in transitions)),
    )


def _finalize_transitions(
    transitions: list[Transition],
    *,
    success: bool,
    success_reward: float = 0.0,
    failure_reward: float = -100.0,
) -> None:
    if not transitions:
        return
    last = transitions[-1]
    last.success = bool(success)
    if success:
        last.reward_env = float(success_reward)
        last.terminated = True
        last.truncated = False
    else:
        last.reward_env = float(failure_reward)
        last.terminated = False
        last.truncated = True


def _record_step(
    env: MujocoPickPlaceTeleopEnv,
    provider: LeaderTeleopProvider,
    obs: dict[str, np.ndarray],
    *,
    residual_policy: TorchResidualPolicy | None = None,
    residual_enabled: bool = False,
    residual_scale: float = 1.0,
    residual_limit: float = 0.3,
) -> tuple[dict[str, np.ndarray], Transition, bool, dict[str, np.ndarray | float | bool]]:
    leader_target = provider.get_joint_target()
    teleop_active = False
    if leader_target is None:
        base_joint_target, base_gripper_target = env.get_joint_target()
    else:
        base_joint_target, base_gripper_target, teleop_active = leader_target
    base_action = env.action_from_joint_target(base_joint_target, base_gripper_target)

    obs_copy = _clone_obs(obs)
    residual_action = np.zeros(env.action_dim, dtype=np.float32)
    if residual_enabled and teleop_active and residual_policy is not None:
        residual_action = np.clip(
            residual_policy.get_action(obs_copy, base_action) * float(residual_scale),
            -float(residual_limit),
            float(residual_limit),
        ).astype(np.float32)
        next_obs, _, _, truncated, info = env.step_base_joint_target(
            base_joint_target,
            base_gripper_target,
            residual_action,
            base_action=base_action,
        )
    else:
        next_obs, _, _, truncated, info = env.step_absolute_joint_target(
            base_joint_target,
            base_gripper_target,
        )

    transition = Transition(
        observation_state=flatten_observation(obs_copy),
        next_observation_state=flatten_observation(next_obs),
        base_action=np.asarray(base_action, dtype=np.float32).copy(),
        next_base_action=np.zeros_like(base_action, dtype=np.float32),
        residual_action=np.asarray(info.get("residual_action", np.zeros_like(base_action)), dtype=np.float32).copy(),
        realized_action=np.asarray(info["realized_action"], dtype=np.float32).copy(),
        reward_env=-1.0,
        terminated=False,
        truncated=bool(truncated),
        base_joint_target=np.asarray(base_joint_target, dtype=np.float32).reshape(6).copy(),
        base_gripper_target=np.asarray([base_gripper_target], dtype=np.float32),
        success=False,
        control_dt=float(env.cfg.control_dt),
    )
    return next_obs, transition, bool(teleop_active), info


def _run_headless_episode(
    env: MujocoPickPlaceTeleopEnv,
    provider: LeaderTeleopProvider,
    *,
    seed: int,
    should_stop,
    residual_policy: TorchResidualPolicy | None = None,
    residual_enabled: bool = False,
    residual_scale: float = 1.0,
    residual_limit: float = 0.1,
) -> tuple[EpisodeRollout, str]:
    obs, _ = env.reset(seed=seed)
    provider.reset()
    if residual_policy is not None:
        residual_policy.reset()
    obs_keys = tuple(obs.keys())
    transitions: list[Transition] = []

    for _ in range(env.cfg.max_steps):
        if should_stop():
            _finalize_transitions(transitions, success=True)
            return _build_rollout(transitions, obs_keys), "manual_save"
        obs, transition, _, _ = _record_step(
            env,
            provider,
            obs,
            residual_policy=residual_policy,
            residual_enabled=residual_enabled,
            residual_scale=residual_scale,
            residual_limit=residual_limit,
        )
        transitions.append(transition)
        if transition.truncated:
            _finalize_transitions(transitions, success=False)
            return _build_rollout(transitions, obs_keys), "max_steps"

    _finalize_transitions(transitions, success=False)
    return _build_rollout(transitions, obs_keys), "max_steps"


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
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


def _scale_delta_action(
    action: np.ndarray,
    *,
    translation_step: float,
    rotation_step: float,
    gripper_step: float,
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(7)
    scale = np.asarray(
        [
            translation_step,
            translation_step,
            translation_step,
            rotation_step,
            rotation_step,
            rotation_step,
            gripper_step,
        ],
        dtype=np.float32,
    )
    return (action * scale).astype(np.float32)


def _extract_follower_task_state(obs: dict[str, np.ndarray] | None) -> np.ndarray | None:
    if obs is None:
        return None
    required = ("ee_pos", "ee_quat", "gripper_pos")
    if any(key not in obs for key in required):
        return None
    ee_pos = np.asarray(obs["ee_pos"], dtype=np.float32).reshape(3)
    ee_rot = _quat_to_rotvec(np.asarray(obs["ee_quat"], dtype=np.float32).reshape(4))
    gripper = np.asarray(obs["gripper_pos"], dtype=np.float32).reshape(1)
    return np.concatenate([ee_pos, ee_rot, gripper], axis=0).astype(np.float32)


def _compose_target_task_state(
    obs: dict[str, np.ndarray] | None,
    action: np.ndarray | None,
    *,
    translation_step: float,
    rotation_step: float,
    gripper_step: float,
) -> np.ndarray | None:
    if obs is None or action is None:
        return None
    required = ("ee_pos", "ee_quat", "gripper_pos")
    if any(key not in obs for key in required):
        return None

    ee_pos = np.asarray(obs["ee_pos"], dtype=np.float32).reshape(3)
    ee_quat = _quat_normalize(np.asarray(obs["ee_quat"], dtype=np.float32).reshape(4))
    gripper = float(np.asarray(obs["gripper_pos"], dtype=np.float32).reshape(-1)[0])
    action = np.asarray(action, dtype=np.float32).reshape(7)

    target_pos = ee_pos + action[:3] * float(translation_step)
    dq = _axis_angle_to_quat(action[3:6] * float(rotation_step))
    target_quat = _quat_normalize(_quat_mul(dq, ee_quat))
    target_rot = _quat_to_rotvec(target_quat)
    target_gripper = np.asarray([gripper + float(action[6]) * float(gripper_step)], dtype=np.float32)
    return np.concatenate([target_pos, target_rot, target_gripper], axis=0).astype(np.float32)


def _render_overlay(
    viewer,
    *,
    mode: str,
    teleop_active: bool,
    residual_enabled: bool,
    saved_count: int,
    target_count: int,
    steps: int,
    total_return: float,
    message: str,
    follower_state: np.ndarray | None = None,
    leader_delta: np.ndarray | None = None,
    base_target_state: np.ndarray | None = None,
    base_action: np.ndarray | None = None,
    residual_delta: np.ndarray | None = None,
    residual_action: np.ndarray | None = None,
    command_target_state: np.ndarray | None = None,
    command_delta: np.ndarray | None = None,
    commanded_action: np.ndarray | None = None,
    realized_action: np.ndarray | None = None,
    box_pos: np.ndarray | None = None,
    basket_pos: np.ndarray | None = None,
    trainer_status: str | None = None,
    show_details: bool = False,
) -> None:
    def _fmt_vec(vec: np.ndarray, digits: int = 3) -> str:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        return "[" + ", ".join(f"{float(value):+.{digits}f}" for value in arr) + "]"

    def _append_vec(
        lines: list[str],
        label: str,
        vec: np.ndarray | None,
        *,
        digits: int = 3,
        items_per_line: int = 4,
    ) -> None:
        if vec is None:
            return
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size <= items_per_line:
            lines.append(f"{label:<11}{_fmt_vec(arr, digits=digits)}")
            return

        chunks = [
            arr[start : start + items_per_line]
            for start in range(0, arr.size, items_per_line)
        ]
        for idx, chunk in enumerate(chunks):
            prefix = f"{label:<11}" if idx == 0 else " " * 11
            text = ", ".join(f"{float(value):+.{digits}f}" for value in chunk)
            if idx == 0:
                lines.append(f"{prefix}[{text},")
            elif idx == len(chunks) - 1:
                lines.append(f"{prefix}{text}]")
            else:
                lines.append(f"{prefix}{text},")

    left_lines = [
        "Teleop Recorder",
        f"Mode: {mode}",
        f"Leader: {'ACTIVE' if teleop_active else 'IDLE'}",
        f"Residual: {'ON' if residual_enabled else 'OFF'}",
        f"Saved: {saved_count}/{target_count}",
        f"Steps: {steps}",
        f"Return: {total_return:.2f}",
    ]
    if show_details:
        _append_vec(left_lines, "Follower:", follower_state, digits=3)
        _append_vec(left_lines, "Leader d:", leader_delta, digits=4)
        _append_vec(left_lines, "Base tgt:", base_target_state, digits=3)
        _append_vec(left_lines, "Base act:", base_action, digits=3)
        _append_vec(left_lines, "Residual d:", residual_delta, digits=4)
        _append_vec(left_lines, "Residual a:", residual_action, digits=3)
        _append_vec(left_lines, "Cmd tgt:", command_target_state, digits=3)
        _append_vec(left_lines, "Cmd d:", command_delta, digits=4)
        _append_vec(left_lines, "Cmd act:", commanded_action, digits=3)
        _append_vec(left_lines, "Applied:", realized_action, digits=3)
        if box_pos is not None:
            left_lines.append(f"Box:    {np.round(np.asarray(box_pos, dtype=np.float32), 3).tolist()}")
        if basket_pos is not None:
            left_lines.append(f"Basket: {np.round(np.asarray(basket_pos, dtype=np.float32), 3).tolist()}")
    left_lines.extend(
        [
            "",
            "Controls",
            "SPACE  preview:start",
            "O      toggle residual",
            "Q      recording:discard",
            "S      recording:save",
        ]
    )
    if trainer_status:
        left_lines.extend(["", f"Trainer: {trainer_status}"])
    if not show_details:
        left_lines.extend(["", "Overlay details: hidden"])
    right_lines = ["", "", "", "", "", "", "", "", message]

    set_texts = getattr(viewer, "set_texts", None)
    if callable(set_texts):
        set_texts(
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "\n".join(left_lines),
                "\n".join(right_lines),
            )
        )
        return

    viewport = getattr(viewer, "viewport", None)
    ctx = getattr(viewer, "ctx", None)
    if viewport is None or ctx is None:
        return
    viewer.user_scn.ngeom = 0
    mujoco.mjr_overlay(
        mujoco.mjtFont.mjFONT_NORMAL,
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        viewport,
        "\n".join(left_lines),
        "\n".join(right_lines),
        ctx,
    )


def _next_episode_index(output_dir: Path) -> int:
    max_index = -1
    for path in output_dir.glob("episode_*.npz"):
        suffix = path.stem.removeprefix("episode_")
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return max_index + 1


def _resolve_runtime_checkpoint(checkpoint: Path | None, train_save_path: Path | None, realtime_train: bool) -> Path | None:
    if checkpoint is not None:
        return checkpoint.resolve()
    if realtime_train and train_save_path is not None and train_save_path.exists():
        return train_save_path.resolve()
    return None


def _build_checkpoint_residual_policy(
    checkpoint_path: Path,
    *,
    obs: dict[str, np.ndarray],
    action_dim: int,
    default_residual_limit: float,
    translation_step: float,
    rotation_step: float,
    gripper_step: float,
    cpu: bool = False,
) -> TorchResidualPolicy:
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    checkpoint = torch.load(checkpoint_path.resolve(), map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    obs_mode = str(cfg.get("obs_mode", TELEOP_RESIDUAL_OBS_MODE))
    history_len = int(cfg.get("history_len", 4))
    residual_limit = float(cfg.get("residual_limit", default_residual_limit))
    hidden_dim = int(cfg.get("hidden_dim", 256))
    obs_dim = infer_policy_obs_dim(
        obs,
        action_dim=action_dim,
        obs_mode=obs_mode,
        history_len=history_len,
        translation_step=float(cfg.get("translation_step", translation_step)),
        rotation_step=float(cfg.get("rotation_step", rotation_step)),
        gripper_step=float(cfg.get("gripper_step", gripper_step)),
    )

    state_normalizer = None
    if "state_mean" in checkpoint and "state_std" in checkpoint and bool(cfg.get("normalize_state", True)):
        state_normalizer = TensorNormalizer(checkpoint["state_mean"], checkpoint["state_std"], device)

    actor = MLPActor(
        obs_dim,
        action_dim,
        hidden_dim=hidden_dim,
        residual_limit=residual_limit,
    ).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    print(
        f"[INFO] Loaded residual checkpoint={checkpoint_path} "
        f"device={device} obs_mode={obs_mode} history_len={history_len} residual_limit={residual_limit:.3f}"
    )
    return TorchResidualPolicy(
        actor,
        device,
        noise_scale=0.0,
        residual_limit=residual_limit,
        state_normalizer=state_normalizer,
        history_len=history_len,
        translation_step=float(cfg.get("translation_step", translation_step)),
        rotation_step=float(cfg.get("rotation_step", rotation_step)),
        gripper_step=float(cfg.get("gripper_step", gripper_step)),
        obs_mode=obs_mode,
    )


def main(args: argparse.Namespace) -> None:
    if not args.headless:
        _ensure_mjpython_on_macos()

    reward_model = PickPlaceReward()
    env_cfg = PickPlaceTaskConfig(
        xml_path=str(args.xml.resolve()),
        seed=args.seed,
        max_steps=args.max_steps,
        control_dt=args.control_dt,
        randomize_box_position=(not args.fixed_scene) and args.fixed_box_xy is None,
        randomize_basket_position=not args.fixed_scene,
        fixed_box_xy=tuple(float(v) for v in args.fixed_box_xy) if args.fixed_box_xy is not None else None,
        terminate_on_success=False,
    )
    env = MujocoPickPlaceTeleopEnv(env_cfg, reward=reward_model)
    obs, info = env.reset(seed=args.seed)
    provider = LeaderTeleopProvider(
        port=args.port,
        variant=args.variant,
        gripper_type=args.gripper_type,
        xml_path=str(args.xml.resolve()),
        translation_step=env.cfg.translation_step,
        rotation_step=env.cfg.rotation_step,
        gripper_step=env.cfg.gripper_step,
        deadman_required=not args.no_deadman,
        debug=args.debug,
    )

    output_dir = (args.output_dir or (PROJECT_ROOT / "logs" / "teleop_dataset")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_save_path = args.train_save_path.resolve()
    runtime_checkpoint = _resolve_runtime_checkpoint(args.checkpoint, train_save_path, args.realtime_train)
    next_episode_index = _next_episode_index(output_dir)
    session_saved_count = 0
    assist_policy: TorchResidualPolicy | None = None
    trainer: RealtimeResidualTrainer | None = None
    trainer_status = "disabled"
    trainer_queue: queue.Queue[EpisodeRollout | None] | None = None
    trainer_thread: threading.Thread | None = None
    trainer_state_lock = threading.Lock()
    trainer_state: dict[str, object] = {
        "status": trainer_status,
        "latest_policy": None,
        "policy_version": 0,
    }
    applied_policy_version = 0

    if args.realtime_train:
        if runtime_checkpoint is not None and args.checkpoint is None and runtime_checkpoint == train_save_path:
            print(f"[INFO] Resuming realtime trainer from existing checkpoint: {runtime_checkpoint}")
        train_cfg = TrainConfig(
            dataset_dirs=(output_dir,),
            save_path=train_save_path,
            xml_path=str(args.xml.resolve()),
            init_checkpoint=runtime_checkpoint,
            seed=args.seed,
            fixed_scene=bool(args.fixed_scene),
            fixed_box_xy=tuple(float(v) for v in args.fixed_box_xy) if args.fixed_box_xy is not None else None,
            batch_size=args.train_batch_size,
            offline_fraction=args.train_offline_fraction,
            critic_warmup_steps=args.train_critic_warmup_steps,
            actor_lr=args.train_actor_lr,
            critic_lr=args.train_critic_lr,
            buffer_capacity=args.train_buffer_capacity,
            residual_limit=args.residual_limit,
            residual_mag_reg_weight=args.train_residual_mag_reg_weight,
            offline_zero_bc_weight=args.train_offline_zero_bc_weight,
            teacher_bc_weight=args.train_teacher_bc_weight,
            hidden_dim=args.train_hidden_dim,
            obs_mode=TELEOP_RESIDUAL_OBS_MODE,
            history_len=args.train_history_len,
            translation_step=env.cfg.translation_step,
            rotation_step=env.cfg.rotation_step,
            gripper_step=env.cfg.gripper_step,
            normalize_state=not args.disable_state_norm,
            normalize_reward=not args.disable_reward_norm,
            log_interval=args.train_log_interval,
        )
        trainer = RealtimeResidualTrainer(
            cfg=train_cfg,
            obs_example=_clone_obs(obs),
            action_dim=env.action_dim,
        )
        assist_policy = trainer.build_policy(noise_scale=0.0)
        trainer_status = (
            f"ready total={trainer.total_transitions} "
            f"offline={len(trainer.offline_rb)} online={len(trainer.online_rb)}"
        )
        trainer_state["status"] = trainer_status
        print(
            f"[INFO] Realtime trainer enabled save_path={train_save_path} "
            f"total_transitions={trainer.total_transitions}"
        )
    elif runtime_checkpoint is not None:
        assist_policy = _build_checkpoint_residual_policy(
            runtime_checkpoint,
            obs=_clone_obs(obs),
            action_dim=env.action_dim,
            default_residual_limit=args.residual_limit,
            translation_step=env.cfg.translation_step,
            rotation_step=env.cfg.rotation_step,
            gripper_step=env.cfg.gripper_step,
            cpu=args.cpu,
        )
        trainer_status = f"checkpoint={runtime_checkpoint.name}"
        trainer_state["status"] = trainer_status

    residual_enabled = bool(args.enable_residual_on_start and assist_policy is not None)
    if args.enable_residual_on_start and assist_policy is None:
        print("[WARN] Residual start requested but no checkpoint/trainer is available; starting with residual OFF")

    print("[INFO] Recording controls:")
    print("       SPACE: start recording on the current live-aligned scene")
    print("       O: toggle residual assist on/off")
    print("       S: save the current recording immediately")
    print("       Q: discard current recording and load a fresh preview scene")
    print("[INFO] Collection mode: teleop_live")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Existing episodes: {next_episode_index}")
    print(f"[INFO] New recordings will start at episode_{next_episode_index:04d}.npz")
    print(f"[INFO] Residual assist: {'enabled' if residual_enabled else 'disabled'} at startup")
    if runtime_checkpoint is not None:
        print(f"[INFO] Residual checkpoint source: {runtime_checkpoint}")
    if args.realtime_train:
        print(
            f"[INFO] Realtime train: save_path={train_save_path} "
            f"updates_per_save={args.train_updates_per_save} "
            f"actor_updates_per_save={args.train_actor_updates_per_save} "
            f"min_transitions={args.train_min_transitions} "
            f"critic_warmup_steps={args.train_critic_warmup_steps}"
        )
    if args.fixed_scene or args.fixed_box_xy is not None:
        box_xy = (
            tuple(float(v) for v in args.fixed_box_xy)
            if args.fixed_box_xy is not None
            else tuple(
                0.5 * (float(low) + float(high))
                for low, high in zip(env.cfg.box_spawn_xy_low, env.cfg.box_spawn_xy_high)
            )
        )
        basket_mode = "xml_default" if args.fixed_scene else "random_jitter"
        print(
            "[INFO] Scene reset: fixed"
            f" box_xy=({box_xy[0]:.3f}, {box_xy[1]:.3f})"
            f" basket={basket_mode}"
        )
    else:
        print(
            "[INFO] Box reset range:"
            f" x={env.cfg.box_spawn_xy_low[0]:.3f}..{env.cfg.box_spawn_xy_high[0]:.3f},"
            f" y={env.cfg.box_spawn_xy_low[1]:.3f}..{env.cfg.box_spawn_xy_high[1]:.3f}"
        )
    print(
        f"[INFO] Episode horizon: {env.cfg.max_steps} steps"
        f" (~{env.cfg.max_steps * env.cfg.control_dt:.1f}s at control_dt={env.cfg.control_dt:.3f})"
    )

    def _set_trainer_state(
        *,
        status: str | None = None,
        latest_policy: TorchResidualPolicy | None = None,
        policy_version_delta: int = 0,
    ) -> None:
        with trainer_state_lock:
            if status is not None:
                trainer_state["status"] = status
            if latest_policy is not None:
                trainer_state["latest_policy"] = latest_policy
            if policy_version_delta:
                trainer_state["policy_version"] = int(trainer_state["policy_version"]) + int(policy_version_delta)

    def _sync_trainer_state(*, allow_policy_swap: bool) -> None:
        nonlocal assist_policy, trainer_status, applied_policy_version
        with trainer_state_lock:
            trainer_status = str(trainer_state.get("status", trainer_status))
            latest_policy = trainer_state.get("latest_policy")
            policy_version = int(trainer_state.get("policy_version", 0))
        if allow_policy_swap and latest_policy is not None and policy_version != applied_policy_version:
            assist_policy = latest_policy  # type: ignore[assignment]
            assist_policy.reset()
            applied_policy_version = policy_version
            print(f"[INFO] Applied latest realtime residual policy snapshot v{policy_version}")

    def _trainer_worker_loop() -> None:
        assert trainer is not None
        assert trainer_queue is not None
        while True:
            rollout = trainer_queue.get()
            if rollout is None:
                trainer_queue.task_done()
                return

            queue_left = trainer_queue.qsize()
            try:
                _set_trainer_state(status=f"ingest queued={queue_left}")
                trainer.ingest_rollout(rollout, is_teacher=False, as_online=True)
                if trainer.total_transitions < args.train_min_transitions:
                    _set_trainer_state(status=f"waiting total={trainer.total_transitions}/{args.train_min_transitions}")
                    print(
                        f"[INFO] Waiting for data: {trainer.total_transitions}/{args.train_min_transitions} "
                        f"transitions before training starts"
                    )
                    continue

                phase = "critic_warmup" if not trainer.critic_warmup_done else "training"
                _set_trainer_state(
                    status=(
                        f"{phase} queued={queue_left} "
                        f"updates={args.train_updates_per_save} total={trainer.total_transitions}"
                    )
                )
                metrics = trainer.train(
                    num_updates=args.train_updates_per_save,
                    actor_updates=args.train_actor_updates_per_save,
                )
                # Only deploy policy snapshot after critic warmup is complete
                # (during warmup the actor is not being trained, so deploying it is pointless)
                if trainer.critic_warmup_done:
                    latest_policy = trainer.build_policy(noise_scale=0.0)
                    policy_version_delta = 1
                else:
                    latest_policy = None
                    policy_version_delta = 0
                checkpoint_path = trainer.save_checkpoint(
                    extra_payload={
                        "realtime_metrics": metrics,
                        "realtime_total_transitions": float(trainer.total_transitions),
                        "realtime_online_transitions": float(len(trainer.online_rb)),
                        "realtime_saved_episodes": float(session_saved_count),
                        "critic_warmup_done": trainer.critic_warmup_done,
                    },
                    path=train_save_path,
                )
                phase_after = "ready" if trainer.critic_warmup_done else "critic_warmup"
                _set_trainer_state(
                    status=(
                        f"{phase_after} queued={trainer_queue.qsize()} "
                        f"updates={trainer.update_steps} total={trainer.total_transitions} ckpt={checkpoint_path.name}"
                    ),
                    latest_policy=latest_policy,
                    policy_version_delta=policy_version_delta,
                )
                if metrics:
                    print(format_metrics(f"[realtime {phase_after}]", metrics))
                print(f"[INFO] Realtime checkpoint updated: {checkpoint_path}")
            except Exception as exc:
                _set_trainer_state(status=f"error {type(exc).__name__}: {exc}")
                print(f"[ERROR] Realtime trainer failed: {type(exc).__name__}: {exc}")
            finally:
                trainer_queue.task_done()

    if trainer is not None:
        trainer_queue = queue.Queue()
        trainer_thread = threading.Thread(
            target=_trainer_worker_loop,
            name="realtime-residual-trainer",
            daemon=True,
        )
        trainer_thread.start()

    def save_rollout(rollout: EpisodeRollout, *, terminated_by: str, viewer_mode: bool) -> None:
        nonlocal next_episode_index, session_saved_count, trainer_status
        path = output_dir / f"episode_{next_episode_index:04d}.npz"
        residual_used = any(float(np.linalg.norm(t.residual_action)) > 1e-6 for t in rollout.transitions)
        rollout.meta.update(
            {
                "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "port": args.port,
                "variant": args.variant,
                "gripper_type": args.gripper_type,
                "viewer": bool(viewer_mode),
                "terminated_by": terminated_by,
                "collection_mode": "teleop_live",
                "fixed_scene": bool(args.fixed_scene),
                "fixed_box_xy": [float(v) for v in args.fixed_box_xy] if args.fixed_box_xy is not None else [],
                "residual_used": bool(residual_used),
                "realtime_train": bool(args.realtime_train),
                "train_save_path": str(train_save_path) if args.realtime_train else "",
            }
        )
        save_episode_npz(path, rollout)
        session_saved_count += 1
        next_episode_index += 1
        print(
            f"[{session_saved_count}/{args.num_episodes}] saved {path.name}"
            f" steps={rollout.steps} success={rollout.success} return={rollout.episode_return_env:.3f}"
        )
        if trainer_queue is None:
            return

        trainer_queue.put(rollout)
        trainer_status = f"queued episodes={trainer_queue.qsize()}"
        _set_trainer_state(status=trainer_status)
        print(
            f"[INFO] Realtime training queued:"
            f" episode={path.name} queue_size={trainer_queue.qsize()}"
        )

    try:
        if args.headless:
            for _ in range(args.num_episodes):
                provider.wait_for_start_signal()
                provider.clear_pending_start_stop()
                rollout, terminated_by = _run_headless_episode(
                    env,
                    provider,
                    seed=args.seed + next_episode_index,
                    should_stop=provider.consume_stop_signal,
                    residual_policy=assist_policy,
                    residual_enabled=residual_enabled,
                    residual_scale=args.residual_scale,
                    residual_limit=args.residual_limit,
                )
                if rollout.steps == 0:
                    print(f"[WARN] episode_{next_episode_index:04d} is empty; waiting for the next start signal")
                    continue
                save_rollout(rollout, terminated_by=terminated_by, viewer_mode=False)
                _sync_trainer_state(allow_policy_swap=True)
            return

        def key_callback(keycode):
            try:
                key = chr(keycode)
            except ValueError:
                return
            if key == " ":
                viewer._teleop_record_start_round = True
            elif key.lower() == "o":
                viewer._teleop_toggle_residual = True
            elif key.lower() == "q":
                viewer._teleop_record_discard_round = True
            elif key.lower() == "s":
                viewer._teleop_record_stop_and_keep = True

        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
            viewer._teleop_record_start_round = False
            viewer._teleop_record_discard_round = False
            viewer._teleop_record_stop_and_keep = False
            viewer._teleop_toggle_residual = False

            seed_index = 1
            obs_keys = tuple(obs.keys())
            transitions: list[Transition] = []
            pending_terminated_by = "manual_save"
            last_base_action = np.zeros(env.action_dim, dtype=np.float32)
            last_residual_action = np.zeros(env.action_dim, dtype=np.float32)
            last_commanded_action = np.zeros(env.action_dim, dtype=np.float32)
            last_realized_action = np.zeros(env.action_dim, dtype=np.float32)
            last_follower_state = _extract_follower_task_state(obs)
            last_leader_delta = _scale_delta_action(
                last_base_action,
                translation_step=env.cfg.translation_step,
                rotation_step=env.cfg.rotation_step,
                gripper_step=env.cfg.gripper_step,
            )
            last_base_target_state = _compose_target_task_state(
                obs,
                last_base_action,
                translation_step=env.cfg.translation_step,
                rotation_step=env.cfg.rotation_step,
                gripper_step=env.cfg.gripper_step,
            )
            last_residual_delta = _scale_delta_action(
                last_residual_action,
                translation_step=env.cfg.translation_step,
                rotation_step=env.cfg.rotation_step,
                gripper_step=env.cfg.gripper_step,
            )
            last_command_delta = _scale_delta_action(
                last_commanded_action,
                translation_step=env.cfg.translation_step,
                rotation_step=env.cfg.rotation_step,
                gripper_step=env.cfg.gripper_step,
            )
            last_command_target_state = _compose_target_task_state(
                obs,
                last_commanded_action,
                translation_step=env.cfg.translation_step,
                rotation_step=env.cfg.rotation_step,
                gripper_step=env.cfg.gripper_step,
            )
            last_teleop_active = False
            reward_env_sum = 0.0
            mode = "PREVIEW"
            info_message = "Preview current scene, align the leader, then press SPACE to start"

            print(
                "[INFO] Initial scene:"
                f" box={np.round(info['box_pos'], 4)}"
                f" basket={np.round(info['basket_pos'], 4)}"
            )

            def load_preview_scene(reason: str) -> None:
                nonlocal obs, info, obs_keys, seed_index
                nonlocal transitions, reward_env_sum, mode, info_message, pending_terminated_by
                nonlocal last_base_action, last_residual_action, last_commanded_action
                nonlocal last_realized_action, last_teleop_active
                nonlocal last_follower_state, last_leader_delta, last_base_target_state
                nonlocal last_residual_delta, last_command_delta, last_command_target_state

                obs, info = env.reset(seed=args.seed + seed_index)
                seed_index += 1
                obs_keys = tuple(obs.keys())
                provider.reset()
                if assist_policy is not None:
                    assist_policy.reset()
                transitions = []
                pending_terminated_by = "manual_save"
                reward_env_sum = 0.0
                last_base_action = np.zeros(env.action_dim, dtype=np.float32)
                last_residual_action = np.zeros(env.action_dim, dtype=np.float32)
                last_commanded_action = np.zeros(env.action_dim, dtype=np.float32)
                last_realized_action = np.zeros(env.action_dim, dtype=np.float32)
                last_follower_state = _extract_follower_task_state(obs)
                last_leader_delta = _scale_delta_action(
                    last_base_action,
                    translation_step=env.cfg.translation_step,
                    rotation_step=env.cfg.rotation_step,
                    gripper_step=env.cfg.gripper_step,
                )
                last_base_target_state = _compose_target_task_state(
                    obs,
                    last_base_action,
                    translation_step=env.cfg.translation_step,
                    rotation_step=env.cfg.rotation_step,
                    gripper_step=env.cfg.gripper_step,
                )
                last_residual_delta = _scale_delta_action(
                    last_residual_action,
                    translation_step=env.cfg.translation_step,
                    rotation_step=env.cfg.rotation_step,
                    gripper_step=env.cfg.gripper_step,
                )
                last_command_delta = _scale_delta_action(
                    last_commanded_action,
                    translation_step=env.cfg.translation_step,
                    rotation_step=env.cfg.rotation_step,
                    gripper_step=env.cfg.gripper_step,
                )
                last_command_target_state = _compose_target_task_state(
                    obs,
                    last_commanded_action,
                    translation_step=env.cfg.translation_step,
                    rotation_step=env.cfg.rotation_step,
                    gripper_step=env.cfg.gripper_step,
                )
                last_teleop_active = False
                mode = "PREVIEW"
                info_message = reason
                print(
                    "[INFO] New scene:"
                    f" box={np.round(info['box_pos'], 4)}"
                    f" basket={np.round(info['basket_pos'], 4)}"
                )

            while viewer.is_running() and session_saved_count < args.num_episodes:
                step_start = time.perf_counter()
                _sync_trainer_state(allow_policy_swap=(mode != "RECORDING"))

                if mode == "PREVIEW":
                    leader_target = provider.get_joint_target()
                    if leader_target is not None:
                        leader_joint_pos, leader_gripper, teleop_active = leader_target
                        preview_obs_for_display = _clone_obs(obs)
                        last_follower_state = _extract_follower_task_state(preview_obs_for_display)
                        preview_base_action = env.action_from_joint_target(leader_joint_pos, leader_gripper)
                        last_base_action = preview_base_action.copy()
                        last_leader_delta = _scale_delta_action(
                            preview_base_action,
                            translation_step=env.cfg.translation_step,
                            rotation_step=env.cfg.rotation_step,
                            gripper_step=env.cfg.gripper_step,
                        )
                        last_base_target_state = _compose_target_task_state(
                            preview_obs_for_display,
                            preview_base_action,
                            translation_step=env.cfg.translation_step,
                            rotation_step=env.cfg.rotation_step,
                            gripper_step=env.cfg.gripper_step,
                        )
                        preview_residual_action = np.zeros(env.action_dim, dtype=np.float32)
                        if residual_enabled and teleop_active and assist_policy is not None:
                            preview_residual_action = np.clip(
                                assist_policy.get_action(preview_obs_for_display, preview_base_action)
                                * float(args.residual_scale),
                                -float(args.residual_limit),
                                float(args.residual_limit),
                            ).astype(np.float32)
                            obs, preview_info = env.sync_base_joint_target(
                                leader_joint_pos,
                                leader_gripper,
                                base_action=preview_base_action,
                                residual_action=preview_residual_action,
                            )
                            preview_commanded_action = np.asarray(
                                preview_info.get("commanded_action", preview_base_action + preview_residual_action),
                                dtype=np.float32,
                            ).copy()
                            preview_realized_action = np.asarray(
                                preview_info.get("realized_action", preview_commanded_action),
                                dtype=np.float32,
                            ).copy()
                        else:
                            preview_commanded_action = env.action_from_joint_target(leader_joint_pos, leader_gripper)
                            obs = env.sync_absolute_joint_target(leader_joint_pos, leader_gripper)
                            preview_realized_action = preview_commanded_action.copy()

                        last_residual_action = preview_residual_action.copy()
                        last_residual_delta = _scale_delta_action(
                            last_residual_action,
                            translation_step=env.cfg.translation_step,
                            rotation_step=env.cfg.rotation_step,
                            gripper_step=env.cfg.gripper_step,
                        )
                        last_commanded_action = preview_commanded_action.copy()
                        last_command_delta = _scale_delta_action(
                            last_commanded_action,
                            translation_step=env.cfg.translation_step,
                            rotation_step=env.cfg.rotation_step,
                            gripper_step=env.cfg.gripper_step,
                        )
                        last_command_target_state = _compose_target_task_state(
                            preview_obs_for_display,
                            last_commanded_action,
                            translation_step=env.cfg.translation_step,
                            rotation_step=env.cfg.rotation_step,
                            gripper_step=env.cfg.gripper_step,
                        )
                        last_realized_action = preview_realized_action.copy()
                        last_teleop_active = bool(teleop_active)
                    else:
                        last_teleop_active = False

                if viewer._teleop_toggle_residual:
                    viewer._teleop_toggle_residual = False
                    if assist_policy is None:
                        info_message = "Residual unavailable: load checkpoint or enable realtime trainer"
                        print("[WARN] Residual toggle ignored because no checkpoint/trainer is available")
                    else:
                        residual_enabled = not residual_enabled
                        assist_policy.reset()
                        info_message = f"Residual {'enabled' if residual_enabled else 'disabled'}"
                        print(f"[INFO] Residual assist {'enabled' if residual_enabled else 'disabled'}")

                if viewer._teleop_record_discard_round:
                    viewer._teleop_record_discard_round = False
                    if mode in {"RECORDING", "REVIEW"}:
                        print("[INFO] Current trial discarded")
                        load_preview_scene("Discarded. Preview next scene, then press SPACE to start")

                if viewer._teleop_record_stop_and_keep:
                    viewer._teleop_record_stop_and_keep = False
                    if mode in {"RECORDING", "REVIEW"}:
                        if not transitions:
                            print("[WARN] Current trial is empty; nothing to save")
                            info_message = "Current trial is empty"
                        else:
                            if mode == "RECORDING":
                                _finalize_transitions(transitions, success=True)
                            rollout = _build_rollout(transitions, obs_keys)
                            save_rollout(rollout, terminated_by=pending_terminated_by, viewer_mode=True)
                            if session_saved_count >= args.num_episodes:
                                info_message = "Target number of episodes reached"
                                viewer.sync()
                                break
                            load_preview_scene("Saved. Preview next scene, then press SPACE to start")

                if viewer._teleop_record_start_round:
                    viewer._teleop_record_start_round = False
                    if mode == "PREVIEW":
                        transitions = []
                        pending_terminated_by = "manual_save"
                        reward_env_sum = 0.0
                        last_base_action = np.zeros(env.action_dim, dtype=np.float32)
                        last_residual_action = np.zeros(env.action_dim, dtype=np.float32)
                        last_commanded_action = np.zeros(env.action_dim, dtype=np.float32)
                        last_realized_action = np.zeros(env.action_dim, dtype=np.float32)
                        last_follower_state = _extract_follower_task_state(obs)
                        last_leader_delta = _scale_delta_action(
                            last_base_action,
                            translation_step=env.cfg.translation_step,
                            rotation_step=env.cfg.rotation_step,
                            gripper_step=env.cfg.gripper_step,
                        )
                        last_base_target_state = _compose_target_task_state(
                            obs,
                            last_base_action,
                            translation_step=env.cfg.translation_step,
                            rotation_step=env.cfg.rotation_step,
                            gripper_step=env.cfg.gripper_step,
                        )
                        last_residual_delta = _scale_delta_action(
                            last_residual_action,
                            translation_step=env.cfg.translation_step,
                            rotation_step=env.cfg.rotation_step,
                            gripper_step=env.cfg.gripper_step,
                        )
                        last_command_delta = _scale_delta_action(
                            last_commanded_action,
                            translation_step=env.cfg.translation_step,
                            rotation_step=env.cfg.rotation_step,
                            gripper_step=env.cfg.gripper_step,
                        )
                        last_command_target_state = _compose_target_task_state(
                            obs,
                            last_commanded_action,
                            translation_step=env.cfg.translation_step,
                            rotation_step=env.cfg.rotation_step,
                            gripper_step=env.cfg.gripper_step,
                        )
                        if assist_policy is not None:
                            assist_policy.reset()
                        info_message = (
                            "Recording with residual assist"
                            if residual_enabled and assist_policy is not None
                            else "Recording pure teleop"
                        )
                        mode = "RECORDING"
                        print(
                            "[INFO] Recording started"
                            f" residual={'ON' if residual_enabled and assist_policy is not None else 'OFF'}"
                        )

                if mode == "RECORDING":
                    obs, transition, teleop_active, step_info = _record_step(
                        env,
                        provider,
                        obs,
                        residual_policy=assist_policy,
                        residual_enabled=residual_enabled,
                        residual_scale=args.residual_scale,
                        residual_limit=args.residual_limit,
                    )
                    transitions.append(transition)
                    reward_env_sum += float(transition.reward_env)
                    prev_obs_for_display = unflatten_observation(transition.observation_state)
                    last_base_action = transition.base_action.copy()
                    last_follower_state = _extract_follower_task_state(prev_obs_for_display)
                    last_leader_delta = _scale_delta_action(
                        transition.base_action,
                        translation_step=env.cfg.translation_step,
                        rotation_step=env.cfg.rotation_step,
                        gripper_step=env.cfg.gripper_step,
                    )
                    last_base_target_state = _compose_target_task_state(
                        prev_obs_for_display,
                        transition.base_action,
                        translation_step=env.cfg.translation_step,
                        rotation_step=env.cfg.rotation_step,
                        gripper_step=env.cfg.gripper_step,
                    )
                    last_residual_action = np.asarray(
                        step_info.get("residual_action", np.zeros(env.action_dim, dtype=np.float32)),
                        dtype=np.float32,
                    ).copy()
                    last_residual_delta = _scale_delta_action(
                        last_residual_action,
                        translation_step=env.cfg.translation_step,
                        rotation_step=env.cfg.rotation_step,
                        gripper_step=env.cfg.gripper_step,
                    )
                    last_commanded_action = np.asarray(
                        step_info.get("commanded_action", transition.base_action + transition.residual_action),
                        dtype=np.float32,
                    ).copy()
                    last_command_delta = _scale_delta_action(
                        last_commanded_action,
                        translation_step=env.cfg.translation_step,
                        rotation_step=env.cfg.rotation_step,
                        gripper_step=env.cfg.gripper_step,
                    )
                    last_command_target_state = _compose_target_task_state(
                        prev_obs_for_display,
                        last_commanded_action,
                        translation_step=env.cfg.translation_step,
                        rotation_step=env.cfg.rotation_step,
                        gripper_step=env.cfg.gripper_step,
                    )
                    last_realized_action = transition.realized_action.copy()
                    last_teleop_active = teleop_active

                    if transition.truncated:
                        _finalize_transitions(transitions, success=False)
                        reward_env_sum = float(sum(t.reward_env for t in transitions))
                        pending_terminated_by = "max_steps"
                        mode = "REVIEW"
                        info_message = (
                            f"Episode finished by {pending_terminated_by}. "
                            "Press S to save or Q to discard"
                        )
                        print(f"[INFO] Round finished by {pending_terminated_by}; waiting for manual save/discard")

                _render_overlay(
                    viewer,
                    mode=mode,
                    teleop_active=last_teleop_active,
                    residual_enabled=residual_enabled and assist_policy is not None,
                    saved_count=session_saved_count,
                    target_count=args.num_episodes,
                    steps=len(transitions) if mode == "RECORDING" else 0,
                    total_return=reward_env_sum if mode == "RECORDING" else 0.0,
                    message=info_message,
                    follower_state=last_follower_state,
                    leader_delta=last_leader_delta,
                    base_target_state=last_base_target_state,
                    base_action=last_base_action,
                    residual_delta=last_residual_delta,
                    residual_action=last_residual_action,
                    command_target_state=last_command_target_state,
                    command_delta=last_command_delta,
                    commanded_action=last_commanded_action,
                    realized_action=last_realized_action,
                    box_pos=obs.get("box_pos") if isinstance(obs, dict) else None,
                    basket_pos=obs.get("basket_pos") if isinstance(obs, dict) else None,
                    trainer_status=trainer_status,
                    show_details=bool(args.show_overlay_details),
                )
                viewer.sync()

                sleep_dt = 0.02 if mode != "RECORDING" else max(env.cfg.control_dt - (time.perf_counter() - step_start), 0.0)
                if sleep_dt > 0:
                    precise_sleep(sleep_dt)
    finally:
        if trainer_queue is not None:
            try:
                trainer_queue.put_nowait(None)
            except Exception:
                pass
        provider.close()
        env.close()


if __name__ == "__main__":
    default_xml = PROJECT_ROOT / "assets" / "mujoco" / "Alicia_D_v5_6" / "gripper_50mm" / "alicia_d_follower.xml"
    default_train_save = PROJECT_ROOT / "logs" / "residual_td3_live.pt"
    parser = argparse.ArgumentParser(description="Record teleoperation episodes and optionally train a residual policy online.")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--xml", type=Path, default=default_xml)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum control steps per episode")
    parser.add_argument("--control_dt", type=float, default=0.05, help="Control interval in seconds")
    parser.add_argument(
        "--fixed_scene",
        action="store_true",
        help="Disable box spawn randomization and basket jitter. The box resets to the spawn-range center and the basket stays at the XML default.",
    )
    parser.add_argument(
        "--fixed_box_xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Optional fixed box XY reset position. Implies a non-random box position while keeping the basket behavior unchanged unless --fixed_scene is also set.",
    )
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--variant", type=str, default="leader")
    parser.add_argument("--gripper_type", type=str, default="50mm")
    parser.add_argument("--no_deadman", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional residual checkpoint used for assist inference or trainer init")
    parser.add_argument("--enable-residual-on-start", action="store_true", help="Start the recorder with residual assist enabled")
    parser.add_argument("--residual-limit", type=float, default=0.1, help="Clamp residual action per dimension during recording")
    parser.add_argument("--residual-scale", type=float, default=1.0, help="Scale the policy residual before clamping")
    parser.add_argument("--cpu", action="store_true", help="Force checkpoint residual inference onto CPU")
    parser.add_argument("--realtime-train", action="store_true", help="After every saved episode, immediately update the residual network")
    parser.add_argument("--train-save-path", type=Path, default=default_train_save, help="Checkpoint path updated by realtime training")
    parser.add_argument("--train-updates-per-save", type=int, default=200, help="Gradient updates run after each saved episode")
    parser.add_argument("--train-actor-updates-per-save", type=int, default=100, help="How many of the per-save updates also update the actor")
    parser.add_argument("--train-min-transitions", type=int, default=2000, help="Start realtime updates only after this many transitions are available (recommend >= 2000 for stable learning)")
    parser.add_argument("--train-critic-warmup-steps", type=int, default=1000, help="Critic-only gradient steps before enabling actor updates (prevents actor learning from unreliable Q-values)")
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--train-offline-fraction", type=float, default=0.5)
    parser.add_argument("--train-actor-lr", type=float, default=3e-4)
    parser.add_argument("--train-critic-lr", type=float, default=3e-4)
    parser.add_argument("--train-buffer-capacity", type=int, default=200_000)
    parser.add_argument("--train-hidden-dim", type=int, default=256)
    parser.add_argument("--train-history-len", type=int, default=4)
    parser.add_argument("--train-residual-mag-reg-weight", type=float, default=1.0)
    parser.add_argument("--train-offline-zero-bc-weight", type=float, default=5.0)
    parser.add_argument("--train-teacher-bc-weight", type=float, default=2.0)
    parser.add_argument("--train-log-interval", type=int, default=50)
    parser.add_argument("--disable-state-norm", action="store_true")
    parser.add_argument("--disable-reward-norm", action="store_true")
    parser.add_argument(
        "--show-overlay-details",
        action="store_true",
        help="Show detailed action/target vectors in the MuJoCo HUD. Default keeps the overlay compact.",
    )
    parser.add_argument("--headless", action="store_true", help="Disable the MuJoCo viewer and record in the terminal only")
    main(parser.parse_args())
