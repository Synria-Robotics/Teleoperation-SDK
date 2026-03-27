from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teleop_sdk.data import EpisodeRollout, Transition, save_episode_npz
from teleop_sdk.envs import MujocoPickPlaceTeleopEnv, PickPlaceTaskConfig, flatten_observation
from teleop_sdk.providers import DemoSnapPhasePolicy, HeuristicResidualPolicy, LeaderTeleopProvider, StrongDemoResidualPolicy, ZeroResidualPolicy
from teleop_sdk.rewards import PickPlaceReward
from teleop_sdk.runners import SharedControlRunner
from train_residual_td3_teleop import MLPActor, TensorNormalizer, TorchResidualPolicy
from utils.fps_utils import precise_sleep

MJ_REEXEC_ENV = "TELEOP_SDK_MJPYTHON_REEXEC"


def _is_assisted_mode(args: argparse.Namespace) -> bool:
    return args.checkpoint is not None or args.assist_policy in {"heuristic", "demo_strong", "demo_snap"}


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
    return {k: np.asarray(v, dtype=np.float32).copy() for k, v in obs.items()}


def _build_rollout(
    transitions: list[Transition],
    obs_keys: tuple[str, ...],
    reward_env_sum: float,
    reward_total_sum: float,
) -> EpisodeRollout:
    for idx in range(len(transitions) - 1):
        transitions[idx].next_base_action = transitions[idx + 1].base_action.copy()
    if transitions:
        transitions[-1].next_base_action = np.zeros_like(transitions[-1].base_action)
    success = any(t.success for t in transitions)
    return EpisodeRollout(
        transitions=transitions,
        obs_keys=obs_keys,
        success=success,
        episode_return_env=reward_env_sum,
        episode_return_total=reward_total_sum,
    )


def _render_overlay(
    viewer,
    *,
    mode: str,
    policy_mode: str,
    copilot_enabled: bool,
    teleop_active: bool,
    residual_status: str,
    current_alpha: float,
    saved_count: int,
    target_count: int,
    steps: int,
    total_return: float,
    message: str,
    residual_norm: float = 0.0,
    base_action: np.ndarray | None = None,
    residual_action: np.ndarray | None = None,
    exec_delta: np.ndarray | None = None,
    box_pos: np.ndarray | None = None,
    basket_pos: np.ndarray | None = None,
) -> None:
    def _fmt_vec(vec: np.ndarray, digits: int = 2) -> str:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        return "[" + ", ".join(f"{float(x):.{digits}f}" for x in arr) + "]"

    left_lines = [
        "Teleop Recorder",
        f"Mode: {mode}",
        f"Policy: {policy_mode}",
        f"Copilot: {'ON' if copilot_enabled else 'OFF'}",
        f"Leader: {'ACTIVE' if teleop_active else 'IDLE'}",
        f"Residual status: {residual_status}",
        f"Alpha: {current_alpha:.3f}",
        f"Saved: {saved_count}/{target_count}",
        f"Steps: {steps}",
        f"Return: {total_return:.2f}",
        f"Residual: {residual_norm:.3f}",
    ]
    if base_action is not None:
        left_lines.append(f"Human/base:   {_fmt_vec(base_action, digits=2)}")
    if residual_action is not None:
        left_lines.append(f"Residual cmd: {_fmt_vec(residual_action, digits=2)}")
        cmd_delta = float(current_alpha) * np.asarray(residual_action, dtype=np.float32)
        left_lines.append(f"Cmd delta:    {_fmt_vec(cmd_delta, digits=2)}")
    if exec_delta is not None:
        left_lines.append(f"Applied delta:{_fmt_vec(exec_delta, digits=2)}")
    if residual_action is not None and exec_delta is not None:
        denom = float(np.linalg.norm(float(current_alpha) * np.asarray(residual_action, dtype=np.float32)))
        numer = float(np.linalg.norm(np.asarray(exec_delta, dtype=np.float32)))
        ratio = numer / max(denom, 1e-6)
        left_lines.append(f"Realize ratio:{ratio:.2f}")
    if box_pos is not None:
        left_lines.append(f"Box:    {np.round(np.asarray(box_pos, dtype=np.float32), 3).tolist()}")
    if basket_pos is not None:
        left_lines.append(f"Basket: {np.round(np.asarray(basket_pos, dtype=np.float32), 3).tolist()}")
    left_lines.extend(
        [
            "",
            "Controls",
            "SPACE  preview:start  review:save+next",
            "Q      recording/review:discard",
            "S      recording:stop and keep",
            "O      toggle copilot",
        ]
    )
    right_lines = ["", "", "", "", "", "", "", "", "", "", message]

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


def _load_residual_policy(
    checkpoint_path: Path | None,
    *,
    obs_dim: int,
    action_dim: int,
    residual_limit_override: float | None,
    alpha_override: float | None,
    assist_policy: str,
    heuristic_reach_gain: float,
    heuristic_place_gain: float,
    heuristic_near_box_threshold: float,
    heuristic_lifted_height: float,
    heuristic_idle_deadband: float,
    demo_align_gain: float,
    demo_near_align_gain: float,
    demo_near_align_floor: float,
    demo_descend_gain: float,
    demo_close_gain: float,
    demo_lift_gain: float,
    demo_place_gain: float,
    demo_near_place_gain: float,
    demo_near_place_floor: float,
    demo_place_descend_gain: float,
    demo_grasp_xy_threshold: float,
    demo_descend_xy_threshold: float,
    demo_descend_z_threshold: float,
    demo_grasp_xyz_threshold: float,
    demo_close_xy_threshold: float,
    demo_close_z_threshold: float,
    demo_place_xy_threshold: float,
    demo_place_descend_xy_threshold: float,
    demo_lifted_height: float,
    demo_idle_deadband: float,
    demo_snap_grasp_xy_threshold: float,
    demo_snap_place_xy_threshold: float,
    demo_snap_xy_scale: float,
    demo_snap_rot_scale: float,
    demo_snap_descend_gain: float,
    demo_snap_close_gain: float,
    demo_snap_lift_gain: float,
) -> tuple[object, float, float, str, str | None]:
    if assist_policy == "heuristic":
        residual_limit = 0.08 if residual_limit_override is None else float(residual_limit_override)
        alpha = 0.05 if alpha_override is None else float(alpha_override)
        policy = HeuristicResidualPolicy(
            residual_limit=residual_limit,
            reach_gain=heuristic_reach_gain,
            place_gain=heuristic_place_gain,
            near_box_threshold=heuristic_near_box_threshold,
            lifted_height=heuristic_lifted_height,
            idle_deadband=heuristic_idle_deadband,
        )
        label = (
            "HeuristicAssist("
            f"alpha={alpha:.2f}, limit={residual_limit:.2f}, "
            f"reach={heuristic_reach_gain:.2f}, place={heuristic_place_gain:.2f}"
            ")"
        )
        return policy, alpha, residual_limit, label, None

    if assist_policy == "demo_strong":
        residual_limit = 0.80 if residual_limit_override is None else float(residual_limit_override)
        alpha = 0.75 if alpha_override is None else float(alpha_override)
        policy = StrongDemoResidualPolicy(
            residual_limit=residual_limit,
            align_gain=demo_align_gain,
            near_align_gain=demo_near_align_gain,
            near_align_floor=demo_near_align_floor,
            descend_gain=demo_descend_gain,
            close_gain=demo_close_gain,
            lift_gain=demo_lift_gain,
            place_gain=demo_place_gain,
            near_place_gain=demo_near_place_gain,
            near_place_floor=demo_near_place_floor,
            place_descend_gain=demo_place_descend_gain,
            grasp_xy_threshold=demo_grasp_xy_threshold,
            descend_xy_threshold=demo_descend_xy_threshold,
            descend_z_threshold=demo_descend_z_threshold,
            grasp_xyz_threshold=demo_grasp_xyz_threshold,
            close_xy_threshold=demo_close_xy_threshold,
            close_z_threshold=demo_close_z_threshold,
            place_xy_threshold=demo_place_xy_threshold,
            place_descend_xy_threshold=demo_place_descend_xy_threshold,
            lifted_height=demo_lifted_height,
            idle_deadband=demo_idle_deadband,
        )
        label = (
            "StrongDemoAssist("
            f"alpha={alpha:.2f}, limit={residual_limit:.2f}, "
            f"align={demo_align_gain:.2f}, close={demo_close_gain:.2f}, place={demo_place_gain:.2f}"
            ")"
        )
        return policy, alpha, residual_limit, label, None

    if assist_policy == "demo_snap":
        residual_limit = 0.90 if residual_limit_override is None else float(residual_limit_override)
        alpha = 1.00 if alpha_override is None else float(alpha_override)
        policy = DemoSnapPhasePolicy(
            residual_limit=residual_limit,
            align_floor=max(demo_near_align_floor, 0.55),
            place_floor=max(demo_near_place_floor, 0.30),
            descend_speed=max(demo_snap_descend_gain * 0.45, 0.75),
            lift_speed=max(demo_snap_lift_gain * 0.60, 0.80),
            close_speed=max(demo_snap_close_gain, 1.0),
            release_speed=1.0,
            grasp_align_xy=max(demo_snap_grasp_xy_threshold, 0.16),
            grasp_descend_xy=0.05,
            grasp_close_xy=0.02,
            grasp_close_z=0.03,
            lift_height=max(demo_lifted_height, 0.11),
            place_align_xy=max(demo_snap_place_xy_threshold, 0.16),
            place_descend_xy=0.06,
            place_release_xy=0.035,
            place_release_z=0.035,
            human_xy_scale=demo_snap_xy_scale,
            human_rot_scale=demo_snap_rot_scale,
        )
        label = f"DemoSnapPhase(alpha={alpha:.2f}, limit={residual_limit:.2f})"
        return policy, alpha, residual_limit, label, None

    if checkpoint_path is None:
        alpha = 0.0 if alpha_override is None else float(alpha_override)
        return ZeroResidualPolicy(), alpha, 0.0, "TeleopOnly", None

    checkpoint = torch.load(checkpoint_path.resolve(), map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    residual_limit = (
        float(residual_limit_override)
        if residual_limit_override is not None
        else float(cfg.get("residual_limit", 0.35))
    )
    alpha = float(alpha_override) if alpha_override is not None else float(cfg.get("alpha", 0.2))
    idle_deadband = float(cfg.get("idle_deadband", 0.05))
    idle_scale_span = float(cfg.get("idle_scale_span", 0.20))
    state_normalizer = None
    if "state_mean" in checkpoint and "state_std" in checkpoint and bool(cfg.get("normalize_state", True)):
        state_normalizer = TensorNormalizer(checkpoint["state_mean"], checkpoint["state_std"], torch.device("cpu"))

    actor = MLPActor(obs_dim, action_dim, residual_limit=residual_limit)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    policy = TorchResidualPolicy(
        actor,
        torch.device("cpu"),
        noise_scale=0.0,
        residual_limit=residual_limit,
        idle_deadband=idle_deadband,
        idle_scale_span=idle_scale_span,
        state_normalizer=state_normalizer,
    )
    label = f"Assisted(alpha={alpha:.2f}, limit={residual_limit:.2f})"
    return policy, alpha, residual_limit, label, str(checkpoint_path.resolve())


def _next_episode_index(output_dir: Path) -> int:
    max_index = -1
    for path in output_dir.glob("episode_*.npz"):
        suffix = path.stem.removeprefix("episode_")
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return max_index + 1


def main(args: argparse.Namespace) -> None:
    if not args.headless:
        _ensure_mjpython_on_macos()

    reward_model = PickPlaceReward()
    env_cfg = PickPlaceTaskConfig(xml_path=str(args.xml.resolve()), seed=args.seed)
    env = MujocoPickPlaceTeleopEnv(env_cfg, reward=reward_model)
    obs, _ = env.reset(seed=args.seed)
    obs_dim = int(flatten_observation(obs).shape[0])
    action_dim = env.action_dim
    residual_policy, assist_alpha, residual_limit, policy_mode, checkpoint_meta = _load_residual_policy(
        args.checkpoint,
        obs_dim=obs_dim,
        action_dim=action_dim,
        residual_limit_override=args.residual_limit,
        alpha_override=args.alpha,
        assist_policy=args.assist_policy,
        heuristic_reach_gain=args.heuristic_reach_gain,
        heuristic_place_gain=args.heuristic_place_gain,
        heuristic_near_box_threshold=args.heuristic_near_box_threshold,
        heuristic_lifted_height=args.heuristic_lifted_height,
        heuristic_idle_deadband=args.heuristic_idle_deadband,
        demo_align_gain=args.demo_align_gain,
        demo_near_align_gain=args.demo_near_align_gain,
        demo_near_align_floor=args.demo_near_align_floor,
        demo_descend_gain=args.demo_descend_gain,
        demo_close_gain=args.demo_close_gain,
        demo_lift_gain=args.demo_lift_gain,
        demo_place_gain=args.demo_place_gain,
        demo_near_place_gain=args.demo_near_place_gain,
        demo_near_place_floor=args.demo_near_place_floor,
        demo_place_descend_gain=args.demo_place_descend_gain,
        demo_grasp_xy_threshold=args.demo_grasp_xy_threshold,
        demo_descend_xy_threshold=args.demo_descend_xy_threshold,
        demo_descend_z_threshold=args.demo_descend_z_threshold,
        demo_grasp_xyz_threshold=args.demo_grasp_xyz_threshold,
        demo_close_xy_threshold=args.demo_close_xy_threshold,
        demo_close_z_threshold=args.demo_close_z_threshold,
        demo_place_xy_threshold=args.demo_place_xy_threshold,
        demo_place_descend_xy_threshold=args.demo_place_descend_xy_threshold,
        demo_lifted_height=args.demo_lifted_height,
        demo_idle_deadband=args.demo_idle_deadband,
        demo_snap_grasp_xy_threshold=args.demo_snap_grasp_xy_threshold,
        demo_snap_place_xy_threshold=args.demo_snap_place_xy_threshold,
        demo_snap_xy_scale=args.demo_snap_xy_scale,
        demo_snap_rot_scale=args.demo_snap_rot_scale,
        demo_snap_descend_gain=args.demo_snap_descend_gain,
        demo_snap_close_gain=args.demo_snap_close_gain,
        demo_snap_lift_gain=args.demo_snap_lift_gain,
    )
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
    runner = SharedControlRunner(env, provider, residual_policy, reward_model.compute_total_reward)

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (
            PROJECT_ROOT
            / "logs"
            / ("assisted_teleop_dataset" if _is_assisted_mode(args) else "teleop_dataset")
        ).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    next_episode_index = _next_episode_index(output_dir)
    print("[INFO] Recording controls:")
    print("       SPACE: start recording on the current live-aligned scene")
    print("       S: stop current recording and keep it for review")
    print("       Q: discard current recording and load a fresh preview scene")
    if _is_assisted_mode(args):
        print("       O: toggle copilot on/off")
    print("       MuJoCo follower stays live-aligned to the leader in preview")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Existing episodes: {next_episode_index}")
    print(f"[INFO] New recordings will start at episode_{next_episode_index:04d}.npz")
    print(f"[INFO] Policy mode: {policy_mode}")
    if checkpoint_meta is not None:
        print(f"[INFO] Assisted checkpoint: {checkpoint_meta}")
        print(f"[INFO] Assisted alpha: {assist_alpha:.3f}")
        print(f"[INFO] Residual limit: {residual_limit:.3f}")
    print(
        "[INFO] Box reset range:"
        f" x={env.cfg.box_spawn_xy_low[0]:.3f}..{env.cfg.box_spawn_xy_high[0]:.3f},"
        f" y={env.cfg.box_spawn_xy_low[1]:.3f}..{env.cfg.box_spawn_xy_high[1]:.3f}"
    )

    try:
        if args.headless:
            session_saved_count = 0
            for _ in range(args.num_episodes):
                provider.wait_for_start_signal()
                provider.clear_pending_start_stop()
                rollout = runner.run_episode(
                    alpha=assist_alpha,
                    seed=args.seed + next_episode_index,
                    should_stop=provider.consume_stop_signal,
                )
                if rollout.steps == 0:
                    print(f"[WARN] episode_{next_episode_index:04d} is empty; waiting for the next start signal")
                    continue
                path = output_dir / f"episode_{next_episode_index:04d}.npz"
                rollout.meta.update(
                    {
                        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "port": args.port,
                        "variant": args.variant,
                        "gripper_type": args.gripper_type,
                        "start_stop_button": "button2",
                        "viewer": False,
                        "collection_mode": "assisted_teleop" if _is_assisted_mode(args) else "teleop_only",
                        "assist_alpha": float(assist_alpha),
                        "residual_limit": float(residual_limit),
                        "checkpoint": checkpoint_meta or "",
                        "heuristic_reach_gain": float(args.heuristic_reach_gain),
                        "heuristic_place_gain": float(args.heuristic_place_gain),
                        "heuristic_near_box_threshold": float(args.heuristic_near_box_threshold),
                        "heuristic_lifted_height": float(args.heuristic_lifted_height),
                        "heuristic_idle_deadband": float(args.heuristic_idle_deadband),
                        "demo_align_gain": float(args.demo_align_gain),
                        "demo_near_align_gain": float(args.demo_near_align_gain),
                        "demo_near_align_floor": float(args.demo_near_align_floor),
                        "demo_descend_gain": float(args.demo_descend_gain),
                        "demo_close_gain": float(args.demo_close_gain),
                        "demo_lift_gain": float(args.demo_lift_gain),
                        "demo_place_gain": float(args.demo_place_gain),
                        "demo_near_place_gain": float(args.demo_near_place_gain),
                        "demo_near_place_floor": float(args.demo_near_place_floor),
                        "demo_place_descend_gain": float(args.demo_place_descend_gain),
                        "demo_grasp_xy_threshold": float(args.demo_grasp_xy_threshold),
                        "demo_descend_xy_threshold": float(args.demo_descend_xy_threshold),
                        "demo_descend_z_threshold": float(args.demo_descend_z_threshold),
                        "demo_grasp_xyz_threshold": float(args.demo_grasp_xyz_threshold),
                        "demo_close_xy_threshold": float(args.demo_close_xy_threshold),
                        "demo_close_z_threshold": float(args.demo_close_z_threshold),
                        "demo_place_xy_threshold": float(args.demo_place_xy_threshold),
                        "demo_place_descend_xy_threshold": float(args.demo_place_descend_xy_threshold),
                        "demo_lifted_height": float(args.demo_lifted_height),
                        "demo_idle_deadband": float(args.demo_idle_deadband),
                        "demo_snap_grasp_xy_threshold": float(args.demo_snap_grasp_xy_threshold),
                        "demo_snap_place_xy_threshold": float(args.demo_snap_place_xy_threshold),
                        "demo_snap_xy_scale": float(args.demo_snap_xy_scale),
                        "demo_snap_rot_scale": float(args.demo_snap_rot_scale),
                        "demo_snap_descend_gain": float(args.demo_snap_descend_gain),
                        "demo_snap_close_gain": float(args.demo_snap_close_gain),
                        "demo_snap_lift_gain": float(args.demo_snap_lift_gain),
                    }
                )
                save_episode_npz(path, rollout)
                session_saved_count += 1
                next_episode_index += 1
                print(
                    f"[{session_saved_count}/{args.num_episodes}] saved {path.name}"
                    f" success={rollout.success} return={rollout.episode_return_total:.3f}"
                )
            return

        def key_callback(keycode):
            try:
                key = chr(keycode)
            except ValueError:
                return
            if key == " ":
                viewer._teleop_record_start_round = True
            elif key.lower() == "q":
                viewer._teleop_record_discard_round = True
            elif key.lower() == "s":
                viewer._teleop_record_stop_and_keep = True
            elif key.lower() == "o":
                viewer._teleop_record_toggle_copilot = True

        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
            viewer._teleop_record_start_round = False
            viewer._teleop_record_discard_round = False
            viewer._teleop_record_stop_and_keep = False
            viewer._teleop_record_toggle_copilot = False
            print("[INFO] MuJoCo viewer launched for teleop recording")

            mode = "PREVIEW"
            info_message = "Preview scene. Follower is live-aligned to leader. Press SPACE to start recording"
            obs, info = env.reset(seed=args.seed)
            obs_keys = tuple(obs.keys())
            provider.reset()
            session_saved_count = 0
            seed_index = 1
            transitions: list[Transition] = []
            prev_residual = np.zeros(7, dtype=np.float32)
            last_residual_norm = 0.0
            last_base_action = np.zeros(7, dtype=np.float32)
            last_residual_action = np.zeros(7, dtype=np.float32)
            last_exec_delta = np.zeros(7, dtype=np.float32)
            last_alpha = 0.0
            last_teleop_active = False
            last_residual_status = "preview"
            reward_env_sum = 0.0
            reward_total_sum = 0.0
            pending_rollout: EpisodeRollout | None = None
            pending_reason = ""
            copilot_enabled = False
            recording_started_at = 0.0

            print(
                "[INFO] Initial scene:"
                f" box={np.round(info['box_pos'], 4)}"
                f" basket={np.round(info['basket_pos'], 4)}"
            )
            if _is_assisted_mode(args):
                print("[INFO] Copilot is OFF by default. Press O to enable/disable it.")

            def load_preview_scene(reason: str) -> None:
                nonlocal obs, info, obs_keys, seed_index
                nonlocal transitions, pending_rollout, reward_env_sum, reward_total_sum, prev_residual, pending_reason, mode, info_message, last_residual_norm, last_base_action, last_residual_action, last_exec_delta, last_alpha, last_teleop_active, last_residual_status

                obs, info = env.reset(seed=args.seed + seed_index)
                seed_index += 1
                obs_keys = tuple(obs.keys())
                provider.reset()
                residual_policy.reset()
                transitions = []
                pending_rollout = None
                reward_env_sum = 0.0
                reward_total_sum = 0.0
                prev_residual = np.zeros(7, dtype=np.float32)
                last_residual_norm = 0.0
                last_base_action = np.zeros(7, dtype=np.float32)
                last_residual_action = np.zeros(7, dtype=np.float32)
                last_exec_delta = np.zeros(7, dtype=np.float32)
                last_alpha = 0.0
                last_teleop_active = False
                last_residual_status = "preview"
                pending_reason = ""
                mode = "PREVIEW"
                info_message = reason
                print(
                    "[INFO] New scene:"
                    f" box={np.round(info['box_pos'], 4)}"
                    f" basket={np.round(info['basket_pos'], 4)}"
                )

            while viewer.is_running() and session_saved_count < args.num_episodes:
                step_start = time.perf_counter()
                if mode == "PREVIEW":
                    leader_target = provider.get_joint_target()
                    if leader_target is not None:
                        leader_joint_pos, leader_gripper, _ = leader_target
                        obs = env.sync_absolute_joint_target(leader_joint_pos, leader_gripper)
                    last_teleop_active = False
                    last_residual_status = "preview"

                if viewer._teleop_record_discard_round:
                    viewer._teleop_record_discard_round = False
                    if mode in {"RECORDING", "REVIEW"}:
                        print("[INFO] Current trial discarded")
                        load_preview_scene("Discarded. Preview next scene, then press SPACE to start")

                if viewer._teleop_record_toggle_copilot:
                    viewer._teleop_record_toggle_copilot = False
                    if _is_assisted_mode(args):
                        copilot_enabled = not copilot_enabled
                        state_text = "ON" if copilot_enabled else "OFF"
                        info_message = f"Copilot toggled {state_text}"
                        print(f"[INFO] Copilot toggled {state_text}")

                if viewer._teleop_record_stop_and_keep:
                    viewer._teleop_record_stop_and_keep = False
                    if mode == "RECORDING":
                        pending_rollout = _build_rollout(transitions, obs_keys, reward_env_sum, reward_total_sum)
                        pending_reason = "manual_stop"
                        mode = "REVIEW"
                        info_message = "Review trial. SPACE saves and loads next preview; Q discards"
                        print("[INFO] Current trial stopped and kept for review")

                if viewer._teleop_record_start_round:
                    viewer._teleop_record_start_round = False
                    if mode == "PREVIEW":
                        transitions = []
                        pending_rollout = None
                        reward_env_sum = 0.0
                        reward_total_sum = 0.0
                        prev_residual = np.zeros(7, dtype=np.float32)
                        last_residual_norm = 0.0
                        last_base_action = np.zeros(7, dtype=np.float32)
                        last_residual_action = np.zeros(7, dtype=np.float32)
                        last_exec_delta = np.zeros(7, dtype=np.float32)
                        last_alpha = 0.0
                        last_teleop_active = False
                        last_residual_status = "recording_not_started"
                        residual_policy.reset()
                        recording_started_at = time.perf_counter()
                        mode = "RECORDING"
                        if not _is_assisted_mode(args):
                            info_message = "Recording pure teleop with absolute live sync to leader"
                            print("[INFO] Recording started in pure teleop mode")
                        else:
                            state_text = "ON" if copilot_enabled else "OFF"
                            info_message = f"Recording assisted teleop with copilot {state_text} (alpha={assist_alpha:.2f})"
                            print(f"[INFO] Recording started in assisted mode with copilot {state_text} and alpha={assist_alpha:.2f}")
                    elif mode == "REVIEW":
                        if pending_rollout is not None and pending_rollout.steps > 0:
                            path = output_dir / f"episode_{next_episode_index:04d}.npz"
                            pending_rollout.meta.update(
                                {
                                    "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "port": args.port,
                                    "variant": args.variant,
                                    "gripper_type": args.gripper_type,
                                    "start_stop_button": "space",
                                    "discard_button": "q",
                                    "manual_stop_button": "s",
                                    "viewer": True,
                                    "terminated_by": pending_reason or "unknown",
                                    "collection_mode": "assisted_teleop" if _is_assisted_mode(args) else "teleop_only",
                                    "assist_alpha": float(assist_alpha),
                                    "residual_limit": float(residual_limit),
                                    "checkpoint": checkpoint_meta or "",
                                    "copilot_enabled_at_save": bool(copilot_enabled),
                                    "heuristic_reach_gain": float(args.heuristic_reach_gain),
                                    "heuristic_place_gain": float(args.heuristic_place_gain),
                                    "heuristic_near_box_threshold": float(args.heuristic_near_box_threshold),
                                    "heuristic_lifted_height": float(args.heuristic_lifted_height),
                                    "heuristic_idle_deadband": float(args.heuristic_idle_deadband),
                                    "demo_align_gain": float(args.demo_align_gain),
                                    "demo_near_align_gain": float(args.demo_near_align_gain),
                                    "demo_near_align_floor": float(args.demo_near_align_floor),
                                    "demo_descend_gain": float(args.demo_descend_gain),
                                    "demo_close_gain": float(args.demo_close_gain),
                                    "demo_lift_gain": float(args.demo_lift_gain),
                                    "demo_place_gain": float(args.demo_place_gain),
                                    "demo_near_place_gain": float(args.demo_near_place_gain),
                                    "demo_near_place_floor": float(args.demo_near_place_floor),
                                    "demo_place_descend_gain": float(args.demo_place_descend_gain),
                                    "demo_grasp_xy_threshold": float(args.demo_grasp_xy_threshold),
                                    "demo_descend_xy_threshold": float(args.demo_descend_xy_threshold),
                                    "demo_descend_z_threshold": float(args.demo_descend_z_threshold),
                                    "demo_grasp_xyz_threshold": float(args.demo_grasp_xyz_threshold),
                                    "demo_close_xy_threshold": float(args.demo_close_xy_threshold),
                                    "demo_close_z_threshold": float(args.demo_close_z_threshold),
                                    "demo_place_xy_threshold": float(args.demo_place_xy_threshold),
                                    "demo_place_descend_xy_threshold": float(args.demo_place_descend_xy_threshold),
                                    "demo_lifted_height": float(args.demo_lifted_height),
                                    "demo_idle_deadband": float(args.demo_idle_deadband),
                                    "demo_snap_grasp_xy_threshold": float(args.demo_snap_grasp_xy_threshold),
                                    "demo_snap_place_xy_threshold": float(args.demo_snap_place_xy_threshold),
                                    "demo_snap_xy_scale": float(args.demo_snap_xy_scale),
                                    "demo_snap_rot_scale": float(args.demo_snap_rot_scale),
                                    "demo_snap_descend_gain": float(args.demo_snap_descend_gain),
                                    "demo_snap_close_gain": float(args.demo_snap_close_gain),
                                    "demo_snap_lift_gain": float(args.demo_snap_lift_gain),
                                }
                            )
                            save_episode_npz(path, pending_rollout)
                            session_saved_count += 1
                            next_episode_index += 1
                            print(
                                f"[{session_saved_count}/{args.num_episodes}] saved {path.name}"
                                f" steps={pending_rollout.steps} success={pending_rollout.success}"
                                f" return={pending_rollout.episode_return_total:.3f}"
                            )
                            if session_saved_count >= args.num_episodes:
                                info_message = "Target number of episodes reached"
                                mode = "REVIEW"
                                pending_rollout = None
                                viewer.sync()
                                break
                        else:
                            print("[WARN] Pending rollout is empty; skipping save")
                        load_preview_scene("Preview scene. Follower is live-aligned to leader. Press SPACE to start recording")

                if mode == "RECORDING":
                    obs_copy = _clone_obs(obs)
                    if args.assist_policy == "demo_snap":
                        target_joint_pos, target_gripper = env.get_joint_target()
                        raw_base_action = provider.get_action(obs_copy)
                        teleop_active = bool(provider.is_active())
                    else:
                        leader_target = provider.get_joint_target()
                        if leader_target is None:
                            target_joint_pos, target_gripper = env.get_joint_target()
                            teleop_active = False
                        else:
                            leader_joint_pos, leader_gripper, teleop_active = leader_target
                            target_joint_pos = leader_joint_pos
                            target_gripper = leader_gripper

                    if not _is_assisted_mode(args):
                        next_obs, reward_env, terminated, truncated, step_info = env.step_absolute_joint_target(
                            target_joint_pos,
                            target_gripper,
                        )
                        base_action = np.asarray(step_info.get("base_action", step_info["realized_action"]), dtype=np.float32).copy()
                        residual_action = np.zeros_like(base_action)
                        current_alpha = 0.0
                        residual_status = "teleop_only"
                    else:
                        if teleop_active:
                            if args.assist_policy == "demo_snap":
                                base_action = np.asarray(raw_base_action, dtype=np.float32).copy()
                                base_scale = np.asarray(residual_policy.get_base_scale(obs_copy, base_action), dtype=np.float32)
                                gated_base_action = np.clip(base_action * base_scale, -1.0, 1.0)
                                snap_status = getattr(residual_policy, "phase", "snap")
                            else:
                                base_action = env.action_from_joint_target(target_joint_pos, target_gripper)
                                gated_base_action = base_action
                                snap_status = "snap_disabled"
                            if copilot_enabled:
                                policy_started_at = time.perf_counter()
                                residual_action = np.clip(residual_policy.get_action(obs_copy, base_action), -1.0, 1.0).astype(np.float32)
                                policy_latency_ms = (time.perf_counter() - policy_started_at) * 1000.0
                                current_alpha = assist_alpha
                                residual_status = (
                                    "copilot_active"
                                    if args.assist_policy != "demo_snap"
                                    else getattr(residual_policy, "phase", snap_status)
                                )
                            else:
                                residual_action = np.zeros(env.action_dim, dtype=np.float32)
                                policy_latency_ms = 0.0
                                current_alpha = 0.0
                                residual_status = "copilot_off"
                        else:
                            base_action = np.zeros(env.action_dim, dtype=np.float32)
                            gated_base_action = base_action
                            residual_action = np.zeros(env.action_dim, dtype=np.float32)
                            policy_latency_ms = 0.0
                            current_alpha = 0.0
                            residual_status = "leader_idle_deadman"
                        if args.assist_policy == "demo_snap":
                            next_obs, reward_env, terminated, truncated, step_info = env.step(
                                gated_base_action,
                                residual_action,
                                alpha=current_alpha,
                            )
                            step_info["base_action"] = base_action.copy()
                            step_info["snap_base_action"] = gated_base_action.copy()
                        else:
                            next_obs, reward_env, terminated, truncated, step_info = env.step_base_joint_target(
                                target_joint_pos,
                                target_gripper,
                                residual_action,
                                alpha=current_alpha,
                                base_action=base_action,
                            )
                        base_action = np.asarray(step_info.get("base_action", base_action), dtype=np.float32).copy()
                    next_base_action = np.zeros_like(base_action)
                    realized_action = np.asarray(step_info["realized_action"], dtype=np.float32).copy()
                    base_action = np.asarray(step_info.get("base_action", base_action), dtype=np.float32).copy()
                    episode_time_s = max(0.0, time.perf_counter() - recording_started_at)
                    reward_total, extra_terms = reward_model.compute_total_reward(
                        reward_env=reward_env,
                        base_action=base_action,
                        residual_action=residual_action,
                        prev_residual_action=prev_residual,
                    )

                    transitions.append(
                        Transition(
                            observation_state=flatten_observation(obs_copy),
                            next_observation_state=flatten_observation(next_obs),
                            base_action=base_action,
                            next_base_action=next_base_action.copy(),
                            residual_action=residual_action,
                            prev_residual_action=prev_residual.copy(),
                            commanded_action=np.asarray(step_info.get("commanded_action", realized_action), dtype=np.float32).copy(),
                            realized_action=realized_action,
                            reward_env=float(reward_env),
                            reward_total=float(reward_total),
                            terminated=bool(terminated),
                            truncated=bool(truncated),
                            base_joint_target=np.asarray(
                                step_info.get("target_joint_pos", env.get_joint_target()[0]),
                                dtype=np.float32,
                            ).copy(),
                            base_gripper_target=np.asarray(
                                [step_info.get("target_gripper", env.get_joint_target()[1])],
                                dtype=np.float32,
                            ),
                            teleop_active=bool(teleop_active),
                            success=bool(step_info.get("success", False)),
                            alpha=float(step_info.get("alpha", assist_alpha)),
                            copilot_enabled=bool(_is_assisted_mode(args) and copilot_enabled),
                            conflict_score=float(extra_terms.get("conflict_score", 0.0)),
                            correction_score=float(extra_terms.get("correction_score", 0.0)),
                            residual_norm=float(extra_terms.get("residual_norm", np.linalg.norm(residual_action))),
                            episode_time_s=float(episode_time_s),
                            control_dt=float(env.cfg.control_dt),
                            policy_latency_ms=float(policy_latency_ms if _is_assisted_mode(args) else 0.0),
                        )
                    )
                    prev_residual = residual_action.copy()
                    last_residual_norm = float(np.linalg.norm(residual_action))
                    last_base_action = base_action.copy()
                    last_residual_action = residual_action.copy()
                    last_exec_delta = (realized_action - base_action).astype(np.float32)
                    last_alpha = float(step_info.get("alpha", assist_alpha))
                    last_teleop_active = bool(teleop_active)
                    last_residual_status = residual_status
                    reward_env_sum += float(reward_env)
                    reward_total_sum += float(reward_total)
                    obs = next_obs

                    if terminated or truncated:
                        pending_rollout = _build_rollout(transitions, obs_keys, reward_env_sum, reward_total_sum)
                        pending_reason = "success" if terminated else "max_steps"
                        mode = "REVIEW"
                        info_message = "Round finished. SPACE saves and loads next preview; Q discards"
                        print(f"[INFO] Round finished by {pending_reason}")

                current_steps = len(transitions) if mode == "RECORDING" else (pending_rollout.steps if pending_rollout else 0)
                current_return = reward_total_sum if mode == "RECORDING" else (pending_rollout.episode_return_total if pending_rollout else 0.0)
                current_box_pos = obs.get("box_pos") if isinstance(obs, dict) else None
                current_basket_pos = obs.get("basket_pos") if isinstance(obs, dict) else None
                _render_overlay(
                    viewer,
                    mode=mode,
                    policy_mode=policy_mode,
                    copilot_enabled=copilot_enabled,
                    teleop_active=last_teleop_active,
                    residual_status=last_residual_status,
                    current_alpha=last_alpha,
                    saved_count=session_saved_count,
                    target_count=args.num_episodes,
                    steps=current_steps,
                    total_return=current_return,
                    message=info_message,
                    residual_norm=last_residual_norm,
                    base_action=last_base_action,
                    residual_action=last_residual_action,
                    exec_delta=last_exec_delta,
                    box_pos=current_box_pos,
                    basket_pos=current_basket_pos,
                )
                viewer.sync()

                sleep_dt = 0.02 if mode != "RECORDING" else max(env.cfg.control_dt - (time.perf_counter() - step_start), 0.0)
                if sleep_dt > 0:
                    precise_sleep(sleep_dt)
    finally:
        provider.close()
        env.close()


if __name__ == "__main__":
    default_xml = PROJECT_ROOT / "assets" / "mujoco" / "Alicia_D_v5_6" / "gripper_50mm" / "alicia_d_follower.xml"
    parser = argparse.ArgumentParser(description="Record pure teleoperation episodes for MuJoCo pick-place residual RL.")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--xml", type=Path, default=default_xml)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--variant", type=str, default="leader")
    parser.add_argument("--gripper_type", type=str, default="50mm")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional residual policy checkpoint for assisted teleop collection")
    parser.add_argument(
        "--assist_policy",
        type=str,
        default="checkpoint",
        choices=("checkpoint", "heuristic", "demo_strong", "demo_snap"),
        help="Which assist policy to use during assisted teleop collection",
    )
    parser.add_argument("--alpha", type=float, default=None, help="Override assist alpha; defaults to checkpoint config")
    parser.add_argument("--residual_limit", type=float, default=None, help="Override residual action limit; defaults to checkpoint config")
    parser.add_argument("--heuristic_reach_gain", type=float, default=0.12, help="Heuristic assist gain before lifting")
    parser.add_argument("--heuristic_place_gain", type=float, default=0.14, help="Heuristic assist gain after lifting")
    parser.add_argument(
        "--heuristic_near_box_threshold",
        type=float,
        default=0.12,
        help="Distance threshold for tapering reach assistance near the box",
    )
    parser.add_argument(
        "--heuristic_lifted_height",
        type=float,
        default=0.10,
        help="Box height threshold that switches assist from reaching to placing",
    )
    parser.add_argument(
        "--heuristic_idle_deadband",
        type=float,
        default=0.02,
        help="Suppress heuristic assist when the operator is nearly idle",
    )
    parser.add_argument("--demo_align_gain", type=float, default=1.35, help="Strong demo XY alignment gain before lift")
    parser.add_argument("--demo_near_align_gain", type=float, default=2.40, help="Near-target XY alignment gain before grasp")
    parser.add_argument("--demo_near_align_floor", type=float, default=0.38, help="Minimum XY pull magnitude near grasp alignment")
    parser.add_argument("--demo_descend_gain", type=float, default=0.72, help="Strong demo descend gain near the grasp point")
    parser.add_argument("--demo_close_gain", type=float, default=1.20, help="Strong demo gripper-close bias at grasp or release")
    parser.add_argument("--demo_lift_gain", type=float, default=0.85, help="Strong demo upward bias right after a confident grasp")
    parser.add_argument("--demo_place_gain", type=float, default=1.35, help="Strong demo XY placement alignment gain after lift")
    parser.add_argument("--demo_near_place_gain", type=float, default=1.80, help="Near-target XY placement gain after lift")
    parser.add_argument("--demo_near_place_floor", type=float, default=0.30, help="Minimum XY pull magnitude near place alignment")
    parser.add_argument(
        "--demo_place_descend_gain",
        type=float,
        default=0.60,
        help="Strong demo descend gain when aligned over the basket",
    )
    parser.add_argument(
        "--demo_grasp_xy_threshold",
        type=float,
        default=0.09,
        help="XY distance threshold that enables stronger near-grasp XY alignment",
    )
    parser.add_argument(
        "--demo_descend_xy_threshold",
        type=float,
        default=0.020,
        help="XY distance threshold that enables descend help near the box",
    )
    parser.add_argument(
        "--demo_descend_z_threshold",
        type=float,
        default=0.055,
        help="Z error threshold that enables descend help near the box",
    )
    parser.add_argument(
        "--demo_grasp_xyz_threshold",
        type=float,
        default=0.040,
        help="XYZ distance threshold that enables strong gripper-close assistance",
    )
    parser.add_argument(
        "--demo_close_xy_threshold",
        type=float,
        default=0.020,
        help="XY distance threshold that enables strong gripper-close assistance",
    )
    parser.add_argument(
        "--demo_close_z_threshold",
        type=float,
        default=0.022,
        help="Z error threshold that enables strong gripper-close assistance",
    )
    parser.add_argument(
        "--demo_place_xy_threshold",
        type=float,
        default=0.12,
        help="XY distance threshold that enables stronger near-place XY alignment",
    )
    parser.add_argument(
        "--demo_place_descend_xy_threshold",
        type=float,
        default=0.055,
        help="XY distance threshold that enables placement descend help",
    )
    parser.add_argument(
        "--demo_lifted_height",
        type=float,
        default=0.09,
        help="Box height threshold that switches the strong demo helper into place mode",
    )
    parser.add_argument(
        "--demo_idle_deadband",
        type=float,
        default=0.02,
        help="Suppress strong demo assist when the operator is nearly idle",
    )
    parser.add_argument(
        "--demo_snap_grasp_xy_threshold",
        type=float,
        default=0.10,
        help="In demo_snap mode, suppress human XY when box XY error is below this threshold",
    )
    parser.add_argument(
        "--demo_snap_place_xy_threshold",
        type=float,
        default=0.14,
        help="In demo_snap mode, suppress human XY when basket XY error is below this threshold",
    )
    parser.add_argument(
        "--demo_snap_xy_scale",
        type=float,
        default=0.18,
        help="In demo_snap mode, scale factor applied to human XY near the target",
    )
    parser.add_argument(
        "--demo_snap_rot_scale",
        type=float,
        default=0.35,
        help="In demo_snap mode, scale factor applied to human rotation near the target",
    )
    parser.add_argument(
        "--demo_snap_descend_gain",
        type=float,
        default=2.0,
        help="In demo_snap mode, extra automatic descend gain after XY snap engages",
    )
    parser.add_argument(
        "--demo_snap_close_gain",
        type=float,
        default=0.9,
        help="In demo_snap mode, extra automatic close bias once grasp alignment is tight",
    )
    parser.add_argument(
        "--demo_snap_lift_gain",
        type=float,
        default=0.8,
        help="In demo_snap mode, extra upward lift bias once the grasp is secured",
    )
    parser.add_argument("--no_deadman", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Disable the MuJoCo viewer and record in the terminal only")
    main(parser.parse_args())
