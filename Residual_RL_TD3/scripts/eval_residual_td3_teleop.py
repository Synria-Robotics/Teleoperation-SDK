from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from teleop_sdk.envs import MujocoPickPlaceTeleopEnv
from teleop_sdk.rewards import PickPlaceReward
from train_residual_td3_teleop import (
    MLPActor,
    ReplayBuffer,
    TensorNormalizer,
    TELEOP_RESIDUAL_OBS_MODE,
    TorchResidualPolicy,
    TrainConfig,
    ZeroResidualPolicy,
    build_pick_place_task_config,
    evaluate_policy,
    infer_policy_obs_dim,
    load_dataset_into_buffer,
    normalize_fixed_box_xy,
    set_seed,
)


def _resolve_dataset_dirs(dataset_dir: Path | None, dataset_dirs: Sequence[Path] | None) -> tuple[Path, ...]:
    if dataset_dirs is not None:
        return tuple(path.resolve() for path in dataset_dirs)
    if dataset_dir is not None:
        return (dataset_dir.resolve(),)
    raise FileNotFoundError("Provide --dataset_dir or --dataset_dirs")


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model = PickPlaceReward()
    checkpoint = torch.load(args.checkpoint.resolve(), map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    checkpoint_fixed_scene = bool(cfg.get("fixed_scene", False))
    checkpoint_fixed_box_xy = normalize_fixed_box_xy(cfg.get("fixed_box_xy"))
    fixed_scene = checkpoint_fixed_scene if args.fixed_scene is None else bool(args.fixed_scene)
    fixed_box_xy = normalize_fixed_box_xy(args.fixed_box_xy) if args.fixed_box_xy is not None else checkpoint_fixed_box_xy
    env = MujocoPickPlaceTeleopEnv(
        build_pick_place_task_config(
            str(args.xml.resolve()),
            seed=args.seed,
            fixed_scene=fixed_scene,
            fixed_box_xy=fixed_box_xy,
        ),
        reward=reward_model,
    )
    obs, _ = env.reset(seed=args.seed)
    action_dim = env.action_dim

    offline_rb = ReplayBuffer(1)
    dataset_dirs = _resolve_dataset_dirs(args.dataset_dir, args.dataset_dirs)
    eval_cfg = TrainConfig(
        dataset_dirs=dataset_dirs,
        save_path=Path("/tmp/eval_unused.pt"),
        xml_path=str(args.xml.resolve()),
        history_len=int(cfg.get("history_len", 4)),
        translation_step=float(cfg.get("translation_step", env.cfg.translation_step)),
        rotation_step=float(cfg.get("rotation_step", env.cfg.rotation_step)),
        gripper_step=float(cfg.get("gripper_step", env.cfg.gripper_step)),
    )
    obs_mode = str(cfg.get("obs_mode", TELEOP_RESIDUAL_OBS_MODE))
    obs_dim = infer_policy_obs_dim(
        obs,
        action_dim=action_dim,
        obs_mode=obs_mode,
        history_len=eval_cfg.history_len,
        translation_step=eval_cfg.translation_step,
        rotation_step=eval_cfg.rotation_step,
        gripper_step=eval_cfg.gripper_step,
    )
    eval_cfg.obs_mode = obs_mode
    playback_episodes = load_dataset_into_buffer(dataset_dirs, offline_rb, eval_cfg)

    history_len = int(cfg.get("history_len", 4))
    hidden_dim = int(cfg.get("hidden_dim", 256))
    residual_limit = float(cfg.get("residual_limit", args.residual_limit))

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

    baseline = evaluate_policy(
        env,
        playback_episodes,
        ZeroResidualPolicy(),
        eval_episodes=args.eval_episodes,
    )
    assisted = evaluate_policy(
        env,
        playback_episodes,
        TorchResidualPolicy(
            actor,
            device,
            residual_limit=residual_limit,
            state_normalizer=state_normalizer,
            history_len=history_len,
            translation_step=eval_cfg.translation_step,
            rotation_step=eval_cfg.rotation_step,
            gripper_step=eval_cfg.gripper_step,
            obs_mode=obs_mode,
        ),
        eval_episodes=args.eval_episodes,
    )
    print("Baseline metrics:", baseline)
    print("Assisted metrics:", assisted)


if __name__ == "__main__":
    default_xml = PROJECT_ROOT / "assets" / "mujoco" / "Alicia_D_v5_6" / "gripper_50mm" / "alicia_d_follower.xml"
    parser = argparse.ArgumentParser(description="Evaluate a trained teleoperation residual TD3 policy.")
    parser.add_argument("--dataset_dir", type=Path, default=None)
    parser.add_argument("--dataset_dirs", type=Path, nargs="+", default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=default_xml)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fixed_scene",
        action="store_true",
        default=None,
        help="Use a fixed rollout/eval scene. If omitted, inherit from checkpoint config when available.",
    )
    parser.add_argument(
        "--fixed_box_xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Optional fixed box XY for rollout/eval. If omitted, inherit from checkpoint config when available.",
    )
    parser.add_argument("--eval_episodes", type=int, default=0, help="How many playback episodes to evaluate; <=0 means all")
    parser.add_argument("--residual_limit", type=float, default=0.1)
    main(parser.parse_args())
