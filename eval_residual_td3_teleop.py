from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from teleop_sdk.envs import MujocoPickPlaceTeleopEnv, PickPlaceTaskConfig, flatten_observation
from teleop_sdk.rewards import PickPlaceReward
from train_residual_td3_teleop import (
    MLPActor,
    ReplayBuffer,
    TensorNormalizer,
    TorchResidualPolicy,
    ZeroResidualPolicy,
    evaluate_policy,
    load_dataset_into_buffer,
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
    env = MujocoPickPlaceTeleopEnv(PickPlaceTaskConfig(xml_path=str(args.xml.resolve()), seed=args.seed), reward=reward_model)
    obs, _ = env.reset(seed=args.seed)
    obs_dim = int(flatten_observation(obs).shape[0])
    action_dim = env.action_dim

    offline_rb = ReplayBuffer(1)
    dataset_dirs = _resolve_dataset_dirs(args.dataset_dir, args.dataset_dirs)
    playback_episodes = load_dataset_into_buffer(dataset_dirs, offline_rb)

    checkpoint = torch.load(args.checkpoint.resolve(), map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    state_normalizer = None
    if "state_mean" in checkpoint and "state_std" in checkpoint and bool(cfg.get("normalize_state", True)):
        state_normalizer = TensorNormalizer(checkpoint["state_mean"], checkpoint["state_std"], device)
    actor = MLPActor(obs_dim, action_dim, residual_limit=args.residual_limit).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    baseline = evaluate_policy(
        env,
        playback_episodes,
        ZeroResidualPolicy(),
        reward_model,
        alpha=args.alpha,
        eval_episodes=args.eval_episodes,
    )
    assisted = evaluate_policy(
        env,
        playback_episodes,
        TorchResidualPolicy(
            actor,
            device,
            residual_limit=args.residual_limit,
            idle_deadband=float(cfg.get("idle_deadband", 0.05)),
            idle_scale_span=float(cfg.get("idle_scale_span", 0.20)),
            state_normalizer=state_normalizer,
        ),
        reward_model,
        alpha=args.alpha,
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
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--eval_episodes", type=int, default=0, help="How many playback episodes to evaluate; <=0 means all")
    parser.add_argument("--residual_limit", type=float, default=0.35)
    main(parser.parse_args())
