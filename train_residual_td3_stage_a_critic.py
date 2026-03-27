from __future__ import annotations

import argparse
from pathlib import Path

from train_residual_td3_teleop import main as run_td3


PROJECT_ROOT = Path(__file__).resolve().parent


def build_args(parsed: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_dir=None,
        dataset_dirs=[parsed.raw_dataset_dir.resolve()],
        xml=parsed.xml.resolve(),
        save_path=parsed.save_path.resolve(),
        init_checkpoint=None,
        seed=parsed.seed,
        alpha=parsed.alpha,
        reward_mode="success_only",
        batch_size=parsed.batch_size,
        offline_fraction=1.0,
        replay_mix_mode="offline_only",
        offline_updates=parsed.offline_updates,
        critic_warmup_steps=10_000_000,
        baseline_episodes=0,
        assisted_episodes=0,
        updates_per_episode=0,
        eval_episodes=parsed.eval_episodes,
        actor_lr=parsed.actor_lr,
        critic_lr=parsed.critic_lr,
        critic_loss=parsed.critic_loss,
        residual_limit=parsed.residual_limit,
        exploration_noise=0.0,
        zero_reg=0.0,
        conflict_reg=0.0,
        teleop_zero_reg=0.0,
        assisted_bc_reg=0.0,
        idle_zero_reg=0.0,
        idle_deadband=parsed.idle_deadband,
        idle_scale_span=parsed.idle_scale_span,
        actor_bc_warmup_steps=10_000_000,
        disable_state_norm=not parsed.state_norm,
        disable_reward_norm=not parsed.reward_norm,
        filter_assist_by_success=False,
        filter_assist_min_cmd_delta=0.0,
        filter_assist_alpha=None,
        filter_assist_reach_gain=None,
        filter_assist_place_gain=None,
        log_interval=parsed.log_interval,
    )


def parse_args() -> argparse.Namespace:
    default_xml = PROJECT_ROOT / "assets" / "mujoco" / "Alicia_D_v5_6" / "gripper_50mm" / "alicia_d_follower.xml"
    default_raw = PROJECT_ROOT / "logs" / "teleop_dataset"
    default_save = PROJECT_ROOT / "logs" / "residual_td3_stage_a_critic.pt"
    parser = argparse.ArgumentParser(description="Stage A: offline critic pretrain from raw teleop only.")
    parser.add_argument("--raw_dataset_dir", type=Path, default=default_raw)
    parser.add_argument("--xml", type=Path, default=default_xml)
    parser.add_argument("--save_path", type=Path, default=default_save)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--offline_updates", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_loss", type=str, default="huber", choices=("huber", "mse"))
    parser.add_argument("--residual_limit", type=float, default=0.25)
    parser.add_argument("--idle_deadband", type=float, default=0.05)
    parser.add_argument("--idle_scale_span", type=float, default=0.20)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--state_norm", action="store_true", default=True)
    parser.add_argument("--reward_norm", action="store_true", default=True)
    parser.add_argument("--no_state_norm", dest="state_norm", action="store_false")
    parser.add_argument("--no_reward_norm", dest="reward_norm", action="store_false")
    parser.add_argument("--log_interval", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    run_td3(build_args(parse_args()))
