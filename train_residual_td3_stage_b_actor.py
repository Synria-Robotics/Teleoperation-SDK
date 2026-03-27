from __future__ import annotations

import argparse
from pathlib import Path

from train_residual_td3_teleop import main as run_td3


PROJECT_ROOT = Path(__file__).resolve().parent


def build_args(parsed: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_dir=None,
        dataset_dirs=[parsed.raw_dataset_dir.resolve(), parsed.assist_dataset_dir.resolve()],
        xml=parsed.xml.resolve(),
        save_path=parsed.save_path.resolve(),
        init_checkpoint=parsed.init_checkpoint.resolve() if parsed.init_checkpoint is not None else None,
        seed=parsed.seed,
        alpha=parsed.alpha,
        reward_mode="success_only",
        batch_size=parsed.batch_size,
        offline_fraction=1.0,
        replay_mix_mode="offline_only",
        offline_updates=parsed.offline_updates,
        critic_warmup_steps=parsed.critic_warmup_steps,
        baseline_episodes=0,
        assisted_episodes=0,
        updates_per_episode=0,
        eval_episodes=parsed.eval_episodes,
        actor_lr=parsed.actor_lr,
        critic_lr=parsed.critic_lr,
        critic_loss=parsed.critic_loss,
        residual_limit=parsed.residual_limit,
        exploration_noise=0.0,
        zero_reg=parsed.zero_reg,
        conflict_reg=parsed.conflict_reg,
        teleop_zero_reg=parsed.teleop_zero_reg,
        assisted_bc_reg=parsed.assisted_bc_reg,
        idle_zero_reg=parsed.idle_zero_reg,
        idle_deadband=parsed.idle_deadband,
        idle_scale_span=parsed.idle_scale_span,
        actor_bc_warmup_steps=parsed.actor_bc_warmup_steps,
        disable_state_norm=not parsed.state_norm,
        disable_reward_norm=not parsed.reward_norm,
        filter_assist_by_success=parsed.filter_assist_by_success,
        filter_assist_min_cmd_delta=parsed.filter_assist_min_cmd_delta,
        filter_assist_alpha=parsed.filter_assist_alpha,
        filter_assist_reach_gain=parsed.filter_assist_reach_gain,
        filter_assist_place_gain=parsed.filter_assist_place_gain,
        log_interval=parsed.log_interval,
    )


def parse_args() -> argparse.Namespace:
    default_xml = PROJECT_ROOT / "assets" / "mujoco" / "Alicia_D_v5_6" / "gripper_50mm" / "alicia_d_follower.xml"
    default_raw = PROJECT_ROOT / "logs" / "teleop_dataset"
    default_assist = PROJECT_ROOT / "logs" / "assisted_teleop_dataset"
    default_init = PROJECT_ROOT / "logs" / "residual_td3_critic_sparse.pt"
    default_save = PROJECT_ROOT / "logs" / "residual_td3_stage_b_actor.pt"
    parser = argparse.ArgumentParser(description="Stage B: offline actor training from raw + filtered assist data.")
    parser.add_argument("--raw_dataset_dir", type=Path, default=default_raw)
    parser.add_argument("--assist_dataset_dir", type=Path, default=default_assist)
    parser.add_argument("--xml", type=Path, default=default_xml)
    parser.add_argument("--save_path", type=Path, default=default_save)
    parser.add_argument("--init_checkpoint", type=Path, default=default_init)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--offline_updates", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--critic_warmup_steps", type=int, default=0)
    parser.add_argument("--actor_bc_warmup_steps", type=int, default=1500)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--critic_loss", type=str, default="huber", choices=("huber", "mse"))
    parser.add_argument("--residual_limit", type=float, default=0.25)
    parser.add_argument("--zero_reg", type=float, default=0.02)
    parser.add_argument("--conflict_reg", type=float, default=0.05)
    parser.add_argument("--teleop_zero_reg", type=float, default=0.50)
    parser.add_argument("--assisted_bc_reg", type=float, default=0.25)
    parser.add_argument("--idle_zero_reg", type=float, default=0.50)
    parser.add_argument("--idle_deadband", type=float, default=0.05)
    parser.add_argument("--idle_scale_span", type=float, default=0.20)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--state_norm", action="store_true", default=True)
    parser.add_argument("--reward_norm", action="store_true", default=True)
    parser.add_argument("--no_state_norm", dest="state_norm", action="store_false")
    parser.add_argument("--no_reward_norm", dest="reward_norm", action="store_false")
    parser.add_argument("--filter_assist_by_success", action="store_true", default=True)
    parser.add_argument("--keep_failed_assist", dest="filter_assist_by_success", action="store_false")
    parser.add_argument("--filter_assist_min_cmd_delta", type=float, default=0.001)
    parser.add_argument("--filter_assist_alpha", type=float, default=0.25)
    parser.add_argument("--filter_assist_reach_gain", type=float, default=0.12)
    parser.add_argument("--filter_assist_place_gain", type=float, default=0.14)
    parser.add_argument("--log_interval", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    run_td3(build_args(parse_args()))
