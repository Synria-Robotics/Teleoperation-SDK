from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from teleop_sdk.envs import MujocoPickPlaceTeleopEnv, PickPlaceTaskConfig, flatten_observation
from teleop_sdk.rewards import PickPlaceReward
from train_residual_td3_teleop import (
    MLPActor,
    MLPCritic,
    MetricTracker,
    ReplayBuffer,
    ScalarNormalizer,
    TensorNormalizer,
    TorchResidualPolicy,
    TrainConfig,
    ZeroResidualPolicy,
    apply_idle_gate_tensor,
    clip_commanded_action_tensor,
    clip_residual_tensor,
    compute_actor_loss,
    compute_critic_loss,
    compute_reward_stats,
    compute_state_stats,
    evaluate_policy,
    format_metrics,
    load_dataset_into_buffer,
    maybe_load_init_checkpoint,
    prepare_batch,
    sample_mixed_batch,
    set_seed,
    soft_update,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def _make_load_cfg(
    *,
    dataset_dirs: Sequence[Path],
    xml_path: str,
    save_path: Path,
    filter_assist_by_success: bool,
    filter_assist_min_cmd_delta: float,
    filter_assist_alpha: float | None,
    filter_assist_reach_gain: float | None,
    filter_assist_place_gain: float | None,
) -> TrainConfig:
    return TrainConfig(
        dataset_dirs=tuple(dataset_dirs),
        save_path=save_path,
        xml_path=xml_path,
        filter_assist_by_success=filter_assist_by_success,
        filter_assist_min_cmd_delta=filter_assist_min_cmd_delta,
        filter_assist_alpha=filter_assist_alpha,
        filter_assist_reach_gain=filter_assist_reach_gain,
        filter_assist_place_gain=filter_assist_place_gain,
    )


def _resolve_dirs(paths: Sequence[Path]) -> tuple[Path, ...]:
    return tuple(path.resolve() for path in paths)


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    offline_dirs = _resolve_dirs(args.offline_dataset_dirs)
    online_dirs = _resolve_dirs(args.online_dataset_dirs)
    xml_path = str(args.xml.resolve())
    save_path = args.save_path.resolve()
    init_checkpoint = args.init_checkpoint.resolve() if args.init_checkpoint is not None else None

    reward_model = PickPlaceReward()
    env = MujocoPickPlaceTeleopEnv(PickPlaceTaskConfig(xml_path=xml_path, seed=args.seed), reward=reward_model)
    obs, _ = env.reset(seed=args.seed)
    obs_dim = int(flatten_observation(obs).shape[0])
    action_dim = env.action_dim

    offline_rb = ReplayBuffer(args.buffer_capacity)
    online_rb = ReplayBuffer(args.buffer_capacity)

    offline_cfg = _make_load_cfg(
        dataset_dirs=offline_dirs,
        xml_path=xml_path,
        save_path=save_path,
        filter_assist_by_success=args.offline_filter_assist_by_success,
        filter_assist_min_cmd_delta=args.offline_filter_assist_min_cmd_delta,
        filter_assist_alpha=args.offline_filter_assist_alpha,
        filter_assist_reach_gain=args.offline_filter_assist_reach_gain,
        filter_assist_place_gain=args.offline_filter_assist_place_gain,
    )
    online_cfg = _make_load_cfg(
        dataset_dirs=online_dirs,
        xml_path=xml_path,
        save_path=save_path,
        filter_assist_by_success=args.online_filter_assist_by_success,
        filter_assist_min_cmd_delta=args.online_filter_assist_min_cmd_delta,
        filter_assist_alpha=args.online_filter_assist_alpha,
        filter_assist_reach_gain=args.online_filter_assist_reach_gain,
        filter_assist_place_gain=args.online_filter_assist_place_gain,
    )

    offline_episodes = load_dataset_into_buffer(offline_dirs, offline_rb, offline_cfg)
    online_episodes = load_dataset_into_buffer(online_dirs, online_rb, online_cfg)
    all_episodes = [*offline_episodes, *online_episodes]
    state_stats = compute_state_stats(all_episodes)
    reward_mean, reward_std = compute_reward_stats(all_episodes, "success_only")
    state_normalizer = TensorNormalizer(state_stats.mean, state_stats.std, device) if not args.disable_state_norm else None
    reward_normalizer = ScalarNormalizer(reward_mean, reward_std) if not args.disable_reward_norm else None

    print(
        format_metrics(
            "Dataset stats",
            {
                "offline_episodes": float(len(offline_episodes)),
                "online_episodes": float(len(online_episodes)),
                "offline_transitions": float(len(offline_rb)),
                "online_transitions": float(len(online_rb)),
                "state_std_mean": float(np.mean(state_stats.std)),
                "reward_mean": reward_mean,
                "reward_std": reward_std,
            },
        )
    )
    print(
        f"Replay mixing mode={args.replay_mix_mode} offline_fraction={args.offline_fraction:.2f} "
        f"state_norm={not args.disable_state_norm} reward_norm={not args.disable_reward_norm} reward_mode=success_only"
    )

    actor = MLPActor(obs_dim, action_dim, hidden_dim=args.hidden_dim, residual_limit=args.residual_limit).to(device)
    critic = MLPCritic(obs_dim, action_dim, hidden_dim=args.hidden_dim).to(device)
    actor_target = MLPActor(obs_dim, action_dim, hidden_dim=args.hidden_dim, residual_limit=args.residual_limit).to(device)
    critic_target = MLPCritic(obs_dim, action_dim, hidden_dim=args.hidden_dim).to(device)
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())
    maybe_load_init_checkpoint(
        init_checkpoint,
        actor=actor,
        critic=critic,
        actor_target=actor_target,
        critic_target=critic_target,
    )

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    cfg = TrainConfig(
        dataset_dirs=offline_dirs,
        save_path=save_path,
        xml_path=xml_path,
        init_checkpoint=init_checkpoint,
        seed=args.seed,
        alpha=args.alpha,
        reward_mode="success_only",
        batch_size=args.batch_size,
        offline_fraction=args.offline_fraction,
        replay_mix_mode=args.replay_mix_mode,
        offline_updates=0,
        updates_per_episode=0,
        critic_warmup_steps=args.critic_warmup_steps,
        baseline_episodes=0,
        assisted_episodes=0,
        eval_episodes=args.eval_episodes,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        critic_loss=args.critic_loss,
        buffer_capacity=args.buffer_capacity,
        residual_limit=args.residual_limit,
        exploration_noise=args.exploration_noise,
        policy_delay=args.policy_delay,
        hidden_dim=args.hidden_dim,
        zero_reg=args.zero_reg,
        conflict_reg=args.conflict_reg,
        teleop_zero_reg=args.teleop_zero_reg,
        assisted_bc_reg=args.assisted_bc_reg,
        idle_zero_reg=args.idle_zero_reg,
        idle_deadband=args.idle_deadband,
        idle_scale_span=args.idle_scale_span,
        actor_bc_warmup_steps=args.actor_bc_warmup_steps,
        normalize_state=not args.disable_state_norm,
        normalize_reward=not args.disable_reward_norm,
        log_interval=args.log_interval,
    )

    tracker = MetricTracker()
    train_steps = 0
    for _ in range(args.total_updates):
        batch = sample_mixed_batch(
            offline_rb,
            online_rb,
            args.batch_size,
            args.offline_fraction,
            args.replay_mix_mode,
        )
        prepared = prepare_batch(
            batch,
            device=device,
            state_normalizer=state_normalizer,
            reward_normalizer=reward_normalizer,
            reward_mode="success_only",
        )
        obs_flat = prepared["obs_flat"]
        next_obs_flat = prepared["next_obs_flat"]
        base_action = prepared["base_action"]
        next_base_action = prepared["next_base_action"]
        commanded_action = prepared["commanded_action"]
        reward_total = prepared["reward_total"]
        done = prepared["done"]

        with torch.no_grad():
            next_residual = clip_residual_tensor(actor_target(next_obs_flat, next_base_action), cfg.residual_limit)
            next_residual = apply_idle_gate_tensor(next_residual, next_base_action, cfg.idle_deadband, cfg.idle_scale_span)
            next_commanded_action = clip_commanded_action_tensor(next_base_action + cfg.alpha * next_residual)
            target_q1, target_q2 = critic_target(next_obs_flat, next_commanded_action)
            target_q = reward_total + cfg.gamma * (1.0 - done) * torch.minimum(target_q1, target_q2)

        q1, q2 = critic(obs_flat, commanded_action)
        critic_loss = compute_critic_loss(q1, q2, target_q, loss_type=cfg.critic_loss)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        if train_steps >= cfg.critic_warmup_steps and train_steps % cfg.policy_delay == 0:
            use_q_loss = train_steps >= (cfg.critic_warmup_steps + cfg.actor_bc_warmup_steps)
            actor_loss = compute_actor_loss(
                actor,
                critic,
                obs_flat,
                base_action,
                prepared["logged_residual"],
                prepared["is_assisted"],
                cfg,
                use_q_loss=use_q_loss,
            )
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            soft_update(actor_target, actor, cfg.tau)
            soft_update(critic_target, critic, cfg.tau)
            actor_loss_value = float(actor_loss.item())
        else:
            soft_update(critic_target, critic, cfg.tau)
            actor_loss_value = float("nan")

        td1 = q1 - target_q
        td2 = q2 - target_q
        if args.replay_mix_mode == "offline_only":
            actual_online_fraction = 0.0
        elif args.replay_mix_mode == "online_only":
            actual_online_fraction = 1.0
        elif args.replay_mix_mode == "available_balanced":
            actual_online_fraction = 0.0 if len(online_rb) == 0 else float(args.batch_size - (args.batch_size // 2)) / float(args.batch_size)
        else:
            actual_online_fraction = float(min(len(online_rb), int(round(args.batch_size * (1.0 - args.offline_fraction))))) / float(args.batch_size)
        tracker.update(
            critic_loss=float(critic_loss.item()),
            q1=float(q1.mean().item()),
            q2=float(q2.mean().item()),
            q_gap=float((q1 - q2).abs().mean().item()),
            target_q=float(target_q.mean().item()),
            target_std=float(target_q.std().item()),
            td_abs_mean=float(0.5 * (td1.abs().mean().item() + td2.abs().mean().item())),
            td_abs_max=float(torch.maximum(td1.abs().max(), td2.abs().max()).item()),
            reward=float(reward_total.mean().item()),
            assisted_fraction=float(prepared["is_assisted"].mean().item()),
            online_fraction=max(0.0, actual_online_fraction),
        )
        if not np.isnan(actor_loss_value):
            tracker.update(actor_loss=actor_loss_value)

        train_steps += 1
        if train_steps % args.log_interval == 0:
            print(format_metrics(f"[train step {train_steps}]", tracker.summary()))
            tracker.reset()

    eval_metrics = evaluate_policy(
        env,
        all_episodes,
        TorchResidualPolicy(
            actor,
            device,
            residual_limit=args.residual_limit,
            idle_deadband=args.idle_deadband,
            idle_scale_span=args.idle_scale_span,
            state_normalizer=state_normalizer,
        ),
        reward_model,
        alpha=args.alpha,
        eval_episodes=args.eval_episodes,
    )
    baseline_metrics = evaluate_policy(
        env,
        all_episodes,
        ZeroResidualPolicy(),
        reward_model,
        alpha=args.alpha,
        eval_episodes=args.eval_episodes,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "config": {
            **{k: v for k, v in vars(args).items() if not isinstance(v, Path)},
            "offline_dataset_dirs": [str(path) for path in offline_dirs],
            "online_dataset_dirs": [str(path) for path in online_dirs],
            "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
        },
        "eval_metrics": eval_metrics,
        "baseline_metrics": baseline_metrics,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "state_mean": state_stats.mean,
        "state_std": state_stats.std,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
    }
    torch.save(payload, save_path)
    print("Saved checkpoint to", save_path)
    print("Baseline metrics:", baseline_metrics)
    print("Assisted metrics:", eval_metrics)


if __name__ == "__main__":
    default_xml = PROJECT_ROOT / "assets" / "mujoco" / "Alicia_D_v5_6" / "gripper_50mm" / "alicia_d_follower.xml"
    default_offline = [
        PROJECT_ROOT / "logs" / "teleop_dataset",
        PROJECT_ROOT / "logs" / "assisted_teleop_dataset",
    ]
    default_online = [PROJECT_ROOT / "logs" / "assisted_teleop_dataset_v2_real"]
    default_init = PROJECT_ROOT / "logs" / "residual_td3_stage_b_actor_v2.pt"
    default_save = PROJECT_ROOT / "logs" / "residual_td3_stage_c_real_online.pt"

    parser = argparse.ArgumentParser(description="Stage C real online RL: mixed finetuning from offline data plus real collected online data.")
    parser.add_argument("--offline_dataset_dirs", type=Path, nargs="+", default=default_offline)
    parser.add_argument("--online_dataset_dirs", type=Path, nargs="+", default=default_online)
    parser.add_argument("--xml", type=Path, default=default_xml)
    parser.add_argument("--save_path", type=Path, default=default_save)
    parser.add_argument("--init_checkpoint", type=Path, default=default_init)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--total_updates", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--offline_fraction", type=float, default=0.75)
    parser.add_argument(
        "--replay_mix_mode",
        type=str,
        default="fixed_offline_fraction",
        choices=("fixed_offline_fraction", "available_balanced", "offline_only", "online_only"),
    )
    parser.add_argument("--critic_warmup_steps", type=int, default=0)
    parser.add_argument("--actor_bc_warmup_steps", type=int, default=300)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    parser.add_argument("--critic_loss", type=str, default="huber", choices=("huber", "mse"))
    parser.add_argument("--residual_limit", type=float, default=0.12)
    parser.add_argument("--exploration_noise", type=float, default=0.0)
    parser.add_argument("--zero_reg", type=float, default=0.02)
    parser.add_argument("--conflict_reg", type=float, default=0.05)
    parser.add_argument("--teleop_zero_reg", type=float, default=0.5)
    parser.add_argument("--assisted_bc_reg", type=float, default=0.05)
    parser.add_argument("--idle_zero_reg", type=float, default=0.5)
    parser.add_argument("--idle_deadband", type=float, default=0.05)
    parser.add_argument("--idle_scale_span", type=float, default=0.20)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--buffer_capacity", type=int, default=200_000)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--disable_state_norm", action="store_true")
    parser.add_argument("--disable_reward_norm", action="store_true")
    parser.add_argument("--offline_filter_assist_by_success", action="store_true", default=False)
    parser.add_argument("--offline_filter_assist_min_cmd_delta", type=float, default=0.0)
    parser.add_argument("--offline_filter_assist_alpha", type=float, default=None)
    parser.add_argument("--offline_filter_assist_reach_gain", type=float, default=None)
    parser.add_argument("--offline_filter_assist_place_gain", type=float, default=None)
    parser.add_argument("--online_filter_assist_by_success", action="store_true", default=False)
    parser.add_argument("--online_filter_assist_min_cmd_delta", type=float, default=0.0)
    parser.add_argument("--online_filter_assist_alpha", type=float, default=None)
    parser.add_argument("--online_filter_assist_reach_gain", type=float, default=None)
    parser.add_argument("--online_filter_assist_place_gain", type=float, default=None)
    parser.add_argument("--policy_delay", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=200)
    main(parser.parse_args())
