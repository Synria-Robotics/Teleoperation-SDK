# Teleop Residual RL 设计文档

## 1. 文档目标

本文档为 `Teleoperation-SDK` 提供一套共享遥操作残差强化学习（Residual RL）方案。核心目标是在保证主从跟随稳定性的前提下，让 RL 策略作为“副驾驶”（Copilot）提供智能微调，

## 2. 输入空间与输出空间

## 2.1 输入空间

当前这版 policy observation 按下面这个口径定义：

```text
x_t = [
  p_t^F(3),
  q_t^F(4),
  g_t^F(1),
  Δp_t^H(3),
  Δrot_t^H(3),
  Δg_t^H(1),
  r_{t-K}(7),
  ...,
  r_{t-1}(7),
]
```

总维度：

```text
3 + 4 + 1 + 3 + 3 + 1 + 7K = 15 + 7K
```

例如默认 `K = 4` 时：

```text
15 + 7 * 4 = 43
```

这里：

- `p_t^F, q_t^F, g_t^F`
  表示从臂当前末端位姿与夹爪状态，也就是 `pos3 + quat4 + gripper1`。

- `Δp_t^H, Δrot_t^H, Δg_t^H`
  表示主臂当前 task-space 增量意图，也就是 `pos3 + rot3 + gripper1`。这里旋转不再用四元数绝对姿态，而是直接用 3 维旋转增量表示。

- `r_{t-K:t-1}`
  表示过去 `K` 帧残差动作历史，每一帧都是 `7` 维 `pos3 + rot3 + gripper1`。

这意味着当前 Actor / Critic 主要关注的是：

1. follower 当前末端状态。
2. 操作者当前给出的主臂增量意图。
3. 策略自己最近 `K` 步做过哪些残差修正。

这个定义更贴合 residual teleop 的语义：策略只看从臂当前状态、人的瞬时增量命令，以及自己最近的修正历史，不再额外依赖主臂绝对位姿或主从误差项的重复展开。

---


## 2.2 动作空间

当前 Actor 的输出空间就是 7 维归一化 task-space 增量动作：

```text
r_t = [dx, dy, dz, d_rx, d_ry, d_rz, d_g]
```

也就是：

```text
pos3 + rot3 + gripper1
```

其中：

- 前 3 维是平移增量。
- 中间 3 维是旋转增量，使用 rotvec / axis-angle 的 3 维表示。
- 最后 1 维是夹爪增量。


## 3. Reward 设计

## 3.1 当前reward

默认 reward 为：

```text
r_t^{env} =
  0,     if success
  -100,  if failed truncation
  -1,    otherwise
```

对应默认参数：

```text
step_penalty = -1.0
success_reward = 0.0
failure_penalty = -100.0
```

因此最大化累计回报等价于：

```text
尽快成功，并尽量避免失败
```

这是一个“最小时间步数”的任务目标。

---

## 4. Actor-Critic 设计

这一版建议采用：

```text
Residual Actor + Baseline Differential Critic
```

核心思想很简单：

- Actor 只负责输出 residual。
- Critic 不只看“加了 residual 之后好不好”，还要看“相比不帮忙，到底多好了多少”。

---

## 4.1 Actor

Actor 的输入是当前 observation `x_t`，输出是 7 维 residual action：

```text
r_t = actor(x_t)
```

实际执行动作是：

```text
a_t^{assist} = clip(b_t + r_t)
```

这里：

- `b_t` 是当前 baseline action，也就是主臂给出的增量命令。
- `r_t` 是策略给出的修正量。

Actor 不重新生成完整动作，而是在 `b_t` 上做小修正。

---

## 4.2 Critic

这里不建议只学一个普通 critic：

```text
Q(x_t, a_t^{assist})
```

因为它会把“人本来就能完成任务”的价值也一起算进去，导致 residual 很难知道自己到底有没有额外帮上忙。

更适合 teleop 的做法是学两个 critic：

```text
Q_ast(x_t, a_t^{assist})
Q_bas(x_t, b_t)
```

其中：

- `Q_ast` 评估“加了 residual 之后”的未来回报。
- `Q_bas` 评估“如果这一步不加 residual，只执行 baseline”的未来回报。

---

## 4.3 Actor 目标

Actor 不直接追求最大化 `Q_ast`，而是追求：

```text
A_help = Q_ast - Q_bas
```

它的含义就是：

```text
这一时刻加 residual，比不加 residual 多创造了多少价值
```

所以 residual policy 追求的不是“总成功率本身”，而是：

```text
比只靠 baseline 更好多少
```

这很适合 teleop 场景，因为人的输入始终存在，而 residual 的职责只是提供额外帮助。


## 5. 训练流程

训练整体分 4 步：

1. 构建 offline replay。
2. 把 online replay 填到 `learning_starts`。
3. 做 `critic_warmup_steps` 次 critic-only update。
4. 进入持续的在线 rollout + mixed replay update。

---

## 5.1 阶段 A：Offline Replay 构建

先把 teleop 日志整理成离线 transition：

```text
(x_t, b_t, a_t, reward_t, x_{t+1}, done)
```

其中：

- `x_t` 是当前 observation。
- `b_t` 是当前 baseline action。
- `a_t` 优先使用实际执行动作 `a_t^{exec}`；如果拿不到，再退化为 `a_t^{cmd}`。

这一阶段的作用是提供离线先验，并统计归一化参数。

---

## 5.2 阶段 B：Online Replay Warm-up

接下来先把 online replay 填到 `learning_starts`：

```text
while len(online_replay) < learning_starts:
    b_t = current base action
    r_t = exploration noise
    a_t = clip(b_t + r_t)
    execute a_t
    push transition to online replay
```

这一步的作用是让 online replay 尽快覆盖当前闭环动力学分布。

---

## 5.3 阶段 C：Critic Warmup

当 online replay 足够大之后，先做 critic-only warmup：

```text
for step in range(critic_warmup_steps):
    B_on  ~ online replay
    B_off ~ offline replay
    B = concat(B_on, B_off)
    update Q_ast
    update Q_bas
    freeze actor
```

batch 按 `offline_fraction` 混合：

```text
online_batch_size  = batch_size * (1 - offline_fraction)
offline_batch_size = batch_size * offline_fraction
```

目的就是先把 critic 训稳，再让 Actor 开始追 `A_help`。

---

## 5.4 阶段 D：持续 Rollout + Mixed Replay Update

之后进入持续循环：

```text
while global_step < total_timesteps:
    rollout current residual policy
    push transitions to online replay

    if update time:
        for i in range(num_updates_per_iteration):
            sample mixed batch from online + offline replay
            update Q_ast
            update Q_bas
            if actor cadence reached:
                update actor with A_help = Q_ast - Q_bas
                soft-update targets
```

对 teleop 来说，一步 rollout 可以简单写成：

```text
b_t = current base action
r_t = actor(x_t)
a_t = clip(b_t + r_t + exploration)
execute a_t
```
