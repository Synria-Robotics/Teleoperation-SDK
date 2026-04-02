# Residual Copilot 项目深度分析

> **论文**: *Efficient and Reliable Teleoperation through Real2Sim2Real Shared Autonomy*
> **机构**: Columbia University & University of Pennsylvania

---

## 目录

- [1. 项目整体思路](#1-项目整体思路)
- [2. 系统架构](#2-系统架构)
- [3. 训练流程详解](#3-训练流程详解)
- [4. Actor-Critic 设计](#4-actor-critic-设计)
- [5. Loss 设计](#5-loss-设计)
- [6. Reward 设计](#6-reward-设计)
- [7. 控制栈详解](#7-控制栈详解)
- [8. Domain Randomization 设计](#8-domain-randomization-设计)
- [9. Pilot 模型详解](#9-pilot-模型详解)
- [10. 真机 Teleoperation Residual 设计建议](#10-真机-teleoperation-residual-设计建议)

---

## 1. 项目整体思路

### 1.1 核心思想

本项目提出了 **Residual Copilot** 框架，一种 **Real2Sim2Real 共享自主** (Shared Autonomy) 方法，用于提升遥操作（Teleoperation）的效率和可靠性。

核心思路是：
1. **人类操作者** 提供粗糙的 base action（通过遥操作设备）
2. **RL copilot** 在 base action 的基础上叠加一个小的 **残差修正** (residual correction)
3. 最终的动作 = base action + residual action

这样做的好处是：
- RL 只需学习一个**小幅度的修正量**，不需要从头学习完整的操控策略
- 保留了人类操作者的意图和大方向控制权
- 通过 Sim2Real transfer，在仿真中训练的 copilot 可以直接部署到真机

### 1.2 Real2Sim2Real 流程

```
真机遥操作数据收集 → 构建仿真环境 → 训练 Pilot 模型（模拟人类）→ 训练 Residual RL → 部署到真机
```

1. **Real → Sim**: 收集真机遥操作数据，在 Isaac Sim 中重建任务场景
2. **Sim 训练**: 用收集的数据训练 Pilot 模型（模拟人类操作），然后训练 RL Residual Copilot
3. **Sim → Real**: 将训练好的 RL Copilot 部署到真机，实时修正人类操作

### 1.3 三个装配任务

| 任务 | 描述 | 时长 | 成功阈值 |
|------|------|------|----------|
| **PegInsert** | 8mm 圆柱销钉插入孔中 | 20s | 高度差 < 0.04 |
| **GearMesh** | 中等齿轮啮合到齿轮底座上 | 20s | 高度差 < 0.05 |
| **NutThread** | 螺母拧入螺栓 | 40s | 累计旋转 ≥ 90° |

---

## 2. 系统架构

### 2.1 整体控制栈

```
观测 obs (11D) → Pilot (KNN / BC DiffusionPolicy) → base_action (8D: pos3 + quat4 + gripper1)
                                                         │
                                                    [可选: +noise 或 +lag 扰动]
                                                         │
                                                         ▼
观测 obs (35D policy / 更多 critic) → RL Residual Policy → residual_action (7D: pos3 + rot3 + gripper1)
                                                         │
                                                         ▼
                              _apply_residual() → 合成 action (8D)
                                                         │
                                                         ▼
                    admittance control (导纳控制) → 笛卡尔空间目标
                                                         │
                                                         ▼
                       IK Controller (Pinocchio/SAPIEN) → 关节目标
                                                         │
                                                         ▼
                              Isaac Sim 物理仿真 → 机器人运动
```

### 2.2 残差合成方式 (`_apply_residual`)

```python
# 位置残差: 直接加法
final_pos = base_pos + residual_pos * pos_threshold  # pos_threshold = [0.03, 0.03, 0.03]

# 旋转残差: 轴角 → 四元数乘法
rot_actions = residual_rot * rot_threshold  # rot_threshold = [0.5, 0.5, 0.5]
rot_quat = quat_from_angle_axis(angle, axis)
final_quat = rot_quat * base_quat

# 夹爪残差: 直接加法 + clamp
final_gripper = clamp(base_gripper + residual_grip * grip_threshold, 0, 1)
```

**关键设计点**:
- Residual action 空间是 7D（不是 8D），因为旋转用 3D 轴角表示（而非 4D 四元数）
- 通过 `threshold` 参数限制残差幅度，保证 RL 的修正是 **小范围的**
- 使用 EMA 平滑: `residual = ema_factor * new_action + (1 - ema_factor) * prev_residual`，其中 `ema_factor = 0.2`

### 2.3 文件结构

| 文件 | 职责 |
|------|------|
| `source/xarm_assembly_env/xarm_env.py` | 主环境：场景、step 循环、reward、reset |
| `source/xarm_assembly_env/xarm_env_cfg.py` | 所有配置（Env、DomainRand、Control 等） |
| `source/xarm_assembly_env/assembly_tasks_cfg.py` | 任务特定的资产 + reward 配置 |
| `source/utils/control.py` | 导纳控制 + IK 控制器 |
| `source/pilot_models/knn_pilot.py` | KNN pilot 模型 |
| `source/pilot_models/bc_pilot.py` | DiffusionPolicy BC pilot |
| `scripts/train.py` | RL 训练入口 |
| `scripts/play.py` | 推理/评估入口 |

---

## 3. 训练流程详解

### 3.1 RL 训练框架

- **算法**: PPO (via  RL-Games 库)
- **仿真引擎**: Isaac Sim 5.1.0 / Isaac Lab 2.3.2
- **并行环境数**: 默认 128 个
- **每步物理仿真**: dt = 1/120s，decimation = 8（即每 8 个物理步一个策略步）

### 3.2 训练入口 (`scripts/train.py`)

```bash
python scripts/train.py \
  --task XArm-GearMesh-Residual \
  --pilot kNNPilot \
  --num_envs 128 \
  --headless
```

训练流程:

```
1. 解析命令行参数，设置 pilot 类型
2. 通过 Hydra 配置系统加载 env_cfg 和 agent_cfg
3. 创建 Isaac Sim 环境 (gymnasium.make)
4. 包装为 RlGamesVecEnvWrapper
5. 创建 RL-Games Runner
6. 调用 runner.run() 开始训练循环
```

### 3.3 环境 Step 循环

每个训练 step 的流程:

```
1. _get_observations()
   ├── 获取 pilot 的 base_action (KNN/BC)
   ├── [可选] 对 base_action 添加 noise/lag 扰动
   ├── 构建 obs_dict (policy obs) 和 state_dict (critic state)
   └── 返回 {"policy": obs_35D, "critic": state_ND}

2. RL Policy 输出 residual_action (7D, 值域 [-1, 1])

3. _pre_physics_step(action)
   ├── EMA 平滑残差动作
   ├── _apply_residual(): 合成 base + residual = env_actions
   ├── 导纳控制: 计算笛卡尔空间目标
   └── IK: 计算关节目标

4. _apply_action() × decimation(8次)
   ├── 在 decimation 内线性插值关节目标
   └── 发送关节位置目标给仿真器

5. _get_dones()
   ├── 检查任务成功
   ├── 检查终止条件（距离过远、倾斜过大、掉落等）
   └── 更新滚动成功率

6. _get_rewards()
   └── 计算多项 reward 信号
```

### 3.4 PPO 超参数

| 参数 | 值 | 说明 |
|------|------|------|
| `gamma` | 0.995 | 折扣因子（较高，鼓励长远规划） |
| `tau` (GAE λ) | 0.95 | GAE 参数 |
| `learning_rate` | 1e-4 | 初始学习率 |
| `lr_schedule` | adaptive | 自适应学习率（基于 KL 散度） |
| `kl_threshold` | 0.008 | KL 散度阈值 |
| `e_clip` | 0.2 | PPO clip 参数 |
| `horizon_length` | 128 | 采样 horizon |
| `minibatch_size` | 512 | 小批量大小 |
| `mini_epochs` | 4 | 每次更新的 epoch 数 |
| `critic_coef` | 2 | Critic loss 权重 |
| `entropy_coef` | 0.0 | 熵正则化系数（关闭） |
| `grad_norm` | 1.0 | 梯度裁剪 |
| `max_epochs` | 400 | 最大训练 epoch |
| `normalize_input` | True | 输入归一化 |
| `normalize_value` | True | 值函数归一化 |
| `normalize_advantage` | True | 优势函数归一化 |
| `bounds_loss_coef` | 0.0001 | 动作边界惩罚系数 |

---

## 4. Actor-Critic 设计

### 4.1 非对称 Actor-Critic (Asymmetric)

**这是本项目最重要的设计之一**：Actor（策略网络）和 Critic（价值网络）接收**不同的输入**。

#### Actor (Policy) 观测空间 — 35D

Actor 只能看到**噪声版本**的观测（模拟真机部署时的信息局限性）:

| 维度 | 内容 | 说明 |
|------|------|------|
| 0:3 | `fingertip_pos` | 指尖位置 |
| 3:7 | `fingertip_quat` | 指尖四元数 |
| 7:8 | `gripper` | 夹爪开合度 |
| 8:11 | `fingertip_pos_rel_fixed` | 指尖相对固定物体位置（**带噪声**） |
| 11:14 | `fingertip_pos_rel_held` | 指尖相对抓取物体位置（**带噪声**） |
| 14:17 | `ee_linvel` | 末端执行器线速度（**有限差分**） |
| 17:20 | `ee_angvel` | 末端执行器角速度（**有限差分**） |
| 20:23 | `base_fingertip_pos` | Pilot base action 目标位置 |
| 23:27 | `base_fingertip_quat` | Pilot base action 目标四元数 |
| 27:28 | `base_gripper` | Pilot base action 夹爪 |
| 28:35 | `prev_actions` | 上一步残差动作（7D） |

**关键**: Actor 的观测中包含了 Pilot 的 base action，使得 RL 能够「看到」人类/pilot 想去哪里，从而做出针对性的修正。

#### Critic (Value) 状态空间 — 更丰富

Critic 可以访问**特权信息**（ground truth，仿真中可用但真机不可用的信息）:

| 额外信息 | 维度 | 说明 |
|----------|------|------|
| `joint_pos` | 7D | 真实关节角度 |
| `held_pos` | 3D | 抓取物体真实位置 |
| `held_pos_rel_fixed` | 3D | 抓取物体相对固定物体真实位置 |
| `held_quat` | 4D | 抓取物体真实四元数 |
| `fixed_pos` | 3D | 固定物体真实位置 |
| `fixed_quat` | 4D | 固定物体真实四元数 |
| `pos_threshold` | 3D | 位置动作阈值 |
| `rot_threshold` | 3D | 旋转动作阈值 |
| `ee_linvel` (GT) | 3D | 真实末端线速度（非有限差分） |
| `ee_angvel` (GT) | 3D | 真实末端角速度 |

**非对称设计的意义**:
- Critic 有更完整的信息，可以更准确地估计状态值
- Actor 只有部署时可获取的信息，保证了 sim-to-real 的可迁移性
- Actor 看到的物体位置是**带噪声的**，Critic 看到的是**真实的**

### 4.2 网络架构

```yaml
# Actor-Critic 共享网络（shared=True 时）/ 分离网络
network:
  name: actor_critic
  separate: False  # Actor 和 Critic 不共享网络

  # MLP 结构
  mlp:
    units: [512, 128, 64]  # 三层 MLP
    activation: elu

  # LSTM (可选，用于处理序列)
  rnn:
    name: lstm
    units: 1024
    layers: 2
    before_mlp: True
    concat_input: True
    layer_norm: True

  # 连续动作空间配置
  space:
    continuous:
      mu_activation: None    # 均值无激活（线性输出）
      sigma_activation: None # 标准差无激活
      fixed_sigma: False     # 可学习的标准差
```

**中心化价值网络 (Central Value Config)**:

项目使用了 RL-Games 的 **central value** 功能，即 Critic 有独立的网络和训练配置：

```yaml
central_value_config:
  minibatch_size: 512
  mini_epochs: 4
  learning_rate: 1e-4
  lr_schedule: adaptive
  kl_threshold: 0.008

  network:
    name: actor_critic
    central_value: True  # 标记为中心化价值网络
    mlp:
      units: [512, 128, 64]
      activation: elu
    rnn:
      name: lstm
      units: 1024
      layers: 2
```

### 4.3 动作空间

- **维度**: 7D (pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, gripper)
- **范围**: [-1, 1] (clip_actions: 1.0)
- **含义**: 归一化的残差修正量，通过 threshold 缩放到实际幅度
  - 位置: ±3cm (`pos_action_threshold = [0.03, 0.03, 0.03]`)
  - 旋转: ±0.5 rad ≈ ±28.6° (`rot_action_threshold = [0.5, 0.5, 0.5]`)
  - 夹爪: ±0.1 (`gripper_action_threshold = [0.1]`)

---

## 5. Loss 设计

### 5.1 PPO Loss 组成

RL-Games 框架下的标准 PPO Loss：

```
Total Loss = Policy Loss + critic_coef × Value Loss + bounds_loss_coef × Bounds Loss
```

#### 5.1.1 Policy Loss (Actor Loss)

```python
# 标准 PPO Clipped Objective
ratio = exp(log_prob_new - log_prob_old)
surr1 = ratio * advantage
surr2 = clamp(ratio, 1 - e_clip, 1 + e_clip) * advantage  # e_clip = 0.2
policy_loss = -min(surr1, surr2)
```

- 使用 **adaptive learning rate**: 如果 KL 散度超过 `kl_threshold (0.008)`，降低学习率；反之提高
- `entropy_coef = 0.0`: 不使用熵正则化（残差策略不需要过多探索）
- `normalize_advantage = True`: 对 advantage 进行归一化

#### 5.1.2 Value Loss (Critic Loss)

```python
# 使用 clip_value = True 的 clipped value loss
value_pred_clipped = old_values + clamp(value_pred - old_values, -e_clip, e_clip)
value_loss = max((value_pred - returns)^2, (value_pred_clipped - returns)^2)

# critic_coef = 2 (Critic loss 权重是 Actor 的 2 倍)
```

- 使用 **value_bootstrap = True**: 对未结束的 episode 进行 value bootstrapping
- 使用 **normalize_value = True**: 对值函数输出进行归一化

#### 5.1.3 Bounds Loss

```python
# bounds_loss_coef = 0.0001
# 惩罚动作超出 [-1, 1] 范围
bounds_loss = sum(max(0, |action| - 1)^2)
```

### 5.2 GAE (Generalized Advantage Estimation)

```python
# gamma = 0.995, tau (lambda) = 0.95
delta = reward + gamma * V(s') - V(s)
advantage = sum_{l=0}^{T} (gamma * tau)^l * delta_{t+l}
returns = advantage + V(s)
```

---

## 6. Reward 设计

### 6.1 Reward 组成

奖励函数由 7 个分项组成:

```python
reward = (
    - action_norm      * action_norm_scale        # 动作范数惩罚
    - tilt_penalty     * tilt_penalty_scale        # 倾斜惩罚
    - force_penalty    * force_penalty_scale       # 接触力惩罚
    - action_smoothing * action_smoothing_scale    # 动作平滑惩罚
    + xy_align         * xy_aligned_scale          # XY 对齐奖励
    - terminated       * termination_scale         # 失败终止惩罚
    + task_success     * task_success_scale        # 任务成功奖励
)
```

### 6.2 各项 Reward 详解

#### (1) 动作范数惩罚 (`action_norm`)

```python
action_norm = ||residual_actions||_2 / sqrt(action_space)
reward -= action_norm * 0.3
```

**目的**: 鼓励 RL 尽可能少干预，只在需要的时候输出非零残差。这是 residual RL 中最重要的正则化——保证人类操作者仍有主要控制权。

#### (2) 倾斜惩罚 (`tilt_penalty`)

```python
# 计算动作四元数的 z 轴方向与垂直向下方向的夹角
a_local = [0, 0, 1]  # 局部 z 轴
a_world = quat_rotate(env_actions_quat, a_local)
z_down = [0, 0, -1]
tilt_penalty = arccos(dot(a_world, z_down))
reward -= tilt_penalty * 1.0
```

**目的**: 惩罚抓取物体过度倾斜，防止物体掉落。scale = 1.0 表示这是一个重要的约束。

#### (3) 接触力惩罚 (`force_penalty`)

```python
F = ||held_asset_force||_2
force_penalty = clamp((F - 10.0) / 20.0, 0, 1)  # F>10N 开始惩罚，F>30N 饱和
reward -= force_penalty * 0.2
```

**目的**: 防止过大的接触力，保护物体和机器人。使用 clamp 使惩罚有上界。

#### (4) 动作平滑惩罚 (`action_smoothing`)

```python
action_smoothing = ||prev_actions - residual_actions||_2
reward -= action_smoothing * 0.1
```

**目的**: 鼓励相邻步之间的动作变化平滑，避免抖动。

#### (5) XY 对齐奖励 (`xy_align`)

```python
xy_align_thresh = 0.005  # 5mm
xy_aligned = (
    (||target_xy - held_xy|| < thresh).float() +
    (||fingertip_xy - held_xy|| < thresh).float()
)
reward += xy_aligned * 0.05
```

**目的**: 鼓励水平方向的精确对齐。分两项：物体对齐 + 指尖对齐。

#### (6) 终止惩罚 (`terminated`)

```python
# 失败终止时（非成功终止）施加惩罚
terminated_penalty = (reset_buf & NOT ep_succeeded).float()
reward -= terminated_penalty * 50.0
```

**目的**: 高惩罚值鼓励 RL 避免导致失败的动作（如物体掉落、过度倾斜）。

#### (7) 任务成功奖励 (`task_success`)

```python
# 仅在首次成功时给予奖励
reward += first_success.float() * 30.0
```

**目的**: 给予大额一次性奖励，驱动策略完成任务。使用 `first_success` 防止重复奖励。

### 6.3 默认 Reward Scale

| 奖励项 | Scale | 类型 |
|--------|-------|------|
| `task_success_reward_scale` | 30.0 | 正奖励（稀疏） |
| `termination_reward_scale` | 50.0 | 负惩罚（稀疏） |
| `tilt_penalty_reward_scale` | 1.0 | 负惩罚（稠密） |
| `action_norm_reward_scale` | 0.3 | 负惩罚（稠密） |
| `force_penalty_reward_scale` | 0.2 | 负惩罚（稠密） |
| `action_smoothing_reward_scale` | 0.1 | 负惩罚（稠密） |
| `xy_aligned_reward_scale` | 0.05 | 正奖励（稠密） |

### 6.4 终止条件

| 条件 | 说明 |
|------|------|
| 指尖与物体距离 > 0.2m | 物体掉落或远离 |
| 任务成功 | 正常结束 |
| 倾斜 > 60° (PegInsert) | 物体翻倒 |
| 物体下落 (NutThread) | 螺母掉落 |
| 时间超限 | 超过 episode 时长 |

---

## 7. 控制栈详解

### 7.1 导纳控制 (Admittance Control)

项目使用**任务空间导纳控制**作为底层控制器，而非直接发送关节目标：

```python
# 导纳控制方程: M*a + D*v + K*e = F_ext
# 其中:
#   M: 虚拟质量 (mx, mr)
#   D: 虚拟阻尼 (dx, dr)，设为临界阻尼 d = sqrt(K*M)
#   K: 虚拟刚度 (Kx, Kr)
#   e: 位姿误差 (当前 - 目标)
#   F_ext: 外部力

# 默认参数:
Kx = 200.0   # 平移刚度
Kr = 100.0   # 旋转刚度
mx = 0.125   # 平移虚拟质量
mr = 0.015   # 旋转虚拟质量
dx = sqrt(Kx * mx)  # 临界阻尼
dr = sqrt(Kr * mr)
```

**为什么用导纳控制?**
- 提供柔顺性：在接触装配任务中，允许一定的力反馈
- 滤波效果：平滑不连续的目标指令
- Sim2Real 鲁棒性：通过随机化导纳参数提高迁移性

### 7.2 IK 控制器

使用 SAPIEN 的 Pinocchio 模型进行逆运动学求解:

```python
class IK_Controller:
    def compute_ik(self, init_qpos, cartesian_target):
        # 对每个环境独立求解 IK
        for i in range(num_envs):
            # 构建 4x4 变换矩阵
            tf = eye(4)
            tf[:3, :3] = quat_to_rot_matrix(cartesian_target[i, 3:7])
            tf[:3, 3] = cartesian_target[i, :3]
            # SAPIEN IK 求解
            qpos = pinocchio_model.compute_inverse_kinematics(...)
            # 验证 FK 精度
            if pose_diff > 0.01 or rot_diff > 0.01:
                return initial_qpos  # IK 失败则保持当前位置
```

### 7.3 Decimation 插值

```python
# decimation = 8: 每 8 个物理步执行一个策略步
# 在 decimation 内线性插值关节目标
ratio = (curr_decimation + 1) / decimation
qpos_target = ratio * qpos_targets + (1 - ratio) * starting_qpos
```

---

## 8. Domain Randomization 设计

### 8.1 控制器参数随机化

```python
# 每次 episode reset 时随机化导纳控制参数
Kx_range = [190, 210]    # ±5%
Kr_range = [95, 105]     # ±5%
mx_range = [0.11875, 0.13125]  # ±5%
mr_range = [0.01425, 0.01575]  # ±5%
```

**这是 Sim2Real 鲁棒性的主要来源**，模拟真机控制器参数的不确定性。

### 8.2 观测噪声

```python
# 固定/抓取物体位置观测噪声 (σ=2mm)
fixed_asset_pos_noise = N(0, [0.002, 0.002, 0.002])
held_asset_pos_noise = N(0, [0.002, 0.002, 0.002])
```

### 8.3 数据增强

```python
# 几何增强：XY 平移 + Z 微调 + Yaw 旋转
pos_xy_aug = 0.02   # 2cm XY 随机平移
pos_z_aug = 0.002   # 2mm Z 随机平移
rot_aug = 2.0°      # 2° Yaw 随机旋转
```

### 8.4 Pilot 动作扰动

为了让 RL copilot 学会应对各种质量的人类操作:

**Noisy Pilot** (模拟不稳定操作):
```python
# 随机噪声叠加到 base action
eps ~ Uniform(-0.6, 0.6)  # 较大的噪声范围
prob_on = 0.5              # 50% 概率启动噪声
gate_alpha = 0.8           # EMA 平滑噪声开关
```

**Laggy Pilot** (模拟延迟操作):
```python
# 以一定概率重复上一步的 base action
lag_prob = 0.8  # 80% 概率使用上一步动作
```

---

## 9. Pilot 模型详解

### 9.1 KNN Pilot

**最常用的 pilot 模型**，基于最近邻检索:

```
当前观测 → 在训练数据中查找 K=10 个最近邻 → 软加权采样 → 返回对应动作序列
```

关键参数:
- `K = 10`: 近邻数
- `tau = 0.3`: softmax 温度（控制采样的随机性）
- `min_horizon = 1, max_horizon = 15`: 每次查询返回 1~15 步的动作序列
- `interp_gamma = 0.5`: chunk 间过渡的非线性插值参数

距离度量:
```python
dist = 5.0 * ||pos_diff|| + 0.02 * angle_diff(deg) + 5.0 * |gripper_diff|
```

### 9.2 BC Pilot (DiffusionPolicy)

基于 LeRobot 的 DiffusionPolicy:
- 输入: 机器人状态 (14D) + 环境状态 (6D)
- 输出: 完整的 8D 动作 (pos3 + quat4 + gripper1)
- 两种变体: `bc_teleop` (遥操作数据训练) 和 `bc_expert` (专家数据训练)

---

## 10. 真机 Teleoperation Residual 设计建议

### 10.1 整体架构建议

```
人类遥操作设备 → base_action (目标末端位姿) 
                    │
                    ▼
机器人状态传感器 → 观测构建 → RL Residual Policy → residual_action (7D)
                    │
                    ▼
              base_action + residual → 导纳控制 → IK → 关节控制 → 机器人
```

### 10.2 关键设计要点

#### (1) 观测空间设计

**必须仔细设计哪些信息作为输入**:

```
真机可用信息 (对应 Actor obs):
├── 末端执行器位姿 (来自FK或外部跟踪): fingertip_pos (3D), fingertip_quat (4D)
├── 夹爪开合度: gripper (1D)
├── 人类 base action: base_pos (3D), base_quat (4D), base_gripper (1D)
└── 上一步残差动作: prev_residual (7D)
```

**注意事项**:
- **必须包含 base_action**: 这让 RL 知道人类想做什么，是残差策略的核心输入

#### (2) 残差幅度限制

```python
# 关键: 限制残差的最大幅度
pos_threshold = [0.03, 0.03, 0.03]   # 最大 ±3cm 位置修正
rot_threshold = [0.5, 0.5, 0.5]      # 最大 ±28.6° 旋转修正
gripper_threshold = [0.1]             # 最大 ±10% 夹爪修正
```

**建议**: 
- 初始部署时可以适当减小 threshold，确保安全
- 可以渐进地增加 threshold 来提升修正能力
- 夹爪 threshold 保持较小，避免意外松开物体

#### (3) 导纳控制参数

```python
# 真机默认参数 (与仿真一致)
Kx = 200.0, Kr = 100.0
mx = 0.125, mr = 0.015
# 阻尼为临界阻尼
dx = sqrt(Kx * mx), dr = sqrt(Kr * mr)
```

**建议**:
- 真机部署时先用默认参数
- 如果机器人响应太灵敏，增大 mx/mr（虚拟质量）
- 如果机器人响应太迟钝，增大 Kx/Kr（虚拟刚度）
- 训练时的 ±5% 随机化已经覆盖了一定的真机参数不确定性

#### (4) EMA 平滑

```python
ema_factor = 0.2
residual = 0.2 * new_action + 0.8 * prev_residual
```

**建议**:
- 真机部署时 **一定要保持 EMA 平滑**，否则动作会抖动
- 如果发现修正太激进，可以降低 ema_factor（如 0.1）
- 如果发现修正太迟钝，可以提高 ema_factor（如 0.3）

#### (5) Fingertip2EEF 偏移

```python
# 仿真和真机的偏移不同！
sim_fingertip2eef = [0.0, 0.0, 0.165]   # 仿真中指尖到 link7 的偏移
real_fingertip2eef = [0.0, 0.0, 0.23]    # 真机中指尖到 link7 的偏移
```

**注意**: 部署时必须使用 `real_fingertip2eef`，这个偏移直接影响目标位置计算。

### 10.3 训练阶段的注意事项

#### (1) 数据收集

```
1. 在真机上使用遥操作设备执行任务
2. 记录每步: 末端位姿、夹爪状态、速度、物体位置、遥操作目标位姿
3. 每条轨迹标记成功/失败
4. 建议收集 50-200 条成功轨迹
```

#### (2) Pilot 模型选择

- **推荐使用 KNN Pilot**: 简单、无需额外训练、数据效率高
- 如果数据量充足 (>500条)，可以考虑 DiffusionPolicy
- KNN Pilot 的 `knn_tau` 参数控制随机性，训练时适当增大可增加探索

#### (3) Pilot 扰动策略

训练 Residual RL 时，**必须**对 pilot 添加扰动:

```python
# 推荐: Noisy Pilot (模拟人类操作的不稳定性)
pilot_type = "noisy"
base_noise_range = [-0.6, 0.6]
base_noise_prob = 0.5

# 或: Laggy Pilot (模拟人类操作的延迟)
pilot_type = "laggy"
base_lag_prob = 0.8
```

**原因**: 如果 RL 只在完美的 pilot 上训练，部署到真人操作时泛化能力会很差。

#### (4) Domain Randomization 策略

```
必须开启:
├── 导纳控制参数随机化 (rand_ctrl=True)
├── 观测噪声 (fixed/held_asset_pos noise)
└── 数据增强 (aug_data=True)

可选:
├── 摩擦力随机化
└── 物体质量随机化
```

### 10.4 部署阶段的注意事项

#### (1) 推理频率

```python
# 仿真: dt=1/120s, decimation=8 → 策略频率 = 120/8 = 15 Hz
# 真机部署需要匹配 ~15 Hz 的控制频率
```

**建议**: 确保从观测采集到动作输出的延迟 < 60ms (在 15Hz 下留有余量)

#### (2) 安全机制

```python
# 1. 动作幅度 clamp
residual = clamp(residual, -1, 1)

# 2. 最终位姿 clamp (防止超出工作空间)
final_pos = clamp(final_pos, workspace_min, workspace_max)

# 3. IK 验证
if pose_diff > 0.01:
    return current_qpos  # IK 失败则保持不动

# 4. 急停机制
if contact_force > threshold:
    stop()
```

#### (3) 关键 Pipeline

```python
def step(human_action, robot_state, objects_state):
    """一步真机 residual copilot 推理"""
    
    # 1. 构建观测
    obs = build_observation(
        fingertip_pos=robot_state.ee_pos + real_fingertip2eef_offset,
        fingertip_quat=robot_state.ee_quat,
        gripper=robot_state.gripper_pos / 1.6,
        base_action=human_action,  # 人类遥操作的目标位姿
        prev_residual=prev_residual_action,
    )
    
    # 2. RL 推理
    residual_action = rl_policy.infer(obs)  # 7D, [-1, 1]
    
    # 3. EMA 平滑
    residual_action = 0.2 * residual_action + 0.8 * prev_residual
    
    # 4. 合成最终动作
    final_pos = human_action.pos + residual_action[:3] * pos_threshold
    final_quat = quat_mul(axis_angle_to_quat(residual_action[3:6] * rot_threshold), human_action.quat)
    final_gripper = clamp(human_action.gripper + residual_action[6] * gripper_threshold, 0, 1)
    
    # 5. 导纳控制
    cartesian_target, velocity = admittance_control(
        current_pos, current_quat,
        final_pos, final_quat,
        velocity, ext_force, dt,
        Kx=200, Kr=100, mx=0.125, mr=0.015
    )
    
    # 6. IK 求解
    joint_targets = ik_solver.solve(current_joints, cartesian_target)
    
    # 7. 发送关节命令
    robot.set_joint_position(joint_targets)
    
    return residual_action  # 保存作为下一步的 prev_residual
```

### 10.5 常见问题与调试建议

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 残差修正过于激进 | ema_factor 过大 / threshold 过大 | 降低 ema_factor 或 threshold |
| 残差修正无效果 | threshold 过小 / 模型未收敛 | 检查训练曲线 / 增大 threshold |
| 机器人抖动 | 缺少 EMA 平滑 / 控制频率不匹配 | 确保 EMA 开启 / 匹配 15Hz |
| Sim2Real gap | DR 不充分 | 增大参数随机化范围 |
| 物体位置估计不准 | 视觉系统误差 | 训练时增大 obs noise (>2mm) |
| IK 求解失败 | 目标位姿超出可达空间 | 添加工作空间裁剪 |

### 10.6 进阶: 多任务/新任务扩展

1. **新任务**: 按照 `assembly_tasks_cfg.py` 的模式定义新的 `AssemblyTask`
2. **新 Pilot**: 在 `pilot_models/` 中实现 `predict()` 和 `reset()` 接口
3. **Vision-based**: 可以扩展到视觉观测，使用 `vision_dp_cfg.json` 配置

---

## 附录: 项目依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | 3.11 | 运行环境 |
| Isaac Sim | 5.1.0 | 物理仿真 |
| Isaac Lab | 2.3.2 | RL 环境框架 |
| PyTorch | 2.7.0 (cu128) | 深度学习 |
| RL-Games | (Isaac 分支) | PPO 训练 |
| SAPIEN | 2.2.2 | IK 求解 |
| LeRobot | (本地) | DiffusionPolicy |
| scikit-learn | - | KNN |
