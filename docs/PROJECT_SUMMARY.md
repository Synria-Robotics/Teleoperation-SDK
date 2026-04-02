# ResFiT 项目深度解析 —— Residual Off-Policy RL for Finetuning Behavior Cloning Policies

> 论文: https://arxiv.org/abs/2509.19301  
> 官网: https://residual-offpolicy-rl.github.io/

---

## 一、项目整体思路

### 1.1 核心问题

Behavior Cloning (BC) 策略通过模仿学习训练，能够学到不错的基础行为，但由于**分布偏移 (distribution shift)** 和**复合误差 (compounding error)**，在复杂的操作任务中成功率有限（如 50-70%）。直接用 RL 从头训练又面临**探索困难**和**样本效率低**的问题。

### 1.2 核心思想：Residual RL + Off-Policy 微调

ResFiT 的核心思想是：

```
最终动作 = BC策略动作(base_action) + 残差策略动作(residual_action)
```

**不修改原始 BC 策略**，而是训练一个独立的残差策略 (residual policy) 来**修正 BC 策略的不足**。核心优势：

1. **保留 BC 策略的先验知识**：BC 策略提供了合理的动作基线，残差策略只需学习"修正量"
2. **大幅缩小探索空间**：残差动作的范围被限制在较小的 `action_scale`（如 0.1~0.2），减轻了探索难度
3. **Off-Policy 训练**：使用 TD3 + Replay Buffer 的 off-policy 算法，大幅提高样本效率
4. **混合在线/离线数据**：同时利用离线演示数据和在线交互数据（RLPD 范式）

### 1.3 系统架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                       训练流程总览                                │
│                                                                 │
│  阶段1: BC 策略预训练 (ACT / Diffusion Policy)                   │
│    ↓                                                            │
│  阶段2: 在线 warm-up（随机残差探索，填充 Replay Buffer）           │
│    ↓                                                            │
│  阶段3: Residual TD3 训练                                       │
│    ├── Critic Warmup（仅更新 Critic）                            │
│    └── 正式训练（交替更新 Critic 和 Actor）                       │
│                                                                 │
│  执行时:                                                        │
│    obs → BC策略 → base_action                                   │
│    obs + base_action → Residual Actor → residual_action         │
│    env.step(base_action + residual_action)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、三阶段训练流程详解

### 阶段一：BC 策略预训练

**入口脚本**: `resfit/lerobot/scripts/train_bc_dexmg.py`

使用 ACT (Action Chunking Transformer) 在人类演示数据上进行监督学习：

- **数据**: LeRobot 格式的人类演示数据集（来自 HuggingFace）
- **模型**: ACT Policy，输入图像观察 + 机器人状态，输出动作序列（action chunking）
- **Loss**: L1 Loss + KL散度（VAE 正则化）
- **训练量**: 约 200k steps

训练完成后，BC 策略权重保存到 wandb，供后续 Residual RL 阶段加载。

### 阶段二：在线 Warm-up（随机残差探索）

**代码位置**: `train_residual_td3.py` 第 718-792 行

在正式 RL 训练之前，先用随机残差动作填充 Online Replay Buffer：

```python
# 两种 warm-up 模式:
if cfg.algo.use_base_policy_for_warmup:
    # 模式1: base_action + 随机噪声（推荐）
    rand_actions = (torch.rand(...) * 2 - 1) * random_action_noise_scale
else:
    # 模式2: 纯随机动作（取消 base_action 的效果）
    rand_actions = pure_random - base_action
```

- **目的**: 收集足够的初始经验数据（默认 10,000 步），让 Critic 有数据可学
- **noise scale**: 默认 0.2，即残差动作在 `[-0.2, 0.2]` 范围内
- **缓存机制**: 支持将 warm-up 数据缓存到磁盘/HuggingFace Hub，避免重复收集

### 阶段三：Residual TD3 训练（核心）

正式训练分为两个子阶段：

#### 子阶段 3a: Critic Warmup（仅更新 Critic）

```python
# 代码位置: train_residual_td3.py 第 867-936 行
_run_critic_warmup(...)  # 默认 10,000 步 critic-only 更新
```

- 仅更新 Critic 网络（和 Encoder），不更新 Actor
- **目的**: 让 Critic 先学到合理的 Q-value 估计，再用 Q-value 来指导 Actor 优化
- 防止 Actor 在 Critic 不准确时学到错误的梯度方向

#### 子阶段 3b: 联合训练（交替更新 Critic 和 Actor）

```python
# 主循环: train_residual_td3.py 第 939-1178 行
while global_step <= total_timesteps:
    # (1) 采集: agent.act() → env.step()
    # (2) 存储到 online replay buffer
    # (3) 定期评估
    # (4) 从 online + offline buffer 混合采样，更新网络
```

**关键超参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_updates_per_iteration` | 4 | 每个环境步骤后执行的梯度更新次数 (UTD ratio) |
| `actor_updates_per_iteration` | 1 | 每个更新周期中 Actor 更新的次数 |
| `offline_fraction` | 0.5 | 每个 mini-batch 中离线数据的比例 |
| `n_step` | 5 | N-step return 的步数 |
| `gamma` | 0.995 | 折扣因子 |
| `batch_size` | 256 | mini-batch 大小 |
| `action_scale` | 0.1~0.2 | 残差动作的最大幅度 |
| `actor_lr` | 1e-6 | Actor 学习率（非常小，防止策略崩溃） |
| `critic_lr` | 1e-4 | Critic 学习率 |

---

## 三、Actor-Critic 设计详解

### 3.1 QAgent（核心 Agent 类）

**文件**: `resfit/rl_finetuning/off_policy/rl/q_agent.py`

```
QAgent
├── encoders (nn.ModuleList)     # 每个相机一个 ViT encoder
├── actor (Actor)                 # 残差策略网络
├── actor_target (Actor)          # Actor 的 target 网络（EMA）
├── critic (Critic)               # Q-value 网络集合
├── critic_target (Critic)        # Critic 的 target 网络（EMA）
├── encoder_opt                   # Encoder 优化器
├── actor_opt                     # Actor 优化器
└── critic_opt                    # Critic 优化器
```

### 3.2 Encoder（视觉编码器）

**文件**: `resfit/rl_finetuning/off_policy/networks/encoder.py`

- **架构**: MinViT（轻量化 Vision Transformer）
- **输入**: 84×84 RGB 图像（每个相机一个 Encoder）
- **输出**: `[B, num_patches, embed_dim]` 形状的特征（默认 81 patches × 128 dim）
- **多相机支持**: 多个 Encoder 的特征在 patch 维度上拼接
- **预处理**: 图像归一化到 `[-0.5, 0.5]`，训练时使用 RandomShiftsAug 数据增强

### 3.3 Actor（残差策略）

**文件**: `resfit/rl_finetuning/off_policy/rl/actor.py`

```
Actor 网络结构:
    ┌───────────┐
    │ 视觉特征   │──→ 压缩层(SpatialEmb/Linear) ──→ ┐
    │ 机器人状态  │─────────────────────────────────→ ├─→ MLP ──→ Tanh ──→ × action_scale
    │ base_action│─────────────────────────────────→ ┘     (多层FC)
    └───────────┘
```

**核心设计要点**:

1. **Residual Actor 模式**: 当 `residual_actor=True` 时，Actor 的输入还包括 `base_action`
   ```python
   if residual_actor:
       self.prop_dim += action_dim  # 扩展输入维度以容纳 base_action
   ```
   
2. **Action Scale 缩放**: 输出经过 `Tanh` 后乘以 `action_scale`（默认 0.1~0.2）
   ```python
   scaled_mu = mu * self.cfg.action_scale  # Tanh 输出 ∈ [-1,1] → [-0.1, 0.1]
   ```

3. **最后一层零初始化**: `actor_last_layer_init_scale=0.0`
   - **极其关键**: 确保训练开始时残差策略输出接近零，不破坏 BC 策略的行为
   ```python
   ActorConfig(
       action_scale=0.1,
       actor_last_layer_init_scale=0.0,  # 重要！残差从零开始
   )
   ```

4. **动作分布**: 使用 `TruncatedNormal` 分布进行探索
   ```python
   action_dist = utils.TruncatedNormal(scaled_mu, std)
   ```

### 3.4 Critic（Q-Value 网络）

**文件**: `resfit/rl_finetuning/off_policy/rl/critic.py`

```
Critic 网络结构:
    ┌──────────────────┐
    │ 视觉特征 + 状态    │
    │ + 动作            │──→ SpatialEmb Trunk (共享) ──→ K 个 MLP Head (vmap并行)
    └──────────────────┘                                     ↓
                                                        Q1, Q2, ..., QK
```

**核心设计要点**:

1. **多头 Ensemble（RED-Q 风格）**: 默认 10 个 Q-head
   - **共享 Trunk**: SpatialEmb 空间嵌入层在所有 Head 之间共享
   - **独立 Head**: 每个 Head 是独立的 MLP，通过 `vmap` 高效并行计算
   
2. **Target Q 计算**: 取随机 2 个 Head 的最小值（消除 Q-value 过估计）
   ```python
   num_heads = min(self.cfg.min_q_heads, q_out.shape[0])  # 默认 2
   idx = torch.randperm(q_out.shape[0])[:num_heads]
   return torch.min(q_out.index_select(0, idx), dim=0).values
   ```

3. **Policy Gradient Q 值**: 支持多种策略梯度类型
   - `ensemble_mean`: 所有 Head 的均值（默认，RED-Q 方式）
   - `min_random_pair`: 随机 2 个 Head 的最小值（保守）
   - `q1`: 只用第一个 Head（标准 TD3）

4. **Soft Target Update (EMA)**:
   ```python
   utils.soft_update_params(self.critic, self.critic_target, tau=0.005)
   ```

---

## 四、Loss 设计详解

### 4.1 Critic Loss（Q-Value 学习）

**位置**: `q_agent.py > update_critic()`

支持三种 Critic Loss 类型：

#### (a) MSE Loss（默认）

```python
# 标准 TD-error 的均方误差
q_all = critic(feat, state, action)  # [K, B, 1]
td_errors = torch.abs(q_all - target_q.unsqueeze(0)).mean(dim=0)  # [B]
critic_loss = (td_errors ** 2).mean()
```

#### (b) HL-Gauss Loss（Histogram Loss with Gaussian Smoothing）

将 Q-value 回归问题转化为分类问题：
- 将 Q-value 范围 `[0, 1]` 离散化为 `n_bins` 个 bin
- 使用高斯平滑的 soft label 作为目标分布
- 通过交叉熵损失训练

```python
# 使用 Gaussian CDF 将连续 target 转换为概率分布
tgt_probs = self._target_to_probs(target.detach())
log_probs = F.log_softmax(logits, dim=-1)
loss = -(tgt_probs * log_probs).sum(dim=-1).mean()
```

#### (c) C51 Loss（Categorical DQN）

另一种分布式 RL 方法，将 Q-value 分布建模为固定支撑点上的离散分布。

#### Target Q 计算（所有 Loss 共享）

```python
with torch.no_grad():
    # 1. Target Actor 预测下一步的残差动作
    next_residual = actor_target(next_obs, stddev, clip)
    
    # 2. 组合动作 = base_action + residual（带裁剪）
    next_action = clamp(next_obs["base_action"] + next_residual, -1, 1)
    
    # 3. Target Critic 评估 Q-value
    target_q_min = critic_target.q_value(next_obs["feat"], next_obs["state"], next_action)
    
    # 4. Bellman 等式
    target_q = reward + gamma * nonterminal * target_q_min
```

### 4.2 Actor Loss（策略优化）

**位置**: `q_agent.py > _compute_actor_loss()`

```python
# 1. Actor 预测残差动作
action_pred = actor(obs, stddev=0.0)

# 2. L2 正则化（限制残差幅度）
action_l2_penalty = action_l2_reg_weight * mean(sum(action_pred²))

# 3. 组合动作 = base_action + residual
combined_action = clamp(obs["base_action"] + action_pred, -1, 1)

# 4. Q-value 作为策略梯度（最大化 Q 值）
q = critic.q_value_for_policy(feat, state, combined_action)
actor_loss_base = -q.mean()

# 5. 总损失
actor_loss_total = actor_loss_base + action_l2_penalty
```

**关键点**:
- Actor Loss 不反向传播到 Encoder（`obs["feat"] = obs["feat"].detach()`）
- L2 正则化防止残差动作过大，保持 BC 策略的基础行为
- Actor 学习率极小（1e-6），避免策略突变

### 4.3 可选的 BC 正则化 Loss

**位置**: `q_agent.py > update_actor_rft()`

```python
# 在 actor loss 基础上加入 BC loss
bc_loss = MSE(actor_pred_action, offline_gt_action)
total_loss = actor_loss + bc_loss_coef * ratio * bc_loss
```

其中 `ratio` 可以动态调整：当参考策略的 Q 值高于当前策略时，加大 BC 正则化权重。

---

## 五、Reward 设计

### 5.1 稀疏二值奖励

本项目使用**极其简单的稀疏奖励**：

```python
reward = 1.0 if task_success else 0.0
```

- **成功**: reward = 1（仅在 episode 最后一步）
- **失败**: reward = 0（整个 episode 全为 0）
- **无中间奖励**: 无整形奖励 (no reward shaping)

### 5.2 N-step Return 处理

为了克服稀疏奖励的信用分配问题，使用 **N-step Return**（默认 n=5）:

```python
# MultiStepTransform: 将连续 n 步的奖励累加并折扣
# R_n = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^(n-1)*r_{t+n-1}
# 并将 next_obs 替换为 t+n 时刻的观察
```

- **gamma**: 0.995（高折扣因子，适合长 horizon 任务）
- **效果**: 将成功奖励向前传播 n 步，加速 Critic 学习

### 5.3 Q-Target Clipping

可选地将 Q target 裁剪到奖励范围 `[0, 1]`：

```python
if clip_q_target_to_reward_range:
    target_q = clamp(target_q, min=0, max=1)
```

---

## 六、Residual 环境包装器

**文件**: `resfit/rl_finetuning/wrappers/residual_env_wrapper.py`

`BasePolicyVecEnvWrapper` 是整个残差学习框架的核心桥梁：

```
                  ┌─────────────────────────────────────────┐
                  │        BasePolicyVecEnvWrapper           │
  residual_action │                                         │
  ──────────────→ │  combined = base_action + residual       │
                  │  env_action = unscale(combined)          │
                  │  obs, reward, ... = vec_env.step(action) │
                  │  next_base = base_policy(obs)            │
                  │  augmented_obs = obs + next_base_action  │
  ←────────────── │                                         │
  augmented_obs   └─────────────────────────────────────────┘
```

**关键流程**:

1. **reset()**: 
   - 重置环境 → 获取 raw_obs
   - 重置 BC 策略 → 用当前 obs 推理得到 base_action
   - 将 base_action 归一化到 `[-1, 1]` 并附加到 obs

2. **step(residual_action)**:
   - `combined = base_action + residual_action`（归一化空间相加）
   - `env_action = unscale(combined)` 转换到原始动作空间
   - 执行环境步骤
   - 用新 obs 重新推理 BC 策略得到下一步的 base_action
   - 处理 episode 终止和重置

### 动作归一化

**文件**: `resfit/rl_finetuning/utils/normalization.py`

```python
# ActionScaler: 将原始动作空间映射到 [-1, 1]
scaled = 2 * (action - min) / (max - min) - 1

# StateStandardizer: 状态标准化 (z-score)
standardized = (state - mean) / std
```

- 动作范围通过 `action_scale` 参数额外扩展（默认 0.1~0.2）
- 确保每个维度有最小范围，防止归一化数值不稳定

---

## 七、Replay Buffer 设计

### 7.1 Online Replay Buffer

- 存储在线交互经验
- 支持 Prioritized Experience Replay (PER)
- 使用 `MultiStepTransform` 自动计算 N-step return

### 7.2 Offline Replay Buffer

- 存储离线演示数据
- 两种标注模式：
  - **GT-as-base**: 用 GT 动作作为 base_action（残差学习输出零）
  - **Base-policy-as-base**: 用 BC 策略推理作为 base_action（更贴近在线训练）

### 7.3 混合采样

```python
online_batch = online_rb.sample(batch_size * (1 - offline_fraction))
offline_batch = offline_rb.sample(batch_size * offline_fraction)
batch = concat([online_batch, offline_batch])
```

---

## 八、面向 Teleoperation 的 Residual 设计方案

### 8.1 场景说明

在遥操作场景中，人类操作员通过遥操作设备（手柄、VR控制器等）实时控制机器人。Residual Policy 的目标是**实时修正人类的遥操作动作**，提升任务成功率和操作精度。

与 BC 场景的关键区别：

| 维度 | BC Residual | Teleoperation Residual |
|------|-------------|----------------------|
| Base Action 来源 | BC 策略推理 | 人类实时输入 |
| Base Action 质量 | 一致（策略确定性推理） | 不稳定（人类操作有抖动/延迟） |
| 实时性要求 | 可接受一定延迟 | 严格要求低延迟 |
| 训练数据 | 离线数据 + 在线模拟 | 真机遥操作数据 |
| 安全性 | 模拟环境无风险 | 真机需要安全约束 |

### 8.2 整体架构设计

```
人类遥操作 → teleop_action (base_action)
                    │
                    ↓
    ┌───────────────────────────────────────┐
    │  obs + teleop_action → Residual Actor │
    │  residual_action = Actor(obs, teleop) │
    │  final = clamp(teleop + residual, -1, 1) │
    └───────────────────────────────────────┘
                    │
                    ↓
              机器人执行 final_action
```

### 8.3 关键设计要点

#### (1) 环境包装器改造

需要将 `BasePolicyVecEnvWrapper` 中的 BC 策略替换为人类遥操作输入接口：

```python
class TeleoperationResidualWrapper:
    def step(self, residual_action, teleop_action):
        """
        Args:
            residual_action: 残差策略输出
            teleop_action: 人类遥操作输入（已归一化）
        """
        combined = torch.clamp(teleop_action + residual_action, -1, 1)
        env_action = self.action_scaler.unscale(combined)
        return self.env.step(env_action)
```

#### (2) Action Scale 需要更小

由于人类操作已经是"有意图的"动作，残差修正幅度应更小：

```python
ActorConfig(
    action_scale=0.05 ~ 0.1,       # 比 BC residual 更小
    actor_last_layer_init_scale=0.0, # 仍然从零开始
    action_l2_reg_weight=0.1,        # 更强的 L2 正则化
)
```

#### (3) 延迟处理（关键！）

遥操作存在通信延迟，残差策略需要考虑：

- **观察延迟**: 残差策略看到的图像可能比人类看到的滞后 1-2 帧
- **动作延迟**: 残差修正可能在人类已改变意图后才生效
- **建议**: 
  - 在 obs 中加入时间戳和历史动作信息
  - 使用较短的 action chunking 或逐帧决策
  - 预测未来 1-2 步的修正

#### (4) 安全约束（真机必需）

```python
class SafeResidualActor(Actor):
    def forward(self, obs, std):
        action_dist = super().forward(obs, std)
        # 硬性安全约束
        safe_action = self.safety_filter(action_dist.mean, obs)
        return TruncatedNormal(safe_action, std)
    
    def safety_filter(self, action, obs):
        # 1. 力矩限制
        action = torch.clamp(action, -self.max_torque, self.max_torque)
        # 2. 关节限位检查
        # 3. 碰撞检测
        # 4. 速度平滑
        return action
```

#### (5) Reward 设计建议

遥操作场景下可以使用更丰富的奖励：

```python
def compute_teleop_reward(obs, action, next_obs):
    # 1. 任务成功奖励（稀疏）
    task_reward = 1.0 if task_success else 0.0
    
    # 2. 辅助奖励（密集，可选）
    # - 末端执行器接近目标
    distance_reward = -distance_to_target
    # - 动作平滑性
    smoothness_reward = -torch.norm(action - prev_action)
    # - 力反馈（如果有）
    force_reward = -excessive_force_penalty
    
    return task_reward + 0.1 * distance_reward + 0.01 * smoothness_reward
```

#### (6) 在线学习 vs 离线学习

**推荐的训练流程**:

1. **阶段1: 离线预训练 Critic**
   - 使用人类遥操作的录制数据
   - 离线训练 Critic 学习 Q-value
   - Actor 保持零初始化

2. **阶段2: 在线微调（人在环）**
   - 人类继续遥操作，同时残差策略实时修正
   - 收集在线数据到 Replay Buffer
   - 混合在线+离线数据更新网络
   - **渐进增大 action_scale**：从 0.01 逐步增大到 0.1

3. **阶段3: 完全自主（可选）**
   - 将训练好的 Residual Policy 与固定的 BC Policy 结合
   - 完全脱离人类操作

#### (7) 网络结构建议

```python
# Teleoperation Residual Agent 配置
agent_config = QAgentConfig(
    actor_lr=1e-7,                    # 更小的学习率
    critic_lr=1e-4,
    critic_target_tau=0.001,          # 更慢的 target 更新
    actor=ActorConfig(
        action_scale=0.05,            # 更小的残差幅度
        actor_last_layer_init_scale=0.0,
        action_l2_reg_weight=0.1,     # 更强的正则化
        hidden_dim=512,               # 可适当减小网络
        num_layers=2,
    ),
    critic=CriticConfig(
        num_q=5,                      # 可减少 Q-head 数量（减少计算量，满足实时性）
        hidden_dim=512,
    ),
)

algo_config = ResidualTD3AlgoConfig(
    n_step=3,                         # 更短的 n-step（遥操作 episode 可能更短）
    gamma=0.99,
    critic_warmup_steps=5000,
    num_updates_per_iteration=2,      # 减少更新次数（保证实时性）
    progressive_clipping_steps=50000, # 渐进增大残差幅度
)
```

### 8.4 注意事项总结

| 注意点 | 说明 |
|--------|------|
| **零初始化** | 残差策略最后一层必须零初始化，确保初始输出为零 |
| **小 action scale** | 遥操作场景下残差幅度应更小（0.05~0.1） |
| **强正则化** | L2 正则化防止残差过大覆盖人类意图 |
| **低延迟推理** | 减少 Q-head 数量、减小网络规模，优化推理速度 |
| **安全约束** | 必须加入力矩限制、关节限位、碰撞检测 |
| **渐进式训练** | 使用 `progressive_clipping_steps` 渐进增大残差幅度 |
| **通信延迟** | 考虑 obs 和 action 的延迟，可加入历史信息 |
| **人类意图理解** | 残差策略应理解人类的操作意图而非简单修正 |
| **数据效率** | 真机数据宝贵，应充分利用离线数据预训练 |
| **回退机制** | 提供一键关闭残差修正的机制，保障安全 |

### 8.5 与 ResFiT 的关键差异总结

```
ResFiT (BC → Residual):
  - base_action 来自固定的 BC 策略（确定性、可复现）
  - 可在模拟器中大量并行采集数据
  - 不需要考虑实时性和安全性

Teleoperation Residual:
  - base_action 来自人类（随机性、不可复现）
  - 数据采集依赖真机，成本高
  - 必须满足实时性和安全性要求
  - 需要处理通信延迟和人机协作
```

---

## 九、代码文件结构参考

```
resfit/
├── lerobot/                          # BC 策略训练
│   ├── configs/policies.py           # 策略配置
│   ├── policies/
│   │   ├── act/                      # ACT 策略
│   │   │   ├── configuration_act.py
│   │   │   └── modeling_act.py
│   │   └── diffusion/               # Diffusion Policy
│   ├── scripts/train_bc_dexmg.py    # BC 训练脚本
│   └── utils/load_policy.py         # 策略加载工具
│
├── rl_finetuning/                    # RL 微调（核心）
│   ├── config/
│   │   ├── rlpd.py                  # RLPD 基础配置
│   │   ├── residual_td3.py          # Residual TD3 配置
│   │   └── performance.py           # 性能配置
│   ├── off_policy/
│   │   ├── common_utils/            # 通用工具
│   │   ├── networks/
│   │   │   ├── encoder.py           # ViT 视觉编码器
│   │   │   └── min_vit.py           # 轻量化 ViT
│   │   └── rl/
│   │       ├── actor.py             # Actor 网络
│   │       ├── critic.py            # Critic 网络（HLGauss/C51/MSE）
│   │       └── q_agent.py           # QAgent（核心 Agent）
│   ├── scripts/
│   │   ├── train_residual_td3.py    # Residual TD3 训练脚本（核心）
│   │   └── train_rlpd_dexmg.py     # RLPD baseline
│   ├── utils/
│   │   ├── normalization.py         # 动作/状态归一化
│   │   ├── rb_transforms.py         # N-step Return Transform
│   │   ├── evaluate_dexmg.py        # 评估工具
│   │   └── checkpoint.py            # 检查点管理
│   └── wrappers/
│       └── residual_env_wrapper.py  # 残差环境包装器（核心）
│
└── dexmg/environments/dexmg.py      # 环境定义
```

---

## 十、关键设计总结

| 设计决策 | 选择 | 理由 |
|---------|------|------|
| Base Policy | ACT (Action Chunking Transformer) | 性能好，支持 temporal ensemble |
| RL 算法 | TD3 (Twin Delayed DDPG) | Off-policy，连续动作空间 |
| Critic Ensemble | 10 个 Q-head (RED-Q 风格) | 减少 Q-value 过估计 |
| 视觉编码器 | MinViT (轻量化 ViT) | 平衡性能和计算效率 |
| Replay Buffer | 混合在线/离线 (RLPD) | 充分利用离线数据 |
| N-step Return | n=5, γ=0.995 | 解决稀疏奖励的信用分配 |
| Action Scale | 0.1~0.2 | 限制残差幅度，保护 BC 行为 |
| 零初始化 | 最后一层 weight=0 | 训练初期残差为零 |
| Critic Warmup | 10k 步 | 先让 Critic 稳定再训 Actor |
| Actor LR | 1e-6（极小） | 防止策略突变 |
