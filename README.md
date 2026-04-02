# Teleoperation-SDK

Alicia-D 机械臂遥操作独立工具包。当前包含：

| Demo | 功能 |
|------|------|
| **Demo 01** — `01_demo_realtime_joint_plot.py` | 实时关节角度绘图 + 手柄输入状态面板 |
| **Demo 02** — `02_demo_mujoco_follower.py` | Leader → MuJoCo Follower 遥操作仿真 |
| **Residual RL** — `record_teleop.py` / `train_residual_td3_teleop.py` / `eval_residual_td3_teleop.py` | 基于 teleoperation 的 MuJoCo shared-control 数据采集、训练与评估 |

---

## 目录结构

```
Teleoperation-SDK/
├── 01_demo_realtime_joint_plot.py       # Demo 01：实时关节监控
├── 02_demo_mujoco_follower.py           # Demo 02：MuJoCo 遥操作
├── record_teleop.py                     # 采集 teleop 数据集
├── train_residual_td3_teleop.py         # 训练 residual TD3 copilot
├── eval_residual_td3_teleop.py          # 评估训练好的 residual policy
├── assets/
│   └── mujoco/
│       └── Alicia_D_v5_6/
│           └── gripper_50mm/
│               ├── alicia_d_follower.xml    # MuJoCo MJCF 模型
│               ├── base_link.STL            # 机器人 mesh 文件
│               ├── link1.STL ~ link6.STL
│               ├── left_gripper_50mm.STL
│               └── right_gripper_50mm.STL
├── teleop_sdk/
│   ├── data/                            # rollout 数据结构与 .npz I/O
│   ├── envs/                            # MuJoCo pick-place teleop 环境
│   ├── providers/                       # human / residual action providers
│   ├── rewards/                         # task reward + anti-conflict shaping
│   └── runners/                         # shared-control rollout driver
├── utils/
│   ├── __init__.py
│   └── fps_utils.py                     # precise_sleep 高精度定时
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## 1. 系统要求

| 项目 | 要求 |
|------|------|
| **操作系统** | Ubuntu 20.04 / 22.04（推荐） |
| **Python** | 3.10 – 3.12（推荐 3.11） |
| **硬件** | Alicia-D Leader 机械臂，USB 串口连接 |
| **GPU**（可选） | NVIDIA GPU + CUDA 12.x（仅影响 torch 安装方式） |

---

## 2. 环境配置（从零开始）

### 2.1 安装 Conda（Miniconda）

> 如已安装 Conda，跳到 2.2。

请按官方文档完成安装与验证：[Conda 环境管理 | Synria Robotics 文档中心](https://docs.sparklingrobo.com/info/development/conda-installation)。安装完成后 **重启终端**（或按文档配置 PATH），确保 `conda --version` 可用。

### 2.2 创建并激活 Python 环境

```bash
conda create -n teleop python=3.11 -y
conda activate teleop
```

> 后续所有命令均在 `teleop` 环境中执行。

### 2.3 安装依赖

```bash
cd /path/to/Teleoperation-SDK
pip install .
```

### 2.4 验证安装

```bash
python -c "
import torch, numpy, matplotlib, mujoco, alicia_d_sdk, serial
print('torch       ', torch.__version__)
print('numpy       ', numpy.__version__)
print('matplotlib  ', matplotlib.__version__)
print('mujoco      ', mujoco.__version__)
print('alicia_d_sdk  OK')
print('pyserial      OK')
print()
print('All dependencies installed successfully!')
"
```

所有行均无报错即配置完成。

### 2.5 串口权限（首次需要）

```bash
sudo usermod -aG dialout $USER
```

执行后 **注销并重新登录** 使权限生效，否则会遇到 `Permission denied: '/dev/ttyACM0'`。

---

## 3. 运行

> **前提**：确保已激活正确的 conda 环境 (`conda activate teleop`)，且 Leader 机械臂已通过 USB 连接。

### Demo 01 — 实时关节监控

```bash
python 01_demo_realtime_joint_plot.py
```

弹出 matplotlib 窗口，实时显示 6 轴关节角度曲线和手柄输入状态。

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | `/dev/ttyACM0` | Leader 串口 |
| `--variant` | `leader` | 机型变体 |
| `--fps` | `20` | 采样频率 (Hz) |
| `--plot_fps` | `15` | 绘图刷新率 (Hz) |
| `--history_sec` | `15` | 时间窗口 (秒) |
| `--format` | `deg` | 角度单位：`deg` / `rad` |
| `--disable_torque` | — | 启动时关闭力矩（需手持机械臂） |

**示例：**

```bash
# 使用第二个串口，关闭力矩，30秒时间窗口
python 01_demo_realtime_joint_plot.py --port /dev/ttyACM1 --disable_torque --history_sec 30
```

### Demo 02 — MuJoCo Follower 遥操作

```bash
python 02_demo_mujoco_follower.py
```

弹出 MuJoCo 3D 仿真窗口，Leader 机械臂的动作实时映射到虚拟 Follower 上。

**操作方式：**

| 操作 | 效果 |
|------|------|
| **按住左键** | 启用遥操作（死人开关） |
| **松开左键** | Follower 冻结在当前姿态 |
| **扳机（模拟量）** | 控制夹爪开合 |
| **按 R** | 热重载 MuJoCo XML 模型 |
| **关闭窗口** | 退出程序 |

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | `/dev/ttyACM0` | Leader 串口 |
| `--variant` | `leader` | 机型变体 |
| `--fps` | `50` | Leader 轮询频率 (Hz) |
| `--xml` | `assets/.../alicia_d_follower.xml` | MuJoCo 模型路径（默认使用仓库内置模型） |
| `--disable_torque` | — | 启动时关闭力矩 |
| `--debug` | — | 打印调试信息 |

**示例：**

```bash
# 关闭力矩 + 调试模式
python 02_demo_mujoco_follower.py --disable_torque --debug

# 使用自定义 XML 模型
python 02_demo_mujoco_follower.py --xml /path/to/custom_model.xml
```

### Teleoperation Residual RL（MuJoCo 第一阶段）

这部分不是用 BC policy 当 base，而是围绕 `human teleop + residual copilot` 设计：

```text
commanded_action = clip(base_action + residual_action)
```

环境是仓库内新增的 `teleop_sdk.envs.MujocoPickPlaceTeleopEnv`，第一版任务为单臂 pick-place，状态里显式包含：

- robot proprioception
- end-effector pose
- box / basket pose

这里建议直接采用 ResFiT 风格的数据语义：

- `observation.state` / `observation_state`：低维状态向量
- `base_action`：唯一的基础动作语义。对 teleop 来说，它就是人类当前输入，不再单独并列保留一个 `human_action`
- `residual_action`：copilot 输出
- `realized_action`：控制器和仿真真正执行出来的动作

#### 1. 采集纯 teleop 数据

```bash
python record_teleop.py --output_dir logs/teleop_dataset
```

每个 episode 会保存成一个 `episode_XXXX.npz`，字段包括：

- `observation_state`
- `next_observation_state`
- `base_action`
- `next_base_action`
- `residual_action`
- `realized_action`
- `reward_env`
- `terminated`
- `truncated`
- `base_joint_target`
- `base_gripper_target`

#### 2. 训练 residual TD3

```bash
python train_residual_td3_teleop.py \
  --dataset_dir logs/teleop_dataset \
  --save_path logs/residual_td3_teleop.pt
```

训练流程固定为：

1. 构建 offline replay
2. 用 exploration residual 把 online replay 填到 `learning_starts`
3. 做 `critic_warmup_steps` 次 critic-only update
4. 进入 rollout + mixed replay update

当前实现按设计文档使用 differential critic：

- `Q_ast(x_t, a_t^{assist})`
- `Q_bas(x_t, b_t)`

actor 目标是最大化 `Q_ast - Q_bas`，不再包含额外正则项。

reward 只保留环境任务回报：

- `reward_env`

#### 3. 评估 residual copilot

```bash
python eval_residual_td3_teleop.py \
  --dataset_dir logs/teleop_dataset \
  --checkpoint logs/residual_td3_teleop.pt
```

评估会同时输出：

- human-only baseline
- assisted residual policy

用于对比：

- success rate
- average return

---

## 4. 常见问题

### 启动报错

| 报错信息 | 原因 | 解决方法 |
|----------|------|----------|
| `ImportError: torch_utils 需要 PyTorch` | 未安装 torch | 在仓库目录执行 `pip install .`（或 `pip install -e .`） |
| `ModuleNotFoundError: No module named 'torch'` | 同上 | 同上 |
| `ModuleNotFoundError: No module named 'mujoco'` | 未安装 mujoco | 同上 |
| `ModuleNotFoundError: No module named 'alicia_d_sdk'` | SDK 未安装 | 同上 |
| `Permission denied: '/dev/ttyACM0'` | 串口权限不足 | 见 [2.5 串口权限](#25-串口权限首次需要) |
| `device disconnected or multiple access on port` | 机械臂未连接 / 串口被占用 | 检查 USB 连接，确认无其他程序占用串口 |

### 其他问题

**Q: 不连接机械臂能运行 Demo 02 吗？**

可以启动，MuJoCo 窗口会正常显示，但终端会持续打印串口错误，且无法控制虚拟机械臂。

**Q: 支持哪些 `--variant` 选项？**

`leader`（默认）、`gripper_50mm`、`gripper_100mm`、`leader_ur`、`vertical_50mm`。

**Q: 如何自定义 MuJoCo 场景？**

编辑 `assets/mujoco/Alicia_D_v5_6/gripper_50mm/alicia_d_follower.xml`，
运行时按 **R** 键热重载，无需重启程序。

---

## 5. 依赖关系

```
Teleoperation-SDK
├── numpy
├── matplotlib        ← Demo 01 绑图
├── mujoco ≥ 3.0      ← Demo 02 仿真
├── pyserial           ← 串口通信
└── alicia_d_sdk       ← 机器人 SDK（本地源码安装）
     └── synria-robocore
          └── torch    ← 间接依赖（import 时强制加载）
```

---

## 许可证

GNU General Public License v3.0 — 详见各源文件头部声明。

© 2026 Synria Robotics Co., Ltd.
