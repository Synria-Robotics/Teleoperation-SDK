# Teleoperation-SDK

Alicia-D 机械臂遥操作独立工具包。提供两个即用示例：

| Demo | 功能 |
|------|------|
| **Demo 01** — `01_demo_realtime_joint_plot.py` | 实时关节角度绘图 + 手柄输入状态面板 |
| **Demo 02** — `02_demo_mujoco_follower.py` | Leader → MuJoCo Follower 遥操作仿真 |

---

## 目录结构

```
Teleoperation-SDK/
├── 01_demo_realtime_joint_plot.py       # Demo 01：实时关节监控
├── 02_demo_mujoco_follower.py           # Demo 02：MuJoCo 遥操作
├── assets/
│   └── mujoco/
│       └── Alicia_D_v5_6/
│           └── gripper_50mm/
│               ├── alicia_d_follower.xml    # MuJoCo MJCF 模型
│               ├── base_link.STL            # 机器人 mesh 文件
│               ├── link1.STL ~ link6.STL
│               ├── left_gripper_50mm.STL
│               └── right_gripper_50mm.STL
├── teleop_utils/
│   ├── __init__.py
│   └── fps_utils.py                     # precise_sleep 高精度定时
├── pyproject.toml
├── LICENSE
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

### 2.2 安装步骤

获取代码和相应 examples 例程：

```bash
git clone https://github.com/Synria-Robotics/Teleoperation-SDK.git
cd Teleoperation-SDK

# 2. 创建 Python 环境（推荐使用 Conda）
conda create -n teleop python=3.11 -y
conda activate teleop
```

#### 方法一：从源码安装（开发模式）

如果您需要修改源码或参与开发：

```bash
cd Teleoperation-SDK
pip install -e .
```

#### 方法二：从 PyPI 安装

```bash
pip install teleoperation-sdk
```

### 2.3 验证安装

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

### 2.4 串口权限（首次需要）

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

---

## 4. 常见问题

### 启动报错

| 报错信息 | 原因 | 解决方法 |
|----------|------|----------|
| `ImportError: torch_utils 需要 PyTorch` | 未安装 torch | 在仓库目录执行 `pip install .`（或 `pip install -e .`） |
| `ModuleNotFoundError: No module named 'torch'` | 同上 | 同上 |
| `ModuleNotFoundError: No module named 'mujoco'` | 未安装 mujoco | 同上 |
| `ModuleNotFoundError: No module named 'alicia_d_sdk'` | SDK 未安装 | 同上 |
| `Permission denied: '/dev/ttyACM0'` | 串口权限不足 | 见 [2.4 串口权限](#24-串口权限首次需要) |
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
