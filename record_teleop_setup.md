# `record_teleop.py` 环境安装与使用

本文档面向 `record_teleop.py`，用于说明环境安装、依赖验证和常用启动方式。

## 1. 适用场景

`record_teleop.py` 用于采集 Alicia-D Leader 遥操作数据，并可选开启：

- MuJoCo 可视化录制
- residual assist 推理
- 每次保存 episode 后实时训练 residual policy

## 2. 环境要求

- 操作系统：Ubuntu 20.04/22.04 优先，macOS 也可用
- Python：3.11 推荐
- 硬件：Alicia-D Leader 机械臂
- 串口：默认 `/dev/ttyACM0`
- GPU：可选，仅影响 PyTorch 训练速度

说明：

- 有界面模式依赖 `mujoco.viewer`
- macOS 下开启 viewer 时，脚本会自动尝试用 `mjpython` 重启
- 无界面模式不需要 MuJoCo viewer，但仍需要 Leader 硬件输入

## 3. 安装步骤

### 3.1 创建 Conda 环境

```bash
conda create -n teleop python=3.11 -y
conda activate teleop
```

### 3.2 安装 Alicia-D SDK

在仓库根目录执行：

```bash
pip install -e Alicia-D-SDK
```

### 3.3 安装当前项目

```bash
pip install -e .
```

如果你使用 NVIDIA GPU，也可以先按自己的 CUDA 版本安装对应的 `torch`，再执行上面的命令。

### 3.4 验证依赖

```bash
python -c "
import mujoco, numpy, torch, serial, alicia_d_sdk
print('mujoco      ', mujoco.__version__)
print('numpy       ', numpy.__version__)
print('torch       ', torch.__version__)
print('pyserial    OK')
print('alicia_d_sdk OK')
"
```

如果没有报错，说明基础环境已经可用。

### 3.5 Linux 串口权限

首次使用建议执行：

```bash
sudo usermod -aG dialout $USER
```

执行后重新登录终端会话，否则可能出现串口权限不足。

## 4. 快速开始

### 4.1 最简单的可视化录制

```bash
python record_teleop.py --output_dir logs/teleop_dataset
```

启动后会打开 MuJoCo 窗口，常用按键如下：

- `SPACE`：开始录制当前场景
- `O`：切换 residual assist 开关
- `S`：保存当前录制
- `Q`：丢弃当前录制并刷新场景

### 4.2 无界面录制

```bash
python record_teleop.py --headless --output_dir logs/teleop_dataset
```

默认带 deadman 逻辑：

- 按下 Leader 激活信号后开始一轮录制
- 松开激活信号后结束当前 episode

如果你不想使用 deadman，可以加：

```bash
python record_teleop.py --headless --no_deadman
```

## 5. 常用命令

### 5.1 固定场景采集

```bash
python record_teleop.py \
  --output_dir logs/teleop_dataset \
  --fixed_scene
```

### 5.2 指定 box 初始位置

```bash
python record_teleop.py \
  --output_dir logs/teleop_dataset \
  --fixed_box_xy 0.10 0.15
```

### 5.3 加载已有 residual checkpoint 辅助录制

```bash
python record_teleop.py \
  --output_dir logs/teleop_dataset \
  --checkpoint logs/residual_td3_live.pt \
  --enable-residual-on-start
```

### 5.4 边录制边实时训练

```bash
python record_teleop.py \
  --output_dir logs/teleop_dataset \
  --realtime-train \
  --train-save-path logs/residual_td3_live.pt
```

### 5.5 查看详细 HUD 信息

```bash
python record_teleop.py \
  --output_dir logs/teleop_dataset \
  --show-overlay-details
```

## 6. 常用参数

| 参数 | 说明 |
|------|------|
| `--output_dir` | episode `.npz` 输出目录 |
| `--xml` | MuJoCo XML 路径 |
| `--num_episodes` | 本次目标录制数量 |
| `--max_steps` | 单个 episode 最大步数 |
| `--control_dt` | 控制周期，默认 `0.05` 秒 |
| `--port` | Leader 串口，默认 `/dev/ttyACM0` |
| `--variant` | Leader 机型，默认 `leader` |
| `--gripper_type` | 夹爪类型，默认 `50mm` |
| `--checkpoint` | residual policy checkpoint 路径 |
| `--enable-residual-on-start` | 启动时直接开启 residual assist |
| `--residual-limit` | residual 单维裁剪范围 |
| `--residual-scale` | residual 输出缩放系数 |
| `--realtime-train` | 每保存一个 episode 后立即训练 |
| `--headless` | 关闭 MuJoCo viewer，只在终端录制 |

## 7. 输出内容

默认输出目录为：

```bash
logs/teleop_dataset
```

保存文件格式为：

```text
episode_0000.npz
episode_0001.npz
episode_0002.npz
...
```

如果启用了实时训练，默认 checkpoint 会写到：

```bash
logs/residual_td3_live.pt
```

## 8. 常见问题

### 8.1 `ModuleNotFoundError: No module named 'alicia_d_sdk'`

执行：

```bash
pip install -e Alicia-D-SDK
```

### 8.2 `ModuleNotFoundError: No module named 'mujoco'`

执行：

```bash
pip install mujoco
```

或者重新执行：

```bash
pip install -e .
```

### 8.3 `Permission denied: '/dev/ttyACM0'`

说明当前用户没有串口权限，参考上面的 `dialout` 配置。

### 8.4 macOS 下 viewer 无法启动

确认当前环境里可以找到 `mjpython`：

```bash
which mjpython
```

如果没有，一般重新安装 `mujoco` 后即可。

## 9. 推荐启动顺序

1. `conda activate teleop`
2. 连接 Leader 机械臂
3. 先运行纯录制模式确认串口和场景正常
4. 再尝试 `--checkpoint` 或 `--realtime-train`

