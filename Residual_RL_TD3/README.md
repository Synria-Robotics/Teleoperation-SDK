# Residual_RL_TD3

这个模块现在按训练使用方式收敛成两套环境：

- `env/mujoco_pick_place_env.py`
  MuJoCo 训练/评估环境
- `env/real_pick_place_env.py`
  真机训练/评估环境
- `scripts/train_residual_td3_teleop.py`
  residual TD3 训练脚本

`env/` 里不再保留额外的包装层。

依赖边界：

- 允许依赖 `Alicia-D-SDK`
- 不依赖 `teleop_sdk`
- `env/mujoco_pick_place_env.py` 直接本地实现，不依赖 `Alicia-D-Mujoco-SDK`

最小用法：

```python
from Residual_RL_TD3.env import MujocoPickPlaceEnv

env = MujocoPickPlaceEnv()
obs, info = env.reset()

step = env.step(base_action, residual_action)
```

训练示例：

```bash
python Residual_RL_TD3/scripts/train_residual_td3_teleop.py \
  --dataset_dir logs/teleop_dataset \
  --save_path logs/residual_td3_teleop.pt \
  --fixed_scene
```
