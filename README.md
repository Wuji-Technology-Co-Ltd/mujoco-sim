# WujiHand Mujoco仿真与控制

在Mujoco仿真平台里查看和控制 WujiHand

## 环境设置
```bash
pip install mujoco numpy
```
## `demo_sim.py`启动仿真

可在Mujoco GUI 右侧control项控制关节运动

```
python demo_sim.py
```

默认是打开右手的模型，如需指定左手，可以在命令行输入参数

```bash
python demo_sim.py -s left
# or
python demo_sim.py --side left
```

## `demo_trajectory.py` 加载预定运动轨迹

```
python demo_trajectory.py -t trajectory/fist.npz
python demo_trajectory.py -t trajectory/pinch.npz
python demo_trajectory.py -t trajectory/wave.npz
```
