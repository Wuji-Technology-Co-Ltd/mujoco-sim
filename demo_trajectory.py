#!/usr/bin/env python3
"""
演示程序：加载并播放预定义轨迹
"""
import argparse
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
import sys

# 导入轨迹播放器
from modules.trajectory_player import TrajectoryPlayer, create_sample_trajectories


def parse_args():
    parser = argparse.ArgumentParser(description="WujiHand 轨迹播放演示")
    parser.add_argument(
        "-s", "--side",
        choices=["left", "right"],
        default="right",
        help="选择左手或右手模型（默认: right）"
    )
    parser.add_argument(
        "-t", "--trajectory",
        type=str,
        default="trajectories/fist.npz",
        help="轨迹文件路径（默认: trajectories/fist.npz）"
    )
    parser.add_argument(
        "-l", "--loop",
        action="store_true",
        help="循环播放轨迹"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="播放速度倍数（默认: 1.0）"
    )
    parser.add_argument(
        "--nogui",
        action="store_true",
        help="无GUI模式（仅仿真，不打开viewer）"
    )
    parser.add_argument(
        "--create-samples",
        action="store_true",
        help="创建示例轨迹文件后退出"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 如果只是创建示例轨迹
    if args.create_samples:
        create_sample_trajectories()
        return
    
    print(f"WujiHand 轨迹播放演示 - {args.side}手")
    
    # 加载MuJoCo模型
    mjcf_path = Path(__file__).parent / "xml" / f"{args.side}.xml"
    
    if not mjcf_path.exists():
        print(f"错误: 模型文件不存在: {mjcf_path}")
        return
    
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    
    # 初始化到中位
    print("初始化关节位置...")
    for i in range(model.nu):
        if model.actuator_ctrllimited[i]:
            ctrl_range = model.actuator_ctrlrange[i]
            data.ctrl[i] = (ctrl_range[0] + ctrl_range[1]) / 2
        else:
            data.ctrl[i] = 0.0
    
    # 预热仿真
    for _ in range(100):
        mujoco.mj_step(model, data)
    
    # 创建轨迹播放器
    player = TrajectoryPlayer(model, data)
    
    # 加载轨迹
    traj_path = Path(args.trajectory)
    if not traj_path.exists():
        print(f"错误: 轨迹文件不存在: {traj_path}")
        print("提示: 运行 'python demo_trajectory.py --create-samples' 创建示例轨迹")
        return
    
    try:
        trajectory, dt = player.load_trajectory(traj_path)
    except Exception as e:
        print(f"加载轨迹失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"轨迹时长: {trajectory.shape[0] * dt:.2f}秒")
    print(f"播放设置: 速度={args.speed}x, 循环={'是' if args.loop else '否'}")
    
    if args.nogui:
        # 无GUI模式：阻塞式播放
        print("开始播放（无GUI）...")
        player.play_trajectory(trajectory, dt, loop=args.loop, speed=args.speed)
    else:
        # 带GUI模式：使用viewer
        print("启动MuJoCo Viewer...")
        print("提示: 在viewer中可以暂停/恢复、调整视角")
        
        # 使用生成器控制轨迹
        traj_gen = player.get_trajectory_generator(trajectory, dt, 
                                                   loop=args.loop, speed=args.speed)
        
        # 自定义仿真循环
        frame_count = 0
        
        def controller(model, data):
            """在每个仿真步调用"""
            nonlocal frame_count
            try:
                positions = next(traj_gen)
                data.ctrl[:] = positions
                frame_count += 1
            except StopIteration:
                print(f"\n轨迹播放完成 ({frame_count} 帧)")
                # 保持最后的姿态
                pass
        
        # 启动viewer并注入控制器
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("Viewer已启动，按ESC退出")
            
            while viewer.is_running():
                step_start = data.time
                
                # 调用控制器
                controller(model, data)
                
                # 仿真步进
                mujoco.mj_step(model, data)
                
                # 同步viewer
                viewer.sync()
                
                # 控制实时播放速度
                time_until_next_step = model.opt.timestep - (data.time - step_start)
                if time_until_next_step > 0:
                    import time
                    time.sleep(time_until_next_step)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序已停止")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)