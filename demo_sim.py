#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
import argparse

def initialize_safe_control(model, data):
        
        # 设置关节位置为中位
        for i in range(model.nu):
            if model.actuator_ctrllimited[i]:
                ctrl_range = model.actuator_ctrlrange[i]
                data.ctrl[i] = (ctrl_range[0] + ctrl_range[1]) / 2
            else:
                data.ctrl[i] = 0.0

        for _ in range(100):
            mujoco.mj_step(model, data)
            
        print("控制目标初始化完成")

def parse_args():
    parser = argparse.ArgumentParser(description="WujiHand MuJoCo 仿真启动器")
    parser.add_argument(
        "-s", "--side",
        choices=["left", "right"],
        default="right",
        help="选择加载的手模型 xml（默认: right）"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    print("WujiHand MuJoCo仿真启动器")
    mjcf_path = Path(__file__).parent / "xml" / f"{args.side}.xml"

    if not mjcf_path.exists():
        print(f"错误: 模型文件不存在: {mjcf_path}")
        print("请确认仓库结构包含 xml/left.xml 或 xml/right.xml，或使用正确路径。")
        return
    
    # 加载MuJoCo模型
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    
    initialize_safe_control(model, data)
    print("启动交互viewer")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
