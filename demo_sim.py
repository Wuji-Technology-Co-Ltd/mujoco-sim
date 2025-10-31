#!/usr/bin/env python3
"""
Demo launcher: load and run the WujiHand model in MuJoCo.
"""
import mujoco
import mujoco.viewer
from pathlib import Path
import argparse

def initialize_safe_control(model, data):
    """
    Initialize actuator controls to a safe mid-range position and
    step the simulator briefly to apply the values.
    """
    # Set actuators to mid-point when control range is limited, otherwise zero.
    for i in range(model.nu):
        if model.actuator_ctrllimited[i]:
            ctrl_range = model.actuator_ctrlrange[i]
            data.ctrl[i] = (ctrl_range[0] + ctrl_range[1]) / 2
        else:
            data.ctrl[i] = 0.0

    # Warm up the simulation for a few steps
    for _ in range(100):
        mujoco.mj_step(model, data)

    print("Control targets initialized")

def parse_args():
    parser = argparse.ArgumentParser(description="WujiHand MuJoCo simulation launcher")
    parser.add_argument(
        "-s", "--side",
        choices=["left", "right"],
        default="right",
        help="Which hand model to load (default: right)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    print("WujiHand MuJoCo simulation launcher")
    mjcf_path = Path(__file__).parent / "xml" / f"{args.side}.xml"

    if not mjcf_path.exists():
        print(f"Error: model file not found: {mjcf_path}")
        print("Ensure repository contains xml/left.xml or xml/right.xml or provide a correct path.")
        return

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)

    initialize_safe_control(model, data)
    print("Launching interactive viewer")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
