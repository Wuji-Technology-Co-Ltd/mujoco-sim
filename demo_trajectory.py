#!/usr/bin/env python3
"""
Demo: load and play predefined trajectories.
"""
import argparse
import mujoco
import mujoco.viewer
from pathlib import Path
import sys

# Import the trajectory player
from modules.trajectory_player import TrajectoryPlayer, create_sample_trajectories


def parse_args():
    parser = argparse.ArgumentParser(description="WujiHand trajectory playback demo")
    parser.add_argument(
        "-s", "--side",
        choices=["left", "right"],
        default="right",
        help="Which hand model to use (default: right)"
    )
    parser.add_argument(
        "-t", "--trajectory",
        type=str,
        default="trajectories/wave.npy",
        help="Path to trajectory file (default: trajectories/wave.npy)"
    )
    parser.add_argument(
        "-l", "--loop",
        action="store_true",
        help="Loop the trajectory playback"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--nogui",
        action="store_true",
        help="Run without GUI (simulation only)"
    )
    parser.add_argument(
        "--create-samples",
        action="store_true",
        help="Create sample trajectory files and exit"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # If requested, only create sample trajectories
    if args.create_samples:
        create_sample_trajectories()
        return

    print(f"WujiHand trajectory playback demo - {args.side} hand")

    # Load MuJoCo model
    mjcf_path = Path(__file__).parent / "xml" / f"{args.side}.xml"

    if not mjcf_path.exists():
        print(f"Error: model file not found: {mjcf_path}")
        return

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)

    # Initialize to mid positions
    print("Initializing joint positions...")
    for i in range(model.nu):
        if model.actuator_ctrllimited[i]:
            ctrl_range = model.actuator_ctrlrange[i]
            data.ctrl[i] = (ctrl_range[0] + ctrl_range[1]) / 2
        else:
            data.ctrl[i] = 0.0

    # Warm up simulation
    for _ in range(100):
        mujoco.mj_step(model, data)

    # Create the trajectory player
    player = TrajectoryPlayer(model, data)

    # Load trajectory file
    traj_path = Path(args.trajectory)
    if not traj_path.exists():
        print(f"Error: trajectory file not found: {traj_path}")
        print("Tip: run 'python demo_trajectory.py --create-samples' to create sample trajectories")
        return

    try:
        trajectory, dt = player.load_trajectory(traj_path)
    except Exception as e:
        print(f"Failed to load trajectory: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Trajectory duration: {trajectory.shape[0] * dt:.2f} seconds")
    print(f"Playback settings: speed={args.speed}x, loop={'yes' if args.loop else 'no'}")

    if args.nogui:
        # Non-GUI blocking playback
        print("Starting playback (nogui)...")
        player.play_trajectory(trajectory, dt, loop=args.loop, speed=args.speed)
    else:
        # GUI playback using viewer
        print("Launching MuJoCo Viewer...")
        print("Tip: you can pause/resume and change the view in the viewer")

        # Use a generator to control trajectory playback
        traj_gen = player.get_trajectory_generator(trajectory, dt,
                                                   loop=args.loop, speed=args.speed)

        # Custom simulation loop
        frame_count = 0

        def controller(model, data):
            """Called on each simulation step."""
            nonlocal frame_count
            try:
                positions = next(traj_gen)
                data.ctrl[:] = positions
                frame_count += 1
            except StopIteration:
                print(f"\nTrajectory playback finished ({frame_count} frames)")
                # Keep the final pose
                pass

        # Launch viewer and inject controller
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("Viewer started, press ESC to exit")

            while viewer.is_running():
                step_start = data.time

                # Call the controller
                controller(model, data)

                # Step the simulator
                mujoco.mj_step(model, data)

                # Sync viewer
                viewer.sync()

                # Control real-time playback speed
                time_until_next_step = model.opt.timestep - (data.time - step_start)
                if time_until_next_step > 0:
                    import time
                    time.sleep(time_until_next_step)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)