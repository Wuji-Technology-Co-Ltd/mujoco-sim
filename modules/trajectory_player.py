#!/usr/bin/env python3
"""
Trajectory player - used to load and play predefined joint trajectories.
Supports `.npy` format.
"""
import numpy as np
import mujoco
import time
from pathlib import Path
from typing import Union, Tuple


class TrajectoryPlayer:
    """Dexterous hand trajectory player."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        Initialize the trajectory player.

        Args:
            model: MuJoCo model
            data: MuJoCo data object
        """
        self.model = model
        self.data = data
        self.num_actuators = model.nu

        # Log the actual actuator count
        print(f"Info: model has {self.num_actuators} actuators")

    def load_trajectory_from_numpy(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, float]:
        """
        Load trajectory from .npy files.

        Args:
            filepath: path to trajectory file

        Returns:
            trajectory: array with shape (T, num_actuators)
            dt: time step in seconds
        """
        filepath = Path(filepath)

        if filepath.suffix.lower() != '.npy':
            raise ValueError(f"Only .npy trajectory files are supported. Got: {filepath.suffix}")

        # load as ndarray
        trajectory = np.load(filepath, allow_pickle=False)
        dt = 0.02  # default timestep (50 Hz)

        if trajectory.ndim != 2 or trajectory.shape[1] != self.num_actuators:
            raise ValueError(f"Trajectory shape mismatch: {trajectory.shape}, expected (T, {self.num_actuators})")

        print(f"Loaded trajectory: {trajectory.shape[0]} frames, dt={dt}s")
        return trajectory, dt



    def load_trajectory(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, float]:
        """
        Load a trajectory file. Only .npy is supported.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Trajectory file not found: {filepath}")

        if filepath.suffix.lower() != '.npy':
            raise ValueError(f"Unsupported file format: {filepath.suffix}. Only .npy is supported.")

        return self.load_trajectory_from_numpy(filepath)

    def set_joint_positions(self, positions: np.ndarray):
        """
        Set joint target positions.

        Args:
            positions: 1D array with length num_actuators
        """
        if positions.shape[0] != self.num_actuators:
            raise ValueError(f"Position array length mismatch: {positions.shape[0]}, expected {self.num_actuators}")

        # Directly set control targets
        self.data.ctrl[:] = positions

    def play_trajectory(self, trajectory: np.ndarray, dt: float,
                       loop: bool = False, speed: float = 1.0,
                       verbose: bool = True):
        """
        Play trajectory in blocking mode (for non-GUI usage).

        Args:
            trajectory: (T, num_actuators) trajectory array
            dt: timestep in seconds
            loop: whether to loop playback
            speed: playback speed multiplier (1.0 = normal)
            verbose: print progress if True
        """
        actual_dt = dt / speed
        num_frames = trajectory.shape[0]

        if verbose:
            print(f"Starting playback: {num_frames} frames, duration {num_frames * dt:.2f}s")

        frame_idx = 0
        try:
            while True:
                start_time = time.time()

                # Apply positions
                self.set_joint_positions(trajectory[frame_idx])

                # Step simulator
                mujoco.mj_step(self.model, self.data)

                # Control playback timing
                elapsed = time.time() - start_time
                sleep_time = max(0, actual_dt - elapsed)
                time.sleep(sleep_time)

                # Advance frame
                frame_idx += 1
                if frame_idx >= num_frames:
                    if loop:
                        frame_idx = 0
                        if verbose:
                            print("Looping trajectory...")
                    else:
                        if verbose:
                            print("Playback finished")
                        break

        except KeyboardInterrupt:
            if verbose:
                print("\nPlayback interrupted")

    def get_trajectory_generator(self, trajectory: np.ndarray, dt: float,
                                 loop: bool = False, speed: float = 1.0):
        """
        Return a generator for trajectory frames (used with viewer integration).

        Args:
            trajectory: (T, num_actuators) trajectory array
            dt: timestep in seconds
            loop: whether to loop playback
            speed: playback speed multiplier

        Yields:
            1D array of joint positions (length num_actuators)
        """
        actual_dt = dt / speed
        num_frames = trajectory.shape[0]
        frame_idx = 0
        last_time = time.time()

        while True:
            current_time = time.time()
            if current_time - last_time >= actual_dt:
                yield trajectory[frame_idx]
                last_time = current_time
                frame_idx += 1
                if frame_idx >= num_frames:
                    if loop:
                        frame_idx = 0
                    else:
                        return
            else:
                # Avoid busy-waiting
                time.sleep(min(0.001, actual_dt / 10.0))


def create_sample_trajectories(output_dir: Union[str, Path] = "trajectories"):
    """
    Create sample trajectory files. Saves only .npy files (keeps loader backward-compatible).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Creating sample trajectories in: {output_dir}")

    # Wave gesture: fingers curl in sequence
    def generate_wave_trajectory(num_frames=150):
        """Generate a wave trajectory"""
        trajectory = np.zeros((num_frames, 20))

        for i in range(num_frames):
            for finger in range(5):
                # Each finger delayed by 1/5 of the cycle
                phase = (i / num_frames - finger * 0.2) * 2 * np.pi
                flex = (np.sin(phase) + 1) / 2  # ranges from 0 to 1

                trajectory[i, finger*4 + 2] = 0.8 * flex
                trajectory[i, finger*4 + 3] = 0.8 * flex

        return trajectory

    traj = generate_wave_trajectory()

    # Save only .npy file
    np.save(output_dir / "wave.npy", traj)

    print(f"  - Created wave trajectory ({traj.shape[0]} frames) in {output_dir}")
    print("Sample trajectories creation complete!")


if __name__ == "__main__":
    print("Trajectory player module")
    print("Creating sample trajectories...")
    create_sample_trajectories()