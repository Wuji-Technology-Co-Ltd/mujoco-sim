
#!/usr/bin/env python3
"""
轨迹播放器 - 用于加载和播放预定义的关节轨迹
支持多种轨迹格式：numpy数组、CSV、JSON等
"""
import numpy as np
import mujoco
import time
from pathlib import Path
from typing import Union, List, Tuple
import json


class TrajectoryPlayer:
    """灵巧手轨迹播放器"""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        初始化轨迹播放器
        
        Args:
            model: MuJoCo模型
            data: MuJoCo数据对象
        """
        self.model = model
        self.data = data
        self.num_actuators = model.nu
        
        # 验证模型是否符合预期（20个执行器）
        if self.num_actuators != 20:
            print(f"警告: 模型有 {self.num_actuators} 个执行器，预期为20")
    
    def load_trajectory_from_numpy(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, float]:
        """
        从.npy或.npz文件加载轨迹
        
        Args:
            filepath: 轨迹文件路径
            
        Returns:
            trajectory: shape=(T, 20)的轨迹数组
            dt: 时间步长（秒）
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.npy':
            trajectory = np.load(filepath)
            dt = 0.02  # 默认50Hz
        elif filepath.suffix == '.npz':
            data = np.load(filepath)
            trajectory = data['positions']
            dt = float(data.get('dt', 0.02))
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")
        
        # 验证形状
        if trajectory.ndim != 2 or trajectory.shape[1] != 20:
            raise ValueError(f"轨迹形状错误: {trajectory.shape}，期望 (T, 20)")
        
        print(f"已加载轨迹: {trajectory.shape[0]} 帧, dt={dt}s")
        return trajectory, dt
    
    def load_trajectory_from_csv(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, float]:
        """
        从CSV文件加载轨迹
        CSV格式: 每行20列代表20个关节位置，第一行可选时间步
        """
        filepath = Path(filepath)
        data = np.loadtxt(filepath, delimiter=',')
        
        # 检查第一行是否是单个数字（时间步）
        if data.shape[0] > 1 and data.shape[1] == 1:
            dt = float(data[0, 0])
            trajectory = data[1:]
        else:
            dt = 0.02
            trajectory = data
        
        if trajectory.shape[1] != 20:
            raise ValueError(f"CSV列数错误: {trajectory.shape[1]}，期望20列")
        
        print(f"已加载轨迹: {trajectory.shape[0]} 帧, dt={dt}s")
        return trajectory, dt
    
    def load_trajectory_from_json(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, float]:
        """
        从JSON文件加载轨迹
        JSON格式: {"dt": 0.02, "positions": [[...], [...], ...]}
        """
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        trajectory = np.array(data['positions'])
        dt = data.get('dt', 0.02)
        
        if trajectory.shape[1] != 20:
            raise ValueError(f"轨迹形状错误: {trajectory.shape}，期望 (T, 20)")
        
        print(f"已加载轨迹: {trajectory.shape[0]} 帧, dt={dt}s")
        return trajectory, dt
    
    def load_trajectory(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, float]:
        """
        自动识别格式并加载轨迹
        
        Args:
            filepath: 轨迹文件路径
            
        Returns:
            trajectory: shape=(T, 20)的轨迹数组
            dt: 时间步长（秒）
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"轨迹文件不存在: {filepath}")
        
        suffix = filepath.suffix.lower()
        
        if suffix in ['.npy', '.npz']:
            return self.load_trajectory_from_numpy(filepath)
        elif suffix == '.csv':
            return self.load_trajectory_from_csv(filepath)
        elif suffix == '.json':
            return self.load_trajectory_from_json(filepath)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
    
    def set_joint_positions(self, positions: np.ndarray):
        """
        设置关节目标位置
        
        Args:
            positions: shape=(20,)的关节位置数组
        """
        if positions.shape[0] != self.num_actuators:
            raise ValueError(f"位置数组长度错误: {positions.shape[0]}，期望 {self.num_actuators}")
        
        # 直接设置控制目标
        self.data.ctrl[:] = positions
    
    def play_trajectory(self, trajectory: np.ndarray, dt: float, 
                       loop: bool = False, speed: float = 1.0,
                       verbose: bool = True):
        """
        播放轨迹（阻塞式，用于非GUI模式）
        
        Args:
            trajectory: shape=(T, 20)的轨迹数组
            dt: 时间步长（秒）
            loop: 是否循环播放
            speed: 播放速度倍数（1.0=正常速度，2.0=2倍速）
            verbose: 是否打印进度
        """
        actual_dt = dt / speed
        num_frames = trajectory.shape[0]
        
        if verbose:
            print(f"开始播放轨迹: {num_frames} 帧, 持续 {num_frames * dt:.2f}秒")
        
        frame_idx = 0
        try:
            while True:
                start_time = time.time()
                
                # 设置目标位置
                self.set_joint_positions(trajectory[frame_idx])
                
                # 仿真步进
                mujoco.mj_step(self.model, self.data)
                
                # 控制播放速度
                elapsed = time.time() - start_time
                sleep_time = max(0, actual_dt - elapsed)
                time.sleep(sleep_time)
                
                # 更新帧索引
                frame_idx += 1
                if frame_idx >= num_frames:
                    if loop:
                        frame_idx = 0
                        if verbose:
                            print("轨迹循环播放...")
                    else:
                        if verbose:
                            print("轨迹播放完成")
                        break
                
        except KeyboardInterrupt:
            if verbose:
                print("\n播放已停止")
    
    def get_trajectory_generator(self, trajectory: np.ndarray, dt: float,
                                loop: bool = False, speed: float = 1.0):
        """
        返回轨迹生成器（用于与viewer集成）
        
        Args:
            trajectory: shape=(T, 20)的轨迹数组
            dt: 时间步长（秒）
            loop: 是否循环播放
            speed: 播放速度倍数
            
        Yields:
            关节位置数组 shape=(20,)
        """
        actual_dt = dt / speed
        num_frames = trajectory.shape[0]
        frame_idx = 0
        last_time = time.time()
        
        while True:
            current_time = time.time()
            
            # 检查是否到达下一帧的时间
            if current_time - last_time >= actual_dt:
                yield trajectory[frame_idx]
                last_time = current_time
                
                frame_idx += 1
                if frame_idx >= num_frames:
                    if loop:
                        frame_idx = 0
                    else:
                        break
            else:
                # 保持当前帧
                yield trajectory[frame_idx]


def create_sample_trajectories(output_dir: Union[str, Path] = "trajectories"):
    """
    创建示例轨迹文件
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"创建示例轨迹到: {output_dir}")
    
    # 1. 握拳动作（从展开到握紧）
    def generate_fist_trajectory(num_frames=100):
        """生成握拳轨迹"""
        trajectory = np.zeros((num_frames, 20))
        
        for i in range(num_frames):
            t = i / (num_frames - 1)  # 0到1
            
            # 每根手指的4个关节
            for finger in range(5):
                # joint1 (abduction): 保持中位
                trajectory[i, finger*4 + 0] = 0.3 * t if finger == 0 else 0.0
                # joint2 (proximal flex): 从0到0.4
                trajectory[i, finger*4 + 1] = 0.4 * t
                # joint3 (middle flex): 从0到1.0
                trajectory[i, finger*4 + 2] = 1.0 * t
                # joint4 (distal flex): 从0到1.2
                trajectory[i, finger*4 + 3] = 1.2 * t
        
        return trajectory
    
    # 2. 捏取动作（拇指和食指）
    def generate_pinch_trajectory(num_frames=80):
        """生成捏取轨迹"""
        trajectory = np.zeros((num_frames, 20))
        
        for i in range(num_frames):
            t = i / (num_frames - 1)
            
            # 拇指 (finger 0)
            trajectory[i, 0] = 0.8 * t  # abduction
            trajectory[i, 1] = 0.3 * t  # flex
            trajectory[i, 2] = 0.5 * t
            trajectory[i, 3] = 0.4 * t
            
            # 食指 (finger 1)
            trajectory[i, 4] = -0.2 * t  # 向拇指靠拢
            trajectory[i, 5] = 0.0
            trajectory[i, 6] = 0.6 * t
            trajectory[i, 7] = 0.5 * t
            
            # 其他手指微微弯曲
            for finger in range(2, 5):
                trajectory[i, finger*4 + 2] = 0.3 * t
                trajectory[i, finger*4 + 3] = 0.3 * t
        
        return trajectory
    
    # 3. 波浪手势（手指依次弯曲）
    def generate_wave_trajectory(num_frames=150):
        """生成波浪轨迹"""
        trajectory = np.zeros((num_frames, 20))
        
        for i in range(num_frames):
            for finger in range(5):
                # 每根手指延迟1/5周期
                phase = (i / num_frames - finger * 0.2) * 2 * np.pi
                flex = (np.sin(phase) + 1) / 2  # 0到1
                
                trajectory[i, finger*4 + 2] = 0.8 * flex
                trajectory[i, finger*4 + 3] = 0.8 * flex
        
        return trajectory
    
    # 保存为不同格式
    trajectories = {
        'fist': generate_fist_trajectory(),
        'pinch': generate_pinch_trajectory(),
        'wave': generate_wave_trajectory()
    }
    
    for name, traj in trajectories.items():
        # NPZ格式（推荐，包含元数据）
        np.savez(output_dir / f"{name}.npz", 
                positions=traj, dt=0.02, description=name)
        
        # NPY格式（简单）
        np.save(output_dir / f"{name}.npy", traj)
        
        # CSV格式（可读性好）
        np.savetxt(output_dir / f"{name}.csv", traj, delimiter=',', 
                  fmt='%.6f', header='20 joint positions per row')
        
        # JSON格式（跨语言）
        with open(output_dir / f"{name}.json", 'w') as f:
            json.dump({
                'dt': 0.02,
                'description': name,
                'positions': traj.tolist()
            }, f, indent=2)
        
        print(f"  - 已创建 {name} 轨迹 ({traj.shape[0]} 帧)")
    
    print("示例轨迹创建完成！")


if __name__ == "__main__":
    print("轨迹播放器模块")
    print("创建示例轨迹...")
    create_sample_trajectories()