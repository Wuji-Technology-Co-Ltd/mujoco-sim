# mujoco-sim

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Simulation demo (MuJoCo): minimal demo for loading and controlling the Wuji Hand in MuJoCo simulator.

https://github.com/user-attachments/assets/4b3d6d5c-420e-4e15-bbe7-68bcad9729f0

<video src="./assets/video.mp4" controls=""></video>

## Table of Contents

- [Repository Structure](#repository-structure)
- [Usage](#usage)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running](#running)
- [Appendix](#appendix)
- [Contact](#contact)

## Repository Structure

```text
├── assets/
│   └── video.mp4
├── data/
│   └── wave.npy
├── wuji_hand_description/
├── run_sim.py
├── requirements.txt
└── README.md
```

### Directory Description

| Directory | Description |
|-----------|-------------|
| `assets/` | Video demo files |
| `data/` | Trajectory data files |
| `wuji_hand_description/` | Hand model submodule (MJCF, meshes) |
| `run_sim.py` | Main simulation script |
| `requirements.txt` | Python dependencies |

## Usage

### Prerequisites

- Python 3.8+

### Installation

Clone the repository with submodules:

```bash
git clone --recursive https://github.com/wuji-technology/mujoco-sim.git
cd mujoco-sim
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running

Run the simulation with trajectory playback:

```bash
python run_sim.py
```

The script loads the default right hand model and plays the trajectory from `data/wave.npy` in a loop. To use the left hand, edit `side = "left"` in `run_sim.py`.

## Appendix

### Update Models

To update the hand models (MJCF, meshes, etc.) to the latest version from the [description repository](https://github.com/wuji-technology/wuji-hand-description):

```bash
git submodule update --remote
```

## Contact

For any questions, please contact support@wuji.tech.
