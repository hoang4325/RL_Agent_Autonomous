# CARLA Urban RL (End-to-End)

Train an **urban driving behavior policy** in **CARLA 0.9.15** using **multi-sensor fusion + LiDAR Euclidean clustering + tracking**.

This project is designed for **Windows CARLA server** + **Python client training** (can be on same machine).

## What you get

- Spawns an ego vehicle + **all common sensors**:
  - RGB, Depth, Semantic Segmentation
  - LiDAR ray_cast (for perception)
  - LiDAR ray_cast_semantic (for validation/metrics)
  - Radar
  - GNSS, IMU
  - Collision, Lane invasion
- Perception pipeline:
  - ROI filter -> voxel downsample -> ground removal -> **adaptive Euclidean clustering** -> bbox + heuristic class
  - **Semantic LiDAR validation** (purity / under/over-seg checks)
- Multi-object tracking: Kalman CV + nearest-neighbor association
- Fusion/world-model:
  - lane-relative gaps (front/left/right), relative speeds, TTC
- Gymnasium environment + SB3 PPO training

> Note: This is a **research/learning** stack, not a production autonomy system.

---

## 1) Start CARLA 0.9.15 server (Windows)

Open a terminal in your CARLA folder and run:

```bat
CarlaUE4.exe -quality-level=Low -fps=20 -carla-world-port=2000 -windowed -ResX=800 -ResY=600
```

If your GPU is weak, you can lower fps / resolution.

---

## 2) Python environment (Windows)

### 2.1 Use the correct Python version
CARLA provides a PythonAPI egg like:

`PythonAPI\carla\dist\carla-0.9.15-py3.X-win-amd64.egg`

Use the **matching Python X**.

### 2.2 Create env + install deps

```bat
conda create -n carla_urban_rl python=3.7 -y
conda activate carla_urban_rl
pip install -U pip
pip install -r requirements.txt
```

### 2.3 Add CARLA PythonAPI to PYTHONPATH
PowerShell example:

```powershell
$env:PYTHONPATH="$env:PYTHONPATH;D:\CARLA_0.9.15\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg"
```

Test:

```bat
python -c "import carla; print('CARLA OK', carla.__file__)"
```

---

## 3) Configure

Edit YAML files in `config/`:

- `carla.yaml`: host/port, sync, no_rendering, town
- `sensors.yaml`: sensor specs
- `scenarios.yaml`: traffic count, episode length, curriculum switches
- `rl.yaml`: PPO hyperparams and observation toggles

---

## 4) Train

```bat
python scripts\train_ppo.py --carla config\carla.yaml --sensors config\sensors.yaml --scenarios config\scenarios.yaml --rl config\rl.yaml
```

Tensorboard:

```bat
tensorboard --logdir runs
```

---

## 5) Evaluate (play)

```bat
python scripts\eval_policy.py --model checkpoints\ppo_urban_final.zip --carla config\carla.yaml --sensors config\sensors.yaml --scenarios config\scenarios.yaml --rl config\rl.yaml
```

---

## Notes / Practical tips

- Training speed is mostly limited by CARLA tick. Use `no_rendering_mode: true` for faster training.
- Start simple: low traffic, then increase traffic in `scenarios.yaml`.
- This environment trains a **high-level behavior policy**. A classical controller executes steering/speed.

