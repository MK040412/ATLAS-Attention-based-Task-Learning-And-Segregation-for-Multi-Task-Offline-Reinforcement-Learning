ㅊ # Ant MoE SAC Agent with DPMM Clustering

This repository implements a Mixture-of-Experts (MoE) SAC agent with dynamic expert management
using a Dirichlet Process Mixture Model (birth and merge), integrated with NVIDIA Isaac Lab's
Ant environment for instruction-conditioned robotic tasks.

## Contents
- `ant_push_env.py`: Custom Isaac Lab Ant environment with a cube object and push task.
- `moe_sac_agent.py`: FusionEncoder, SACExpert class, replay buffer, clustering logic.
- `train.py`: Training script that initializes simulation, environment, agent, and runs episodes.
- `README.md`: This file.

## Setup
1. Install NVIDIA Isaac Sim and Isaac Lab according to the [Isaac Lab Quickstart Guide].
2. Ensure `PYTHONPATH` includes Isaac Lab's `source` directory.
3. Install Python dependencies:
   ```
   pip install torch gymnasium transformers sentence-transformers numpy
   ```
4. Unzip the package and run training:
   ```
   python train.py
   ```

## Usage
- Adjust `DIST_THRESHOLD` and `MERGE_THRESHOLD` in `moe_sac_agent.py` for clustering sensitivity.
- Modify `num_episodes` and `max_steps` in `train.py` for experiment length.
- Monitor log outputs to see when new experts are created and merged.

