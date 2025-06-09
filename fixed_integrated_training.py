#!/usr/bin/env python3
"""
수정된 RTX 4060 최적화 통합 학습 스크립트
"""

import argparse
import torch
import numpy as np
import yaml
import os
import gc
import time
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Fixed RTX 4060 Integrated Training")
    parser.add_argument("--tasks", type=str, default="tasks.yaml")
    parser.add_argument("--save_dir", type=str, default="./rtx4060_outputs")
    parser.add_argument("--num_envs", type=int, default=1)  # 단일 환경으로 시작
    parser.add_argument("--episodes_per_task", type=int, default=20)  # 에피소드 수 축소
    parser.add_argument("--max_steps", type=int, default=200)  # 스텝 수 축소
    parser.add_argument("--tau_birth", type=float, default=1e-2)  # 임계값 완화
    parser.add_argument("--tau_merge_kl", type=float, default=0.5)  # 병합 임계값 완화
    parser.add_argument("--batch_size", type=int, default=16)  # 배치 크기 축소
    parser.add_argument("--update_freq", type=int, default=5)
    parser.add_argument("--merge_freq", type=int, default=10)
    parser.add_argument("--checkpoint_freq", type=int, default=10)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # RTX 4060 메모리 최적화
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = False
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

    app = AppLauncher(vars(args))
    sim = app.app

    from fixed_ant_push_env import IsaacAntPushEnv
    from integrated_kl_dpmm_moe_sac import IntegratedKLDPMMMoESAC

    print(f"🚀 Fixed RTX 4060 Training Started")
    print(f"  Single environment mode")
    print(f"  Episodes per task: {args.episodes_per_task}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Relaxed thresholds: birth={args.tau_birth}, merge={args.tau_merge_kl}")

    os.makedirs(args.save_dir, exist_ok=True)
    tasks = yaml.safe_load(open(args.tasks))
    
    agent = None
    all_stats = {
        'task_names': [],
        'episode_rewards': [],
        'cluster_evolution': [],
        'expert_evolution': [],
        'successful_episodes': 0,
        'total_episodes': 0
    }

    try:
        for task_idx, task in enumerate(tasks):
            print(f"\n{'='*60}")
            print(f"🎯 Task {task_idx + 1}/{len(tasks)}: {task['name']}")
            print(f"📝 Instruction: {task['instruction']}")
            print(f"{'='*60}")
            
            # 환경 생성
            env = IsaacAntPushEnv(custom_cfg=task['env_cfg'], num_envs=args.num_envs)
            
            # 초기 상태로 차원 확인
            state, _ = env.reset()
            state_dim = state.shape[0] if hasattr(state, 'shape') else len(state)
            
            # action space 차원 확인
            if hasattr(env.action_space, 'shape'):
                if len(env.action_space.shape) == 2:
                    action_dim = env.action_space.shape[1]
                else:
                    action_dim = env.action_space.shape[0]
            else:
                action_dim = 8  # 기본값