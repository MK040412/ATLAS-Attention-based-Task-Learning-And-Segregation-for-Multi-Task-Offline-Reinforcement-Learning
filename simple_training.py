#!/usr/bin/env python3
"""
가장 간단한 MoE 학습 스크립트
"""

import argparse
import numpy as np
import yaml
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Simple MoE Training")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=300)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    from fixed_ant_push_env import IsaacAntPushEnv
    from cleanrl_moe_agent import CleanRLMoEAgent

    print(f"🚀 Simple MoE Training")
    
    # 간단한 태스크들
    tasks = [
        {
            'name': 'push_forward',
            'instruction': 'Push the cube forward',
            'env_cfg': {'cube_position': [2.0, 0.0, 0.1], 'goal_position': [3.0, 0.0, 0.1]}
        },
        {
            'name': 'push_left',
            'instruction': 'Push the cube to the left',
            'env_cfg': {'cube_position': [2.0, 0.0, 0.1], 'goal_position': [2.0, 1.0, 0.1]}
        }
    ]
    
    # 환경 생성
    env = IsaacAntPushEnv(custom_cfg=tasks[0]['env_cfg'], num_envs=1)
    
    # 초기 상태로 차원 확인
    state, _ = env.reset()
    state_dim = len(state)
    action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else 8
    
    # 에이전트 생성
    agent = CleanRLMoEAgent(state_dim, action_dim)
    
    try:
        for task in tasks:
            print(f"\n🎯 Task: {task['name']}")
            print(f"📝 Instruction: {task['instruction']}")
            
            # 환경 재설정
            env.close()
            env = IsaacAntPushEnv(custom_cfg=task['env_cfg'], num_envs=1)
            
            episode_rewards = []
            
            for episode in range(args.episodes):
                state, _ = env.