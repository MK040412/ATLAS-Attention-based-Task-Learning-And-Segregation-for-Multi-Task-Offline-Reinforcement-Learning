#!/usr/bin/env python3
"""
환경만 테스트하는 스크립트
"""

import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Test Environment Only")
    parser.add_argument("--num_envs", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        from ant_push_env_fixed import IsaacAntPushEnv
        
        print("Testing environment creation...")
        
        # 테스트 설정
        test_cfg = {
            'cube_position': [2.0, 0.0, 0.1],
            'goal_position': [3.0, 0.0, 0.1]
        }
        
        # 환경 생성
        env = IsaacAntPushEnv(custom_cfg=test_cfg, num_envs=args.num_envs)
        print("Environment created successfully!")
        
        # 환경 정보 출력
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # 리셋 테스트
        print("\nTesting reset...")
        obs, info = env.reset()
        print(f"Reset successful!")
        print(f"Observation shape: {obs.shape}")
        print(f"Observation sample: {obs[:10]}")  # 처음 10개 값만 출력
        
        # 스텝 테스트
        print("\nTesting steps...")
        for step in range(10):
            # 랜덤 액션
            action = np.random.randn(env.action_space.shape[0]) * 0.1
            
            obs, reward, done, info = env.step(action)
            
            print(f"Step {step + 1}:")
            print(f"  Action shape: {action.shape}")
            print(f"  Obs shape: {obs.shape}")
            print(f"  Reward: {reward:.3f}")
            print(f"  Done: {done}")
            
            if done:
                print("Episode finished, resetting...")
                obs, info = env.reset()
        
        print("\nEnvironment test completed successfully!")
        
        # 환경 종료
        env.close()
        
    except Exception as e:
        print(f"Environment test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        sim.close()

if __name__ == "__main__":
    main()