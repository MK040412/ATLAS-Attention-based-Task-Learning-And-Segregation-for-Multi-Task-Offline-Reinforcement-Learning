#!/usr/bin/env python3
"""
최소한의 환경 테스트
"""

import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Minimal Environment Test")
    parser.add_argument("--num_envs", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        from ant_push_env import IsaacAntPushEnv
        from rtx4060_sac_expert_fixed import RTX4060SACExpert
        
        print("=== Minimal Environment Test ===")
        
        # 환경 생성
        env_cfg = {
            'cube_position': [1.5, 0.0, 0.1],
            'goal_position': [2.0, 0.0, 0.1]
        }
        
        env = IsaacAntPushEnv(custom_cfg=env_cfg, num_envs=args.num_envs)
        print("✓ Environment created")
        
        # 초기 리셋
        state, info = env.reset()
        print(f"✓ Initial reset: state shape {state.shape}")
        print(f"  State type: {type(state)}")
        print(f"  State dtype: {state.dtype if hasattr(state, 'dtype') else 'N/A'}")
        
        # 전문가 생성
        state_dim = len(state)
        action_dim = env.action_space.shape[0] if len(env.action_space.shape) == 1 else env.action_space.shape[1]
        
        expert = RTX4060SACExpert(
            state_dim=state_dim,
            action_dim=action_dim,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"✓ Expert created: state_dim={state_dim}, action_dim={action_dim}")
        
        # 몇 스텝 실행
        print("\n--- Step Test ---")
        for step in range(5):
            try:
                # 상태 확인
                print(f"Step {step + 1}:")
                print(f"  State: {type(state)}, shape: {state.shape}")
                
                # 액션 선택
                action = expert.select_action(state, deterministic=True)
                print(f"  Action: {type(action)}, shape: {action.shape}")
                
                # 환경 스텝
                next_state, reward, done, info = env.step(action)
                print(f"  Next state: {type(next_state)}, shape: {next_state.shape}")
                print(f"  Reward: {reward}, Done: {done}")
                
                state = next_state
                
                if done:
                    print("  Episode finished, resetting...")
                    state, _ = env.reset()
                
            except Exception as e:
                print(f"  ✗ Step {step + 1} failed: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print("\n✓ Minimal test completed")
        
        # 정리
        expert.cleanup()
        env.close()
        
    except Exception as e:
        print(f"✗ Minimal test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        sim.close()

if __name__ == "__main__":
    main()