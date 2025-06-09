#!/usr/bin/env python3
"""
수정된 최소한의 환경 테스트
"""

import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Fixed Minimal Environment Test")
    parser.add_argument("--num_envs", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        from ant_push_env_fixed import IsaacAntPushEnv
        from rtx4060_sac_expert_fixed import RTX4060SACExpert
        
        print("=== Fixed Minimal Environment Test ===")
        
        # 환경 생성
        env_cfg = {
            'cube_position': [1.5, 0.0, 0.1],
            'goal_position': [2.0, 0.0, 0.1]
        }
        
        print("Creating environment...")
        env = IsaacAntPushEnv(custom_cfg=env_cfg, num_envs=args.num_envs)
        print("✓ Environment created")
        
        # 초기 리셋
        print("Resetting environment...")
        state, info = env.reset()
        print(f"✓ Initial reset successful")
        print(f"  State shape: {state.shape}")
        print(f"  State type: {type(state)}")
        print(f"  State dtype: {state.dtype}")
        print(f"  State range: [{state.min():.3f}, {state.max():.3f}]")
        
        # 전문가 생성
        state_dim = len(state)
        action_dim = env.action_space.shape[0] if len(env.action_space.shape) == 1 else env.action_space.shape[1]
        
        print(f"Creating expert with state_dim={state_dim}, action_dim={action_dim}")
        expert = RTX4060SACExpert(
            state_dim=state_dim,
            action_dim=action_dim,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"✓ Expert created")
        
        # 몇 스텝 실행
        print("\n--- Step Test ---")
        total_reward = 0.0
        successful_steps = 0
        
        for step in range(10):
            try:
                print(f"\nStep {step + 1}:")
                
                # 상태 정보 출력 (첫 번째 스텝만)
                if step == 0:
                    print(f"  State details:")
                    print(f"    Shape: {state.shape}")
                    print(f"    First 5 values: {state[:5]}")
                    print(f"    Last 6 values (cube+goal): {state[-6:]}")
                
                # 액션 선택
                action = expert.select_action(state, deterministic=True)
                print(f"  Action: shape={action.shape}, range=[{action.min():.3f}, {action.max():.3f}]")
                
                # 환경 스텝
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                successful_steps += 1
                
                print(f"  Result: reward={reward:.4f}, done={done}")
                
                # 위치 정보 추출 및 출력
                if len(next_state) >= 6:
                    ant_pos = next_state[:3]
                    cube_pos = next_state[-6:-3]
                    goal_pos = next_state[-3:]
                    
                    cube_goal_dist = np.linalg.norm(cube_pos - goal_pos)
                    ant_cube_dist = np.linalg.norm(ant_pos - cube_pos)
                    
                    print(f"  Positions:")
                    print(f"    Ant: [{ant_pos[0]:.2f}, {ant_pos[1]:.2f}, {ant_pos[2]:.2f}]")
                    print(f"    Cube: [{cube_pos[0]:.2f}, {cube_pos[1]:.2f}, {cube_pos[2]:.2f}]")
                    print(f"    Goal: [{goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f}]")
                    print(f"  Distances: cube->goal={cube_goal_dist:.3f}, ant->cube={ant_cube_dist:.3f}")
                
                state = next_state
                
                if done:
                    print("  Episode finished, resetting...")
                    state, _ = env.reset()
                
            except Exception as e:
                print(f"  ✗ Step {step + 1} failed: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # 결과 요약
        print(f"\n--- Test Summary ---")
        print(f"✓ Successful steps: {successful_steps}/10")
        print(f"✓ Total reward: {total_reward:.4f}")
        print(f"✓ Average reward: {total_reward/max(successful_steps, 1):.4f}")
        
        if successful_steps >= 8:
            print("✓ Environment test PASSED!")
        else:
            print("⚠ Environment test partially successful")
        
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