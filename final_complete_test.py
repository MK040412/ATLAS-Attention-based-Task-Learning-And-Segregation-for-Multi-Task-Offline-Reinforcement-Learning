#!/usr/bin/env python3
"""
완전히 수정된 최종 테스트
"""

import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Final Complete Test")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        from ant_push_env_final import IsaacAntPushEnv
        from rtx4060_sac_expert_fixed import RTX4060SACExpert
        
        print("=== Final Complete Test ===")
        
        # 환경 설정
        env_cfg = {
            'cube_position': [1.5, 0.0, 0.1],
            'goal_position': [2.5, 0.0, 0.1]
        }
        
        print("Creating environment...")
        env = IsaacAntPushEnv(custom_cfg=env_cfg, num_envs=args.num_envs)
        print("✓ Environment created")
        
        # 초기 리셋
        print("Resetting environment...")
        state, info = env.reset()
        print(f"✓ Reset successful - State shape: {state.shape}")
        
        # 환경 정보 확인
        env_info = env.get_info()
        print(f"Environment info: {env_info}")
        
        # 전문가 생성
        state_dim = len(state)
        action_space = env.action_space
        
        if hasattr(action_space, 'shape'):
            if len(action_space.shape) == 1:
                action_dim = action_space.shape[0]
            else:
                action_dim = action_space.shape[1]
        else:
            action_dim = 8  # 기본값
        
        print(f"Creating expert: state_dim={state_dim}, action_dim={action_dim}")
        expert = RTX4060SACExpert(
            state_dim=state_dim,
            action_dim=action_dim,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("✓ Expert created")
        
        # 테스트 실행
        print(f"\n--- Running {args.steps} steps ---")
        total_reward = 0.0
        successful_steps = 0
        episode_count = 0
        
        for step in range(args.steps):
            try:
                print(f"\nStep {step + 1}:")
                
                # 상태 확인
                print(f"  State: shape={state.shape}, type={type(state)}")
                if step == 0:
                    print(f"  State sample: {state[:5]} ... {state[-6:]}")
                
                # 액션 선택
                action = expert.select_action(state, deterministic=False)
                print(f"  Action: shape={action.shape}, range=[{action.min():.3f}, {action.max():.3f}]")
                
                # 환경 스텝
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                successful_steps += 1
                
                print(f"  Result: reward={reward:.4f}, done={done}")
                
                # 위치 정보 (가능한 경우)
                try:
                    env_info = env.get_info()
                    if 'cube_position' in env_info and 'ant_position' in env_info:
                        cube_pos = env_info['cube_position']
                        ant_pos = env_info['ant_position']
                        goal_pos = env_info['goal_position']
                        
                        print(f"  Positions:")
                        print(f"    Ant:  [{ant_pos[0]:.2f}, {ant_pos[1]:.2f}, {ant_pos[2]:.2f}]")
                        print(f"    Cube: [{cube_pos[0]:.2f}, {cube_pos[1]:.2f}, {cube_pos[2]:.2f}]")
                        print(f"    Goal: [{goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f}]")
                        print(f"  Distances: to_goal={env_info['distance_to_goal']:.3f}, to_cube={env_info['distance_to_cube']:.3f}")
                except Exception as pos_error:
                    print(f"  Position info unavailable: {pos_error}")
                
                # 경험 저장 (간단한 버전)
                try:
                    expert.replay_buffer.add(state, action, reward, next_state, done)
                    
                    # 주기적 업데이트
                    if step > 0 and step % 5 == 0 and len(expert.replay_buffer) >= 16:
                        loss_info = expert.update(batch_size=8)
                        if loss_info:
                            print(f"  Update: loss={loss_info.get('total_loss', 0):.6f}")
                except Exception as update_error:
                    print(f"  Update failed: {update_error}")
                
                state = next_state
                
                # 에피소드 종료 처리
                if done:
                    episode_count += 1
                    print(f"  Episode {episode_count} finished, resetting...")
                    state, _ = env.reset()
                
            except Exception as step_error:
                print(f"  ✗ Step {step + 1} failed: {step_error}")
                # 복구 시도
                try:
                    state, _ = env.reset()
                    print("  ✓ Recovered with reset")
                except:
                    print("  ✗ Recovery failed")
                    break
        
        # 결과 요약
        print(f"\n--- Test Results ---")
        print(f"✓ Successful steps: {successful_steps}/{args.steps}")
        print(f"✓ Total episodes: {episode_count}")
        print(f"✓ Total reward: {total_reward:.4f}")
        print(f"✓ Average reward: {total_reward/max(successful_steps, 1):.4f}")
        print(f"✓ Success rate: {successful_steps/args.steps*100:.1f}%")
        
        # 최종 환경 정보
        try:
            final_info = env.get_info()
            print(f"✓ Final environment state: {final_info}")
        except:
            pass
        
        # 성공 판정
        if successful_steps >= args.steps * 0.8:
            print("\n🎉 TEST PASSED! Environment is working correctly.")
            return_code = 0
        else:
            print("\n⚠️  TEST PARTIALLY SUCCESSFUL. Some issues remain.")
            return_code = 1
        
        # 정리
        expert.cleanup()
        env.close()
        
        return return_code
        
    except Exception as e:
        print(f"✗ Test failed completely: {e}")
        import traceback
        traceback.print_exc()
        return 2
    
    finally:
        sim.close()

if __name__ == "__main__":
    import sys
    sys.exit(main())