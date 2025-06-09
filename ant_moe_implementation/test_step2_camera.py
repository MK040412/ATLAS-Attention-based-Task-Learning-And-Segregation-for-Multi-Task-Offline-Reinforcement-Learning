import argparse
import torch
import numpy as np  # 추가된 import
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        from ant_push_env_with_camera import IsaacAntPushEnvWithCamera
        from config import CONFIG
        
        print("🧪 Step 2 Test: Camera + Simple Vision")
        
        # 환경 생성
        env = IsaacAntPushEnvWithCamera(
            custom_cfg={
                'cube_position': [2.0, 0.0, 0.1],
                'goal_position': [3.0, 0.0, 0.1]
            },
            num_envs=1
        )
        
        # 초기 리셋
        state, info = env.reset()
        print(f"✓ Initial state shape: {state.shape}")
        
        # 상태 분석
        base_obs_dim = len(state) - 6
        ant_pos = state[:3]
        vision_cube_pos = state[base_obs_dim:base_obs_dim+3]
        goal_pos = state[base_obs_dim+3:base_obs_dim+6]
        
        print(f"📊 State Analysis:")
        print(f"  Total state dim: {len(state)}")
        print(f"  Base obs dim: {base_obs_dim}")
        print(f"  Ant position: {ant_pos}")
        print(f"  Vision cube position: {vision_cube_pos}")
        print(f"  Goal position: {goal_pos}")
        
        # 카메라 테스트
        print(f"\n📷 Camera Test:")
        rgb, depth = env.get_camera_obs()
        print(f"  RGB shape: {rgb.shape}")
        print(f"  Depth shape: {depth.shape}")
        print(f"  RGB range: [{rgb.min()}, {rgb.max()}]")
        print(f"  Depth range: [{depth.min():.2f}, {depth.max():.2f}]")
        
        # 몇 스텝 실행
        for i in range(3):
            action = torch.randn(env.get_action_dim()) * 0.1
            next_state, reward, done, info = env.step(action.numpy())
            
            # 비전 vs 실제 큐브 위치 비교
            vision_pos = next_state[base_obs_dim:base_obs_dim+3]
            actual_pos = env._get_cube_position()
            
            print(f"Step {i+1}:")
            print(f"  Reward: {reward:.3f}")
            print(f"  Vision cube pos: {vision_pos}")
            print(f"  Actual cube pos: {actual_pos}")
            print(f"  Position diff: {np.linalg.norm(vision_pos - actual_pos):.3f}")
            
            if done:
                state, _ = env.reset()
            else:
                state = next_state
        
        print("✅ Step 2 test passed!")
        env.close()
        
    except Exception as e:
        print(f"❌ Step 2 test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        sim.close()

if __name__ == "__main__":
    main()