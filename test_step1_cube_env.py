import argparse
import torch
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        from ant_push_env import IsaacAntPushEnv
        from config import CONFIG
        
        print("🧪 Step 1 Test: Blue Cube Environment")
        
        # 환경 생성
        env = IsaacAntPushEnv(
            custom_cfg={
                'cube_position': [2.0, 0.0, 0.1],
                'goal_position': [3.0, 0.0, 0.1]
            },
            num_envs=1
        )
        
        # 초기 리셋
        state, info = env.reset()
        print(f"✓ Initial state shape: {state.shape}")
        print(f"✓ Action dimension: {env.get_action_dim()}")
        
        # 상태 분석
        base_obs_dim = len(state) - 6
        ant_pos = state[:3]
        cube_pos = state[base_obs_dim:base_obs_dim+3]
        goal_pos = state[base_obs_dim+3:base_obs_dim+6]
        
        print(f"📊 State Analysis:")
        print(f"  Total state dim: {len(state)}")
        print(f"  Base obs dim: {base_obs_dim}")
        print(f"  Ant position: {ant_pos}")
        print(f"  Cube position: {cube_pos}")
        print(f"  Goal position: {goal_pos}")
        
        # 몇 스텝 실행
        for i in range(5):
            # 랜덤 액션
            action = torch.randn(env.get_action_dim()) * 0.1
            next_state, reward, done, info = env.step(action.numpy())
            
            print(f"Step {i+1}: Reward={reward:.3f}, Done={done}")
            
            if done:
                state, _ = env.reset()
            else:
                state = next_state
        
        print("✅ Step 1 test passed!")
        env.close()
        
    except Exception as e:
        print(f"❌ Step 1 test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        sim.close()

if __name__ == "__main__":
    main()