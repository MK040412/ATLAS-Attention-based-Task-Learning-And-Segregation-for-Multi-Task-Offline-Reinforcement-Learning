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
        import numpy as np
        from ant_push_env_absolutely_final import IsaacAntPushEnv
        from memory_optimized_sac_ultra_safe import DimensionExactSACExpert

        print("🧪 Final test starting...")
        
        # 환경 생성
        env = IsaacAntPushEnv(num_envs=1)
        state, _ = env.reset()
        
        print(f"State shape: {state.shape}")
        print(f"Action dim: {env.get_action_dim()}")
        
        # 에이전트 생성
        agent = DimensionExactSACExpert(
            state_dim=len(state),
            action_dim=env.get_action_dim()
        )
        
        # 여러 스텝 테스트
        for i in range(3):
            action = agent.select_action(state)
            print(f"Step {i+1}: Action shape: {action.shape}")
            
            next_state, reward, done, info = env.step(action)
            print(f"  Reward: {reward:.3f}, Done: {done}")
            
            if done:
                state, _ = env.reset()
            else:
                state = next_state
        
        env.close()
        print("✅ Final test passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        sim.close()

if __name__ == "__main__":
    main()