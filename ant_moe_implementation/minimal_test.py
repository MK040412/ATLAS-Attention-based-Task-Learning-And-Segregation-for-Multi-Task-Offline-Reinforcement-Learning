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
        from ant_push_env_ultra_safe import IsaacAntPushEnv
        from memory_optimized_sac_ultra_safe import DimensionExactSACExpert

        print("🧪 Minimal test starting...")
        
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
        
        # 한 스텝만 테스트
        action = agent.select_action(state)
        print(f"Action shape: {action.shape}")
        
        next_state, reward, done, info = env.step(action)
        print(f"Step successful! Reward: {reward}")
        
        env.close()
        print("✅ Minimal test passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        sim.close()

if __name__ == "__main__":
    main()