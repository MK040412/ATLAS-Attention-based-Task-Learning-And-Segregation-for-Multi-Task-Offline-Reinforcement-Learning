#!/usr/bin/env python3
"""
환경만 테스트하는 스크립트
"""

import argparse
import numpy as np
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Test Environment Only")
    parser.add_argument("--steps", type=int, default=50)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        from fixed_simple_gym_wrapper import SimpleAntPushWrapper

        print(f"🧪 Testing Environment Only")
        
        # 환경 생성
        env = SimpleAntPushWrapper(custom_cfg={
            'cube_position': [2.0, 0.0, 0.1],
            'goal_position': [3.0, 0.0, 0.1]
        })
        
        print(f"📊 Environment created:")
        print(f"  Obs space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        # 환경 테스트
        obs, info = env.reset()
        print(f"🔄 Reset successful: obs shape {obs.shape}")
        
        total_reward = 0
        
        for step in range(args.steps):
            # 랜덤 액션
            action = env.action_space.sample()
            
            # 스텝 실행
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # 타입 검증
            print(f"Step {step+1:2d}: "
                  f"obs={obs.shape} "
                  f"reward={reward:.3f} "
                  f"done={done} "
                  f"truncated={truncated} "
                  f"total={total_reward:.3f}")
            
            # 타입 확인
            assert isinstance(obs, np.ndarray), f"obs type error: {type(obs)}"
            assert isinstance(reward, (int, float, np.number)), f"reward type error: {type(reward)}"
            assert isinstance(done, (bool, np.bool_)), f"done type error: {type(done)}"
            assert isinstance(truncated, (bool, np.bool_)), f"truncated type error: {type(truncated)}"
            
            if done or truncated:
                print(f"  Episode ended, resetting...")
                obs, info = env.reset()
                print(f"  Reset: obs shape {obs.shape}")
        
        print(f"✅ Environment test completed successfully!")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Average reward per step: {total_reward/args.steps:.3f}")
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            env.close()
        except:
            pass
        sim.close()

if __name__ == "__main__":
    main()