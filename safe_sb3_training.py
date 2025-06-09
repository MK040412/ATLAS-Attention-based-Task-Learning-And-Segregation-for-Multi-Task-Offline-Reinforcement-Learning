#!/usr/bin/env python3
"""
안전한 SB3 학습 - 차원 문제 해결
"""

import argparse
import os
import numpy as np
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Safe SB3 Training")
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--save_path", type=str, default="./safe_sac_model")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.env_checker import check_env
        from fixed_simple_gym_wrapper import SimpleAntPushWrapper

        print(f"🚀 Safe SB3 Training")
        print(f"  Timesteps: {args.timesteps}")
        print(f"  Save path: {args.save_path}")
        
        # 환경 생성
        print(f"🏗️ Creating environment...")
        env = SimpleAntPushWrapper(custom_cfg={
            'cube_position': [2.0, 0.0, 0.1],
            'goal_position': [3.0, 0.0, 0.1]
        })
        
        print(f"📊 Environment info:")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        # 환경 테스트
        print(f"🧪 Testing environment...")
        try:
            obs, info = env.reset()
            print(f"  Reset successful, obs shape: {obs.shape}")
            
            # 랜덤 액션 테스트
            for i in range(3):
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                print(f"  Step {i+1}: obs shape: {obs.shape}, reward: {reward:.3f}")
                
                if done or truncated:
                    obs, info = env.reset()
                    print(f"  Reset after done, obs shape: {obs.shape}")
            
            print(f"✅ Environment test passed!")
            
        except Exception as e:
            print(f"❌ Environment test failed: {e}")
            return
        
        # 환경 검증 (선택적)
        print(f"🔍 Checking SB3 compatibility...")
        try:
            check_env(env, warn=True)
            print(f"✅ Environment is SB3 compatible!")
        except Exception as e:
            print(f"⚠️ Environment check warning (continuing anyway): {e}")
        
        # SAC 에이전트 생성
        print(f"🤖 Creating SAC agent...")
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=5000,       # 작은 버퍼
            learning_starts=100,    # 빠른 시작
            batch_size=32,          # 작은 배치
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=dict(
                net_arch=[64, 64]   # 작은 네트워크
            ),
            verbose=1,
            device="cuda"
        )
        
        print(f"📊 Model created successfully:")
        print(f"  Device: {model.device}")
        print(f"  Buffer size: {model.buffer_size}")
        
        # 학습 시작
        print(f"🎓 Starting training...")
        model.learn(
            total_timesteps=args.timesteps,
            log_interval=4,
            progress_bar=True
        )
        
        # 모델 저장
        model.save(args.save_path)
        print(f"💾 Model saved to: {args.save_path}")
        
        # 간단한 테스트
        print(f"🧪 Testing trained model...")
        obs, _ = env.reset()
        total_reward = 0
        
        for step in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 10 == 0:
                print(f"  Step {step}: reward={reward:.3f}, total={total_reward:.3f}")
            
            if done or truncated:
                print(f"  Episode finished at step {step}")
                break
        
        print(f"✅ Training completed! Final test reward: {total_reward:.2f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
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