#!/usr/bin/env python3
"""
가장 간단한 SB3 학습
"""

import argparse
import os
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Simple SB3 Training")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--save_path", type=str, default="./simple_sac_model")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        # SB3 import
        from stable_baselines3 import SAC
        from stable_baselines3.common.env_checker import check_env
        from fixed_simple_gym_wrapper import SimpleAntPushWrapper

        print(f"🚀 Simple SB3 Training")
        print(f"  Timesteps: {args.timesteps}")
        print(f"  Save path: {args.save_path}")
        
        # 환경 생성
        env = SimpleAntPushWrapper(custom_cfg={
            'cube_position': [2.0, 0.0, 0.1],
            'goal_position': [3.0, 0.0, 0.1]
        })
        
        # 환경 검증
        print(f"🔍 Checking environment compatibility...")
        try:
            check_env(env)
            print(f"✅ Environment is SB3 compatible!")
        except Exception as e:
            print(f"⚠️ Environment check warning: {e}")
        
        # SAC 에이전트 생성 (최소 설정)
        print(f"🤖 Creating SAC agent...")
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=10000,      # 작은 버퍼
            learning_starts=100,    # 빠른 시작
            batch_size=64,          # 작은 배치
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
        
        print(f"📊 Model info:")
        print(f"  Policy: {model.policy}")
        print(f"  Device: {model.device}")
        print(f"  Buffer size: {model.buffer_size}")
        
        # 학습 시작
        print(f"🎓 Starting training...")
        model.learn(
            total_timesteps=args.timesteps,
            log_interval=10,
            progress_bar=True
        )
        
        # 모델 저장
        model.save(args.save_path)
        print(f"💾 Model saved to: {args.save_path}")
        
        # 간단한 테스트
        print(f"🧪 Testing model...")
        obs, _ = env.reset()
        total_reward = 0
        
        for step in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 20 == 0:
                print(f"  Step {step}: reward={reward:.3f}, total={total_reward:.3f}")
            
            if done or truncated:
                print(f"  Episode finished at step {step}")
                break
        
        print(f"✅ Test completed! Total reward: {total_reward:.2f}")
        
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