#!/usr/bin/env python3
"""
매우 안전한 SB3 학습 - 모든 타입 변환 문제 해결
"""

import argparse
import os
import numpy as np
import torch
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Ultra Safe SB3 Training")
    parser.add_argument("--timesteps", type=int, default=3000)
    parser.add_argument("--save_path", type=str, default="./ultra_safe_sac_model")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.monitor import Monitor
        from fixed_simple_gym_wrapper import SimpleAntPushWrapper

        print(f"🚀 Ultra Safe SB3 Training")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  Timesteps: {args.timesteps}")
        
        # 환경 생성
        print(f"🏗️ Creating environment...")
        base_env = SimpleAntPushWrapper(custom_cfg={
            'cube_position': [2.0, 0.0, 0.1],
            'goal_position': [3.0, 0.0, 0.1]
        })
        
        # Monitor로 래핑 (SB3 호환성 향상)
        env = Monitor(base_env)
        
        print(f"📊 Environment info:")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        # 환경 안전성 테스트
        print(f"🧪 Testing environment safety...")
        test_passed = True
        
        try:
            # 리셋 테스트
            obs, info = env.reset()
            print(f"  ✅ Reset: obs shape {obs.shape}, type {type(obs)}")
            
            # 스텝 테스트
            for i in range(5):
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                
                # 타입 검증
                assert isinstance(obs, np.ndarray), f"obs should be numpy array, got {type(obs)}"
                assert isinstance(reward, (int, float, np.number)), f"reward should be number, got {type(reward)}"
                assert isinstance(done, (bool, np.bool_)), f"done should be bool, got {type(done)}"
                assert isinstance(truncated, (bool, np.bool_)), f"truncated should be bool, got {type(truncated)}"
                
                print(f"  ✅ Step {i+1}: obs {obs.shape}, reward {reward:.3f}, done {done}")
                
                if done or truncated:
                    obs, info = env.reset()
                    print(f"  ✅ Reset after done: obs {obs.shape}")
            
        except Exception as e:
            print(f"  ❌ Environment test failed: {e}")
            test_passed = False
        
        if not test_passed:
            print("❌ Environment test failed, aborting training")
            return
        
        print(f"✅ Environment test passed!")
        
        # SAC 에이전트 생성 (보수적 설정)
        print(f"🤖 Creating SAC agent...")
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=1e-4,     # 낮은 학습률
            buffer_size=2000,       # 작은 버퍼
            learning_starts=50,     # 빠른 시작
            batch_size=16,          # 작은 배치
            tau=0.01,               # 빠른 타겟 업데이트
            gamma=0.95,             # 낮은 할인율
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=dict(
                net_arch=[32, 32]   # 매우 작은 네트워크
            ),
            verbose=1,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print(f"📊 Model info:")
        print(f"  Device: {model.device}")
        print(f"  Buffer size: {model.buffer_size}")
        print(f"  Network architecture: {model.policy.net_arch}")
        
        # 학습 시작
        print(f"🎓 Starting training...")
        try:
            model.learn(
                total_timesteps=args.timesteps,
                log_interval=2,
                progress_bar=True,
                callback=None
            )
            
            print(f"✅ Training completed successfully!")
            
        except Exception as e:
            print(f"⚠️ Training interrupted: {e}")
            print("Attempting to save partial model...")
        
        # 모델 저장
        try:
            model.save(args.save_path)
            print(f"💾 Model saved to: {args.save_path}")
        except Exception as e:
            print(f"⚠️ Model save failed: {e}")
        
        # 학습된 모델 테스트
        print(f"🧪 Testing trained model...")
        try:
            obs, _ = env.reset()
            total_reward = 0
            episode_length = 0
            
            for step in range(100):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                episode_length += 1
                
                if step % 20 == 0:
                    print(f"  Step {step}: reward={reward:.3f}, total={total_reward:.3f}")
                
                if done or truncated:
                    print(f"  Episode finished at step {step}")
                    break
            
            print(f"✅ Test completed!")
            print(f"  Episode length: {episode_length}")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Average reward: {total_reward/episode_length:.3f}")
            
        except Exception as e:
            print(f"⚠️ Testing failed: {e}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            env.close()
            print("🔒 Environment closed")
        except:
            pass
        sim.close()
        print("🔒 Simulator closed")

if __name__ == "__main__":
    main()