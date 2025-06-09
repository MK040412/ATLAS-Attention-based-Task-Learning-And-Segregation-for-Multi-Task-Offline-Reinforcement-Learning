#!/usr/bin/env python3
"""
학습된 모델 테스트
"""

import argparse
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Test Trained Model")
    parser.add_argument("--model_path", type=str, default="./ultra_safe_sac_model")
    parser.add_argument("--episodes", type=int, default=3)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        from stable_baselines3 import SAC
        from fixed_simple_gym_wrapper import SimpleAntPushWrapper
        
        print(f"🧪 Testing trained model: {args.model_path}")
        
        # 환경 생성
        env = SimpleAntPushWrapper()
        
        # 모델 로드
        try:
            model = SAC.load(args.model_path)
            print(f"✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return
        
        # 테스트 실행
        for episode in range(args.episodes):
            obs, _ = env.reset()
            episode_reward = 0
            steps = 0
            
            print(f"\n🎮 Episode {episode + 1}")
            
            for step in range(200):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                
                if step % 40 == 0:
                    print(f"  Step {step}: reward={reward:.3f}, total={episode_reward:.3f}")
                
                if done or truncated:
                    break
            
            print(f"  Episode {episode + 1} completed:")
            print(f"    Steps: {steps}")
            print(f"    Total reward: {episode_reward:.2f}")
            print(f"    Average reward: {episode_reward/steps:.3f}")
        
        print(f"🎉 Testing completed!")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
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