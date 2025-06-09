#!/usr/bin/env python3
"""
SB3 기반 다중 태스크 학습
"""

import argparse
import os
import numpy as np
import yaml
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="SB3 Multi-task Training")
    parser.add_argument("--tasks", type=str, default="tasks.yaml")
    parser.add_argument("--timesteps_per_task", type=int, default=20000)
    parser.add_argument("--save_dir", type=str, default="./sb3_multitask")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
    from simple_gym_wrapper import SimpleAntPushWrapper

    print(f"🚀 SB3 Multi-task Training")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 태스크 로드
    tasks = yaml.safe_load(open(args.tasks))
    print(f"📋 Loaded {len(tasks)} tasks")
    
    # 각 태스크별로 별도 모델 학습
    models = {}
    
    try:
        for task_idx, task in enumerate(tasks):
            print(f"\n{'='*50}")
            print(f"🎯 Task {task_idx + 1}/{len(tasks)}: {task['name']}")
            print(f"📝 Instruction: {task['instruction']}")
            print(f"{'='*50}")
            
            # 태스크별 환경 생성
            def make_task_env():
                return SimpleAntPushWrapper(custom_cfg=task['env_cfg'])
            
            env = DummyVecEnv([make_task_env])
            
            # 태스크별 SAC 모델
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                buffer_size=50000,  # 메모리 절약
                learning_starts=1000,
                batch_size=128,     # 배치 크기 축소
                tau=0.005,
                gamma=0.99,
                policy_kwargs=dict(
                    net_arch=dict(pi=[128, 128], qf=[128, 128])  # 네트워크 크기 축소
                ),
                verbose=1,
                device="cuda"
            )
            
            # 체크포인트 콜백
            task_save_dir = os.path.join(args.save_dir, f"task_{task_idx + 1}")
            os.makedirs(task_save_dir, exist_ok=True)
            
            checkpoint_callback = CheckpointCallback(
                save_freq=5000,
                save_path=task_save_dir,
                name_prefix=f"sac_{task['name']}"
            )
            
            # 학습
            print(f"🎓 Training for {args.timesteps_per_task} timesteps...")
            model.learn(
                total_timesteps=args.timesteps_per_task,
                callback=checkpoint_callback,
                log_interval=5,
                progress_bar=True
            )
            
            # 모델 저장
            model_path = os.path.join(task_save_dir, f"final_{task['name']}")
            model.save(model_path)
            models[task['name']] = model_path
            
            print(f"✅ Task '{task['name']}' completed!")
            
            # 간단한 평가
            obs = env.reset()
            episode_reward = 0
            
            for _ in range(200):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward[0]
                
                if done[0]:
                    break
            
            print(f"  Final evaluation reward: {episode_reward:.2f}")
            
            env.close()
        
        # 모든 태스크 완료 후 요약
        print(f"\n🎉 All tasks completed!")
        print(f"📊 Trained models:")
        for task_name, model_path in models.items():
            print(f"  {task_name}: {model_path}")
        
        # 모델 경로 저장
        model_registry_path = os.path.join(args.save_dir, "model_registry.yaml")
        with open(model_registry_path, 'w') as f:
            yaml.dump(models, f)
        
        print(f"📝 Model registry saved to: {model_registry_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        sim.close()

if __name__ == "__main__":
    main()