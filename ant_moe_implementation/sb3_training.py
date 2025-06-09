#!/usr/bin/env python3
"""
Stable-Baselines3 기반 MoE 학습
"""

import argparse
import numpy as np
import yaml
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="SB3 MoE Training")
    parser.add_argument("--tasks", type=str, default="tasks.yaml")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default="./sb3_outputs")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    from fixed_ant_push_env import IsaacAntPushEnv
    from sb3_moe_agent import SB3MoEAgent
    import os

    print(f"🚀 SB3 MoE Training Started")
    
    os.makedirs(args.save_dir, exist_ok=True)
    tasks = yaml.safe_load(open(args.tasks))
    
    # 환경 생성
    env = IsaacAntPushEnv(custom_cfg=tasks[0]['env_cfg'], num_envs=1)
    
    # 에이전트 생성
    agent = SB3MoEAgent(env)
    
    try:
        for task_idx, task in enumerate(tasks):
            print(f"\n🎯 Task {task_idx + 1}: {task['name']}")
            print(f"📝 Instruction: {task['instruction']}")
            
            # 환경 재설정
            env.close()
            env = IsaacAntPushEnv(custom_cfg=task['env_cfg'], num_envs=1)
            
            task_rewards = []
            
            for episode in range(args.episodes):
                state, _ = env.reset()
                episode_reward = 0
                
                for step in range(args.max_steps):
                    # 행동 선택
                    action, cluster_id, latent_state = agent.select_action(
                        state, task['instruction']
                    )
                    
                    # 환경 스텝
                    next_state, reward, done, info = env.step(action)
                    episode_reward += reward
                    
                    # 전문가 업데이트
                    agent.update_expert(cluster_id, state, action, reward, next_state, done)
                    
                    state = next_state
                    
                    if done:
                        break
                
                task_rewards.append(episode_reward)
                
                if episode % 10 == 0:
                    stats = agent.get_stats()
                    recent_avg = np.mean(task_rewards[-10:])
                    print(f"  Episode {episode}: Reward={episode_reward:.2f}, "
                          f"Avg={recent_avg:.2f}, Clusters={stats['num_clusters']}")
            
            print(f"Task completed! Average reward: {np.mean(task_rewards):.2f}")
            
            # 태스크별 체크포인트 저장
            task_save_dir = os.path.join(args.save_dir, f"task_{task_idx + 1}")
            agent.save(task_save_dir)
        
        # 최종 모델 저장
        final_save_dir = os.path.join(args.save_dir, "final_model")
        agent.save(final_save_dir)
        
        # 최종 통계
        final_stats = agent.get_stats()
        print(f"\n🎉 Training completed!")
        print(f"  Final clusters: {final_stats['num_clusters']}")
        print(f"  Final experts: {final_stats['num_experts']}")
        print(f"  Model saved to: {args.save_dir}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        sim.close()

if __name__ == "__main__":
    main()