#!/usr/bin/env python3
"""
SB3 기반 MoE SAC 통합 학습 (환경 재사용 버전)
"""

import argparse
import os
import yaml
import numpy as np
import torch
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="SB3 MoE SAC Training")
    parser.add_argument("--tasks", type=str, default="tasks.yaml")
    parser.add_argument("--save_dir", type=str, default="./sb3_moe_outputs")
    parser.add_argument("--episodes_per_task", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--update_freq", type=int, default=10)
    parser.add_argument("--cluster_threshold", type=float, default=2.0)
    parser.add_argument("--latent_dim", type=int, default=32)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        from fixed_simple_gym_wrapper import SimpleAntPushWrapper
        from sb3_moe_agent import SB3MoEAgent
        from language_processor import LanguageProcessor
        from config import CONFIG

        print(f"🚀 SB3 MoE SAC Training (Single Environment)")
        print(f"  Tasks file: {args.tasks}")
        print(f"  Episodes per task: {args.episodes_per_task}")
        print(f"  Max steps per episode: {args.max_steps}")
        print(f"  Cluster threshold: {args.cluster_threshold}")
        
        # 출력 디렉토리 생성
        os.makedirs(args.save_dir, exist_ok=True)
        
        # 태스크 로드
        if os.path.exists(args.tasks):
            tasks = yaml.safe_load(open(args.tasks))
        else:
            # 기본 태스크 (동일한 환경 설정으로 통일)
            base_env_cfg = {
                'cube_position': [2.0, 0.0, 0.1],
                'goal_position': [3.0, 0.0, 0.1]
            }
            tasks = [
                {
                    'name': 'push_cube_basic',
                    'instruction': 'Push the blue cube to the goal position',
                    'env_cfg': base_env_cfg,
                    'episodes': args.episodes_per_task
                },
                {
                    'name': 'push_cube_far',
                    'instruction': 'Move the cube to the distant target',
                    'env_cfg': base_env_cfg,  # 같은 환경 사용
                    'episodes': args.episodes_per_task
                },
                {
                    'name': 'push_cube_left',
                    'instruction': 'Push the cube to the left side',
                    'env_cfg': base_env_cfg,  # 같은 환경 사용
                    'episodes': args.episodes_per_task
                },
                {
                    'name': 'push_cube_different',
                    'instruction': 'Move the object to the target location',
                    'env_cfg': base_env_cfg,  # 같은 환경 사용
                    'episodes': args.episodes_per_task
                }
            ]
        
        # 언어 처리기
        lang_processor = LanguageProcessor(device=CONFIG.device)
        
        # 단일 환경으로 모든 태스크 처리
        env = SimpleAntPushWrapper(custom_cfg=tasks[0]['env_cfg'])
        first_instr_emb = lang_processor.encode(tasks[0]['instruction'])
        instr_dim = first_instr_emb.shape[-1]
        
        # MoE 에이전트 생성
        agent = SB3MoEAgent(
            env=env,
            instr_dim=instr_dim,
            latent_dim=args.latent_dim,
            cluster_threshold=args.cluster_threshold
        )
        
        # 학습 통계
        all_stats = {
            'task_rewards': [],
            'cluster_evolution': [],
            'expert_counts': []
        }
        
        # 태스크별 학습 (같은 환경 재사용)
        for task_idx, task in enumerate(tasks):
            print(f"\n{'='*60}")
            print(f"🎯 Task {task_idx + 1}/{len(tasks)}: {task['name']}")
            print(f"📝 Instruction: {task['instruction']}")
            print(f"{'='*60}")
            
            # 지시사항 임베딩
            instr_emb = lang_processor.encode(task['instruction'])
            
            # 태스크 통계
            task_rewards = []
            
            # 에피소드 학습
            for episode in range(task['episodes']):
                print(f"\n--- Episode {episode + 1}/{task['episodes']} ---")
                
                # 환경 리셋 (같은 환경 재사용)
                state, _ = env.reset()
                episode_reward = 0
                step_count = 0
                
                while step_count < args.max_steps:
                    # 행동 선택
                    action, cluster_id = agent.select_action(
                        state, 
                        instr_emb.squeeze(0).detach().cpu().numpy(),
                        deterministic=False
                    )
                    
                    if step_count == 0:
                        print(f"  Assigned to cluster/expert: {cluster_id}")
                    
                    # 환경 스텝
                    next_state, reward, done, truncated, info = env.step(action)
                    episode_reward += reward
                    
                    # 경험 저장
                    agent.store_experience(state, action, reward, next_state, done, cluster_id)
                    
                    state = next_state
                    step_count += 1
                    
                    if done or truncated:
                        break
                
                # 주기적 업데이트
                if episode % args.update_freq == 0:
                    update_results = agent.update_experts(batch_size=32)
                    if update_results:
                        print(f"  📈 Updated experts: {list(update_results.keys())}")
                
                # 통계 수집
                task_rewards.append(episode_reward)
                agent_info = agent.get_info()
                
                # 주기적 출력
                if episode % 10 == 0 or episode == task['episodes'] - 1:
                    print(f"  📊 Episode {episode + 1} Summary:")
                    print(f"    Reward: {episode_reward:.2f}")
                    print(f"    Steps: {step_count}")
                    print(f"    Clusters: {agent_info['dpmm']['num_clusters']}")
                    print(f"    Experts: {agent_info['num_experts']}")
                    print(f"    Total buffer size: {agent_info['total_buffer_size']}")
                    
                    # 최근 평균 보상
                    recent_rewards = task_rewards[-10:]
                    if recent_rewards:
                        print(f"    Recent avg reward: {np.mean(recent_rewards):.2f}")
            
            # 태스크 완료 후 통계
            all_stats['task_rewards'].append(task_rewards)
            all_stats['cluster_evolution'].append(agent.get_info()['dpmm'])
            all_stats['expert_counts'].append(agent.get_info()['num_experts'])
            
            print(f"\n🏁 Task '{task['name']}' completed!")
            print(f"  Average reward: {np.mean(task_rewards):.2f}")
            print(f"  Best reward: {max(task_rewards):.2f}")
            print(f"  Final clusters: {agent.get_info()['dpmm']['num_clusters']}")
            print(f"  Final experts: {agent.get_info()['num_experts']}")
        
        # 최종 모델 저장
        final_save_path = os.path.join(args.save_dir, "final_sb3_moe_model")
        agent.save(final_save_path)
        
        # 학습 통계 저장
        stats_path = os.path.join(args.save_dir, "training_stats.npz")
        np.savez(stats_path, **all_stats)
        
        # 최종 요약
        print(f"\n🎉 All tasks completed!")
        print(f"  Total episodes: {sum(len(rewards) for rewards in all_stats['task_rewards'])}")
        all_rewards = [r for task_rewards in all_stats['task_rewards'] for r in task_rewards]
        print(f"  Overall average reward: {np.mean(all_rewards):.2f}")
        print(f"  Final clusters: {all_stats['cluster_evolution'][-1]['num_clusters']}")
        print(f"  Final experts: {all_stats['expert_counts'][-1]}")
        print(f"  Model saved to: {final_save_path}")
        
        env.close()
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        sim.close()

if __name__ == "__main__":
    main()