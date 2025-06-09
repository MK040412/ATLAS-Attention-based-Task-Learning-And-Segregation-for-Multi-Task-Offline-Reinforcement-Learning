#!/usr/bin/env python3
"""
SB3 기반 KL-DPMM MoE SAC 병렬 학습 (DPMM 문제 해결 버전)
"""

import argparse
import torch
import numpy as np
import yaml
import os
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="SB3-based KL-DPMM MoE SAC Training")
    parser.add_argument("--tasks", type=str, default="tasks.yaml")
    parser.add_argument("--save_dir", type=str, default="./sb3_outputs")
    parser.add_argument("--num_envs", type=int, default=256)  # 병렬 환경
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--tau_birth", type=float, default=1e-3)
    parser.add_argument("--tau_merge_kl", type=float, default=0.1)  # KL divergence 임계값
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--checkpoint_freq", type=int, default=20)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        # SB3 기반 모듈들 import
        from fixed_simple_gym_wrapper import SimpleAntPushWrapper  # SB3 호환 환경
        from sb3_moe_agent import SB3MoEAgent
        from language_processor import LanguageProcessor
        from config import CONFIG

        # 출력 디렉토리 생성
        os.makedirs(args.save_dir, exist_ok=True)

        # 태스크 로드 (기본 태스크 생성)
        if os.path.exists(args.tasks):
            tasks = yaml.safe_load(open(args.tasks))
        else:
            # 기본 태스크들
            tasks = [
                {
                    'name': 'push_cube_basic',
                    'instruction': 'Push the blue cube to the goal position',
                    'env_cfg': {'cube_position': [2.0, 0.0, 0.1], 'goal_position': [3.0, 0.0, 0.1]}
                },
                {
                    'name': 'push_cube_far',
                    'instruction': 'Move the cube to the distant target',
                    'env_cfg': {'cube_position': [2.0, 0.0, 0.1], 'goal_position': [4.0, 0.0, 0.1]}
                },
                {
                    'name': 'push_cube_left',
                    'instruction': 'Push the cube to the left side',
                    'env_cfg': {'cube_position': [2.0, 0.0, 0.1], 'goal_position': [2.0, 2.0, 0.1]}
                }
            ]
        
        # 언어 처리기 초기화
        lang_processor = LanguageProcessor(device=CONFIG.device)
        
        # 통합 에이전트 (아직 초기화하지 않음)
        agent = None
        # 학습 통계
        all_training_stats = {
            'tasks': [],
            'episode_rewards': [],
            'cluster_counts': [],
            'expert_counts': [],
            'dpmm_events': []
        }
        print(f"🚀 Starting SB3-based KL-DPMM MoE-SAC Training")
        print(f"  Tasks: {len(tasks)}")
        print(f"  Episodes per task: {args.episodes}")
        print(f"  KL merge threshold: {args.tau_merge_kl}")
        print(f"  Birth threshold: {args.tau_birth}")
        print(f"  Parallel environments: {args.num_envs}")

        # 태스크별 학습
        for task_idx, task in enumerate(tasks):
            print(f"\n{'='*60}")
            print(f"🎯 Task {task_idx + 1}/{len(tasks)}: {task['name']}")
            print(f"📝 Instruction: {task['instruction']}")
            print(f"{'='*60}")
            
            # SB3 호환 환경 생성 (단일 환경으로 시작)
            env = SimpleAntPushWrapper(custom_cfg=task['env_cfg'])
            
            # 차원 확인
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            
            print(f"🔧 Environment Setup:")
            print(f"  State dim: {state_dim}")
            print(f"  Action dim: {action_dim}")
            
            # 지시사항 임베딩
            instr_emb = lang_processor.encode(task['instruction'])
            instr_dim = instr_emb.shape[-1]
            
            print(f"  Instruction embedding dim: {instr_dim}")
            
            # 에이전트 초기화 (첫 번째 태스크에서만)
            if agent is None:
                agent = SB3MoEAgent(
                    env=env,
                    instr_dim=instr_dim,
                    latent_dim=args.latent_dim,
                    cluster_threshold=args.tau_merge_kl  # KL divergence 임계값
                )
                
                # DPMM 파라미터 설정 (DPMM 문제 해결을 위한 안전한 설정)
                if hasattr(agent, 'set_dpmm_params'):
                    agent.set_dpmm_params(
                        tau_birth=args.tau_birth,
                        tau_merge=args.tau_merge_kl
                    )
                
                print(f"🤖 SB3 Agent initialized with KL-DPMM")
            
            # 태스크별 학습 통계
            task_stats = {
                'task_name': task['name'],
                'episode_rewards': [],
                'cluster_counts': [],
                'expert_counts': [],
                'dpmm_events': []
            }
            
            # 에피소드 학습 (DPMM 안정성을 위한 보수적 접근)
            for episode in range(args.episodes):
                if episode % 10 == 0:
                    print(f"\n--- Episode {episode + 1}/{args.episodes} ---")
                
                # 환경 리셋
                state, _ = env.reset()
                episode_reward = 0
                step_count = 0
                max_steps = 500
                
                while step_count < max_steps:
                    try:
                        # 행동 선택
                        action, cluster_id = agent.select_action(
                            state, 
                            instr_emb.squeeze(0).detach().cpu().numpy(),
                            deterministic=False
                        )
                        
                        if step_count == 0 and episode % 10 == 0:
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
                            
                    except Exception as e:
                        print(f"  ⚠️ Environment step error: {e}")
                        break
                
                # 에피소드 후 업데이트 (더 보수적으로)
                if episode % 10 == 0 and episode > 0:  # 10 에피소드마다 업데이트
                    try:
                        update_info = agent.update_experts(batch_size=32)
                        if update_info and episode % 20 == 0:
                            print(f"  📈 Updated experts: {list(update_info.keys())}")
                    except Exception as e:
                        print(f"  ⚠️ Update error: {e}")
                
                # KL 기반 병합 (DPMM 문제 해결을 위해 더 신중하게)
                if episode % 20 == 0 and episode > 0:  # 20 에피소드마다 병합 시도
                    try:
                        merged = safe_merge_clusters(agent, verbose=(episode % 40 == 0))
                        if merged:
                            task_stats['dpmm_events'].append({
                                'episode': episode,
                                'type': 'merge',
                                'info': merged
                            })
                    except Exception as e:
                        print(f"  ⚠️ Merge error: {e}")
                
                # 통계 수집 (안전하게)
                task_stats['episode_rewards'].append(episode_reward)
                
                try:
                    stats = agent.get_info()
                    task_stats['cluster_counts'].append(stats.get('dpmm', {}).get('num_clusters', 0))
                    task_stats['expert_counts'].append(stats.get('num_experts', 0))
                except:
                    task_stats['cluster_counts'].append(0)
                    task_stats['expert_counts'].append(0)
                
                # 주기적 출력
                if episode % 20 == 0 and episode > 0:
                    try:
                        stats = agent.get_info()
                        recent_rewards = task_stats['episode_rewards'][-10:]
                        print(f"  📊 Episode {episode + 1} Summary:")
                        print(f"    Recent avg reward: {np.mean(recent_rewards):.2f}")
                        print(f"    Episode reward: {episode_reward:.2f}")
                        print(f"    Steps: {step_count}")
                        print(f"    Clusters: {stats.get('dpmm', {}).get('num_clusters', 'N/A')}")
                        print(f"    Experts: {stats.get('num_experts', 'N/A')}")
                    except Exception as e:
                        print(f"  ⚠️ Stats error: {e}")
            
            # 태스크 완료 후 체크포인트 저장
            if (task_idx + 1) % args.checkpoint_freq == 0:
                checkpoint_path = os.path.join(args.save_dir, f"sb3_checkpoint_task_{task_idx + 1}.pt")
                try:
                    agent.save(checkpoint_path)
                    print(f"💾 Checkpoint saved: {checkpoint_path}")
                except Exception as e:
                    print(f"⚠️ Checkpoint save error: {e}")
            
            # 태스크 통계 저장
            all_training_stats['tasks'].append(task_stats)
            all_training_stats['episode_rewards'].extend(task_stats['episode_rewards'])
            all_training_stats['cluster_counts'].extend(task_stats['cluster_counts'])
            all_training_stats['expert_counts'].extend(task_stats['expert_counts'])
            all_training_stats['dpmm_events'].extend(task_stats['dpmm_events'])
            
            # 태스크 요약
            print(f"\n🏁 Task '{task['name']}' completed!")
            print(f"  Average reward: {np.mean(task_stats['episode_rewards']):.2f}")
            print(f"  Final clusters: {task_stats['cluster_counts'][-1] if task_stats['cluster_counts'] else 0}")
            print(f"  Final experts: {task_stats['expert_counts'][-1] if task_stats['expert_counts'] else 0}")
            print(f"  DPMM events: {len(task_stats['dpmm_events'])}")
            
            env.close()
        
        # 최종 체크포인트 저장
        final_checkpoint_path = os.path.join(args.save_dir, "final_sb3_checkpoint.pt")
        try:
            agent.save(final_checkpoint_path)
            print(f"💾 Final checkpoint saved: {final_checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Final checkpoint save error: {e}")
        
        # 학습 통계 저장
        stats_path = os.path.join(args.save_dir, "sb3_training_stats.npz")
        try:
            np.savez(stats_path, **all_training_stats)
            print(f"📊 Training stats saved: {stats_path}")
        except Exception as e:
            print(f"⚠️ Stats save error: {e}")
        
        # 최종 요약
        print(f"\n🎉 All tasks completed!")
        print(f"  Total episodes: {len(all_training_stats['episode_rewards'])}")
        if all_training_stats['episode_rewards']:
            print(f"  Average reward: {np.mean(all_training_stats['episode_rewards']):.2f}")
        print(f"  Final clusters: {all_training_stats['cluster_counts'][-1] if all_training_stats['cluster_counts'] else 0}")
        print(f"  Final experts: {all_training_stats['expert_counts'][-1] if all_training_stats['expert_counts'] else 0}")
        print(f"  Total DPMM events: {len(all_training_stats['dpmm_events'])}")
        print(f"  Checkpoints saved to: {args.save_dir}")

    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        sim.close()

def safe_merge_clusters(agent, verbose=False):
    """DPMM 문제 해결을 위한 안전한 클러스터 병합"""
    try:
        if hasattr(agent, 'merge_clusters'):
            return agent.merge_clusters(verbose=verbose)
        elif hasattr(agent, 'dpmm') and hasattr(agent.dpmm, 'merge'):
            agent.dpmm.merge(verbose=verbose)
            return True
        else:
            if verbose:
                print("  ⚠️ No merge method available")
            return False
    except Exception as e:
        if verbose:
            print(f"  ⚠️ Safe merge failed: {e}")
        return False

def create_default_tasks():
    """기본 태스크 생성"""
    return [
        {
            'name': 'push_cube_basic',
            'instruction': 'Push the blue cube to the goal position',
            'env_cfg': {
                'cube_position': [2.0, 0.0, 0.1],
                'goal_position': [3.0, 0.0, 0.1]
            }
        },
        {
            'name': 'push_cube_far',
            'instruction': 'Move the cube to the distant target location',
            'env_cfg': {
                'cube_position': [2.0, 0.0, 0.1],
                'goal_position': [4.0, 0.0, 0.1]
            }
        },
        {
            'name': 'push_cube_left',
            'instruction': 'Push the cube to the left side target',
            'env_cfg': {
                'cube_position': [2.0, 0.0, 0.1],
                'goal_position': [2.0, 2.0, 0.1]
            }
        },
        {
            'name': 'push_cube_right',
            'instruction': 'Move the object to the right side goal',
            'env_cfg': {
                'cube_position': [2.0, 0.0, 0.1],
                'goal_position': [2.0, -2.0, 0.1]
            }
        }
    ]

if __name__ == "__main__":
    main()
