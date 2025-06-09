#!/usr/bin/env python3
"""
RTX 4060 최적화 통합 KL-DPMM MoE-SAC 학습 스크립트
"""

import argparse
import torch
import numpy as np
import yaml
import os
import gc
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="RTX 4060 Integrated KL-DPMM MoE-SAC Training")
    parser.add_argument("--tasks", type=str, default="tasks.yaml")
    parser.add_argument("--save_dir", type=str, default="./rtx4060_outputs")
    parser.add_argument("--num_envs", type=int, default=64)  # RTX 4060에 맞게 축소
    parser.add_argument("--episodes_per_task", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--tau_birth", type=float, default=1e-3)
    parser.add_argument("--tau_merge_kl", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--update_freq", type=int, default=10)
    parser.add_argument("--merge_freq", type=int, default=15)
    parser.add_argument("--checkpoint_freq", type=int, default=25)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # RTX 4060 메모리 최적화 설정
    torch.backends.cudnn.benchmark = False  # 메모리 절약
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 메모리 분할 최적화
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    app = AppLauncher(vars(args))
    sim = app.app

    # 필요한 모듈들 import
    from ant_push_env import IsaacAntPushEnv
    from integrated_kl_dpmm_moe_sac import IntegratedKLDPMMMoESAC

    print(f"🚀 RTX 4060 Optimized Integrated Training Started")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Episodes per task: {args.episodes_per_task}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Birth threshold: {args.tau_birth}")
    print(f"  KL merge threshold: {args.tau_merge_kl}")

    # 출력 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)

    # 태스크 로드
    tasks = yaml.safe_load(open(args.tasks))
    
    # 통합 에이전트 초기화 (아직 하지 않음)
    agent = None
    
    # 전체 학습 통계
    all_stats = {
        'task_names': [],
        'episode_rewards': [],
        'cluster_evolution': [],
        'expert_evolution': [],
        'merge_events': [],
        'memory_usage': []
    }

    try:
        # 태스크별 학습
        for task_idx, task in enumerate(tasks):
            print(f"\n{'='*60}")
            print(f"🎯 Task {task_idx + 1}/{len(tasks)}: {task['name']}")
            print(f"📝 Instruction: {task['instruction']}")
            print(f"{'='*60}")
            
            # 환경 생성
            env = IsaacAntPushEnv(custom_cfg=task['env_cfg'], num_envs=args.num_envs)
            
            # 초기 상태로 차원 확인
            state, _ = env.reset()
            state_dim = state.shape[0]
            action_dim = env.action_space.shape[1] if len(env.action_space.shape) == 2 else env.action_space.shape[0]
            
            print(f"🔧 Environment Setup:")
            print(f"  State dim: {state_dim}")
            print(f"  Action dim: {action_dim}")
            
            # 에이전트 초기화 (첫 번째 태스크에서만)
            if agent is None:
                agent = IntegratedKLDPMMMoESAC(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    instr_dim=384,  # sentence transformer 출력 차원
                    latent_dim=32,
                    tau_birth=args.tau_birth,
                    tau_merge_kl=args.tau_merge_kl,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                print(f"🤖 Agent initialized")
            
            # 태스크별 통계
            task_stats = {
                'episode_rewards': [],
                'cluster_counts': [],
                'expert_counts': [],
                'merge_events': []
            }
            
            # 에피소드 학습
            for episode in range(args.episodes_per_task):
                print(f"\n--- Episode {episode + 1}/{args.episodes_per_task} ---")
                
                # 환경 리셋
                state, _ = env.reset()
                episode_reward = 0
                step_count = 0
                episode_experiences = []
                
                while step_count < args.max_steps:
                    # 행동 선택
                    action, cluster_id = agent.select_action(
                        state, 
                        task['instruction'],
                        deterministic=False
                    )
                    
                    if step_count == 0:
                        print(f"  🎯 Assigned to cluster/expert: {cluster_id}")
                    
                    # 환경 스텝
                    try:
                        next_state, reward, done, info = env.step(action)
                        episode_reward += reward
                        
                        # 경험 저장
                        agent.store_experience(
                            state, action, reward, next_state, done,
                            task['instruction'], cluster_id
                        )
                        
                        state = next_state
                        step_count += 1
                        
                        if done:
                            break
                            
                    except Exception as e:
                        print(f"  ⚠️ Environment step error: {e}")
                        break
                
                # 주기적 업데이트
                if episode % args.update_freq == 0:
                    update_info = agent.update_experts(batch_size=args.batch_size)
                    if episode % (args.update_freq * 2) == 0:
                        print(f"  📈 Updated {update_info['updated_experts']}/{update_info['total_experts']} experts")
                
                # KL 기반 병합
                if episode % args.merge_freq == 0:
                    merged = agent.merge_clusters(verbose=(episode % (args.merge_freq * 2) == 0))
                    if merged:
                        task_stats['merge_events'].append(episode)
                
                # 통계 수집
                stats = agent.get_statistics()
                task_stats['episode_rewards'].append(episode_reward)
                task_stats['cluster_counts'].append(stats['dpmm']['num_clusters'])
                task_stats['expert_counts'].append(stats['experts']['total_experts'])
                
                # 메모리 정리 (RTX 4060 최적화)
                if episode % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # 주기적 출력
                if episode % 10 == 0 or episode == args.episodes_per_task - 1:
                    print(f"  📊 Episode {episode + 1} Summary:")
                    print(f"    Reward: {episode_reward:.2f}")
                    print(f"    Steps: {step_count}")
                    print(f"    Clusters: {stats['dpmm']['num_clusters']}")
                    print(f"    Experts: {stats['experts']['total_experts']}")
                    print(f"    Total births: {stats['dpmm']['total_births']}")
                    print(f"    Total KL-merges: {stats['dpmm']['total_merges']}")
                    
                    # GPU 메모리 사용량
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
                        print(f"    GPU Memory: {memory_used:.1f} MB")
                        all_stats['memory_usage'].append(memory_used)
                    
                    # 최근 평균 보상
                    recent_rewards = task_stats['episode_rewards'][-10:]
                    if recent_rewards:
                        print(f"    Recent avg reward: {np.mean(recent_rewards):.2f}")
            
            # 태스크 완료 후 체크포인트 저장
            if (task_idx + 1) % args.checkpoint_freq == 0:
                checkpoint_path = os.path.join(args.save_dir, f"checkpoint_task_{task_idx + 1}.pt")
                agent.save_checkpoint(checkpoint_path)
            
            # 전체 통계에 추가
            all_stats['task_names'].append(task['name'])
            all_stats['episode_rewards'].extend(task_stats['episode_rewards'])
            all_stats['cluster_evolution'].extend(task_stats['cluster_counts'])
            all_stats['expert_evolution'].extend(task_stats['expert_counts'])
            all_stats['merge_events'].extend(task_stats['merge_events'])
            
            # 태스크 요약
            print(f"\n🏁 Task '{task['name']}' completed!")
            print(f"  Average reward: {np.mean(task_stats['episode_rewards']):.2f}")
            print(f"  Final clusters: {task_stats['cluster_counts'][-1]}")
            print(f"  Final experts: {task_stats['expert_counts'][-1]}")
            print(f"  KL-merge events: {len(task_stats['merge_events'])}")
            
            env.close()
            
            # 메모리 정리
            torch.cuda.empty_cache()
            gc.collect()
        
        # 최종 체크포인트 저장
        final_checkpoint_path = os.path.join(args.save_dir, "final_checkpoint.pt")
        agent.save_checkpoint(final_checkpoint_path)
        
        # 학습 통계 저장
        stats_path = os.path.join(args.save_dir, "training_stats.npz")
        np.savez(stats_path, **all_stats)
        
        # 최종 분석 및 시각화
        print(f"\n🎉 All tasks completed!")
        analyze_training_results(all_stats, args.save_dir)
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # 에러 발생 시에도 체크포인트 저장 시도
        if agent is not None:
            try:
                emergency_path = os.path.join(args.save_dir, "emergency_checkpoint.pt")
                agent.save_checkpoint(emergency_path)
                print(f"🆘 Emergency checkpoint saved to {emergency_path}")
            except:
                pass
    
    finally:
        sim.close()
        torch.cuda.empty_cache()

def analyze_training_results(stats: dict, save_dir: str):
    """학습 결과 분석 및 시각화"""
    print(f"\n🔬 TRAINING ANALYSIS")
    print(f"="*50)
    
    # 기본 통계
    total_episodes = len(stats['episode_rewards'])
    avg_reward = np.mean(stats['episode_rewards'])
    final_clusters = stats['cluster_evolution'][-1] if stats['cluster_evolution'] else 0
    final_experts = stats['expert_evolution'][-1] if stats['expert_evolution'] else 0
    total_merges = len(stats['merge_events'])
    
    print(f"📊 Basic Statistics:")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Average reward: {avg_reward:.3f}")
    print(f"  Final clusters: {final_clusters}")
    print(f"  Final experts: {final_experts}")
    print(f"  Total KL-merges: {total_merges}")
    
    if stats['memory_usage']:
        avg_memory = np.mean(stats['memory_usage'])
        max_memory = np.max(stats['memory_usage'])
        print(f"  Average GPU memory: {avg_memory:.1f} MB")
        print(f"  Peak GPU memory: {max_memory:.1f} MB")
    
    # 학습 곡선 분석
    if len(stats['episode_rewards']) > 10:
        early_rewards = np.mean(stats['episode_rewards'][:10])
        late_rewards = np.mean(stats['episode_rewards'][-10:])
        improvement = late_rewards - early_rewards
        print(f"  Learning improvement: {improvement:.3f}")
        print(f"  Early avg reward: {early_rewards:.3f}")
        print(f"  Late avg reward: {late_rewards:.3f}")
    
    # 클러스터 진화 분석
    if stats['cluster_evolution']:
        max_clusters = np.max(stats['cluster_evolution'])
        cluster_stability = np.std(stats['cluster_evolution'][-20:]) if len(stats['cluster_evolution']) > 20 else 0
        print(f"  Max clusters reached: {max_clusters}")
        print(f"  Recent cluster stability (std): {cluster_stability:.2f}")
    
    # 간단한 플롯 생성 (matplotlib 사용 가능한 경우)
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 보상 곡선
        axes[0, 0].plot(stats['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # 클러스터 진화
        if stats['cluster_evolution']:
            axes[0, 1].plot(stats['cluster_evolution'])
            axes[0, 1].set_title('Cluster Evolution')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Number of Clusters')
        
        # 전문가 진화
        if stats['expert_evolution']:
            axes[1, 0].plot(stats['expert_evolution'])
            axes[1, 0].set_title('Expert Evolution')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Number of Experts')
        
        # 메모리 사용량
        if stats['memory_usage']:
            axes[1, 1].plot(stats['memory_usage'])
            axes[1, 1].set_title('GPU Memory Usage')
            axes[1, 1].set_xlabel('Checkpoint')
            axes[1, 1].set_ylabel('Memory (MB)')
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, "training_analysis.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training plots saved to {plot_path}")
        
    except ImportError:
        print("📈 Matplotlib not available, skipping plots")
    except Exception as e:
        print(f"📈 Plot generation failed: {e}")

if __name__ == "__main__":
    main()
