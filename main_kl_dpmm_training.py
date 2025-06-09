import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="KL-DPMM MoE SAC Training")
    parser.add_argument("--tasks", type=str, default="tasks.yaml")
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--tau_birth", type=float, default=1e-3)
    parser.add_argument("--tau_merge_kl", type=float, default=0.1)  # KL divergence 임계값
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--checkpoint_freq", type=int, default=20)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    import yaml
    import os
    from ant_push_env import IsaacAntPushEnv
    from config import CONFIG
    from language_processor import LanguageProcessor
    from integrated_dpmm_moe_sac import IntegratedKLDPMMMoESAC  # KL-DPMM 사용

    # 출력 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)

    # 태스크 로드
    tasks = yaml.safe_load(open(args.tasks))
    
    # 언어 처리기 초기화
    lang = LanguageProcessor(device=CONFIG.device)
    
    # 통합 에이전트 (아직 초기화하지 않음)
    agent = None
    
    # 학습 통계
    all_training_stats = {
        'tasks': [],
        'episode_rewards': [],
        'cluster_counts': [],
        'expert_counts': [],
        'merge_events': []
    }

    print(f"🚀 Starting KL-DPMM MoE-SAC Training")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Episodes per task: {args.episodes}")
    print(f"  KL merge threshold: {args.tau_merge_kl}")
    print(f"  Birth threshold: {args.tau_birth}")

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
        print(f"  Action space: {env.action_space.shape}")
        
        # 지시사항 임베딩
        instr_emb = lang.encode(task['instruction'])
        instr_dim = instr_emb.shape[-1]
        
        print(f"  Instruction embedding dim: {instr_dim}")
        
        # 에이전트 초기화 (첫 번째 태스크에서만)
        if agent is None:
            agent = IntegratedKLDPMMMoESAC(
                state_dim=state_dim,
                action_dim=action_dim,
                instr_dim=instr_dim,
                latent_dim=args.latent_dim,
                tau_birth=args.tau_birth,
                tau_merge_kl=args.tau_merge_kl,  # KL divergence 임계값
                device=str(CONFIG.device)
            )
            print(f"🤖 Agent initialized with KL-DPMM")
        
        # 태스크별 학습 통계
        task_stats = {
            'task_name': task['name'],
            'episode_rewards': [],
            'cluster_counts': [],
            'expert_counts': [],
            'kl_merge_events': []
        }
        
        # 에피소드 학습
        for episode in range(args.episodes):
            print(f"\n--- Episode {episode + 1}/{args.episodes} ---")
            
            # 환경 리셋
            state, _ = env.reset()
            episode_reward = 0
            step_count = 0
            max_steps = 500
            
            while step_count < max_steps:
                # 행동 선택
                action, cluster_id = agent.select_action(
                    state, 
                    instr_emb.squeeze(0).detach().cpu().numpy(),
                    deterministic=False
                )
                
                if step_count == 0:
                    print(f"  Assigned to cluster/expert: {cluster_id}")
                
                # 환경 스텝
                try:
                    next_state, reward, done, info = env.step(action)
                    episode_reward += reward
                    
                    # 경험 저장
                    agent.store_experience(state, action, reward, next_state, done, cluster_id)
                    
                    state = next_state
                    step_count += 1
                    
                    if done:
                        break
                        
                except Exception as e:
                    print(f"  ⚠️ Environment step error: {e}")
                    break
            
            # 에피소드 후 업데이트
            if episode % 5 == 0:  # 5 에피소드마다 업데이트
                update_info = agent.update_experts()
                print(f"  📈 Updated {update_info['updated_experts']}/{update_info['total_experts']} experts")
            
            # KL 기반 병합 (더 자주 시도)
            if episode % 3 == 0:  # 3 에피소드마다 병합 시도
                merged = agent.merge_clusters(verbose=(episode % 10 == 0))
                if merged:
                    task_stats['kl_merge_events'].append(episode)
            
            # 통계 수집
            stats = agent.get_statistics()
            task_stats['episode_rewards'].append(episode_reward)
            task_stats['cluster_counts'].append(stats['dpmm']['num_clusters'])
            task_stats['expert_counts'].append(stats['experts']['total_experts'])
            
            # 주기적 출력
            if episode % 10 == 0 or episode == args.episodes - 1:
                print(f"  📊 Episode {episode + 1} Summary:")
                print(f"    Reward: {episode_reward:.2f}")
                print(f"    Steps: {step_count}")
                print(f"    Clusters: {stats['dpmm']['num_clusters']}")
                print(f"    Experts: {stats['experts']['total_experts']}")
                print(f"    Total births: {stats['dpmm']['total_births']}")
                print(f"    Total KL-merges: {stats['dpmm']['total_merges']}")
                
                # 최근 평균 보상
                recent_rewards = task_stats['episode_rewards'][-10:]
                if recent_rewards:
                    print(f"    Recent avg reward: {np.mean(recent_rewards):.2f}")
        
        # 태스크 완료 후 체크포인트 저장
        if (task_idx + 1) % args.checkpoint_freq == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_task_{task_idx + 1}.pt")
            agent.save_checkpoint(checkpoint_path)
        
        # 태스크 통계 저장
        all_training_stats['tasks'].append(task_stats)
        all_training_stats['episode_rewards'].extend(task_stats['episode_rewards'])
        all_training_stats['cluster_counts'].extend(task_stats['cluster_counts'])
        all_training_stats['expert_counts'].extend(task_stats['expert_counts'])
        all_training_stats['merge_events'].extend(task_stats['kl_merge_events'])
        
        # 태스크 요약
        print(f"\n🏁 Task '{task['name']}' completed!")
        print(f"  Average reward: {np.mean(task_stats['episode_rewards']):.2f}")
        print(f"  Final clusters: {task_stats['cluster_counts'][-1]}")
        print(f"  Final experts: {task_stats['expert_counts'][-1]}")
        print(f"  KL-merge events: {len(task_stats['kl_merge_events'])}")
        
        env.close()
    
    # 최종 체크포인트 저장
    final_checkpoint_path = os.path.join(args.save_dir, "final_checkpoint.pt")
    agent.save_checkpoint(final_checkpoint_path)
    
    # 학습 통계 저장
    stats_path = os.path.join(args.save_dir, "training_stats.npz")
    np.savez(stats_path, **all_training_stats)
    
    # 최종 요약
    print(f"\n🎉 All tasks completed!")
    print(f"  Total episodes: {len(all_training_stats['episode_rewards'])}")
    print(f"  Average reward: {np.mean(all_training_stats['episode_rewards']):.2f}")
    print(f"  Final clusters: {all_training_stats['cluster_counts'][-1]}")
    print(f"  Final experts: {all_training_stats['expert_counts'][-1]}")
    print(f"  Total KL-merge events: {len(all_training_stats['merge_events'])}")
    print(f"  Checkpoints saved to: {args.save_dir}")

    sim.close()

if __name__ == "__main__":
    main()