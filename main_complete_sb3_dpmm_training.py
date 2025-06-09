#!/usr/bin/env python3
"""
완전한 SB3-DPMM 통합 학습 (Birth 문제 해결 + 병렬 환경 + 실시간 시각화)
"""

import argparse
import torch
import numpy as np
import yaml
import os
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Complete SB3-DPMM MoE SAC Training (Fixed)")
    parser.add_argument("--tasks", type=str, default="tasks.yaml")
    parser.add_argument("--save_dir", type=str, default="./fixed_sb3_outputs")
    parser.add_argument("--num_envs", type=int, default=256)  # 병렬 환경 지원
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500)
    
    # 🔧 개선된 DPMM 파라미터
    parser.add_argument("--tau_birth", type=float, default=-2.0)
    parser.add_argument("--tau_merge_kl", type=float, default=0.3)
    parser.add_argument("--initial_variance", type=float, default=0.1)
    parser.add_argument("--max_clusters", type=int, default=15)
    parser.add_argument("--target_birth_rate", type=float, default=0.12)
    
    parser.add_argument("--latent_dim", type=int, default=24)
    parser.add_argument("--update_freq", type=int, default=8)
    parser.add_argument("--merge_freq", type=int, default=15)
    parser.add_argument("--checkpoint_freq", type=int, default=20)
    parser.add_argument("--monitor_freq", type=int, default=5)
    parser.add_argument("--plot_freq", type=int, default=25)  # 실시간 플롯 주기
    
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        # 필요한 모듈들 import
        from ant_push_env import IsaacAntPushEnv  # 병렬 환경 지원
        from improved_sb3_dpmm_agent import ImprovedSB3DPMMAgent
        from language_processor import LanguageProcessor
        from experiment_visualization import ExperimentVisualizer, analyze_birth_problem
        from config import CONFIG

        print(f"🚀 Fixed SB3-DPMM Training Started")
        print(f"  🔧 Birth Control Parameters:")
        print(f"    • tau_birth: {args.tau_birth} (was -5.0)")
        print(f"    • tau_merge_kl: {args.tau_merge_kl} (was 0.1)")
        print(f"    • initial_variance: {args.initial_variance}")
        print(f"    • max_clusters: {args.max_clusters}")
        print(f"    • target_birth_rate: {args.target_birth_rate * 100:.1f}%")
        print(f"  📊 Training Parameters:")
        print(f"    • Parallel environments: {args.num_envs}")
        print(f"    • Episodes per task: {args.episodes}")
        print(f"    • Latent dimension: {args.latent_dim}")

        # 출력 디렉토리 생성
        os.makedirs(args.save_dir, exist_ok=True)
        
        # 실시간 시각화 초기화
        visualizer = ExperimentVisualizer(save_dir=args.save_dir)

        # 태스크 로드
        tasks = load_or_create_tasks(args.tasks, args.episodes)
        
        # 언어 처리기
        lang_processor = LanguageProcessor(device=CONFIG.device)
        
        # 통합 에이전트 (첫 번째 태스크에서 초기화)
        agent = None
        
        # 전체 학습 통계 (Birth 모니터링 추가)
        all_training_stats = {
            'tasks': [],
            'episode_rewards': [],
            'cluster_evolution': [],
            'expert_evolution': [],
            'dpmm_events': [],
            'birth_rate_history': [],
            'tau_birth_history': [],
            'merge_history': []
        }

        # 태스크별 학습
        for task_idx, task in enumerate(tasks):
            print(f"\n{'='*70}")
            print(f"🎯 Task {task_idx + 1}/{len(tasks)}: {task['name']}")
            print(f"📝 Instruction: {task['instruction']}")
            print(f"{'='*70}")
            
            # 병렬 환경 생성 (main_kl_dpmm_training.py 방식)
            env = IsaacAntPushEnv(custom_cfg=task['env_cfg'], num_envs=args.num_envs)
            
            # 초기 상태로 차원 확인
            state, _ = env.reset()
            state_dim = state.shape[0] if len(state.shape) == 1 else state.shape[1]
            action_dim = env.action_space.shape[1] if len(env.action_space.shape) == 2 else env.action_space.shape[0]
            
            print(f"🔧 Environment Setup:")
            print(f"  • State dim: {state_dim}")
            print(f"  • Action dim: {action_dim}")
            print(f"  • Parallel envs: {args.num_envs}")
            print(f"  • Action space: {env.action_space.shape}")
            
            # 지시사항 임베딩
            instr_emb = lang_processor.encode(task['instruction'])
            instr_dim = instr_emb.shape[-1]
            
            print(f"  • Instruction embedding dim: {instr_dim}")
            
            # 에이전트 초기화 (첫 번째 태스크에서만)
            if agent is None:
                agent = ImprovedSB3DPMMAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    instr_dim=instr_dim,
                    latent_dim=args.latent_dim,
                    tau_birth=args.tau_birth,
                    tau_merge_kl=args.tau_merge_kl,
                    initial_variance=args.initial_variance,
                    max_clusters=args.max_clusters,
                    target_birth_rate=args.target_birth_rate,
                    device=str(CONFIG.device)
                )
                print(f"🤖 Improved SB3-DPMM Agent initialized with birth control")
            
            # 태스크별 통계
            task_stats = {
                'task_name': task['name'],
                'episode_rewards': [],
                'cluster_counts': [],
                'expert_counts': [],
                'birth_rates': [],
                'dpmm_events': [],
                'early_stop_triggered': False
            }
            
            # Birth rate 경고 카운터
            high_birth_rate_warnings = 0
            
            # 에피소드 학습
            for episode in range(args.episodes):
                print(f"\n--- Episode {episode + 1}/{args.episodes} ---")
                
                # 환경 리셋
                state, _ = env.reset()
                episode_reward = 0
                step_count = 0
                
                while step_count < args.max_steps:
                    try:
                        # 행동 선택 (병렬 환경 처리)
                        if len(state.shape) == 2:  # 병렬 환경
                            # 첫 번째 환경만 사용하거나 평균 사용
                            single_state = state[0] if state.shape[0] > 1 else state.squeeze(0)
                        else:
                            single_state = state
                        
                        action, cluster_id = agent.select_action(
                            single_state, 
                            instr_emb.squeeze(0).detach().cpu().numpy(),
                            deterministic=False
                        )
                        
                        if step_count == 0:
                            print(f"  🎯 Assigned to cluster/expert: {cluster_id}")
                        
                        # 병렬 환경을 위한 액션 확장
                        if args.num_envs > 1:
                            # 모든 환경에 같은 액션 적용
                            action_batch = np.tile(action, (args.num_envs, 1))
                        else:
                            action_batch = action
                        
                        # 환경 스텝
                        next_state, reward, done, info = env.step(action_batch)
                        
                        # 병렬 환경 처리
                        if isinstance(reward, np.ndarray):
                            episode_reward += reward.mean()  # 평균 보상
                            done = done.any()  # 하나라도 끝나면 종료
                        else:
                            episode_reward += reward
                        
                        # 경험 저장 (단일 상태로 변환)
                        if len(next_state.shape) == 2:
                            next_single_state = next_state[0]
                        else:
                            next_single_state = next_state
                        
                        agent.store_experience(
                            single_state, action, reward if not isinstance(reward, np.ndarray) else reward.mean(), 
                            next_single_state, done, cluster_id
                        )
                        
                        state = next_state
                        step_count += 1
                        
                        if done:
                            break
                            
                    except Exception as e:
                        print(f"  ⚠️ Episode step error: {e}")
                        break
                
                # 주기적 업데이트
                if episode % args.update_freq == 0 and episode > 0:
                    try:
                        update_results = agent.update_experts(batch_size=128)
                        if update_results and episode % (args.update_freq * 2) == 0:
                            updated_experts = len([k for k, v in update_results.items() if v is not None])
                            print(f"  📈 Updated {updated_experts} experts")
                    except Exception as e:
                        print(f"  ⚠️ Expert update error: {e}")
                
                # DPMM 클러스터 병합
                if episode % args.merge_freq == 0 and episode > 0:
                    try:
                        merged_count = agent.merge_clusters_aggressive(verbose=(episode % (args.merge_freq * 2) == 0))
                        if merged_count > 0:
                            task_stats['dpmm_events'].append({
                                'episode': episode,
                                'type': 'merge',
                                'count': merged_count
                            })
                            all_training_stats['merge_history'].append(episode)
                    except Exception as e:
                        print(f"  ⚠️ Cluster merge error: {e}")
                
                # 🔍 Birth Rate 모니터링
                if episode % args.monitor_freq == 0 and episode > 0:
                    try:
                        dpmm_stats = agent.get_dpmm_statistics()
                        current_birth_rate = dpmm_stats.get('recent_birth_rate', 0.0)
                        current_tau_birth = dpmm_stats.get('current_tau_birth', args.tau_birth)
                        
                        task_stats['birth_rates'].append(current_birth_rate)
                        all_training_stats['birth_rate_history'].append(current_birth_rate)
                        all_training_stats['tau_birth_history'].append(current_tau_birth)
                        
                        # 높은 birth rate 경고
                        if current_birth_rate > 0.25:
                            high_birth_rate_warnings += 1
                            print(f"  ⚠️ HIGH BIRTH RATE: {current_birth_rate:.1%} (tau_birth: {current_tau_birth:.3f})")
                            
                            if high_birth_rate_warnings >= 3:
                                print(f"  🛑 Triggering emergency cluster merge...")
                                emergency_merges = agent.emergency_cluster_reduction()
                                print(f"  🔧 Emergency merged {emergency_merges} clusters")
                                high_birth_rate_warnings = 0
                        
                        elif episode % (args.monitor_freq * 3) == 0:
                            print(f"  📊 Birth rate: {current_birth_rate:.1%}, Clusters: {dpmm_stats.get('num_clusters', 0)}")
                        
                    except Exception as e:
                        print(f"  ⚠️ Birth rate monitoring error: {e}")
                
                # 통계 수집
                task_stats['episode_rewards'].append(episode_reward)
                
                try:
                    agent_info = agent.get_info()
                    task_stats['cluster_counts'].append(agent_info['dpmm']['num_clusters'])
                    task_stats['expert_counts'].append(agent_info['num_experts'])
                except:
                    task_stats['cluster_counts'].append(0)
                    task_stats['expert_counts'].append(0)
                
                # 클러스터 폭발 조기 감지
                current_clusters = task_stats['cluster_counts'][-1] if task_stats['cluster_counts'] else 0
                if current_clusters > args.max_clusters * 1.5:
                    print(f"  🚨 CLUSTER EXPLOSION DETECTED: {current_clusters} clusters!")
                    print(f"  🛑 Triggering emergency stop and reset...")
                    task_stats['early_stop_triggered'] = True
                    break
                
                # 📊 실시간 시각화 (주기적)
                if episode % args.plot_freq == 0 and episode > 0:
                    try:
                        # 현재까지의 통계로 임시 시각화
                        temp_stats = {
                            'episode_rewards': task_stats['episode_rewards'],
                            'cluster_evolution': task_stats['cluster_counts'],
                            'expert_evolution': task_stats['expert_counts'],
                            'dpmm_events': task_stats['dpmm_events']
                        }
                        
                        print(f"  📈 Generating real-time plots...")
                        visualizer.plot_training_curves(
                            temp_stats, 
                            title=f"Task {task_idx+1}: {task['name']} - Episode {episode}"
                        )
                        
                        if len(task_stats['dpmm_events']) > 0:
                            analyze_birth_problem(temp_stats)
                        
                    except Exception as e:
                        print(f"  ⚠️ Real-time plotting error: {e}")
                
                # 주기적 상세 출력
                if episode % 20 == 0 and episode > 0:
                    try:
                        agent_info = agent.get_info()
                        dpmm_stats = agent.get_dpmm_statistics()
                        recent_rewards = task_stats['episode_rewards'][-10:]
                        
                        print(f"  📊 Episode {episode + 1} Detailed Summary:")
                        print(f"    • Episode reward: {episode_reward:.2f}")
                        print(f"    • Recent avg reward: {np.mean(recent_rewards):.2f}")
                        print(f"    • Steps: {step_count}")
                        print(f"    • Clusters: {agent_info['dpmm']['num_clusters']}")
                        print(f"    • Experts: {agent_info['num_experts']}")
                        print(f"    • Birth rate: {dpmm_stats.get('recent_birth_rate', 0):.1%}")
                        print(f"    • Total births: {dpmm_stats.get('total_births', 0)}")
                        print(f"    • Total merges: {dpmm_stats.get('total_merges', 0)}")
                        print(f"    • Current tau_birth: {dpmm_stats.get('current_tau_birth', 0):.3f}")
                        
                    except Exception as e:
                        print(f"  ⚠️ Stats display error: {e}")
            
            # 태스크 완료 후 체크포인트 저장
            if (task_idx + 1) % args.checkpoint_freq == 0:
                checkpoint_path = os.path.join(args.save_dir, f"fixed_checkpoint_task_{task_idx + 1}")
                try:
                    agent.save(checkpoint_path)
                    print(f"💾 Checkpoint saved: {checkpoint_path}")
                except Exception as e:
                    print(f"⚠️ Checkpoint save error: {e}")
            
            # 태스크 통계 저장
            all_training_stats['tasks'].append(task_stats)
            all_training_stats['episode_rewards'].extend(task_stats['episode_rewards'])
            all_training_stats['cluster_evolution'].extend(task_stats['cluster_counts'])
            all_training_stats['expert_evolution'].extend(task_stats['expert_counts'])
            all_training_stats['dpmm_events'].extend(task_stats['dpmm_events'])
            
            # 태스크 요약 (개선된 출력)
            print(f"\n🏁 Task '{task['name']}' completed!")
            print(f"  📈 Performance:")
            print(f"    • Average reward: {np.mean(task_stats['episode_rewards']):.2f}")
            print(f"    • Best reward: {max(task_stats['episode_rewards']):.2f}")
            print(f"    • Final reward (last 10): {np.mean(task_stats['episode_rewards'][-10:]):.2f}")
            print(f"  🧠 Model Status:")
            print(f"    • Final clusters: {task_stats['cluster_counts'][-1] if task_stats['cluster_counts'] else 0}")
            print(f"    • Final experts: {task_stats['expert_counts'][-1] if task_stats['expert_counts'] else 0}")
            print(f"    • DPMM events: {len(task_stats['dpmm_events'])}")
            print(f"    • Early stop: {'Yes' if task_stats['early_stop_triggered'] else 'No'}")
            
            # 태스크별 상세 시각화
            try:
                print(f"  📊 Generating task summary plots...")
                task_specific_stats = {
                    'episode_rewards': task_stats['episode_rewards'],
                    'cluster_evolution': task_stats['cluster_counts'],
                    'expert_evolution': task_stats['expert_counts'],
                    'dpmm_events': task_stats['dpmm_events']
                }
                
                visualizer.plot_training_curves(
                    task_specific_stats,
                    title=f"Task {task_idx+1} Complete: {task['name']}"
                )
                
                # DPMM 상세 분석
                if len(task_stats['dpmm_events']) > 5:  # 충분한 이벤트가 있을 때만
                    visualizer.plot_dpmm_analysis(task_specific_stats)
                
            except Exception as e:
                print(f"  ⚠️ Task visualization error: {e}")
            
            env.close()
        
        # 최종 모델 저장
        final_checkpoint_path = os.path.join(args.save_dir, "final_fixed_model")
        try:
            agent.save(final_checkpoint_path)
            print(f"💾 Final model saved: {final_checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Final save error: {e}")
        
        # 학습 통계 저장
        stats_path = os.path.join(args.save_dir, "fixed_training_stats.npz")
        try:
            np.savez(stats_path, **all_training_stats)
            print(f"📊 Training stats saved: {stats_path}")
        except Exception as e:
            print(f"⚠️ Stats save error: {e}")
        
        # 🎨 최종 종합 시각화 및 분석
        print(f"\n🎨 Generating comprehensive analysis...")
        try:
            task_names = [task['name'] for task in tasks]
            
            # 종합 보고서 생성
            visualizer.create_comprehensive_report(all_training_stats, task_names)
            
            # Birth 문제 최종 분석
            print(f"\n🔬 Final Birth Problem Analysis:")
            analyze_birth_problem(all_training_stats)
            
            # 하이퍼파라미터 효과 분석 (가능한 경우)
            if len(all_training_stats['birth_rate_history']) > 10:
                print(f"\n📈 Birth Rate Evolution Analysis:")
                birth_rates = all_training_stats['birth_rate_history']
                tau_history = all_training_stats['tau_birth_history']
                
                print(f"  • Initial birth rate: {birth_rates[0]:.1%}")
                print(f"  • Final birth rate: {birth_rates[-1]:.1%}")
                print(f"  • Average birth rate: {np.mean(birth_rates):.1%}")
                print(f"  • Birth rate std: {np.std(birth_rates):.1%}")
                print(f"  • Initial tau_birth: {tau_history[0]:.3f}")
                print(f"  • Final tau_birth: {tau_history[-1]:.3f}")
                print(f"  • Tau adaptation range: {max(tau_history) - min(tau_history):.3f}")
            
        except Exception as e:
            print(f"⚠️ Final visualization error: {e}")
        
        # 최종 요약 (개선된 형식)
        print(f"\n" + "="*80)
        print(f"🎉 FIXED SB3-DPMM TRAINING COMPLETED!")
        print(f"="*80)
        
        total_episodes = len(all_training_stats['episode_rewards'])
        overall_avg_reward = np.mean(all_training_stats['episode_rewards'])
        final_clusters = all_training_stats['cluster_evolution'][-1] if all_training_stats['cluster_evolution'] else 0
        final_experts = all_training_stats['expert_evolution'][-1] if all_training_stats['expert_evolution'] else 0
        total_dpmm_events = len(all_training_stats['dpmm_events'])
        
        print(f"📊 OVERALL STATISTICS:")
        print(f"  • Total episodes: {total_episodes}")
        print(f"  • Overall average reward: {overall_avg_reward:.2f}")
        print(f"  • Best episode reward: {max(all_training_stats['episode_rewards']):.2f}")
        print(f"  • Final clusters: {final_clusters}")
        print(f"  • Final experts: {final_experts}")
        print(f"  • Total DPMM events: {total_dpmm_events}")
        print(f"  • Successful tasks: {len([t for t in all_training_stats['tasks'] if not t.get('early_stop_triggered', False)])}/{len(tasks)}")
        
        print(f"\n🔧 BIRTH CONTROL EFFECTIVENESS:")
        if all_training_stats['birth_rate_history']:
            avg_birth_rate = np.mean(all_training_stats['birth_rate_history'])
            print(f"  • Average birth rate: {avg_birth_rate:.1%} (target: {args.target_birth_rate:.1%})")
            print(f"  • Birth control: {'✅ Effective' if avg_birth_rate < 0.2 else '⚠️ Needs tuning'}")
        
        print(f"\n💾 OUTPUT FILES:")
        print(f"  • Model: {final_checkpoint_path}")
        print(f"  • Statistics: {stats_path}")
        print(f"  • Plots: {args.save_dir}/")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"  1. Review generated plots in {args.save_dir}/")
        print(f"  2. Check birth rate analysis for parameter tuning")
        print(f"  3. Run hyperparameter sweep if needed")
        print(f"  4. Test final model on evaluation tasks")

    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        sim.close()

def load_or_create_tasks(tasks_file: str, episodes_per_task: int):
    """태스크 로드 또는 생성 (개선된 버전)"""
    if os.path.exists(tasks_file):
        try:
            with open(tasks_file, 'r') as f:
                tasks = yaml.safe_load(f)
            print(f"✅ Loaded {len(tasks)} tasks from {tasks_file}")
            return tasks
        except Exception as e:
            print(f"⚠️ Task file load error: {e}, creating default tasks")
    
    # 기본 태스크 생성 (더 다양한 설정)
    default_tasks = [
        {
            'name': 'push_cube_basic',
            'instruction': 'Push the blue cube to the goal position',
            'env_cfg': {
                'cube_position': [2.0, 0.0, 0.1],
                'goal_position': [3.0, 0.0, 0.1],
                'cube_size': 0.2,
                'goal_tolerance': 0.3
            },
            'episodes': episodes_per_task
        },
        {
            'name': 'push_cube_far',
            'instruction': 'Move the cube to the distant target location',
            'env_cfg': {
                'cube_position': [2.0, 0.0, 0.1],
                'goal_position': [4.5, 0.0, 0.1],  # 더 멀리
                'cube_size': 0.2,
                'goal_tolerance': 0.3
            },
            'episodes': episodes_per_task
        },
        {
            'name': 'push_cube_left',
            'instruction': 'Push the cube to the left side target',
            'env_cfg': {
                'cube_position': [2.0, 0.0, 0.1],
                'goal_position': [2.0, 2.5, 0.1],  # 왼쪽으로
                'cube_size': 0.2,
                'goal_tolerance': 0.3
            },
            'episodes': episodes_per_task
        },
        {
            'name': 'push_cube_right',
            'instruction': 'Move the object to the right side goal',
            'env_cfg': {
                'cube_position': [2.0, 0.0, 0.1],
                'goal_position': [2.0, -2.5, 0.1],  # 오른쪽으로
                'cube_size': 0.2,
                'goal_tolerance': 0.3
            },
            'episodes': episodes_per_task
        },
        {
            'name': 'push_cube_diagonal',
            'instruction': 'Push the cube diagonally to the corner target',
            'env_cfg': {
                'cube_position': [2.0, 0.0, 0.1],
                'goal_position': [4.0, 2.0, 0.1],  # 대각선
                'cube_size': 0.2,
                'goal_tolerance': 0.3
            },
            'episodes': episodes_per_task
        },
        {
            'name': 'push_cube_precision',
            'instruction': 'Carefully move the cube to the precise target',
            'env_cfg': {
                'cube_position': [2.0, 0.0, 0.1],
                'goal_position': [3.0, 1.0, 0.1],
                'cube_size': 0.15,  # 작은 큐브
                'goal_tolerance': 0.2  # 정밀한 목표
            },
            'episodes': episodes_per_task
        }
    ]
    
    # 태스크 파일 저장
    try:
        with open(tasks_file, 'w') as f:
            yaml.dump(default_tasks, f, default_flow_style=False)
        print(f"💾 Saved default tasks to {tasks_file}")
    except Exception as e:
        print(f"⚠️ Could not save tasks file: {e}")
    
    print(f"✅ Created {len(default_tasks)} default tasks")
    return default_tasks

if __name__ == "__main__":
    main()
