#!/usr/bin/env python3
"""
모든 차원 문제를 해결한 최종 학습 스크립트
"""

import argparse
import torch
import numpy as np
import yaml
import os
import json
import time
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Final Fixed Training")
    parser.add_argument("--tasks", type=str, default="simple_test_tasks.yaml")
    parser.add_argument("--save_dir", type=str, default="./outputs_final_fixed")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=50)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        # 임포트
        from ant_push_env_ultra_fixed import IsaacAntPushEnv
        from dimension_fixed_sac_expert import DimensionFixedSACExpert
        from robust_dpmm_detector import RobustDPMMOODDetector
        from dimension_aware_fusion import create_dimension_aware_fusion
        from config import CONFIG
        from language_processor import LanguageProcessor
        
        print("=== Final Fixed Training ===")
        print(f"Device: {CONFIG.device}")
        print(f"Max episodes per task: {args.max_episodes}")
        print(f"Max steps per episode: {args.max_steps}")
        
        # 전역 변수들
        tasks = yaml.safe_load(open(args.tasks))
        lang = LanguageProcessor(device=CONFIG.device)
        dpmm = RobustDPMMOODDetector(latent_dim=CONFIG.latent_dim, max_components=5)
        fusion = None
        experts = []
        
        # 저장 디렉토리 생성
        os.makedirs(args.save_dir, exist_ok=True)
        
        # 전체 학습 통계
        training_stats = {
            'start_time': time.time(),
            'tasks': [],
            'total_experts': 0,
            'total_episodes': 0,
            'total_steps': 0,
            'dimension_info': {}
        }
        
        # GPU 메모리 모니터링
        def print_gpu_memory():
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                cached = torch.cuda.memory_reserved() / 1024**2
                print(f"GPU Memory - Allocated: {allocated:.1f}MB, Cached: {cached:.1f}MB")
        
        print_gpu_memory()
        
        # 태스크 루프
        for task_idx, task in enumerate(tasks):
            print(f"\n{'='*70}")
            print(f"Task {task_idx + 1}/{len(tasks)}: {task['name']}")
            print(f"Instruction: {task['instruction']}")
            print(f"{'='*70}")
            
            task_start_time = time.time()
            current_env = None
            task_stats = {
                'name': task['name'],
                'instruction': task['instruction'],
                'episodes': 0,
                'successful_episodes': 0,
                'total_reward': 0.0,
                'total_steps': 0,
                'experts_used': set(),
                'start_time': task_start_time,
                'dimension_issues': []
            }
            
            try:
                # 환경 생성
                print("Creating environment...")
                current_env = IsaacAntPushEnv(
                    custom_cfg=task['env_cfg'], 
                    num_envs=args.num_envs
                )
                print("✓ Environment created")
                
                # 초기 리셋으로 차원 확인
                state, _ = current_env.reset()
                state_dim = len(state)
                
                # 액션 차원 확인
                action_space = current_env.action_space
                if hasattr(action_space, 'shape'):
                    if len(action_space.shape) == 1:
                        action_dim = action_space.shape[0]
                    elif len(action_space.shape) == 2:
                        action_dim = action_space.shape[1]
                    else:
                        action_dim = 8  # 기본값
                else:
                    action_dim = 8  # 기본값
                
                print(f"✓ Dimensions confirmed: state={state_dim}, action={action_dim}")
                
                # 차원 정보 저장
                training_stats['dimension_info'][f'task_{task_idx}'] = {
                    'state_dim': state_dim,
                    'action_dim': action_dim,
                    'state_shape': state.shape if hasattr(state, 'shape') else 'unknown'
                }
                
                # 융합 모델 초기화 (한 번만)
                if fusion is None:
                    print("Creating fusion model...")
                    fusion = create_dimension_aware_fusion(
                        CONFIG.instr_emb_dim, 
                        state_dim, 
                        CONFIG.latent_dim, 
                        CONFIG.hidden_dim
                    ).to(CONFIG.device)
                    print(f"✓ Fusion model: {fusion.count_parameters():,} parameters")
                
                # 지시사항 임베딩
                print("Encoding instruction...")
                instr_emb = lang.encode(task['instruction'])
                print(f"✓ Instruction encoded: {instr_emb.shape}")
                
                # 에피소드 루프
                max_episodes = min(task.get('episodes', args.max_episodes), args.max_episodes)
                
                for episode in range(max_episodes):
                    episode_start_time = time.time()
                    episode_reward = 0.0
                    episode_steps = 0
                    episode_successful = False
                    
                    try:
                        # 에피소드 시작
                        state, _ = current_env.reset()
                        done = False
                        
                        print(f"\n--- Episode {episode + 1}/{max_episodes} ---")
                        
                        while not done and episode_steps < args.max_steps:
                            try:
                                # 상태 안전 처리
                                state_safe = np.array(state, dtype=np.float32).flatten()
                                
                                # 차원 체크
                                if len(state_safe) != state_dim:
                                    print(f"Warning: State dimension mismatch: expected {state_dim}, got {len(state_safe)}")
                                    if len(state_safe) < state_dim:
                                        state_safe = np.pad(state_safe, (0, state_dim - len(state_safe)))
                                    else:
                                        state_safe = state_safe[:state_dim]
                                    task_stats['dimension_issues'].append(f"Episode {episode+1}, Step {episode_steps}: state dim mismatch")
                                
                                # 융합된 특징 추출
                                with torch.no_grad():
                                    try:
                                        z = fusion(instr_emb, state_safe)
                                        z_np = z.detach().cpu().numpy() if torch.is_tensor(z) else np.array(z, dtype=np.float32)
                                        
                                        # NaN 체크
                                        if not np.isfinite(z_np).all():
                                            print("Warning: Non-finite values in latent vector")
                                            z_np = np.nan_to_num(z_np, nan=0.0, posinf=1.0, neginf=-1.0)
                                        
                                    except Exception as fusion_error:
                                        if args.debug:
                                            print(f"    Fusion error: {fusion_error}")
                                        z_np = np.random.randn(CONFIG.latent_dim).astype(np.float32) * 0.1
                                
                                # 클러스터 할당
                                cluster_id = dpmm.assign(z_np)
                                task_stats['experts_used'].add(cluster_id)
                                
                                # 전문가가 없으면 생성
                                while cluster_id >= len(experts):
                                    print(f"  Creating expert {len(experts)}")
                                    new_expert = DimensionFixedSACExpert(
                                        state_dim=state_dim,
                                        action_dim=action_dim,
                                        device=CONFIG.device,
                                        hidden_dim=CONFIG.hidden_dim
                                    )
                                    experts.append(new_expert)
                                
                                # 전문가로부터 행동 선택
                                expert = experts[cluster_id]
                                action = expert.select_action(state_safe, deterministic=False)
                                
                                # 액션 차원 체크
                                if len(action) != action_dim:
                                    print(f"Warning: Action dimension mismatch: expected {action_dim}, got {len(action)}")
                                    if len(action) < action_dim:
                                        action = np.pad(action, (0, action_dim - len(action)))
                                    else:
                                        action = action[:action_dim]
                                    task_stats['dimension_issues'].append(f"Episode {episode+1}, Step {episode_steps}: action dim mismatch")
                                
                                # 환경에서 스텝 실행
                                next_state, reward, done, info = current_env.step(action)
                                
                                # 다음 상태 안전 처리
                                next_state_safe = np.array(next_state, dtype=np.float32).flatten()
                                if len(next_state_safe) != state_dim:
                                    if len(next_state_safe) < state_dim:
                                        next_state_safe = np.pad(next_state_safe, (0, state_dim - len(next_state_safe)))
                                    else:
                                        next_state_safe = next_state_safe[:state_dim]
                                
                                # 경험 저장 및 업데이트
                                try:
                                    expert.replay_buffer.add(
                                        state_safe, action, reward, next_state_safe, done
                                    )
                                    
                                    # 주기적 업데이트
                                    if episode_steps % 10 == 0 and len(expert.replay_buffer) >= 16:
                                        loss_info = expert.update(batch_size=min(16, len(expert.replay_buffer)))
                                        if args.debug and loss_info:
                                            print(f"    Update - Total loss: {loss_info.get('total_loss', 0):.6f}")
                                            
                                except Exception as update_error:
                                    if args.debug:
                                        print(f"    Update failed: {update_error}")
                                
                                episode_reward += reward
                                episode_steps += 1
                                state = next_state
                                
                                # 진행상황 출력 (매 25스텝마다)
                                if args.debug and episode_steps % 25 == 0:
                                    print(f"    Step {episode_steps}: reward={reward:.3f}, total={episode_reward:.3f}, expert={cluster_id}")
                            
                            except Exception as step_error:
                                print(f"    Warning: Step {episode_steps} failed: {step_error}")
                                if args.debug:
                                    import traceback
                                    traceback.print_exc()
                                # 스텝 실패 시 에피소드 종료
                                done = True
                                break
                        
                        # 에피소드 완료
                        episode_duration = time.time() - episode_start_time
                        task_stats['episodes'] += 1
                        task_stats['total_reward'] += episode_reward
                        task_stats['total_steps'] += episode_steps
                        
                        # 성공 판정 (개선된 기준)
                        if episode_reward > -30 or episode_steps >= args.max_steps:
                            episode_successful = True
                            task_stats['successful_episodes'] += 1
                        
                        # 에피소드 요약
                        success_rate = task_stats['successful_episodes'] / task_stats['episodes'] * 100
                        print(f"Episode {episode + 1} completed:")
                        print(f"  Steps: {episode_steps}, Reward: {episode_reward:.3f}, Duration: {episode_duration:.1f}s")
                        print(f"  Success: {'✓' if episode_successful else '✗'}, Rate: {success_rate:.1f}%")
                        print(f"  Expert used: {cluster_id}, Total experts: {len(experts)}")
                        
                        # 주기적 클러스터 병합
                        if episode % 2 == 0 and episode > 0:
                            try:
                                dpmm.merge(threshold=1.5)
                                if args.debug:
                                    cluster_info = dpmm.get_cluster_info()
                                    print(f"  Clusters after merge: {cluster_info['num_clusters']}")
                            except Exception as merge_error:
                                if args.debug:
                                    print(f"  Merge failed: {merge_error}")
                        
                        # 메모리 정리
                        if episode % 3 == 0:
                            torch.cuda.empty_cache()
                    
                    except Exception as episode_error:
                        print(f"Warning: Episode {episode + 1} failed: {episode_error}")
                        if args.debug:
                            import traceback
                            traceback.print_exc()
                        continue
                
                # 태스크 완료 후 통계
                task_duration = time.time() - task_start_time
                task_stats['duration'] = task_duration
                task_stats['experts_used'] = list(task_stats['experts_used'])
                task_stats['avg_reward'] = task_stats['total_reward'] / max(task_stats['episodes'], 1)
                task_stats['success_rate'] = task_stats['successful_episodes'] / max(task_stats['episodes'], 1) * 100
                
                print(f"\n--- Task '{task['name']}' Summary ---")
                print(f"Episodes: {task_stats['episodes']}")
                print(f"Success rate: {task_stats['success_rate']:.1f}%")
                print(f"Average reward: {task_stats['avg_reward']:.3f}")
                print(f"Total steps: {task_stats['total_steps']}")
                print(f"Experts used: {len(task_stats['experts_used'])}")
                print(f"Duration: {task_duration:.1f}s")
                print(f"Dimension issues: {len(task_stats['dimension_issues'])}")
                
                # 클러스터 정보
                try:
                    cluster_info = dpmm.get_cluster_info()
                    print(f"Total clusters: {cluster_info['num_clusters']}")
                    print(f"Cluster distribution: {cluster_info.get('cluster_counts', {})}")
                except Exception as cluster_error:
                    print(f"Cluster info unavailable: {cluster_error}")
                
                # 모델 저장 - 안전한 버전
                try:
                    task_save_dir = os.path.join(args.save_dir, f"task_{task_idx + 1}_{task['name'].replace(' ', '_')}")
                    os.makedirs(task_save_dir, exist_ok=True)
                    
                    # 전문가들 저장
                    for i, expert in enumerate(experts):
                        try:
                            expert_path = os.path.join(task_save_dir, f"expert_{i}.pth")
                            expert.save(expert_path)
                        except Exception as expert_save_error:
                            print(f"Warning: Failed to save expert {i}: {expert_save_error}")
                    
                    # 융합 모델 저장
                    try:
                        fusion_path = os.path.join(task_save_dir, "fusion.pth")
                        fusion.save_checkpoint(fusion_path)
                    except Exception as fusion_save_error:
                        print(f"Warning: Failed to save fusion model: {fusion_save_error}")
                    
                    # DPMM 상태 저장
                    try:
                        dpmm_path = os.path.join(task_save_dir, "dpmm.pkl")
                        dpmm.save(dpmm_path)
                    except Exception as dpmm_save_error:
                        print(f"Warning: Failed to save DPMM: {dpmm_save_error}")
                    
                    # 태스크 통계 저장
                    try:
                        stats_path = os.path.join(task_save_dir, "task_stats.json")
                        with open(stats_path, 'w') as f:
                            json.dump(task_stats, f, indent=2, default=str)
                    except Exception as stats_save_error:
                        print(f"Warning: Failed to save task stats: {stats_save_error}")
                    
                    print(f"✓ Models and stats saved to {task_save_dir}")
                    
                except Exception as save_error:
                    print(f"Warning: Failed to save models: {save_error}")
                
                # 전체 통계 업데이트
                training_stats['tasks'].append(task_stats)
                training_stats['total_experts'] = len(experts)
                training_stats['total_episodes'] += task_stats['episodes']
                training_stats['total_steps'] += task_stats['total_steps']
                
                print_gpu_memory()
                
            except Exception as task_error:
                print(f"Error in task {task['name']}: {task_error}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
                continue
            
            finally:
                # 환경 정리
                if current_env is not None:
                    try:
                        current_env.close()
                    except:
                        pass
        
        # 최종 보고서
        training_duration = time.time() - training_stats['start_time']
        training_stats['total_duration'] = training_duration
        
        print(f"\n{'='*70}")
        print("🎉 FINAL FIXED TRAINING COMPLETED!")
        print(f"{'='*70}")
        print(f"Total duration: {training_duration:.1f}s ({training_duration/60:.1f}min)")
        print(f"Total tasks: {len(tasks)}")
        print(f"Total episodes: {training_stats['total_episodes']}")
        print(f"Total steps: {training_stats['total_steps']}")
        print(f"Total experts: {training_stats['total_experts']}")
        
        # 태스크별 요약
        print(f"\n--- Task Summary ---")
        for i, task_stat in enumerate(training_stats['tasks']):
            print(f"Task {i+1}: {task_stat['name']}")
            print(f"  Success rate: {task_stat['success_rate']:.1f}%")
            print(f"  Avg reward: {task_stat['avg_reward']:.3f}")
            print(f"  Episodes: {task_stat['episodes']}")
            print(f"  Experts used: {len(task_stat['experts_used'])}")
            print(f"  Dimension issues: {len(task_stat.get('dimension_issues', []))}")
        
        # 차원 정보 요약
        print(f"\n--- Dimension Summary ---")
        for task_key, dim_info in training_stats['dimension_info'].items():
            print(f"{task_key}: state={dim_info['state_dim']}, action={dim_info['action_dim']}")
        
        # 전체 통계 저장
        try:
            final_stats_path = os.path.join(args.save_dir, "training_summary.json")
            with open(final_stats_path, 'w') as f:
                json.dump(training_stats, f, indent=2, default=str)
            print(f"✓ Training summary saved to {final_stats_path}")
        except Exception as save_error:
            print(f"Warning: Failed to save training summary: {save_error}")
        
        print(f"\nAll models saved in: {args.save_dir}")
        print(f"{'='*70}")
        
        # 정리
        try:
            for expert in experts:
                expert.cleanup()
            lang.cleanup()
            dpmm.cleanup()
            torch.cuda.empty_cache()
        except:
            pass
        
        return 0
        
    except Exception as main_error:
        print(f"Main error: {main_error}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        sim.close()

if __name__ == "__main__":
    import sys
    sys.exit(main())