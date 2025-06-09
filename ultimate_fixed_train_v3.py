import argparse
import torch
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Ultimate Fixed Ant MoE Training")
    parser.add_argument("--tasks", type=str, default="simple_test_tasks.yaml")
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--num_envs", type=int, default=4)  # 작은 환경 수
    parser.add_argument("--max_episodes", type=int, default=2)  # 적은 에피소드
    parser.add_argument("--max_steps", type=int, default=50)   # 짧은 스텝
    parser.add_argument("--debug", action="store_true")
    
    # AppLauncher 인수 추가
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    print("🚀 Starting Ultimate Fixed Training v3...")
    print(f"📋 Configuration:")
    print(f"  Tasks file: {args.tasks}")
    print(f"  Save directory: {args.save_dir}")
    print(f"  Number of environments: {args.num_envs}")
    print(f"  Max episodes per task: {args.max_episodes}")
    print(f"  Max steps per episode: {args.max_steps}")
    print(f"  Debug mode: {args.debug}")

    # Isaac Sim 초기화
    app = AppLauncher(vars(args))
    sim = app.app

    try:
        # 필요한 모듈들 import
        import yaml
        import numpy as np
        import os
        import time
        from pathlib import Path
        
        from ant_push_env_final_fix import IsaacAntPushEnv
        from config import CONFIG
        from language_processor import LanguageProcessor
        from robust_dpmm_detector import RobustDPMMOODDetector
        from dimension_aware_fusion import DimensionAwareFusion
        from memory_optimized_sac import DimensionExactSACExpert

        # 출력 디렉토리 생성
        os.makedirs(args.save_dir, exist_ok=True)

        # 태스크 로드
        try:
            with open(args.tasks, 'r') as f:
                tasks = yaml.safe_load(f)
            print(f"✓ Loaded {len(tasks)} tasks from {args.tasks}")
        except Exception as e:
            print(f"❌ Failed to load tasks: {e}")
            return

        # 컴포넌트 초기화
        print("\n🔧 Initializing components...")
        
        lang = LanguageProcessor(device=CONFIG.device)
        print("✓ Language processor initialized")
        
        dpmm = RobustDPMMOODDetector(latent_dim=CONFIG.latent_dim)
        print("✓ DPMM detector initialized")
        
        fusion = None
        experts = []
        
        def init_expert(state_dim, action_dim):
            """새로운 전문가 초기화"""
            print(f"  🧠 Creating new expert {len(experts)}")
            expert = DimensionExactSACExpert(
                state_dim=state_dim, 
                action_dim=action_dim, 
                device=CONFIG.device,
                hidden_dim=CONFIG.hidden_dim
            )
            experts.append(expert)
            return expert

        def safe_cleanup():
            """안전한 정리 함수"""
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(0.5)  # 정리 대기
                print("✓ Memory cleaned up")
            except Exception as e:
                print(f"Warning: Cleanup failed: {e}")

        # 메인 학습 루프
        print(f"\n{'='*60}")
        print("🎯 Starting Training Loop")
        print(f"{'='*60}")
        
        global_stats = {
            'total_episodes': 0,
            'total_steps': 0,
            'total_rewards': [],
            'task_results': {}
        }

        # 각 태스크를 별도의 환경에서 실행 (PhysX 오류 방지)
        for task_idx, task in enumerate(tasks):
            print(f"\n{'='*50}")
            print(f"📋 Task {task_idx + 1}/{len(tasks)}: {task['name']}")
            print(f"💬 Instruction: {task['instruction']}")
            print(f"{'='*50}")
            
            task_start_time = time.time()
            env = None
            
            try:
                # 환경 생성
                print("🏗️ Creating environment...")
                env = IsaacAntPushEnv(
                    custom_cfg=task['env_cfg'], 
                    num_envs=args.num_envs
                )
                
                # 초기 리셋으로 차원 정보 얻기
                print("🔄 Getting environment dimensions...")
                state, _ = env.reset()
                state_dim = len(state)
                action_dim = env.get_action_dim()
                
                print(f"📏 Environment dimensions:")
                print(f"  State dim: {state_dim}")
                print(f"  Action dim: {action_dim}")
                
                # Fusion 모델 초기화 (한 번만)
                if fusion is None:
                    print("🔗 Initializing fusion model...")
                    fusion = DimensionAwareFusion(
                        instr_dim=CONFIG.instr_emb_dim,
                        state_dim=state_dim,
                        latent_dim=CONFIG.latent_dim
                    ).to(CONFIG.device)
                    print("✓ Fusion model initialized")
                
                # 지시사항 임베딩
                print("💭 Encoding instruction...")
                instr_emb = lang.encode(task['instruction'])
                print(f"✓ Instruction embedded: {instr_emb.shape}")
                
                # 태스크 통계
                task_stats = {
                    'episodes': 0,
                    'total_steps': 0,
                    'total_reward': 0.0,
                    'episode_rewards': [],
                    'expert_usage': {},
                    'errors': []
                }
                
                # 에피소드 루프
                for episode in range(task['episodes']):
                    print(f"\n--- Episode {episode + 1}/{task['episodes']} ---")
                    
                    episode_start_time = time.time()
                    episode_reward = 0.0
                    episode_steps = 0
                    episode_losses = []
                    
                    # 에피소드 초기화
                    state, _ = env.reset()
                    done = False
                    
                    # 스텝 루프
                    while not done and episode_steps < args.max_steps:
                        try:
                            # 융합된 특징 추출
                            with torch.no_grad():
                                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(CONFIG.device)
                                instr_tensor = instr_emb.unsqueeze(0) if instr_emb.dim() == 1 else instr_emb
                                z = fusion(instr_tensor, state_tensor).squeeze(0)
                            
                            # 클러스터 할당
                            cluster_id = dpmm.assign(z.detach().cpu().numpy())
                            
                            # 전문가 생성 (필요시)
                            if cluster_id >= len(experts):
                                init_expert(state_dim, action_dim)
                            
                            # 행동 선택
                            expert = experts[cluster_id]
                            action = expert.select_action(state, deterministic=False)
                            
                            # 환경 스텝
                            next_state, reward, done, info = env.step(action)
                            
                            # 경험 저장
                            expert.replay_buffer.add(state, action, reward, next_state, done)
                            
                            # 학습 (주기적으로)
                            if episode_steps % 10 == 0 and len(expert.replay_buffer) > 32:
                                loss_info = expert.update(batch_size=32)
                                if loss_info:
                                    episode_losses.append(loss_info)
                            
                            # 통계 업데이트
                            episode_reward += reward
                            episode_steps += 1
                            
                            # 전문가 사용 통계
                            if cluster_id not in task_stats['expert_usage']:
                                task_stats['expert_usage'][cluster_id] = 0
                            task_stats['expert_usage'][cluster_id] += 1
                            
                            # 디버그 출력 (주기적으로)
                            if args.debug and episode_steps % 25 == 0:
                                print(f"    Step {episode_steps}: reward={reward:.3f}, expert={cluster_id}")
                            
                            state = next_state
                            
                        except Exception as step_error:
                            print(f"⚠️ Step error: {step_error}")
                            task_stats['errors'].append(str(step_error))
                            break
                    
                    # 에피소드 완료
                    episode_time = time.time() - episode_start_time
                    
                    # 클러스터 병합
                    dpmm.merge(threshold=1.5)
                    
                    # 통계 업데이트
                    task_stats['episodes'] += 1
                    task_stats['total_steps'] += episode_steps
                    task_stats['total_reward'] += episode_reward
                    task_stats['episode_rewards'].append(episode_reward)
                    
                    global_stats['total_episodes'] += 1
                    global_stats['total_steps'] += episode_steps
                    global_stats['total_rewards'].append(episode_reward)
                    
                    # 에피소드 요약
                    avg_loss = np.mean([l['total_loss'] for l in episode_losses]) if episode_losses else 0.0
                    print(f"  ✓ Episode {episode + 1} completed:")
                    print(f"    Steps: {episode_steps}, Time: {episode_time:.1f}s")
                    print(f"    Reward: {episode_reward:.3f}, Avg Loss: {avg_loss:.6f}")
                    print(f"    Experts used: {len(task_stats['expert_usage'])}")
                
                # 태스크 완료
                task_time = time.time() - task_start_time
                avg_reward = task_stats['total_reward'] / task_stats['episodes'] if task_stats['episodes'] > 0 else 0.0
                
                # 클러스터 정보
                cluster_info = dpmm.get_cluster_info()
                
                print(f"\n🎯 Task '{task['name']}' completed!")
                print(f"  Time: {task_time:.1f}s")
                print(f"  Episodes: {task_stats['episodes']}")
                print(f"  Total steps: {task_stats['total_steps']}")
                print(f"  Average reward: {avg_reward:.3f}")
                print(f"  Total clusters: {cluster_info['num_clusters']}")
                print(f"  Expert usage: {task_stats['expert_usage']}")
                if task_stats['errors']:
                    print(f"  Errors: {len(task_stats['errors'])}")
                
                # 태스크 결과 저장
                global_stats['task_results'][task['name']] = task_stats
                
            except Exception as task_error:
                print(f"❌ Task '{task['name']}' failed: {task_error}")
                global_stats['task_results'][task['name']] = {'error': str(task_error)}
                import traceback
                traceback.print_exc()
                
            finally:
                # 환경 정리 (PhysX 오류 방지)
                if env is not None:
                    try:
                        print("🔄 Closing environment safely...")
                        env.close()
                        print("✓ Environment closed")
                    except Exception as close_error:
                        print(f"Warning: Environment close failed: {close_error}")
                
                # 메모리 정리
                safe_cleanup()
                
                # 태스크 간 대기 시간 (PhysX 안정화)
                if task_idx < len(tasks) - 1:
                    print("⏳ Waiting for system stabilization...")
                    time.sleep(2.0)
        
        # 전체 학습 완료
        print(f"\n{'='*60}")
        print("🎉 Training Completed!")
        print(f"{'='*60}")
        print(f"📊 Global Statistics:")
        print(f"  Total episodes: {global_stats['total_episodes']}")
        print(f"  Total steps: {global_stats['total_steps']}")
        print(f"  Total experts: {len(experts)}")
        print(f"  Final clusters: {dpmm.get_cluster_info()['num_clusters']}")
        
        if global_stats['total_rewards']:
            avg_reward = np.mean(global_stats['total_rewards'])
            print(f"  Average reward: {avg_reward:.3f}")
        
        # 결과 저장
        try:
            results_file = os.path.join(args.save_dir, "training_results.yaml")
            with open(results_file, 'w') as f:
                # numpy 배열을 리스트로 변환
                save_stats = global_stats.copy()
                save_stats['total_rewards'] = [float(r) for r in save_stats['total_rewards']]
                yaml.dump(save_stats, f, default_flow_style=False)
            print(f"✓ Results saved to {results_file}")
            
            # 모델 저장
            models_dir = os.path.join(args.save_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            
            for i, expert in enumerate(experts):
                model_path = os.path.join(models_dir, f"expert_{i}.pth")
                expert.save(model_path)
            
            # DPMM 상태 저장
            dpmm_path = os.path.join(args.save_dir, "dpmm_state.pkl")
            dpmm.save_state(dpmm_path)
            
            print(f"✓ Models saved to {models_dir}")
            print(f"✓ DPMM state saved to {dpmm_path}")
            
        except Exception as save_error:
            print(f"⚠️ Failed to save results: {save_error}")

    except Exception as main_error:
        print(f"❌ Main training error: {main_error}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 시뮬레이션 정리
        print("\n🔄 Final cleanup...")
        try:
            # 최종 메모리 정리
            safe_cleanup()
            
            # 시뮬레이션 종료
            sim.close()
            print("✓ Simulation closed")
        except Exception as close_error:
            print(f"Warning: Final cleanup failed: {close_error}")
        
        print("✅ All cleanup completed!")

if __name__ == "__main__":
    main()