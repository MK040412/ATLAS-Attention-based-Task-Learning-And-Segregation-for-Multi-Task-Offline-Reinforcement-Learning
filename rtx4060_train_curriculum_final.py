import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher

def safe_tensor_conversion(data, device):
    """안전한 텐서 변환"""
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, np.ndarray):
        return torch.FloatTensor(data).to(device)
    else:
        return torch.FloatTensor(np.array(data)).to(device)

def main():
    parser = argparse.ArgumentParser(description="RTX 4060 Optimized Ant MoE Training - Final")
    parser.add_argument("--tasks", type=str, default="simple_tasks.yaml")
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--num_envs", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    import yaml
    import os
    from ant_push_env_fixed import IsaacAntPushEnv
    from config import CONFIG
    from language_processor import LanguageProcessor
    from dpmm_ood_detector import DPMMOODDetector
    from memory_efficient_fusion import create_fusion_encoder
    from rtx4060_sac_expert import RTX4060SACExpert

    # 출력 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)

    # 태스크 로드
    try:
        with open(args.tasks, 'r') as f:
            tasks = yaml.safe_load(f)
        print(f"Loaded {len(tasks)} tasks from {args.tasks}")
    except Exception as e:
        print(f"Error loading tasks file: {e}")
        return

    # 컴포넌트 초기화
    try:
        lang = LanguageProcessor(device=CONFIG.device)
        dpmm = DPMMOODDetector(latent_dim=CONFIG.latent_dim)
        fusion = None
        experts = []
        current_env = None
        print("Components initialized successfully")
    except Exception as e:
        print(f"Error initializing components: {e}")
        return

    def create_expert(state_dim, action_dim):
        """새로운 전문가 생성"""
        expert = RTX4060SACExpert(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=CONFIG.hidden_dim,
            device=CONFIG.device
        )
        experts.append(expert)
        print(f"Created expert {len(experts)}: {expert.count_parameters():,} parameters")
        return expert

    # 메인 학습 루프
    for task_idx, task in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"Task {task_idx + 1}/{len(tasks)}: {task['name']}")
        print(f"Instruction: {task['instruction']}")
        print(f"{'='*60}")

        try:
            # 환경 생성 또는 재사용
            if current_env is None:
                print("Creating new environment...")
                current_env = IsaacAntPushEnv(
                    custom_cfg=task['env_cfg'], 
                    num_envs=args.num_envs
                )
                print("Environment created successfully")
            else:
                print("Reusing existing environment...")
                # 환경 설정 업데이트
                c = task['env_cfg']
                current_env.cube_position = np.array(c.get('cube_position', [2.0,0.0,0.1]), dtype=np.float32)
                current_env.goal_position = np.array(c.get('goal_position', [3.0,0.0,0.1]), dtype=np.float32)

            # 초기 리셋
            state, _ = current_env.reset()
            state_dim = len(state)
            action_dim = current_env.action_space.shape[0] if len(current_env.action_space.shape) == 1 else current_env.action_space.shape[1]
            
            print(f"Environment info: State dim={state_dim}, Action dim={action_dim}")
            
            # Fusion 모델 초기화 (한 번만)
            if fusion is None:
                fusion = create_fusion_encoder(
                    CONFIG.instr_emb_dim, 
                    state_dim, 
                    CONFIG.latent_dim, 
                    CONFIG.hidden_dim
                ).to(CONFIG.device)
                print(f"Fusion encoder created: {fusion.count_parameters():,} parameters")
            
            # 지시사항 임베딩
            instr_emb = lang.encode(task['instruction'])
            
            # 에피소드 루프
            episodes = task.get('episodes', 10)
            successful_episodes = 0
            
            for episode in range(episodes):
                try:
                    # 에피소드 초기화
                    state_cur = state.copy() if hasattr(state, 'copy') else np.array(state)
                    episode_reward = 0.0
                    episode_steps = 0
                    max_steps = 500
                    done = False
                    
                    while not done and episode_steps < max_steps:
                        try:
                            # 안전한 텐서 변환
                            state_tensor = safe_tensor_conversion(state_cur, CONFIG.device)
                            
                            # 융합된 특징 추출
                            with torch.no_grad():
                                z = fusion(instr_emb, state_tensor)
                                z_np = z.detach().cpu().numpy()
                            
                            # 클러스터 할당
                            cluster_id = dpmm.assign(z_np)
                            
                            # 전문가가 없으면 생성
                            while cluster_id >= len(experts):
                                create_expert(state_dim, action_dim)
                            
                            # 전문가로부터 행동 선택
                            expert = experts[cluster_id]
                            
                            # state_cur이 numpy 배열인지 확인
                            if torch.is_tensor(state_cur):
                                state_np = state_cur.detach().cpu().numpy()
                            else:
                                state_np = np.array(state_cur, dtype=np.float32)
                            
                            action = expert.select_action(state_np, deterministic=False)
                            
                            # 환경에서 스탭 실행
                            next_state, reward, done, info = current_env.step(action)
                            episode_reward += reward
                            episode_steps += 1
                            
                            # next_state도 numpy로 변환
                            if torch.is_tensor(next_state):
                                next_state_np = next_state.detach().cpu().numpy()
                            else:
                                next_state_np = np.array(next_state, dtype=np.float32)
                            
                            # 경험 저장
                            expert.replay_buffer.add(state_np, action, reward, next_state_np, done)
                            
                            # 주기적 업데이트
                            if episode_steps % CONFIG.update_frequency == 0 and len(expert.replay_buffer) >= CONFIG.batch_size:
                                try:
                                    loss_info = expert.update(batch_size=CONFIG.batch_size)
                                    if loss_info and episode_steps % 100 == 0:
                                        print(f"  Step {episode_steps}: Loss = {loss_info.get('total_loss', 0):.4f}")
                                except Exception as e:
                                    print(f"Warning: Expert update failed: {e}")
                            
                            state_cur = next_state_np
                            
                        except Exception as e:
                            print(f"Warning: Step failed: {e}")
                            done = True
                            break
                    
                    if episode_steps > 10:  # 최소한의 스텝을 완료한 경우만 성공으로 간주
                        successful_episodes += 1
                    
                    if episode % 10 == 0 or episode == episodes - 1:
                        print(f"Episode {episode + 1}/{episodes}:")
                        print(f"  Steps: {episode_steps}, Reward: {episode_reward:.3f}")
                        print(f"  Expert: {cluster_id}, Success rate: {successful_episodes/(episode+1)*100:.1f}%")
                    
                    # 클러스터 병합 (주기적으로)
                    if episode % CONFIG.merge_frequency == 0:
                        try:
                            dpmm.merge()
                        except Exception as e:
                            print(f"Warning: Cluster merge failed: {e}")
                    
                    # 다음 에피소드를 위한 리셋
                    try:
                        state, _ = current_env.reset()
                    except Exception as e:
                        print(f"Warning: Environment reset failed: {e}")
                        state = np.zeros(state_dim, dtype=np.float32)
                    
                except Exception as e:
                    print(f"Error in episode {episode + 1}: {e}")
                    continue
            
            # 태스크 완료 후 정보 출력
            try:
                cluster_info = dpmm.get_cluster_info()
                print(f"\nTask '{task['name']}' completed!")
                print(f"Success rate: {successful_episodes/episodes*100:.1f}%")
                print(f"Total experts: {len(experts)}")
                print(f"Active clusters: {cluster_info['num_clusters']}")
            except Exception as e:
                print(f"Warning: Failed to get cluster info: {e}")
            
            # 모델 저장
            try:
                task_save_dir = os.path.join(args.save_dir, f"task_{task_idx + 1}_{task['name']}")
                os.makedirs(task_save_dir, exist_ok=True)
                
                # 전문가들 저장
                for i, expert in enumerate(experts):
                    expert_path = os.path.join(task_save_dir, f"expert_{i}.pth")
                    expert.save(expert_path)
                
                # 융합 모델 저장
                fusion_path = os.path.join(task_save_dir, "fusion_encoder.pth")
                fusion.save_checkpoint(fusion_path)
                
                # 태스크 정보 저장
                task_info = {
                    'task_name': task['name'],
                    'instruction': task['instruction'],
                    'episodes': episodes,
                    'successful_episodes': successful_episodes,
                    'success_rate': successful_episodes/episodes*100,
                    'total_experts': len(experts),
                    'clusters': cluster_info['num_clusters'] if 'cluster_info' in locals() else 0
                }
                
                import json
                info_path = os.path.join(task_save_dir, "task_info.json")
                with open(info_path, 'w') as f:
                    json.dump(task_info, f, indent=2)
                
                print(f"Models and info saved to {task_save_dir}")
                
            except Exception as e:
                print(f"Warning: Failed to save models: {e}")
        
        except Exception as e:
            print(f"Error in task {task['name']}: {e}")
            print("Continuing to next task...")
            continue

    # 최종 정리 및 보고서 생성
    try:
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED!")
        print(f"{'='*60}")
        
        # 최종 통계
        total_experts = len(experts)
        final_cluster_info = dpmm.get_cluster_info()
        
        print(f"Final Statistics:")
        print(f"  Total experts trained: {total_experts}")
        print(f"  Final clusters: {final_cluster_info['num_clusters']}")
        print(f"  Models saved in: {args.save_dir}")
        
        # 전체 보고서 생성
        final_report = {
            'training_completed': True,
            'total_tasks': len(tasks),
            'total_experts': total_experts,
            'final_clusters': final_cluster_info['num_clusters'],
            'cluster_distribution': final_cluster_info.get('cluster_counts', []),
            'config': {
                'hidden_dim': CONFIG.hidden_dim,
                'latent_dim': CONFIG.latent_dim,
                'batch_size': CONFIG.batch_size,
                'device': str(CONFIG.device)
            }
        }
        
        import json
        report_path = os.path.join(args.save_dir, "final_training_report.json")
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        print(f"Final report saved to: {report_path}")
        
    except Exception as e:
        print(f"Warning: Failed to generate final report: {e}")
    
    # 정리
    try:
        if current_env is not None:
            current_env.close()
        
        # 전문가들 정리
        for expert in experts:
            expert.cleanup()
        
        # 언어 처리기 정리
        lang.cleanup()
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated() / 1024**2
            print(f"Final GPU memory usage: {final_memory:.1f} MB")
        
    except Exception as e:
        print(f"Warning: Cleanup failed: {e}")
    
    finally:
        sim.close()

if __name__ == "__main__":
    main()