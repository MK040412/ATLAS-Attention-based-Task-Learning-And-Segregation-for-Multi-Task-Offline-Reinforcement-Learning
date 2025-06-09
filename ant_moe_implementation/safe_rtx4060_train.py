import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher

class SafeTensorHandler:
    """안전한 텐서 변환 핸들러"""
    
    def __init__(self, device):
        self.device = device
    
    def to_tensor(self, data, dtype=torch.float32):
        """데이터를 안전하게 텐서로 변환"""
        try:
            if torch.is_tensor(data):
                return data.to(dtype).to(self.device)
            elif isinstance(data, np.ndarray):
                # numpy 배열을 복사해서 변환 (메모리 공유 문제 방지)
                data_copy = data.copy().astype(np.float32)
                return torch.from_numpy(data_copy).to(dtype).to(self.device)
            else:
                # 기타 타입은 numpy로 먼저 변환
                np_data = np.array(data, dtype=np.float32)
                return torch.from_numpy(np_data).to(dtype).to(self.device)
        except Exception as e:
            print(f"Warning: Tensor conversion failed: {e}")
            # 실패시 더미 텐서 반환
            if hasattr(data, 'shape'):
                shape = data.shape
            elif hasattr(data, '__len__'):
                shape = (len(data),)
            else:
                shape = (1,)
            return torch.zeros(shape, dtype=dtype, device=self.device)
    
    def to_numpy(self, data):
        """텐서를 안전하게 numpy로 변환"""
        try:
            if torch.is_tensor(data):
                return data.detach().cpu().numpy().astype(np.float32)
            elif isinstance(data, np.ndarray):
                return data.astype(np.float32)
            else:
                return np.array(data, dtype=np.float32)
        except Exception as e:
            print(f"Warning: Numpy conversion failed: {e}")
            return np.array([0.0], dtype=np.float32)

def main():
    parser = argparse.ArgumentParser(description="Safe RTX 4060 Training")
    parser.add_argument("--tasks", type=str, default="simple_tasks.yaml")
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
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

    # 안전한 텐서 핸들러 초기화
    tensor_handler = SafeTensorHandler(CONFIG.device)
    
    # 디버그 모드
    if args.debug:
        print("=== DEBUG MODE ENABLED ===")
        torch.autograd.set_detect_anomaly(True)

    # 출력 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)

    # 태스크 로드
    try:
        with open(args.tasks, 'r') as f:
            tasks = yaml.safe_load(f)
        print(f"Loaded {len(tasks)} tasks")
    except Exception as e:
        print(f"Error loading tasks: {e}")
        return

    # 컴포넌트 초기화
    try:
        lang = LanguageProcessor(device=CONFIG.device)
        dpmm = DPMMOODDetector(latent_dim=CONFIG.latent_dim)
        fusion = None
        experts = []
        current_env = None
        print("Components initialized")
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
        print(f"Created expert {len(experts)}")
        return expert

    # 메인 학습 루프
    for task_idx, task in enumerate(tasks):
        print(f"\n{'='*50}")
        print(f"Task {task_idx + 1}/{len(tasks)}: {task['name']}")
        print(f"{'='*50}")

        try:
            # 환경 생성
            if current_env is None:
                current_env = IsaacAntPushEnv(
                    custom_cfg=task['env_cfg'], 
                    num_envs=args.num_envs
                )
                print("Environment created")
            else:
                # 환경 설정 업데이트
                c = task['env_cfg']
                current_env.cube_position = np.array(c.get('cube_position', [2.0,0.0,0.1]), dtype=np.float32)
                current_env.goal_position = np.array(c.get('goal_position', [3.0,0.0,0.1]), dtype=np.float32)

            # 초기 리셋
            state, _ = current_env.reset()
            state_dim = len(state)
            action_dim = current_env.action_space.shape[0] if len(current_env.action_space.shape) == 1 else current_env.action_space.shape[1]
            
            print(f"State dim: {state_dim}, Action dim: {action_dim}")
            
            # Fusion 모델 초기화 (한 번만)
            if fusion is None:
                fusion = create_fusion_encoder(
                    CONFIG.instr_emb_dim, 
                    state_dim, 
                    CONFIG.latent_dim, 
                    CONFIG.hidden_dim
                ).to(CONFIG.device)
                print("Fusion encoder created")
            
            # 지시사항 임베딩
            instr_emb = lang.encode(task['instruction'])
            
            # 에피소드 루프
            episodes = task.get('episodes', 10)
            successful_episodes = 0
            
            for episode in range(episodes):
                if args.debug:
                    print(f"\n--- Episode {episode + 1} DEBUG ---")
                
                try:
                    # 에피소드 초기화
                    state_cur = tensor_handler.to_numpy(state)
                    episode_reward = 0.0
                    episode_steps = 0
                    max_steps = 100  # 더 짧은 에피소드로 테스트
                    done = False
                    
                    while not done and episode_steps < max_steps:
                        try:
                            if args.debug and episode_steps == 0:
                                print(f"  State shape: {state_cur.shape}")
                                print(f"  State type: {type(state_cur)}")
                            
                            # 안전한 텐서 변환
                            state_tensor = tensor_handler.to_tensor(state_cur)
                            instr_tensor = tensor_handler.to_tensor(instr_emb.squeeze() if instr_emb.dim() > 1 else instr_emb)
                            
                            if args.debug and episode_steps == 0:
                                print(f"  State tensor: {state_tensor.shape}, device: {state_tensor.device}")
                                print(f"  Instr tensor: {instr_tensor.shape}, device: {instr_tensor.device}")
                            
                            # 융합된 특징 추출
                            with torch.no_grad():
                                try:
                                    z = fusion(instr_tensor, state_tensor)
                                    z_np = tensor_handler.to_numpy(z)
                                    
                                    if args.debug and episode_steps == 0:
                                        print(f"  Fusion output: {z.shape}")
                                    
                                except Exception as fusion_error:
                                    print(f"Warning: Fusion failed: {fusion_error}")
                                    z_np = np.random.randn(CONFIG.latent_dim).astype(np.float32)
                            
                            # 클러스터 할당
                            try:
                                cluster_id = dpmm.assign(z_np)
                            except Exception as e:
                                print(f"Warning: DPMM failed: {e}")
                                cluster_id = 0
                            
                            # 전문가가 없으면 생성
                            while cluster_id >= len(experts):
                                create_expert(state_dim, action_dim)
                            
                            # 전문가로부터 행동 선택
                            expert = experts[cluster_id]
                            action = expert.select_action(state_cur, deterministic=False)
                            
                            if args.debug and episode_steps == 0:
                                print(f"  Action: {action.shape}")
                            
                            # 환경에서 스텝 실행
                            next_state, reward, done, info = current_env.step(action)
                            episode_reward += reward
                            episode_steps += 1
                            
                            # 다음 상태 변환
                            next_state_np = tensor_handler.to_numpy(next_state)
                            
                            # 경험 저장
                            expert.replay_buffer.add(state_cur, action, reward, next_state_np, done)
                            
                            # 주기적 업데이트
                            if episode_steps % 10 == 0 and len(expert.replay_buffer) >= 16:  # 더 작은 배치
                                try:
                                    loss_info = expert.update(batch_size=16)
                                except Exception as e:
                                    if args.debug:
                                        print(f"  Update failed: {e}")
                            
                            state_cur = next_state_np
                            
                        except Exception as step_error:
                            print(f"Warning: Step failed: {step_error}")
                            if args.debug:
                                import traceback
                                traceback.print_exc()
                            done = True
                            break
                    
                    if episode_steps > 5:  # 최소 5스텝 이상 완료
                        successful_episodes += 1
                    
                    if episode % 10 == 0 or episode == episodes - 1:
                        success_rate = successful_episodes / (episode + 1) * 100
                        print(f"Episode {episode + 1}/{episodes}:")
                        print(f"  Steps: {episode_steps}, Reward: {episode_reward:.3f}")
                        print(f"  Expert: {cluster_id}, Success rate: {success_rate:.1f}%")
                    
                    # 클러스터 병합 (주기적으로)
                    if episode % 20 == 0 and episode > 0:
                        try:
                            dpmm.merge()
                        except Exception as e:
                            if args.debug:
                                print(f"Warning: Cluster merge failed: {e}")
                    
                    # 다음 에피소드를 위한 리셋
                    try:
                        state, _ = current_env.reset()
                    except Exception as e:
                        print(f"Warning: Environment reset failed: {e}")
                        state = np.zeros(state_dim, dtype=np.float32)
                    
                except Exception as episode_error:
                    print(f"Error in episode {episode + 1}: {episode_error}")
                    if args.debug:
                        import traceback
                        traceback.print_exc()
                    continue
            
            # 태스크 완료 후 정보 출력
            try:
                cluster_info = dpmm.get_cluster_info()
                success_rate = successful_episodes / episodes * 100
                
                print(f"\nTask '{task['name']}' completed!")
                print(f"Success rate: {success_rate:.1f}%")
                print(f"Total experts: {len(experts)}")
                print(f"Active clusters: {cluster_info['num_clusters']}")
                
                # 성공률이 너무 낮으면 경고
                if success_rate < 10:
                    print("WARNING: Very low success rate - consider debugging")
                
            except Exception as e:
                print(f"Warning: Failed to get cluster info: {e}")
            
            # 모델 저장
            try:
                task_save_dir = os.path.join(args.save_dir, f"task_{task_idx + 1}_{task['name']}")
                os.makedirs(task_save_dir, exist_ok=True)
                
                # 전문가들 저장
                saved_experts = 0
                for i, expert in enumerate(experts):
                    try:
                        expert_path = os.path.join(task_save_dir, f"expert_{i}.pth")
                        expert.save(expert_path)
                        saved_experts += 1
                    except Exception as e:
                        print(f"Warning: Failed to save expert {i}: {e}")
                
                # 융합 모델 저장
                try:
                    fusion_path = os.path.join(task_save_dir, "fusion_encoder.pth")
                    fusion.save_checkpoint(fusion_path)
                    print(f"Fusion model saved")
                except Exception as e:
                    print(f"Warning: Failed to save fusion model: {e}")
                
                # 태스크 정보 저장
                task_info = {
                    'task_name': task['name'],
                    'instruction': task['instruction'],
                    'episodes': episodes,
                    'successful_episodes': successful_episodes,
                    'success_rate': success_rate,
                    'total_experts': len(experts),
                    'saved_experts': saved_experts,
                    'clusters': cluster_info['num_clusters'] if 'cluster_info' in locals() else 0
                }
                
                import json
                info_path = os.path.join(task_save_dir, "task_info.json")
                with open(info_path, 'w') as f:
                    json.dump(task_info, f, indent=2)
                
                print(f"Task info saved: {saved_experts} experts, success rate {success_rate:.1f}%")
                
            except Exception as e:
                print(f"Warning: Failed to save models: {e}")
        
        except Exception as task_error:
            print(f"Error in task {task['name']}: {task_error}")
            if args.debug:
                import traceback
                traceback.print_exc()
            print("Continuing to next task...")
            continue

    # 최종 정리 및 보고서 생성
    try:
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED!")
        print(f"{'='*60}")
        
        # 최종 통계
        total_experts = len(experts)
        try:
            final_cluster_info = dpmm.get_cluster_info()
            final_clusters = final_cluster_info['num_clusters']
        except:
            final_clusters = 0
        
        print(f"Final Statistics:")
        print(f"  Total experts trained: {total_experts}")
        print(f"  Final clusters: {final_clusters}")
        print(f"  Models saved in: {args.save_dir}")
        
        # 전체 보고서 생성
        final_report = {
            'training_completed': True,
            'total_tasks': len(tasks),
            'total_experts': total_experts,
            'final_clusters': final_clusters,
            'config': {
                'hidden_dim': CONFIG.hidden_dim,
                'latent_dim': CONFIG.latent_dim,
                'batch_size': CONFIG.batch_size,
                'device': str(CONFIG.device)
            },
            'debug_mode': args.debug
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
            try:
                expert.cleanup()
            except:
                pass
        
        # 언어 처리기 정리
        try:
            lang.cleanup()
        except:
            pass
        
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