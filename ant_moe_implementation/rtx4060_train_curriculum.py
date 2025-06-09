import argparse
import torch
import yaml
import numpy as np
import gc
import os
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="RTX 4060 Optimized Ant MoE Agent")
    parser.add_argument("--tasks", type=str, default="simple_tasks.yaml")
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--num_envs", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # 출력 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Output directory: {args.save_dir}")

    # RTX 4060 최적화 설정
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    app = AppLauncher(vars(args))
    sim = app.app

    from ant_push_env import IsaacAntPushEnv
    from config import CONFIG
    from language_processor import LanguageProcessor
    from dpmm_ood_detector import DPMMOODDetector
    from memory_efficient_fusion import MemoryEfficientFusionEncoder
    from rtx4060_sac_expert import RTX4060SACExpert
    from lightweight_clip_vision import get_vision_system

    # RTX 4060 최적화된 전역 변수들
    tasks = yaml.safe_load(open(args.tasks))
    lang = LanguageProcessor(device=CONFIG.device)
    dpmm = DPMMOODDetector(latent_dim=CONFIG.latent_dim)
    fusion = None
    experts = []
    vision_system = None

    # 메모리 관리 함수들
    def cleanup_memory():
        """주기적 메모리 정리"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"  Memory cleanup - GPU allocated: {allocated:.2f}GB")

    def init_expert(state_dim, action_dim):
        """새로운 전문가 초기화 (메모리 효율적)"""
        if len(experts) >= CONFIG.max_experts:
            print(f"  Maximum experts ({CONFIG.max_experts}) reached, reusing existing expert")
            return experts[len(experts) % CONFIG.max_experts]
        
        print(f"  Creating new RTX4060 expert {len(experts)}")
        expert = RTX4060SACExpert(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=CONFIG.hidden_dim,
            device=CONFIG.device
        )
        experts.append(expert)
        return expert

    class RTX4060MoEAgent:
        """RTX 4060 최적화된 MoE 에이전트"""
        def __init__(self, state_dim, action_dim, instr_dim, latent_dim, fusion_hidden_dim):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.device = CONFIG.device
            
            # 융합 인코더 (메모리 효율적)
            self.fusion = MemoryEfficientFusionEncoder(
                instr_dim, state_dim, latent_dim, fusion_hidden_dim
            ).to(self.device)
            
            # DPMM 클러스터링
            self.dpmm = dpmm
            
            # 전문가들
            self.experts = experts
            
            print(f"RTX4060 MoE Agent initialized:")
            memory_info = self.fusion.get_memory_usage()
            print(f"  Fusion network: {memory_info['memory_mb']:.1f}MB")
        
        def select_action(self, state, instruction_emb, deterministic=False):
            """행동 선택"""
            with torch.no_grad():
                # 상태와 지시사항을 텐서로 변환
                if not torch.is_tensor(state):
                    state_tensor = torch.FloatTensor(state).to(self.device)
                else:
                    state_tensor = state.to(self.device)
                
                if not torch.is_tensor(instruction_emb):
                    instr_tensor = torch.FloatTensor(instruction_emb).to(self.device)
                else:
                    instr_tensor = instruction_emb.to(self.device)
                
                # 융합된 잠재 상태 계산
                latent_state = self.fusion(instr_tensor, state_tensor)
                
                # 클러스터 할당
                latent_np = latent_state.detach().cpu().numpy()
                cluster_id = self.dpmm.assign(latent_np)
                
                # 전문가가 없으면 생성
                if cluster_id >= len(self.experts):
                    init_expert(self.state_dim, self.action_dim)
                
                # 전문가로부터 행동 선택
                expert = self.experts[cluster_id]
                action = expert.select_action(state, deterministic=deterministic)
                
                return action, cluster_id, latent_np
        
        def store_experience(self, state, action, reward, next_state, done, 
                           instruction_emb, cluster_id, latent_state):
            """경험 저장"""
            if cluster_id < len(self.experts):
                expert = self.experts[cluster_id]
                expert.replay_buffer.add(state, action, reward, next_state, done)
        
        def update_experts(self, batch_size=None):
            """전문가들 업데이트"""
            batch_size = batch_size or CONFIG.batch_size
            losses = {}
            
            for i, expert in enumerate(self.experts):
                if len(expert.replay_buffer) >= batch_size:
                    loss_info = expert.update(batch_size)
                    if loss_info:
                        losses[f'expert_{i}'] = loss_info
            
            return losses
        
        def merge_clusters(self, threshold=None):
            """클러스터 병합"""
            threshold = threshold or CONFIG.merge_threshold
            self.dpmm.merge(threshold)
        
        def get_info(self):
            """에이전트 정보 반환"""
            buffer_sizes = [len(expert.replay_buffer) for expert in self.experts]
            cluster_info = self.dpmm.get_cluster_info()
            
            return {
                'num_experts': len(self.experts),
                'buffer_sizes': buffer_sizes,
                'cluster_info': cluster_info
            }
        
        def save(self, path):
            """모델 저장"""
            try:
                torch.save({
                    'fusion_state_dict': self.fusion.state_dict(),
                    'experts': [
                        {
                            'actor': expert.actor.state_dict(),
                            'critic': expert.critic.state_dict(),
                            'log_alpha': expert.log_alpha.item()
                        } for expert in self.experts
                    ],
                    'dpmm_state': self.dpmm.get_state()
                }, path)
                print(f"Agent saved to {path}")
            except Exception as e:
                print(f"Warning: Failed to save agent: {e}")
        
        def cleanup(self):
            """메모리 정리"""
            for expert in self.experts:
                expert.cleanup()

    # 메인 학습 루프
    agent = None
    
    for task_idx, task in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"Task {task_idx + 1}/{len(tasks)}: {task['name']}")
        print(f"Instruction: {task['instruction']}")
        print(f"{'='*60}")
        
        try:
            # 환경 생성
            env = IsaacAntPushEnv(custom_cfg=task['env_cfg'], num_envs=args.num_envs)
            
            # 초기 리셋으로 상태 차원 확인
            state, _ = env.reset()
            
            if state.ndim == 2:  # 다중 환경
                actual_state_dim = state.shape[1]
                single_env_state = state[0]
            else:  # 단일 환경
                actual_state_dim = state.shape[0]
                single_env_state = state
            
            # action space 확인
            action_space = env.action_space
            if hasattr(action_space, 'shape'):
                if len(action_space.shape) == 2:
                    action_dim = action_space.shape[1]
                else:
                    action_dim = action_space.shape[0]
            else:
                # action space가 Box가 아닌 경우 기본값 사용
                action_dim = 8  # Ant 환경의 기본 action 차원
            
            print(f"Environment setup:")
            print(f"  State dim: {actual_state_dim}, Action dim: {action_dim}")
            print(f"  Num envs: {args.num_envs}")
            
            # 에이전트 초기화 (한 번만)
            if agent is None:
                agent = RTX4060MoEAgent(
                    state_dim=actual_state_dim,
                    action_dim=action_dim,
                    instr_dim=CONFIG.instr_emb_dim,
                    latent_dim=CONFIG.latent_dim,
                    fusion_hidden_dim=CONFIG.hidden_dim
                )
                print(f"RTX4060 MoE Agent initialized")
            
            # 지시사항 임베딩
            instr_emb = lang.encode(task['instruction'])
            print(f"Instruction embedding shape: {instr_emb.shape}")
            
            # 비전 시스템 초기화 (필요한 경우)
            if task['env_cfg'].get('use_vision', False) and vision_system is None:
                vision_system = get_vision_system(CONFIG.device)
                print("Vision system initialized")
            
            # 에피소드 루프
            for episode in range(task['episodes']):
                print(f"\n--- Episode {episode + 1}/{task['episodes']} ---")
                
                # 환경 리셋
                state, _ = env.reset()
                done = False
                step_count = 0
                max_steps = CONFIG.max_episode_length
                episode_reward = 0
                episode_experiences = []
                
                while not done and step_count < max_steps:
                    # 다중 환경 처리
                    if state.ndim == 2:
                        current_state = state[0]  # 첫 번째 환경만 사용
                    else:
                        current_state = state
                    
                    try:
                        # 행동 선택
                        action, cluster_id, latent_state = agent.select_action(
                            current_state, instr_emb, deterministic=False
                        )
                        
                        if step_count == 0:
                            print(f"  Action shape: {action.shape}")
                            print(f"  Assigned to cluster: {cluster_id}")
                        
                        # 환경 스텝 - action을 numpy 배열로 확실히 변환
                        if torch.is_tensor(action):
                            action_np = action.detach().cpu().numpy()
                        else:
                            action_np = np.array(action, dtype=np.float32)
                        
                        # 환경 스텝
                        next_state, reward, done_info, info = env.step(action_np)
                        
                        # 보상 및 종료 처리
                        if isinstance(reward, (np.ndarray, torch.Tensor)):
                            if hasattr(reward, 'item'):
                                reward = reward.item()
                            elif len(reward) > 0:
                                reward = float(reward[0])
                            else:
                                reward = 0.0
                        else:
                            reward = float(reward)
                        
                        if isinstance(done_info, (np.ndarray, torch.Tensor)):
                            if hasattr(done_info, 'item'):
                                done = bool(done_info.item())
                            elif len(done_info) > 0:
                                done = bool(done_info[0])
                            else:
                                done = False
                        else:
                            done = bool(done_info)
                        
                        episode_reward += reward
                        
                        # 경험 저장
                        if next_state.ndim == 2:
                            next_state_single = next_state[0]
                        else:
                            next_state_single = next_state
                        
                        agent.store_experience(
                            state=current_state,
                            action=action_np,
                            reward=reward,
                            next_state=next_state_single,
                            done=done,
                            instruction_emb=instr_emb,
                            cluster_id=cluster_id,
                            latent_state=latent_state
                        )
                        
                        # 상태 업데이트
                        state = next_state
                        step_count += 1
                        
                        # 주기적 정보 출력 (메모리 절약을 위해 덜 빈번하게)
                        if step_count % 100 == 0:
                            # 현재 상태에서 정보 추출
                            if state.ndim == 2:
                                current_obs = state[0]
                            else:
                                current_obs = state
                            
                            try:
                                base_dim = actual_state_dim - 6
                                vision_cube = current_obs[base_dim:base_dim+3]
                                goal_pos = current_obs[base_dim+3:base_dim+6]
                                cube_goal_dist = np.linalg.norm(vision_cube - goal_pos)
                                
                                print(f"    Step {step_count}: Reward={reward:.3f}, Total={episode_reward:.3f}")
                                print(f"    Cube-Goal distance: {cube_goal_dist:.3f}")
                                print(f"    Current cluster: {cluster_id}")
                            except Exception as e:
                                print(f"    Step {step_count}: Reward={reward:.3f}, Total={episode_reward:.3f}")
                        
                    except Exception as e:
                        print(f"Warning: Environment step failed: {e}")
                        break
                
                # 전문가 업데이트 (메모리 효율적)
                if episode % CONFIG.update_frequency == 0 and episode > 0:
                    try:
                        losses = agent.update_experts(batch_size=CONFIG.batch_size)
                        if losses:
                            print(f"  Updated {len(losses)} experts")
                    except Exception as e:
                        print(f"Warning: Expert update failed: {e}")
                
                # 클러스터 병합
                if episode % CONFIG.merge_frequency == 0 and episode > 0:
                    try:
                        agent.merge_clusters(threshold=CONFIG.merge_threshold)
                    except Exception as e:
                        print(f"Warning: Cluster merge failed: {e}")
                
                # 메모리 정리 (주기적)
                if episode % CONFIG.cleanup_frequency == 0:
                    cleanup_memory()
                
                print(f"Episode {episode + 1} Summary:")
                print(f"  Steps: {step_count}, Total Reward: {episode_reward:.3f}")
                if step_count > 0:
                    print(f"  Average Reward: {episode_reward/step_count:.3f}")
                
                # 에이전트 정보 출력 (덜 빈번하게)
                if episode % 20 == 0:
                    try:
                        agent_info = agent.get_info()
                        print(f"  Agent Info: {agent_info['num_experts']} experts, "
                              f"{agent_info['cluster_info']['num_clusters']} clusters")
                        if agent_info['buffer_sizes']:
                            total_experiences = sum(agent_info['buffer_sizes'])
                            print(f"  Total experiences: {total_experiences}")
                    except Exception as e:
                        print(f"Warning: Failed to get agent info: {e}")
            
            # 태스크 완료 정보
            try:
                final_info = agent.get_info()
                print(f"\nTask '{task['name']}' completed!")
                print(f"Final state: {final_info['num_experts']} experts, "
                      f"{final_info['cluster_info']['num_clusters']} clusters")
            except Exception as e:
                print(f"\nTask '{task['name']}' completed with warnings: {e}")
            
            # 모델 저장 (메모리 효율적)
            save_path = f"{args.save_dir}/rtx4060_agent_task_{task_idx}.pt"
            if agent:
                agent.save(save_path)
            
            # 환경 정리
            env.close()
            
        except Exception as e:
            print(f"Error in task {task['name']}: {e}")
            print("Continuing to next task...")
            continue
        
        # 메모리 정리
        cleanup_memory()
    
    # 최종 정리
    try:
        if agent:
            agent.cleanup()
        
        if vision_system:
            vision_system.cleanup()
    except Exception as e:
        print(f"Warning during cleanup: {e}")
    
    sim.close()
    cleanup_memory()
    
    print("\n" + "="*60)
    print("RTX 4060 Optimized Training Completed!")
    print("="*60)

if __name__ == "__main__":
    main()