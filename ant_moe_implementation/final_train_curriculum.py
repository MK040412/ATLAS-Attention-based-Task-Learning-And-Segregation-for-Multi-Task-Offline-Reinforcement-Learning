import argparse
import torch
import yaml
import numpy as np
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Simplified Ant MoE Agent")
    parser.add_argument("--tasks", type=str, default="tasks.yaml")
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--num_envs", type=int, default=1)  # 단일 환경으로 시작
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    from ant_push_env import IsaacAntPushEnv
    from config import CONFIG
    from language_processor import LanguageProcessor
    from simplified_moe_agent import SimplifiedMoEAgent

    # 전역 변수들
    tasks = yaml.safe_load(open(args.tasks))
    lang = LanguageProcessor(device=CONFIG.device)
    agent = None

    # 메인 학습 루프
    for task_idx, task in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"Task {task_idx + 1}/{len(tasks)}: {task['name']}")
        print(f"Instruction: {task['instruction']}")
        print(f"{'='*60}")
        
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
        
        action_space_shape = env.action_space.shape
        if len(action_space_shape) == 2:
            action_dim = action_space_shape[1]
        else:
            action_dim = action_space_shape[0]
        
        print(f"State dim: {actual_state_dim}, Action dim: {action_dim}")
        print(f"State shape: {state.shape}")
        
        # CONFIG 업데이트
        if CONFIG.base_state_dim is None:
            CONFIG.base_state_dim = actual_state_dim
        
        # 에이전트 초기화 (한 번만)
        if agent is None:
            agent = SimplifiedMoEAgent(
                state_dim=actual_state_dim,
                action_dim=action_dim,
                instr_dim=CONFIG.instr_emb_dim,
                latent_dim=CONFIG.latent_dim,
                fusion_hidden_dim=CONFIG.hidden_dim
            )
            print(f"Initialized SimplifiedMoEAgent")
        
        # 지시사항 임베딩
        instr_emb = lang.encode(task['instruction'])
        print(f"Instruction embedding shape: {instr_emb.shape}")
        
        # 에피소드 루프
        for episode in range(task['episodes']):
            print(f"\n--- Episode {episode + 1}/{task['episodes']} ---")
            
            # 환경 리셋
            state, _ = env.reset()
            done = False
            step_count = 0
            max_steps = 1000
            episode_reward = 0
            episode_experiences = []
            
            # 초기 상태 정보
            if episode == 0:
                print(f"  Initial state shape: {state.shape}")
                if state.ndim == 2:
                    sample_state = state[0]
                else:
                    sample_state = state
                
                # 상태 구성 분석
                base_dim = actual_state_dim - 6  # vision_cube(3) + goal(3) 제외
                proprioception = sample_state[:base_dim]
                vision_cube_pos = sample_state[base_dim:base_dim+3]
                goal_pos = sample_state[base_dim+3:base_dim+6]
                
                print(f"  State composition:")
                print(f"    Proprioception dim: {proprioception.shape[0]}")
                print(f"    Vision cube pos: {vision_cube_pos}")
                print(f"    Goal pos: {goal_pos}")
            
            while not done and step_count < max_steps:
                # 다중 환경 처리
                if state.ndim == 2:
                    current_state = state[0]  # 첫 번째 환경만 사용
                    
                    # 행동 선택
                    action, cluster_id, latent_state = agent.select_action(
                        current_state, instr_emb, deterministic=False
                    )
                    
                    # 모든 환경에 같은 행동 적용
                    full_action = np.tile(action, (args.num_envs, 1))
                else:
                    current_state = state
                    action, cluster_id, latent_state = agent.select_action(
                        current_state, instr_emb, deterministic=False
                    )
                    full_action = action
                
                if step_count == 0:
                    print(f"  Action shape: {action.shape}")
                    print(f"  Assigned to cluster: {cluster_id}")
                
                # 환경 스텝
                try:
                    next_state, reward, done_info, info = env.step(full_action)
                except Exception as e:
                    print(f"Warning: Environment step failed: {e}")
                    break
                
                # 보상 및 종료 처리
                if isinstance(reward, np.ndarray):
                    reward = reward[0] if len(reward) > 0 else 0.0
                elif torch.is_tensor(reward):
                    reward = reward.item() if reward.numel() == 1 else reward[0].item()
                else:
                    reward = float(reward)
                
                if isinstance(done_info, np.ndarray):
                    done = done_info[0] if len(done_info) > 0 else False
                elif torch.is_tensor(done_info):
                    done = done_info.item() if done_info.numel() == 1 else done_info[0].item()
                else:
                    done = bool(done_info)
                
                episode_reward += reward
                
                # 경험 저장
                if next_state.ndim == 2:
                    next_state_single = next_state[0]
                else:
                    next_state_single = next_state
                
                episode_experiences.append({
                    'state': current_state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state_single,
                    'done': done,
                    'instruction_emb': instr_emb,
                    'cluster_id': cluster_id,
                    'latent_state': latent_state
                })
                
                # 상태 업데이트
                state = next_state
                step_count += 1
                
                # 주기적 정보 출력
                if step_count % 200 == 0:
                    # 현재 상태에서 정보 추출
                    if state.ndim == 2:
                        current_obs = state[0]
                    else:
                        current_obs = state
                    
                    base_dim = actual_state_dim - 6
                    vision_cube = current_obs[base_dim:base_dim+3]
                    goal_pos = current_obs[base_dim+3:base_dim+6]
                    cube_goal_dist = np.linalg.norm(vision_cube - goal_pos)
                    
                    print(f"    Step {step_count}: Reward={reward:.3f}, Total={episode_reward:.3f}")
                    print(f"    Vision cube pos: [{vision_cube[0]:.2f}, {vision_cube[1]:.2f}, {vision_cube[2]:.2f}]")
                    print(f"    Goal pos: [{goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f}]")
                    print(f"    Cube-Goal distance: {cube_goal_dist:.3f}")
                    print(f"    Current cluster: {cluster_id}")
            
            # 에피소드 종료 후 경험 저장
            for exp in episode_experiences:
                agent.store_experience(**exp)
            
            # 전문가 업데이트
            if episode % CONFIG.update_frequency == 0 and episode > 0:
                losses = agent.update_experts(batch_size=CONFIG.batch_size)
                if losses:
                    print(f"  Updated experts: {list(losses.keys())}")
                    # 평균 손실 출력
                    valid_losses = {k: v for k, v in losses.items() if v is not None}
                    if valid_losses:
                        print(f"  Loss summary: {len(valid_losses)} experts updated")
            
            # 클러스터 병합
            if episode % CONFIG.merge_frequency == 0 and episode > 0:
                agent.merge_clusters(threshold=CONFIG.merge_threshold)
            
            print(f"Episode {episode + 1} Summary:")
            print(f"  Steps: {step_count}, Total Reward: {episode_reward:.3f}")
            if step_count > 0:
                print(f"  Average Reward: {episode_reward/step_count:.3f}")
            
            # 에이전트 정보 출력
            if episode % 10 == 0:
                agent_info = agent.get_info()
                print(f"  Agent Info: {agent_info['num_experts']} experts, "
                      f"{agent_info['cluster_info']['num_clusters']} clusters")
                if agent_info['buffer_sizes']:
                    total_experiences = sum(agent_info['buffer_sizes'])
                    print(f"  Total experiences: {total_experiences}")
        
        # 태스크 완료 정보
        final_info = agent.get_info()
        print(f"\nTask '{task['name']}' completed!")
        print(f"Final state: {final_info['num_experts']} experts, "
              f"{final_info['cluster_info']['num_clusters']} clusters")
        if final_info['cluster_info']['cluster_counts']:
            print(f"Cluster counts: {final_info['cluster_info']['cluster_counts']}")
        
        # 모델 저장
        save_path = f"{args.save_dir}/agent_after_task_{task_idx}.pt"
        try:
            agent.save(save_path)
            print(f"Agent saved to {save_path}")
        except Exception as e:
            print(f"Warning: Failed to save agent: {e}")
        
        env.close()

    sim.close()
    print("\n🎉 Simplified training completed!")

if __name__ == "__main__":
    main()