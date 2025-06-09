import argparse
import torch
import numpy as np
import yaml
import os
from datetime import datetime
import matplotlib.pyplot as plt
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Full Curriculum Training")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="./curriculum_results")
    parser.add_argument("--checkpoint_freq", type=int, default=50)
    parser.add_argument("--eval_freq", type=int, default=20)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        from ant_push_env_improved import IsaacAntPushEnvImproved
        from integrated_moe_agent import IntegratedMoEAgent
        from language_processor import LanguageProcessor
        from improved_reward_system import CurriculumManager
        from config import CONFIG
        
        print("🎓 Full Curriculum Training Started")
        
        # 결과 저장 디렉토리 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(args.save_dir, f"curriculum_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        
        # 언어 처리기 초기화
        lang_processor = LanguageProcessor(device=CONFIG.device)
        
        # 커리큘럼 매니저 초기화
        curriculum = CurriculumManager()
        
        # 환경 생성 (초기 레벨로)
        initial_level = curriculum.get_current_level()
        env = IsaacAntPushEnvImproved(
            custom_cfg=initial_level,
            num_envs=args.num_envs
        )
        
        # 상태 및 액션 차원 확인
        state, _ = env.reset()
        state_dim = state.shape[0]
        action_dim = env.get_action_dim()
        
        print(f"📊 Training Setup:")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Save directory: {save_dir}")
        
        # MoE 에이전트 생성
        agent = IntegratedMoEAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            instr_dim=CONFIG.instr_emb_dim,
            latent_dim=CONFIG.latent_dim,
            hidden_dim=CONFIG.hidden_dim
        )
        
        # 학습 통계 저장
        training_stats = {
            'episodes': [],
            'rewards': [],
            'success_rates': [],
            'steps': [],
            'levels': [],
            'expert_counts': [],
            'cluster_counts': []
        }
        
        # 명령어 템플릿
        instruction_templates = [
            "Push the blue cube to the goal position",
            "Move the cube towards the target location", 
            "Navigate to the blue object and push it to the goal",
            "Guide the cube to reach the destination",
            "Push the blue block to the target area"
        ]
        
        episode = 0
        total_episodes = 1000  # 총 에피소드 수
        
        print(f"\n🚀 Starting curriculum training for {total_episodes} episodes")
        
        while episode < total_episodes:
            # 현재 레벨 정보
            current_level = curriculum.get_current_level()
            
            # 환경 태스크 설정 업데이트
            env.update_task_config(current_level)
            
            # 명령어 선택 (다양성을 위해 랜덤)
            instruction = np.random.choice(instruction_templates)
            instr_emb = lang_processor.encode(instruction)
            
            # 에피소드 실행
            episode_reward, episode_steps, episode_success, episode_info = run_episode(
                env=env,
                agent=agent,
                instruction_emb=instr_emb,
                max_steps=current_level['max_steps'],
                episode_num=episode
            )
            
            # 커리큘럼에 결과 추가
            final_distance = episode_info.get('final_cube_to_goal_distance', float('inf'))
            curriculum.add_episode_result(
                success=episode_success,
                final_distance=final_distance,
                steps=episode_steps,
                total_reward=episode_reward
            )
            
            # 통계 저장
            training_stats['episodes'].append(episode)
            training_stats['rewards'].append(episode_reward)
            training_stats['steps'].append(episode_steps)
            training_stats['levels'].append(curriculum.current_level)
            
            agent_stats = agent.get_statistics()
            training_stats['expert_counts'].append(agent_stats['num_experts'])
            training_stats['cluster_counts'].append(agent_stats['num_clusters'])
            
            # 성공률 계산 (최근 20개 에피소드)
            recent_episodes = min(20, len(training_stats['episodes']))
            if recent_episodes > 0:
                recent_successes = sum(curriculum.episode_results[-recent_episodes:][i]['success'] 
                                     for i in range(recent_episodes))
                success_rate = recent_successes / recent_episodes
                training_stats['success_rates'].append(success_rate)
            else:
                training_stats['success_rates'].append(0.0)
            
            # 진행 상황 출력
            if episode % 10 == 0 or episode_success:
                curriculum_stats = curriculum.get_statistics()
                print(f"\n📈 Episode {episode + 1}/{total_episodes}")
                print(f"  Level: {curriculum_stats['level_name']}")
                print(f"  Reward: {episode_reward:.2f}, Steps: {episode_steps}")
                print(f"  Success: {'✅' if episode_success else '❌'}")
                print(f"  Recent Success Rate: {curriculum_stats['recent_success_rate']:.2%}")
                print(f"  Experts: {agent_stats['num_experts']}, Clusters: {agent_stats['num_clusters']}")
                
                if episode_info:
                    print(f"  Final Distance: {episode_info.get('final_cube_to_goal_distance', 0):.3f}")
                    print(f"  GLIP Error: {episode_info.get('glip_detection_error', 0):.3f}")
            
            # 커리큘럼 레벨 진행 확인
            if curriculum.should_advance_level():
                curriculum.advance_level()
                print(f"🎓 Advanced to level {curriculum.current_level + 1}")
            
            # 체크포인트 저장
            if (episode + 1) % args.checkpoint_freq == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_ep{episode + 1}.pth")
                agent.save_checkpoint(checkpoint_path)
                
                # 통계 저장
                stats_path = os.path.join(save_dir, f"stats_ep{episode + 1}.npz")
                np.savez(stats_path, **training_stats)
                print(f"💾 Saved checkpoint and stats at episode {episode + 1}")
            
            # 평가 및 시각화
            if (episode + 1) % args.eval_freq == 0:
                create_training_plots(training_stats, save_dir, episode + 1)
                
                # 상세 평가 실행
                eval_results = evaluate_agent(env, agent, lang_processor, instruction_templates)
                print(f"🔍 Evaluation Results:")
                for key, value in eval_results.items():
                    print(f"  {key}: {value}")
            
            episode += 1
        
        # 최종 결과 저장
        final_checkpoint = os.path.join(save_dir, "final_model.pth")
        agent.save_checkpoint(final_checkpoint)
        
        final_stats = os.path.join(save_dir, "final_stats.npz")
        np.savez(final_stats, **training_stats)
        
        # 최종 시각화
        create_training_plots(training_stats, save_dir, "final")
        
        # 최종 평가
        print(f"\n🏆 Final Evaluation:")
        final_eval = evaluate_agent(env, agent, lang_processor, instruction_templates, num_episodes=10)
        for key, value in final_eval.items():
            print(f"  {key}: {value}")
        
        print(f"\n✅ Curriculum training completed!")
        print(f"📁 Results saved to: {save_dir}")
        
        env.close()
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        sim.close()

def run_episode(env, agent, instruction_emb, max_steps, episode_num):
    """단일 에피소드 실행"""
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    instr_np = instruction_emb.squeeze(0).cpu().numpy()
    
    episode_experiences = []
    cluster_history = []
    
    while not done and steps < max_steps:
        # 행동 선택
        action, cluster_id = agent.select_action(
            state=state,
            instruction_emb=instr_np,
            deterministic=False
        )
        
        cluster_history.append(cluster_id)
        
        # 환경에서 스텝 실행
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        # 경험 저장
        agent.store_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            instruction_emb=instr_np,
            cluster_id=cluster_id
        )
        
        episode_experiences.append({
            'state': state.copy(),
            'action': action.copy(),
            'reward': reward,
            'next_state': next_state.copy(),
            'done': done,
            'cluster_id': cluster_id
        })
        
        # 에이전트 업데이트 (매 5스텝마다)
        if steps % 5 == 0 and steps > 0:
            agent.update_experts(batch_size=32)
        
        state = next_state
        steps += 1
    
    # 에피소드 종료 후 정보
    success = info.get('success', False)
    final_distance = info.get('cube_to_goal_distance', float('inf'))
    
    episode_info = {
        'final_cube_to_goal_distance': final_distance,
        'glip_detection_error': info.get('glip_detection_error', 0),
        'unique_clusters': len(set(cluster_history)),
        'most_used_cluster': max(set(cluster_history), key=cluster_history.count) if cluster_history else 0,
        'reward_breakdown': info.get('reward_info', {})
    }
    
    return total_reward, steps, success, episode_info

def evaluate_agent(env, agent, lang_processor, instruction_templates, num_episodes=5):
    """에이전트 평가"""
    eval_results = {
        'success_rate': 0.0,
        'average_reward': 0.0,
        'average_steps': 0.0,
        'average_final_distance': 0.0
    }
    
    total_success = 0
    total_reward = 0
    total_steps = 0
    total_distance = 0
    
    for i in range(num_episodes):
        instruction = instruction_templates[i % len(instruction_templates)]
        instr_emb = lang_processor.encode(instruction)
        
        # 결정적 행동으로 평가
        episode_reward, episode_steps, episode_success, episode_info = run_episode(
            env=env,
            agent=agent,
            instruction_emb=instr_emb,
            max_steps=500,
            episode_num=f"eval_{i}"
        )
        
        total_success += int(episode_success)
        total_reward += episode_reward
        total_steps += episode_steps
        total_distance += episode_info.get('final_cube_to_goal_distance', 0)
    
    eval_results['success_rate'] = total_success / num_episodes
    eval_results['average_reward'] = total_reward / num_episodes
    eval_results['average_steps'] = total_steps / num_episodes
    eval_results['average_final_distance'] = total_distance / num_episodes
    
    return eval_results

def create_training_plots(stats, save_dir, episode_suffix):
    """학습 진행 상황 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Training Progress - Episode {episode_suffix}', fontsize=16)
    
    episodes = stats['episodes']
    
    # 1. 보상 그래프
    axes[0, 0].plot(episodes, stats['rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # 2. 성공률 그래프
    if len(stats['success_rates']) > 0:
        axes[0, 1].plot(episodes, stats['success_rates'])
        axes[0, 1].set_title('Success Rate (Recent 20 episodes)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True)
    
    # 3. 스텝 수 그래프
    axes[0, 2].plot(episodes, stats['steps'])
    axes[0, 2].set_title('Episode Steps')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Steps')
    axes[0, 2].grid(True)
    
    # 4. 커리큘럼 레벨
    axes[1, 0].plot(episodes, stats['levels'])
    axes[1, 0].set_title('Curriculum Level')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Level')
    axes[1, 0].grid(True)
    
    # 5. 전문가 수
    axes[1, 1].plot(episodes, stats['expert_counts'])
    axes[1, 1].set_title('Number of Experts')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Expert Count')
    axes[1, 1].grid(True)
    
    # 6. 클러스터 수
    axes[1, 2].plot(episodes, stats['cluster_counts'])
    axes[1, 2].set_title('Number of Clusters')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Cluster Count')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    # 저장
    plot_path = os.path.join(save_dir, f'training_progress_{episode_suffix}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training plots saved to {plot_path}")

if __name__ == "__main__":
    main()