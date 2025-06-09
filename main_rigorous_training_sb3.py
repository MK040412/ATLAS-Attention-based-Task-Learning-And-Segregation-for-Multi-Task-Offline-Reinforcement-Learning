#!/usr/bin/env python3
"""
SB3 기반 엄밀한 DPMM + MoE-SAC 통합 학습 스크립트 (수정됨)
"""

import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="SB3-based Rigorous DPMM-MoE-SAC Training")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--tau_birth", type=float, default=1e-4)
    parser.add_argument("--tau_merge", type=float, default=0.5)
    parser.add_argument("--save_checkpoint", type=str, default="rigorous_sb3_checkpoint.pth")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        # SB3 기반 모듈들 import
        from fixed_simple_gym_wrapper import SimpleAntPushWrapper
        from sb3_moe_agent import SB3MoEAgent
        from language_processor import LanguageProcessor
        from config import CONFIG

        print(f"🚀 Starting SB3-based Rigorous DPMM-MoE-SAC Training")
        print(f"  Episodes: {args.episodes}")
        print(f"  Max steps per episode: {args.max_steps}")
        print(f"  Latent dimension: {args.latent_dim}")
        print(f"  Birth threshold: {args.tau_birth}")
        print(f"  Merge threshold: {args.tau_merge}")

        # 환경 설정 (SB3 호환)
        env_config = {
            'cube_position': [2.0, 0.0, 0.1],
            'goal_position': [3.0, 0.0, 0.1]
        }
        env = SimpleAntPushWrapper(custom_cfg=env_config)
        
        # 언어 처리기
        lang_processor = LanguageProcessor(device=CONFIG.device)
        instruction = "Push the blue cube to the goal position"
        instruction_emb = lang_processor.encode(instruction)
        instr_dim = instruction_emb.shape[-1]
        
        print(f"📊 Environment Dimensions:")
        print(f"  State dimension: {env.observation_space.shape[0]}")
        print(f"  Action dimension: {env.action_space.shape[0]}")
        print(f"  Instruction dimension: {instr_dim}")

        # SB3 기반 MoE 에이전트 초기화 (올바른 파라미터만 사용)
        agent = SB3MoEAgent(
            env=env,
            instr_dim=instr_dim,
            latent_dim=args.latent_dim,
            cluster_threshold=args.tau_merge  # SB3 버전에서 사용하는 파라미터명
        )
        
        # 에이전트에 추가 파라미터 설정 (tau_birth를 별도로 설정)
        if hasattr(agent, 'set_dpmm_params'):
            agent.set_dpmm_params(tau_birth=args.tau_birth, tau_merge=args.tau_merge)
        elif hasattr(agent, 'dpmm'):
            agent.dpmm.tau_birth = args.tau_birth
            agent.dpmm.tau_merge_kl = args.tau_merge

        print(f"✅ SB3 MoE Agent initialized successfully")
        print(f"  Agent type: {type(agent).__name__}")
        
        # 에이전트 정보 확인
        if hasattr(agent, 'get_info'):
            agent_info = agent.get_info()
            print(f"  Initial clusters: {agent_info.get('dpmm', {}).get('num_clusters', 'N/A')}")
            print(f"  Initial experts: {agent_info.get('num_experts', 'N/A')}")

        # 간단한 학습 루프 (먼저 기본 동작 확인)
        training_stats = simple_sb3_training_loop(
            env=env,
            agent=agent,
            instruction_emb=instruction_emb,
            max_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            update_frequency=10
        )

        # 결과 출력
        print(f"\n🎉 Training completed!")
        print(f"  Total episodes: {len(training_stats['episode_rewards'])}")
        print(f"  Average reward: {np.mean(training_stats['episode_rewards']):.2f}")
        
        # 체크포인트 저장
        if args.save_checkpoint:
            agent.save(args.save_checkpoint)
            print(f"💾 SB3 Checkpoint saved to {args.save_checkpoint}")

    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        sim.close()

def simple_sb3_training_loop(env, agent, instruction_emb, max_episodes, max_steps_per_episode, update_frequency):
    """간단한 SB3 기반 학습 루프"""
    
    training_stats = {
        'episode_rewards': [],
        'cluster_evolution': [],
        'expert_counts': []
    }
    
    for episode in range(max_episodes):
        if episode % 10 == 0:
            print(f"\n🎯 Episode {episode + 1}/{max_episodes}")
        
        state, _ = env.reset()
        episode_reward = 0
        step_count = 0
        
        while step_count < max_steps_per_episode:
            try:
                # SB3 기반 행동 선택
                action, cluster_id = agent.select_action(
                    state, 
                    instruction_emb.squeeze(0).detach().cpu().numpy(),
                    deterministic=False
                )
                
                # 환경 스텝
                next_state, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                
                # 경험 저장
                agent.store_experience(state, action, reward, next_state, done, cluster_id)
                
                state = next_state
                step_count += 1
                
                if done or truncated:
                    break
                    
            except Exception as e:
                print(f"  ⚠️ Step error at episode {episode}, step {step_count}: {e}")
                break
        
        # 주기적 업데이트
        if episode % update_frequency == 0 and episode > 0:
            try:
                update_results = agent.update_experts(batch_size=32)
                if update_results and episode % 20 == 0:
                    print(f"  📈 Updated experts: {list(update_results.keys())}")
            except Exception as e:
                print(f"  ⚠️ Update error at episode {episode}: {e}")
        
        # 통계 수집
        training_stats['episode_rewards'].append(episode_reward)
        
        try:
            agent_info = agent.get_info()
            training_stats['cluster_evolution'].append(agent_info.get('dpmm', {}))
            training_stats['expert_counts'].append(agent_info.get('num_experts', 0))
        except:
            training_stats['cluster_evolution'].append({})
            training_stats['expert_counts'].append(0)
        
        # 주기적 출력
        if episode % 20 == 0 and episode > 0:
            recent_rewards = training_stats['episode_rewards'][-10:]
            print(f"  📊 Episode {episode} Summary:")
            print(f"    Recent avg reward: {np.mean(recent_rewards):.2f}")
            print(f"    Episode reward: {episode_reward:.2f}")
            print(f"    Steps: {step_count}")
            
            try:
                agent_info = agent.get_info()
                print(f"    Clusters: {agent_info.get('dpmm', {}).get('num_clusters', 'N/A')}")
                print(f"    Experts: {agent_info.get('num_experts', 'N/A')}")
            except:
                print(f"    Clusters: N/A, Experts: N/A")
    
    return training_stats

if __name__ == "__main__":
    main()