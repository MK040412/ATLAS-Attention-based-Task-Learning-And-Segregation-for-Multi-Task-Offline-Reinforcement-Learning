#!/usr/bin/env python3
"""
엄밀한 DPMM + MoE-SAC 통합 학습 스크립트 (수정됨)
"""

import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Rigorous DPMM-MoE-SAC Training")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--tau_birth", type=float, default=1e-4)
    parser.add_argument("--tau_merge", type=float, default=0.5)
    parser.add_argument("--save_checkpoint", type=str, default="rigorous_checkpoint.pth")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    # 필요한 모듈들 import
    from ant_push_env import IsaacAntPushEnv
    from language_processor import LanguageProcessor
    from integrated_dpmm_moe_sac import IntegratedDPMMMoESAC
    from complete_training_loop import rigorous_training_loop, visualize_cluster_evolution, analyze_dpmm_behavior, mathematical_verification
    from config import CONFIG

    print(f"🚀 Starting Rigorous DPMM-MoE-SAC Training")
    print(f"  Episodes: {args.episodes}")
    print(f"  Max steps per episode: {args.max_steps}")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Birth threshold: {args.tau_birth}")
    print(f"  Merge threshold: {args.tau_merge}")

    # 환경 설정
    env_config = {
        'cube_position': [2.0, 0.0, 0.1],
        'goal_position': [3.0, 0.0, 0.1]
    }
    env = IsaacAntPushEnv(custom_cfg=env_config, num_envs=1)
    
    # 초기 상태로 차원 확인
    state, _ = env.reset()
    state_dim = state.shape[0]
    action_dim = env.action_space.shape[1] if len(env.action_space.shape) == 2 else env.action_space.shape[0]
    
    print(f"📊 Environment Dimensions:")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")

    # 언어 처리기
    lang_processor = LanguageProcessor(device=CONFIG.device)
    instruction = "Push the blue cube to the goal position"
    instruction_emb = lang_processor.encode(instruction)
    instr_dim = instruction_emb.shape[-1]
    
    print(f"  Instruction dimension: {instr_dim}")
    print(f"  Instruction: '{instruction}'")

    # 통합 에이전트 초기화
    agent = IntegratedDPMMMoESAC(
        state_dim=state_dim,
        action_dim=action_dim,
        instr_dim=instr_dim,
        latent_dim=args.latent_dim,
        tau_birth=args.tau_birth,
        tau_merge=args.tau_merge,
        device=str(CONFIG.device)
    )

    try:
        # 엄밀한 학습 실행
        training_stats = rigorous_training_loop(
            env=env,
            agent=agent,
            instruction_emb=instruction_emb,
            max_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            update_frequency=10,
            merge_frequency=20
        )

        # 상세 분석
        print(f"\n" + "="*60)
        print(f"🔬 POST-TRAINING ANALYSIS")
        print(f"="*60)
        
        analyze_dpmm_behavior(agent)
        mathematical_verification(agent)
        
        # 시각화
        visualize_cluster_evolution(training_stats)
        
        # 체크포인트 저장
        if args.save_checkpoint:
            agent.save_checkpoint(args.save_checkpoint)
            print(f"💾 Checkpoint saved to {args.save_checkpoint}")
        
        # 최종 요약
        final_stats = training_stats['final_comprehensive_stats']
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"  Total episodes: {len(training_stats['episode_rewards'])}")
        print(f"  Average reward: {np.mean(training_stats['episode_rewards']):.2f}")
        print(f"  Final clusters: {final_stats['dpmm']['num_clusters']}")
        print(f"  Final experts: {final_stats['total_experts']}")
        print(f"  Total births: {final_stats['dpmm']['total_births']}")
        print(f"  Total merges: {final_stats['dpmm']['total_merges']}")
        
        # 수학적 공식 최종 검증
        print(f"\n✅ Mathematical Formula Verification:")
        print(f"  DPMM Birth Heuristic: max_k ℓ_k < τ_birth = {args.tau_birth}")
        print(f"  DPMM Merge Heuristic: d_ij < τ_merge = {args.tau_merge}")
        print(f"  SAC Critic Loss: L_Q = E[(Q(s,a) - (r + γ(min Q(s',a') - α log π(a'|s'))))²]")
        print(f"  SAC Policy Loss: L_π = E[α log π(a|s) - Q(s,a)]")
        print(f"  Cluster Evolution: K_{{t+1}} = K_t + 1_{{birth}} - 1_{{merge}}")
        
        print(f"\n🎉 Rigorous DPMM-MoE-SAC Training Completed Successfully!")

    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        sim.close()

if __name__ == "__main__":
    main()