import torch
import numpy as np
from typing import Dict, List
from integrated_dpmm_moe_sac import IntegratedDPMMMoESAC

def rigorous_training_loop(env, agent: IntegratedDPMMMoESAC, 
                          instruction_emb, max_episodes: int = 1000,
                          max_steps_per_episode: int = 500,
                          update_frequency: int = 10,
                          merge_frequency: int = 50):
    """엄밀한 Per-Timestep Algorithm 구현"""
    
    print(f"🚀 Starting Rigorous DPMM-MoE-SAC Training")
    print(f"  Max episodes: {max_episodes}")
    print(f"  Max steps per episode: {max_steps_per_episode}")
    print(f"  Update frequency: {update_frequency}")
    print(f"  Merge frequency: {merge_frequency}")
    
    # 통계 추적
    episode_rewards = []
    episode_lengths = []
    cluster_evolution = []
    expert_evolution = []
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_cluster_usage = {}
        
        print(f"\n📍 Episode {episode + 1}/{max_episodes}")
        
        for step in range(max_steps_per_episode):
            # ===== Per-Timestep Algorithm =====
            
            # 1. z_t = FusionEncoder(e, s_t)
            # 2. k_t = DPMM.assign(z_t) # birth or assign  
            # 3. if new cluster k_t: init_expert()
            # 4. expert = experts[k_t]
            # 5. a_t, logp = expert.sample_action(s_t)
            action, cluster_id = agent.select_action(
                state=state,
                instruction_emb=instruction_emb,
                deterministic=False
            )
            
            # 클러스터 사용 통계
            if cluster_id not in episode_cluster_usage:
                episode_cluster_usage[cluster_id] = 0
            episode_cluster_usage[cluster_id] += 1
            
            # 6. s_{t+1}, r_t = env.step(a_t)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            # 7. expert.replay_buffer.add(s_t, a_t, r_t, s_{t+1}, done)
            agent.store_experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                cluster_id=cluster_id
            )
            
            # 8. expert.update() # SAC critic & actor losses
            if step % update_frequency == 0:
                update_losses = agent.update_experts(batch_size=64)
                if step == 0 and update_losses:  # 첫 스텝에서만 출력
                    print(f"  Update losses: {update_losses}")
            
            # 9. if time to merge: DPMM.merge()
            if step % merge_frequency == 0:
                old_cluster_count = agent.dpmm.get_cluster_count()
                agent.merge_clusters()
                new_cluster_count = agent.dpmm.get_cluster_count()
                
                if new_cluster_count != old_cluster_count:
                    print(f"  🔗 Clusters: {old_cluster_count} → {new_cluster_count}")
            
            # 상태 업데이트
            state = next_state
            
            # 에피소드 종료 조건
            if done:
                break
        
        # 에피소드 통계
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        
        # 클러스터/전문가 진화 추적
        evolution = agent.get_cluster_evolution()
        cluster_evolution.append(evolution['current_clusters'])
        expert_evolution.append(len(agent.experts))
        
        # 에피소드 요약
        print(f"  ✅ Reward: {episode_reward:.2f}, Steps: {episode_steps}")
        print(f"  🧠 Clusters: {evolution['current_clusters']}, Experts: {len(agent.experts)}")
        print(f"  📊 Cluster usage: {episode_cluster_usage}")
        
        # 상세 통계 (매 10 에피소드마다)
        if (episode + 1) % 10 == 0:
            comprehensive_stats = agent.get_comprehensive_statistics()
            print(f"\n📈 Episode {episode + 1} Comprehensive Stats:")
            print(f"  DPMM: {comprehensive_stats['dpmm']['num_clusters']} clusters")
            print(f"  Total births: {comprehensive_stats['dpmm']['total_births']}")
            print(f"  Total merges: {comprehensive_stats['dpmm']['total_merges']}")
            print(f"  Expert usage: {comprehensive_stats['expert_usage']}")
            
            # 최근 10 에피소드 평균 성능
            recent_rewards = episode_rewards[-10:]
            avg_reward = np.mean(recent_rewards)
            print(f"  Recent avg reward: {avg_reward:.2f}")
    
    # 최종 통계 및 시각화
    final_stats = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'cluster_evolution': cluster_evolution,
        'expert_evolution': expert_evolution,
        'final_comprehensive_stats': agent.get_comprehensive_statistics()
    }
    
    print(f"\n🎯 Training Completed!")
    print(f"  Total episodes: {len(episode_rewards)}")
    print(f"  Average reward: {np.mean(episode_rewards):.2f}")
    print(f"  Final clusters: {cluster_evolution[-1]}")
    print(f"  Final experts: {expert_evolution[-1]}")
    
    return final_stats

def visualize_cluster_evolution(stats: Dict):
    """클러스터 진화 시각화"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 에피소드별 보상
        axes[0, 0].plot(stats['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # 2. 클러스터 수 진화 (K_t)
        axes[0, 1].plot(stats['cluster_evolution'], label='Clusters', color='blue')
        axes[0, 1].plot(stats['expert_evolution'], label='Experts', color='red')
        axes[0, 1].set_title('Dynamic Behavior: K_t Evolution')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend()
        
        # 3. 에피소드 길이
        axes[1, 0].plot(stats['episode_lengths'])
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        
        # 4. 전문가 사용 분포
        final_stats = stats['final_comprehensive_stats']
        expert_usage = final_stats['expert_usage']
        if expert_usage:
            experts = list(expert_usage.keys())
            usage_counts = list(expert_usage.values())
            axes[1, 1].bar(experts, usage_counts)
            axes[1, 1].set_title('Expert Usage Distribution')
            axes[1, 1].set_xlabel('Expert ID')
            axes[1, 1].set_ylabel('Usage Count')
        
        plt.tight_layout()
        plt.savefig('dpmm_moe_sac_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Visualization saved as 'dpmm_moe_sac_evolution.png'")
        
    except ImportError:
        print("⚠️ matplotlib not available for visualization")

def analyze_dpmm_behavior(agent: IntegratedDPMMMoESAC):
    """DPMM 행동 분석"""
    stats = agent.get_comprehensive_statistics()
    
    print(f"\n🔬 DPMM Behavior Analysis:")
    print(f"=" * 50)
    
    # 클러스터 정보
    dpmm_stats = stats['dpmm']
    print(f"📊 Cluster Statistics:")
    print(f"  Total clusters: {dpmm_stats['num_clusters']}")
    print(f"  Cluster sizes: {dpmm_stats['cluster_sizes']}")
    print(f"  Total births: {dpmm_stats['total_births']}")
    print(f"  Total merges: {dpmm_stats['total_merges']}")
    
    # 클러스터 평균들
    print(f"\n🎯 Cluster Centers (μ_k):")
    for cluster_id, mean in dpmm_stats['cluster_means'].items():
        print(f"  Cluster {cluster_id}: μ = {np.array(mean)}")
    
    # 클러스터 분산들
    print(f"\n📈 Cluster Variances (σ²_k):")
    for cluster_id, var in dpmm_stats['cluster_variances'].items():
        print(f"  Cluster {cluster_id}: σ² = {var:.4f}")
    
    # Birth/Merge 이벤트 분석
    birth_events = agent.dpmm.birth_events
    merge_events = agent.dpmm.merge_events
    
    print(f"\n🐣 Birth Events Analysis:")
    for i, event in enumerate(birth_events[-5:]):  # 최근 5개만
        print(f"  Birth {i}: Cluster {event['cluster_id']} at timestep {event['timestep']}")
    
    print(f"\n🔗 Merge Events Analysis:")
    for i, event in enumerate(merge_events[-3:]):  # 최근 3개만
        print(f"  Merge {i}: {event['old_clusters']} → {event['new_cluster']} at timestep {event['timestep']}")
    
    # 전문가 성능 분석
    print(f"\n🤖 Expert Performance Analysis:")
    for expert_id, expert_stats in stats['experts'].items():
        print(f"  {expert_id}:")
        print(f"    Updates: {expert_stats['update_count']}")
        print(f"    Buffer size: {expert_stats['buffer_size']}")
        print(f"    Current α: {expert_stats['current_alpha']:.4f}")
        
        recent_losses = expert_stats['recent_losses']
        if recent_losses.get('actor_loss'):
            avg_actor_loss = np.mean(recent_losses['actor_loss'])
            avg_critic_loss = np.mean(recent_losses['critic1_loss'])
            print(f"    Recent avg actor loss: {avg_actor_loss:.4f}")
            print(f"    Recent avg critic loss: {avg_critic_loss:.4f}")

def mathematical_verification(agent: IntegratedDPMMMoESAC):
    """수학적 공식 검증"""
    print(f"\n🔬 Mathematical Verification:")
    print(f"=" * 50)
    
    # DPMM 공식 검증
    print(f"📐 DPMM Formulas:")
    for cluster_id, cluster in agent.dpmm.clusters.items():
        print(f"  Cluster {cluster_id}:")
        print(f"    μ_k = Σx_i / N_k")
        print(f"    Current: μ = {cluster.mu}")
        print(f"    N_k = {cluster.N}")
        
        # 온라인 업데이트 공식 확인
        expected_mu = cluster.sum_x / cluster.N
        mu_diff = np.linalg.norm(cluster.mu - expected_mu)
        print(f"    Online update error: ||μ_computed - μ_expected|| = {mu_diff:.8f}")
    
    # SAC 공식 확인
    print(f"\n🎯 SAC Formulas:")
    for expert in agent.experts[:2]:  # 처음 2개 전문가만
        print(f"  Expert {expert.expert_id}:")
        print(f"    Target entropy: H_target = -dim(A) = {expert.target_entropy}")
        print(f"    Current α: {expert.alpha.item():.4f}")
        print(f"    γ (discount): {expert.gamma}")
        print(f"    τ (soft update): {expert.tau}")
    
    # 클러스터 진화 공식: K_{t+1} = K_t + 1_{birth} - 1_{merge}
    evolution = agent.get_cluster_evolution()
    print(f"\n📈 Cluster Evolution Formula: K_{{t+1}} = K_t + 1_{{birth}} - 1_{{merge}}")
    print(f"  Current K_t: {evolution['current_clusters']}")
    print(f"  Total births: {len(agent.dpmm.birth_events)}")
    print(f"  Total merges: {len(agent.dpmm.merge_events)}")
    
    # 이론적 vs 실제 클러스터 수
    theoretical_K = len(agent.dpmm.birth_events) - len(agent.dpmm.merge_events)
    actual_K = evolution['current_clusters']
    print(f"  Theoretical K: {theoretical_K}")
    print(f"  Actual K: {actual_K}")
    print(f"  Difference: {abs(theoretical_K - actual_K)}")