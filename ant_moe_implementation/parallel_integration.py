def train_with_parallel_environments():
    """병렬 환경을 활용한 학습"""
    
    # 병렬 환경 생성 (64개 환경 사용)
    env = RotatingVectorizedWrapper(num_envs=64, rotation_frequency=20)
    
    # 언어 처리기
    from language_processor import LanguageProcessor
    lang_processor = LanguageProcessor(device=CONFIG.device)
    
    # MoE 에이전트 (기존 코드 재사용)
    from sb3_moe_agent import SB3MoEAgent
    
    # 첫 번째 리셋으로 차원 확인
    state, _ = env.reset()
    instruction = "Push the blue cube to the goal position"
    instruction_emb = lang_processor.encode(instruction)
    
    agent = SB3MoEAgent(
        env=env,
        instr_dim=instruction_emb.shape[-1],
        latent_dim=32
    )
    
    print(f"🚀 Parallel MoE Training Started:")
    print(f"  • Parallel environments: {env.num_envs}")
    print(f"  • State dimension: {state.shape}")
    print(f"  • Instruction dimension: {instruction_emb.shape}")
    
    # 병렬 경험 수집기
    parallel_collector = ParallelExperienceCollector(env, agent)
    
    # 학습 루프
    for episode in range(100):
        print(f"\n--- Episode {episode + 1}/100 ---")
        
        # 🔄 환경 로테이션으로 다양성 확보
        if episode % 10 == 0:
            print(f"🔄 Using environment {env.active_env_idx}/{env.num_envs}")
        
        # 단일 에피소드 실행 (선택된 환경에서)
        episode_reward = 0
        step_count = 0
        max_steps = 500
        
        state, _ = env.reset()
        
        while step_count < max_steps:
            # 행동 선택
            action, cluster_id = agent.select_action(
                state, 
                instruction_emb.squeeze(0).detach().cpu().numpy(),
                deterministic=False
            )
            
            if step_count == 0:
                print(f"  🎯 Assigned to cluster/expert: {cluster_id}")
                print(f"  🌍 Active environment: {env.active_env_idx}")
            
            # 환경 스텝 (내부적으로 병렬 처리)
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            # 경험 저장
            agent.store_experience(state, action, reward, next_state, done, cluster_id)
            
            state = next_state
            step_count += 1
            
            if done or truncated:
                break
        
        # 🚀 병렬 배치 경험 수집 (주기적으로)
        if episode % 5 == 0 and episode > 0:
            print(f"  🔄 Collecting parallel batch experiences...")
            batch_experiences = parallel_collector.collect_batch(
                instruction_emb, 
                batch_size=32,  # 32개 환경에서 동시 수집
                steps_per_env=50
            )
            
            # 배치 경험을 각 전문가에게 분배
            agent.store_batch_experiences(batch_experiences)
            print(f"  📦 Collected {len(batch_experiences)} parallel experiences")
        
        # 전문가 업데이트
        if episode % 10 == 0 and episode > 0:
            update_results = agent.update_experts(batch_size=256)
            if update_results:
                updated_count = len([k for k, v in update_results.items() if v is not None])
                print(f"  📈 Updated {updated_count} experts")
        
        # 클러스터 병합
        if episode % 15 == 0 and episode > 0:
            merged = agent.merge_clusters(verbose=(episode % 30 == 0))
            if merged:
                print(f"  🔗 Clusters merged")
        
        # 통계 출력
        if episode % 20 == 0 or episode == 99:
            agent_info = agent.get_info()
            print(f"  📊 Episode {episode + 1} Summary:")
            print(f"    Episode reward: {episode_reward:.2f}")
            print(f"    Steps: {step_count}")
            print(f"    Clusters: {agent_info['dpmm']['num_clusters']}")
            print(f"    Experts: {agent_info['num_experts']}")
            print(f"    Total buffer size: {agent_info['total_buffer_size']}")
            print(f"    Parallel env utilization: {info.get('parallel_rewards_mean', 0):.2f}")
    
    env.close()
    return agent

class ParallelExperienceCollector:
    """병렬 환경에서 경험 수집"""
    
    def __init__(self, vectorized_env, agent):
        self.env = vectorized_env
        self.agent = agent
        self.num_envs = vectorized_env.num_envs
    
    def collect_batch(self, instruction_emb, batch_size: int = 32, steps_per_env: int = 50):
        """배치로 병렬 경험 수집"""
        experiences = []
        
        # 사용할 환경 인덱스들 선택
        env_indices = np.random.choice(self.num_envs, size=min(batch_size, self.num_envs), replace=False)
        
        for env_idx in env_indices:
            # 환경 전환
            original_idx = self.env.active_env_idx
            self.env.active_env_idx = env_idx
            
            try:
                # 짧은 에피소드 실행
                state, _ = self.env.reset()
                
                for step in range(steps_per_env):
                    action, cluster_id = self.agent.select_action(
                        state, 
                        instruction_emb.squeeze(0).detach().cpu().numpy(),
                        deterministic=True  # 탐험보다는 활용
                    )
                    
                    next_state, reward, done, truncated, info = self.env.step(action)
                    
                    experiences.append({
                        'state': state.copy(),
                        'action': action.copy(),
                        'reward': reward,
                        'next_state': next_state.copy(),
                        'done': done or truncated,
                        'cluster_id': cluster_id,
                        'env_idx': env_idx,
                        'instruction_emb': instruction_emb.squeeze(0).detach().cpu().numpy()
                    })
                    
                    state = next_state
                    
                    if done or truncated:
                        break
                        
            except Exception as e:
                print(f"⚠️ Parallel collection error in env {env_idx}: {e}")
            
            # 원래 환경으로 복원
            self.env.active_env_idx = original_idx
        
        return experiences

# 🔧 SB3MoEAgent 확장 (배치 경험 처리)
class EnhancedSB3MoEAgent(SB3MoEAgent):
    """병렬 경험을 처리할 수 있는 확장된 MoE 에이전트"""
    
    def store_batch_experiences(self, batch_experiences):
        """배치 경험들을 각 전문가에게 분배"""
        cluster_experiences = {}
        
        # 클러스터별로 경험 분류
        for exp in batch_experiences:
            cluster_id = exp['cluster_id']
            if cluster_id not in cluster_experiences:
                cluster_experiences[cluster_id] = []
            cluster_experiences[cluster_id].append(exp)
        
        # 각 클러스터의 전문가에게 경험 저장
        for cluster_id, experiences in cluster_experiences.items():
            if cluster_id in self.experts:
                expert = self.experts[cluster_id]
                
                for exp in experiences:
                    if hasattr(expert, 'replay_buffer'):
                        expert.replay_buffer.add(
                            exp['state'], exp['action'], exp['reward'],
                            exp['next_state'], exp['done']
                        )
        
        print(f"  📦 Distributed {len(batch_experiences)} experiences to {len(cluster_experiences)} clusters")
    
    def get_parallel_statistics(self):
        """병렬 환경 관련 통계"""
        base_stats = self.get_info()
        
        # 클러스터별 경험 분포
        cluster_experience_counts = {}
        for cluster_id, expert in self.experts.items():
            if hasattr(expert, 'replay_buffer'):
                cluster_experience_counts[cluster_id] = len(expert.replay_buffer)
        
        base_stats['parallel_stats'] = {
            'cluster_experience_distribution': cluster_experience_counts,
            'total_parallel_experiences': sum(cluster_experience_counts.values()),
            'experience_balance': np.std(list(cluster_experience_counts.values())) if cluster_experience_counts else 0
        }
        
        return base_stats

# 🎯 메인 실행 함수
def main_parallel_training():
    """병렬 환경을 활용한 메인 학습"""
    print("🚀 Starting Parallel MoE Training with IsaacLab")
    
    try:
        # 병렬 학습 실행
        trained_agent = train_with_parallel_environments()
        
        # 최종 통계
        final_stats = trained_agent.get_parallel_statistics()
        print(f"\n🎉 Parallel Training Completed!")
        print(f"  • Final clusters: {final_stats['dpmm']['num_clusters']}")
        print(f"  • Final experts: {final_stats['num_experts']}")
        print(f"  • Total experiences: {final_stats['parallel_stats']['total_parallel_experiences']}")
        print(f"  • Experience balance (std): {final_stats['parallel_stats']['experience_balance']:.2f}")
        
        return trained_agent
        
    except Exception as e:
        print(f"❌ Parallel training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main_parallel_training()