import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=3)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        from ant_push_env_with_glip import IsaacAntPushEnvWithGLIP
        from integrated_moe_agent import IntegratedMoEAgent
        from language_processor import LanguageProcessor
        from config import CONFIG
        
        print("🧪 Step 4 Test: Integrated MoE Agent")
        
        # 언어 처리기 초기화
        lang_processor = LanguageProcessor(device=CONFIG.device)
        
        # 환경 생성
        env = IsaacAntPushEnvWithGLIP(
            custom_cfg={
                'cube_position': [2.0, 0.0, 0.1],
                'goal_position': [3.0, 0.0, 0.1]
            },
            num_envs=1
        )
        
        # 초기 리셋으로 상태 차원 확인
        state, _ = env.reset()
        state_dim = state.shape[0]
        action_dim = env.get_action_dim()
        
        print(f"📊 Environment Info:")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Instruction embedding dim: {CONFIG.instr_emb_dim}")
        
        # MoE 에이전트 생성
        agent = IntegratedMoEAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            instr_dim=CONFIG.instr_emb_dim,
            latent_dim=CONFIG.latent_dim,
            hidden_dim=CONFIG.hidden_dim
        )
        
        # 테스트 명령어들
        test_instructions = [
            "Push the blue cube to the goal position",
            "Move the cube towards the target",
            "Navigate to the blue object and push it"
        ]
        
        print(f"\n🎯 Testing with {len(test_instructions)} different instructions:")
        
        for episode in range(args.episodes):
            instruction = test_instructions[episode % len(test_instructions)]
            print(f"\n=== Episode {episode + 1}: '{instruction}' ===")
            
            # 명령어 임베딩
            instr_emb = lang_processor.encode(instruction)
            print(f"  Instruction embedding shape: {instr_emb.shape}")
            
            # 에피소드 실행
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            max_steps = 200
            
            cluster_history = []
            
            while steps < max_steps:
                # 행동 선택
                action, cluster_id = agent.select_action(
                    state=state,
                    instruction_emb=instr_emb.squeeze(0).cpu().numpy(),
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
                    instruction_emb=instr_emb.squeeze(0).cpu().numpy(),
                    cluster_id=cluster_id
                )
                
                # 에이전트 업데이트 (매 10스텝마다)
                if steps % 10 == 0 and steps > 0:
                    update_results = agent.update_experts(batch_size=32)
                    if update_results and steps % 50 == 0:
                        print(f"    Step {steps}: Updated {len(update_results)} experts")
                
                # 진행 상황 출력
                if steps % 50 == 0:
                    ant_pos = next_state[:3]
                    cube_pos = next_state[-6:-3]
                    goal_pos = next_state[-3:]
                    dist_to_goal = np.linalg.norm(cube_pos - goal_pos)
                    
                    print(f"    Step {steps}: Reward={reward:.3f}, Total={total_reward:.1f}")
                    print(f"      Cluster: {cluster_id}, Dist to goal: {dist_to_goal:.3f}")
                    print(f"      Ant: {ant_pos}, Cube: {cube_pos}")
                
                state = next_state
                steps += 1
                
                if done:
                    break
            
            # 에피소드 통계
            unique_clusters = len(set(cluster_history))
            most_used_cluster = max(set(cluster_history), key=cluster_history.count)
            
            print(f"  Episode Summary:")
            print(f"    Steps: {steps}, Total Reward: {total_reward:.2f}")
            print(f"    Unique clusters used: {unique_clusters}")
            print(f"    Most used cluster: {most_used_cluster}")
            
            agent.episode_count += 1
        
        # 최종 통계
        final_stats = agent.get_statistics()
        print(f"\n📈 Final Statistics:")
        print(f"  Total experts created: {final_stats['num_experts']}")
        print(f"  Total clusters: {final_stats['num_clusters']}")
        print(f"  Cluster counts: {final_stats['cluster_counts']}")
        print(f"  Expert buffer sizes: {final_stats['expert_buffer_sizes']}")
        print(f"  Total steps: {final_stats['total_steps']}")
        print(f"  Total episodes: {final_stats['total_episodes']}")
        
        # 체크포인트 저장 테스트
        checkpoint_path = "./test_checkpoint.pth"
        agent.save_checkpoint(checkpoint_path)
        
        print("✅ Step 4 test passed!")
        env.close()
        
    except Exception as e:
        print(f"❌ Step 4 test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        sim.close()

if __name__ == "__main__":
    main()