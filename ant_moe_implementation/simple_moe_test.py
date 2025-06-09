import argparse
import torch
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=100)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        from working_ant_push_env import WorkingAntPushEnv
        from working_moe_agent import WorkingMoEAgent, WorkingLanguageProcessor
        
        print("🧪 Simple MoE Test")
        
        # 환경 생성 (한 번만)
        env = WorkingAntPushEnv(num_envs=1)
        state, _ = env.reset()
        
        # 에이전트 생성
        moe_agent = WorkingMoEAgent(len(state), env.get_action_dim())
        lang_processor = WorkingLanguageProcessor()
        
        # 다양한 지시사항
        instructions = [
            "Push the cube forward",
            "Move the cube to target", 
            "Guide the cube carefully"
        ]
        
        print(f"🎯 Running {args.episodes} episodes with {len(instructions)} instructions")
        
        for episode in range(args.episodes):
            # 랜덤하게 지시사항 선택
            instruction = instructions[episode % len(instructions)]
            instruction_emb = lang_processor.encode(instruction)
            
            print(f"\nEpisode {episode + 1}: '{instruction}'")
            
            state, _ = env.reset()
            episode_reward = 0
            
            for step in range(args.max_steps):
                action, expert_id = moe_agent.select_action(state, instruction_emb)
                next_state, reward, done, _ = env.step(action)
                
                # 학습
                moe_agent.update_expert(expert_id, state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            print(f"  Reward: {episode_reward:.2f}, Expert: {expert_id}")
        
        # 최종 통계
        stats = moe_agent.get_stats()
        print(f"\n📊 Final Results:")
        print(f"  Experts created: {stats['num_experts']}")
        print(f"  Clusters found: {stats['num_clusters']}")
        print(f"  Expert usage: {stats['expert_usage']}")
        
        env.close()
        print("✅ Test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        sim.close()

if __name__ == "__main__":
    main()