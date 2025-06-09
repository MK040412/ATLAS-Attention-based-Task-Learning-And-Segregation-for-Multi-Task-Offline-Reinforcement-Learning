import argparse
import torch
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        from working_ant_push_env import WorkingAntPushEnv
        from working_moe_agent import WorkingMoEAgent, WorkingLanguageProcessor
        
        print("🧪 Quick MoE Test Starting...")
        
        # 환경 생성
        env = WorkingAntPushEnv(num_envs=1)
        state, _ = env.reset()
        
        print(f"✓ Environment created: state_dim={len(state)}, action_dim={env.get_action_dim()}")
        
        # MoE 에이전트 생성
        moe_agent = WorkingMoEAgent(
            state_dim=len(state),
            action_dim=env.get_action_dim()
        )
        
        # 언어 프로세서
        lang_processor = WorkingLanguageProcessor()
        instruction_emb = lang_processor.encode("Push the cube to the target")
        
        print("✓ MoE Agent and Language Processor created")
        
        # 몇 스텝 테스트
        for i in range(5):
            action, expert_id = moe_agent.select_action(state, instruction_emb)
            print(f"Step {i+1}: Expert {expert_id}, Action shape: {action.shape}")
            
            next_state, reward, done, info = env.step(action)
            print(f"  Reward: {reward:.3f}, Done: {done}")
            
            # 학습 업데이트
            loss_info = moe_agent.update_expert(expert_id, state, action, reward, next_state, done)
            if loss_info:
                print(f"  Loss: {loss_info['total_loss']:.6f}")
            
            if done:
                state, _ = env.reset()
                print("  Environment reset")
            else:
                state = next_state
        
        # 통계 출력
        stats = moe_agent.get_stats()
        print(f"\n📊 Final Stats:")
        print(f"  Experts: {stats['num_experts']}")
        print(f"  Clusters: {stats['num_clusters']}")
        print(f"  Usage: {stats['expert_usage']}")
        
        env.close()
        print("✅ Quick test passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        sim.close()

if __name__ == "__main__":
    main()