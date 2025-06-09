import argparse
import torch
import yaml
import numpy as np
import time
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Working Ant MoE Training")
    parser.add_argument("--tasks", type=str, default="working_tasks.yaml")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=100)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        from working_ant_push_env import WorkingAntPushEnv
        from working_moe_agent import WorkingMoEAgent, WorkingLanguageProcessor
        
        print("🚀 Starting Working Ant MoE Training")
        
        # 언어 프로세서 초기화
        lang_processor = WorkingLanguageProcessor()
        
        # 태스크 정의 (YAML 파일이 없어도 작동)
        tasks = [
            {
                'name': 'basic_push',
                'instruction': 'Push the cube to the target location',
                'env_cfg': {
                    'cube_position': [2.0, 0.0, 0.1],
                    'goal_position': [3.0, 0.0, 0.1]
                },
                'episodes': args.episodes
            },
            {
                'name': 'long_push',
                'instruction': 'Move the cube far to the target location',
                'env_cfg': {
                    'cube_position': [1.0, 0.0, 0.1],
                    'goal_position': [4.0, 0.0, 0.1]
                },
                'episodes': args.episodes
            }
        ]
        
        # MoE 에이전트 초기화 (환경 생성 후)
        moe_agent = None
        
        # 각 태스크 실행
        for task_idx, task in enumerate(tasks):
            print(f"\n{'='*60}")
            print(f"📋 Task {task_idx+1}/{len(tasks)}: {task['name']}")
            print(f"💬 Instruction: {task['instruction']}")
            print(f"{'='*60}")
            
            # 환경 생성
            print("🏗️ Creating environment...")
            env = WorkingAntPushEnv(custom_cfg=task['env_cfg'], num_envs=args.num_envs)
            
            # 첫 번째 태스크에서 MoE 에이전트 초기화
            if moe_agent is None:
                state, _ = env.reset()
                state_dim = len(state)
                action_dim = env.get_action_dim()
                
                moe_agent = WorkingMoEAgent(
                    state_dim=state_dim,
                    action_dim=action_dim
                )
            
            # 지시사항 임베딩
            instruction_emb = lang_processor.encode(task['instruction'])
            
            # 에피소드 실행
            task_start_time = time.time()
            total_steps = 0
            total_reward = 0
            
            for episode in range(task['episodes']):
                print(f"\n--- Episode {episode + 1}/{task['episodes']} ---")
                
                state, _ = env.reset()
                episode_reward = 0
                episode_steps = 0
                episode_losses = []
                used_experts = set()
                
                for step in range(args.max_steps):
                    # 행동 선택
                    action, expert_id = moe_agent.select_action(
                        state, instruction_emb, deterministic=False
                    )
                    used_experts.add(expert_id)
                    
                    # 환경 스텝
                    next_state, reward, done, info = env.step(action)
                    
                    # 경험 저장 및 학습
                    loss_info = moe_agent.update_expert(
                        expert_id, state, action, reward, next_state, done
                    )
                    if loss_info:
                        episode_losses.append(loss_info)
                    
                    episode_reward += reward
                    episode_steps += 1
                    state = next_state
                    
                    if done:
                        break
                
                total_steps += episode_steps
                total_reward += episode_reward
                
                # 에피소드 결과 출력
                avg_loss = np.mean([l['total_loss'] for l in episode_losses]) if episode_losses else 0.0
                print(f"✓ Episode {episode + 1} completed:")
                print(f"    Steps: {episode_steps}, Time: {time.time() - task_start_time:.1f}s")
                print(f"    Reward: {episode_reward:.3f}, Avg Loss: {avg_loss:.6f}")
                print(f"    Experts used: {len(used_experts)}")
            
            # 태스크 완료 통계
            task_time = time.time() - task_start_time
            avg_reward = total_reward / task['episodes']
            stats = moe_agent.get_stats()
            
            print(f"\n🎯 Task '{task['name']}' completed!")
            print(f"  Time: {task_time:.1f}s")
            print(f"  Episodes: {task['episodes']}")
            print(f"  Total steps: {total_steps}")
            print(f"  Average reward: {avg_reward:.3f}")
            print(f"  Total clusters: {stats['num_clusters']}")
            print(f"  Expert usage: {stats['expert_usage']}")
            
            # 환경 정리
            print("🔄 Closing environment safely...")
            env.close()
            print("✓ Environment closed")
            
            # 메모리 정리
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print("✓ Memory cleaned up")
            
            # 시스템 안정화 대기
            if task_idx < len(tasks) - 1:
                print("⏳ Waiting for system stabilization...")
                time.sleep(2)
        
        # 최종 통계
        final_stats = moe_agent.get_stats()
        print(f"\n🏆 Training Completed!")
        print(f"  Total experts created: {final_stats['num_experts']}")
        print(f"  Total clusters discovered: {final_stats['num_clusters']}")
        print(f"  Final expert usage: {final_stats['expert_usage']}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🔄 Shutting down simulation...")
        try:
            sim.close()
            print("✅ Simulation closed successfully")
        except:
            print("⚠️ Simulation close had issues, but continuing...")

if __name__ == "__main__":
    main()