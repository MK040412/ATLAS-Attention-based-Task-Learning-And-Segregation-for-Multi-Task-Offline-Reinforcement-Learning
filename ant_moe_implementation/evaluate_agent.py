import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import yaml
from ant_push_env import IsaacAntPushEnv
from config import CONFIG
from language_processor import LanguageProcessor
from dpmm_ood_detector import DPMMOODDetector
from moe_sac_agent import InstructionFusion, MoESACAgent

def evaluate_performance(tasks_file="tasks.yaml", num_episodes=5):
    """에이전트 성능 평가"""
    
    tasks = yaml.safe_load(open(tasks_file))
    results = defaultdict(list)
    
    # 기존 모델들 로드 (학습된 것이 있다면)
    # 여기서는 간단히 새로 초기화
    
    for task in tasks:
        print(f"\nEvaluating task: {task['name']}")
        
        env = IsaacAntPushEnv(custom_cfg=task['env_cfg'], num_envs=1)
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 1000:
                # 랜덤 액션 (실제로는 학습된 에이전트 사용)
                action = env.action_space.sample()[0]  # (1, 8) -> (8,)
                
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                steps += 1
                state = next_state
            
            results[task['name']].append({
                'reward': total_reward,
                'steps': steps,
                'success': total_reward > -1000  # 임의의 성공 기준
            })
            
            print(f"  Episode {episode+1}: Reward={total_reward:.2f}, Steps={steps}")
        
        env.close()
    
    # 결과 출력
    print("\n=== Evaluation Results ===")
    for task_name, episodes in results.items():
        avg_reward = np.mean([ep['reward'] for ep in episodes])
        avg_steps = np.mean([ep['steps'] for ep in episodes])
        success_rate = np.mean([ep['success'] for ep in episodes])
        
        print(f"{task_name}:")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Steps: {avg_steps:.1f}")
        print(f"  Success Rate: {success_rate:.2f}")
    
    return results

if __name__ == "__main__":
    evaluate_performance()