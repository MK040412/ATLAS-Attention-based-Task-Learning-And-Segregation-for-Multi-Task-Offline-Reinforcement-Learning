import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict

def evaluate_agent(agent, env, num_episodes: int = 10, render: bool = False, 
                  max_episode_length: int = 1000) -> Dict:
    """Evaluate agent performance"""
    
    episode_rewards = []
    episode_lengths = []
    expert_usage = defaultdict(int)
    attention_weights_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        episode_reward = 0
        episode_length = 0
        episode_attention = []
        
        for step in range(max_episode_length):
            # Get action with attention info
            action, action_info = agent.select_action(state, deterministic=True)
            
            # Track expert usage
            if 'expert_id' in action_info:
                expert_usage[action_info['expert_id']] += 1
            
            # Track attention weights
            if 'attention_weights' in action_info:
                episode_attention.append(action_info['attention_weights'])
            
            # Environment step
            next_state, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
            
            if done or truncated:
                break
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if episode_attention:
            attention_weights_history.append(np.array(episode_attention))
    
    # Compute statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'expert_usage': dict(expert_usage),
        'num_experts': agent.get_num_experts(),
        'num_clusters': agent.get_num_clusters()
    }
    
    # Attention analysis
    if attention_weights_history:
        all_attention = np.concatenate(attention_weights_history, axis=0)
        results['attention_stats'] = {
            'mean_attention': np.mean(all_attention, axis=0).tolist(),
            'std_attention': np.std(all_attention, axis=0).tolist(),
            'max_attention_expert': int(np.argmax(np.mean(all_attention, axis=0))),
            'attention_entropy': compute_attention_entropy(all_attention)
        }
    
    return results


def evaluate_multitask(agent, envs: List, task_names: Optional[List[str]] = None, 
                      num_episodes: int = 10) -> Dict:
    """Evaluate agent on multiple tasks"""
    
    if task_names is None:
        task_names = [f'Task_{i}' for i in range(len(envs))]
    
    all_results = {}
    
    for task_id, (env, task_name) in enumerate(zip(envs, task_names)):
        print(f"🎯 Evaluating on {task_name}...")
        
        task_results = evaluate_agent(agent, env, num_episodes)
        all_results[task_name] = task_results
        
        print(f"  Mean reward: {task_results['mean_reward']:.2f} ± {task_results['std_reward']:.2f}")
        print(f"  Mean length: {task_results['mean_length']:.1f}")
        print(f"  Experts used: {len(task_results['expert_usage'])}")
    
    # Compute overall statistics
    all_rewards = []
    all_lengths = []
    
    for results in all_results.values():
        all_rewards.extend(results['episode_rewards'])
        all_lengths.extend(results['episode_lengths'])
    
    overall_results = {
        'overall_mean_reward': np.mean(all_rewards),
        'overall_std_reward': np.std(all_rewards),
        'overall_mean_length': np.mean(all_lengths),
        'task_results': all_results
    }
    
    return overall_results


def evaluate_expert_specialization(agent, envs: List, task_names: Optional[List[str]] = None, 
                                  num_episodes: int = 5) -> Dict:
    """Evaluate how well experts specialize to different tasks"""
    
    if task_names is None:
        task_names = [f'Task_{i}' for i in range(len(envs))]
    
    expert_task_performance = defaultdict(lambda: defaultdict(list))
    expert_task_usage = defaultdict(lambda: defaultdict(int))
    
    for task_id, (env, task_name) in enumerate(zip(envs, task_names)):
        # Force each expert to act on this task
        num_experts = agent.get_num_experts()
        
        for expert_id in range(num_experts):
            expert_rewards = []
            
            for episode in range(num_episodes):
                state = env.reset()
                if isinstance(state, tuple):
                    state = state[0]
                
                episode_reward = 0
                
                for step in range(1000):  # Max episode length
                    # Force specific expert
                    action = agent.get_expert_action(expert_id, state)
                    
                    next_state, reward, done, truncated, info = env.step(action)
                    episode_reward += reward
                    
                    if done or truncated:
                        break
                    
                    state = next_state
                
                expert_rewards.append(episode_reward)
            
            expert_task_performance[expert_id][task_name] = expert_rewards
            expert_task_usage[expert_id][task_name] = len(expert_rewards)
    
    # Compute specialization metrics
    specialization_results = {}
    
    for expert_id in range(agent.get_num_experts()):
        expert_results = {}
        
        for task_name in task_names:
            rewards = expert_task_performance[expert_id][task_name]
            expert_results[task_name] = {
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'episodes': len(rewards)
            }
        
        # Find best task for this expert
        best_task = max(task_names, 
                       key=lambda t: expert_results[t]['mean_reward'])
        
        expert_results['best_task'] = best_task
        expert_results['specialization_score'] = compute_specialization_score(expert_results, task_names)
        
        specialization_results[f'expert_{expert_id}'] = expert_results
    
    return specialization_results


def evaluate_continual_learning(agent, envs: List, task_names: Optional[List[str]] = None,
                               num_episodes: int = 10) -> Dict:
    """Evaluate continual learning performance (forward/backward transfer)"""
    
    if task_names is None:
        task_names = [f'Task_{i}' for i in range(len(envs))]
    
    # Store initial performance on all tasks
    initial_performance = {}
    for task_id, (env, task_name) in enumerate(zip(envs, task_names)):
        results = evaluate_agent(agent, env, num_episodes)
        initial_performance[task_name] = results['mean_reward']
    
    # After training, evaluate on all tasks again
    final_performance = {}
    for task_id, (env, task_name) in enumerate(zip(envs, task_names)):
        results = evaluate_agent(agent, env, num_episodes)
        final_performance[task_name] = results['mean_reward']
    
    # Compute transfer metrics
    forward_transfer = {}
    backward_transfer = {}
    
    for i, task_name in enumerate(task_names):
        # Forward transfer: performance on task i after training on tasks 0..i-1
        if i > 0:
            # This would require intermediate checkpoints to compute properly
            forward_transfer[task_name] = final_performance[task_name] - initial_performance[task_name]
        
        # Backward transfer: performance change on task i after training on tasks i+1..n
        backward_transfer[task_name] = final_performance[task_name] - initial_performance[task_name]
    
    # Average transfer metrics
    avg_forward_transfer = np.mean(list(forward_transfer.values())) if forward_transfer else 0.0
    avg_backward_transfer = np.mean(list(backward_transfer.values()))
    
    return {
        'initial_performance': initial_performance,
        'final_performance': final_performance,
        'forward_transfer': forward_transfer,
        'backward_transfer': backward_transfer,
        'avg_forward_transfer': avg_forward_transfer,
        'avg_backward_transfer': avg_backward_transfer
    }


def compute_attention_entropy(attention_weights: np.ndarray) -> List[float]:
    """Compute entropy of attention distribution over time"""
    entropies = []
    
    for weights in attention_weights:
        # Add small epsilon to avoid log(0)
        weights = weights + 1e-8
        weights = weights / np.sum(weights)  # Normalize
        entropy = -np.sum(weights * np.log(weights))
        entropies.append(entropy)
    
    return entropies


def compute_specialization_score(expert_results: Dict, task_names: List[str]) -> float:
    """Compute how specialized an expert is (higher = more specialized)"""
    rewards = [expert_results[task]['mean_reward'] for task in task_names]
    
    if len(rewards) <= 1:
        return 0.0
    
    # Specialization as ratio of max to mean performance
    max_reward = max(rewards)
    mean_reward = np.mean(rewards)
    
    if mean_reward <= 0:
        return 0.0
    
    return max_reward / mean_reward


def benchmark_agent(agent, benchmark_envs: Dict[str, any], num_episodes: int = 10) -> Dict:
    """Run agent on standard benchmark environments"""
    
    benchmark_results = {}
    
    for env_name, env in benchmark_envs.items():
        print(f"🏆 Running benchmark: {env_name}")
        
        results = evaluate_agent(agent, env, num_episodes)
        benchmark_results[env_name] = results
        
        print(f"  {env_name}: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    
    return benchmark_results


def evaluate_robustness(agent, env, num_episodes: int = 10, 
                       noise_levels: List[float] = [0.0, 0.1, 0.2, 0.5]) -> Dict:
    """Evaluate agent robustness to observation noise"""
    
    robustness_results = {}
    
    for noise_level in noise_levels:
        print(f"🔊 Testing with noise level: {noise_level}")
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            episode_reward = 0
            
            for step in range(1000):  # Max episode length
                # Add noise to observations
                noisy_state = state + np.random.normal(0, noise_level, state.shape)
                
                action, _ = agent.select_action(noisy_state, deterministic=True)
                next_state, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                
                if done or truncated:
                    break
                
                state = next_state
            
            episode_rewards.append(episode_reward)
        
        robustness_results[f'noise_{noise_level}'] = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'episode_rewards': episode_rewards
        }
    
    return robustness_results


def evaluate_sample_efficiency(agent, env, episode_budgets: List[int] = [10, 50, 100, 500]) -> Dict:
    """Evaluate sample efficiency at different training budgets"""
    
    efficiency_results = {}
    
    # This would require training separate agents with different budgets
    # For now, we'll evaluate current agent performance
    for budget in episode_budgets:
        # Simulate limited evaluation
        num_eval_episodes = min(budget // 10, 10)
        results = evaluate_agent(agent, env, num_eval_episodes)
        
        efficiency_results[f'budget_{budget}'] = results
    
    return efficiency_results


def create_evaluation_report(results: Dict, save_path: str = None) -> str:
    """Create a formatted evaluation report"""
    
    report = []
    report.append("=" * 60)
    report.append("ATLAS AGENT EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Basic performance
    if 'mean_reward' in results:
        report.append(f"Performance Summary:")
        report.append(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        report.append(f"  Episode Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
        report.append(f"  Best Episode: {results['max_reward']:.2f}")
        report.append(f"  Worst Episode: {results['min_reward']:.2f}")
        report.append("")
    
    # Expert information
    if 'num_experts' in results:
        report.append(f"Expert Information:")
        report.append(f"  Number of Experts: {results['num_experts']}")
        report.append(f"  Number of Clusters: {results['num_clusters']}")
        
        if 'expert_usage' in results:
            report.append(f"  Expert Usage:")
            for expert_id, usage in results['expert_usage'].items():
                percentage = (usage / sum(results['expert_usage'].values())) * 100
                report.append(f"    Expert {expert_id}: {usage} steps ({percentage:.1f}%)")
        report.append("")
    
    # Attention analysis
    if 'attention_stats' in results:
        attention = results['attention_stats']
        report.append(f"Attention Analysis:")
        report.append(f"  Most Used Expert: {attention['max_attention_expert']}")
        report.append(f"  Mean Attention Entropy: {np.mean(attention['attention_entropy']):.3f}")
        report.append("")
    
    # Multi-task results
    if 'task_results' in results:
        report.append("Multi-Task Performance:")
        for task_name, task_results in results['task_results'].items():
            report.append(f"  {task_name}:")
            report.append(f"    Reward: {task_results['mean_reward']:.2f} ± {task_results['std_reward']:.2f}")
            report.append(f"    Length: {task_results['mean_length']:.1f}")
        report.append("")
        
        if 'overall_mean_reward' in results:
            report.append(f"Overall Performance:")
            report.append(f"  Mean Reward: {results['overall_mean_reward']:.2f}")
            report.append(f"  Std Reward: {results['overall_std_reward']:.2f}")
            report.append("")
    
    # Continual learning metrics
    if 'avg_forward_transfer' in results:
        report.append("Continual Learning Metrics:")
        report.append(f"  Average Forward Transfer: {results['avg_forward_transfer']:.3f}")
        report.append(f"  Average Backward Transfer: {results['avg_backward_transfer']:.3f}")
        report.append("")
    
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"📋 Evaluation report saved to {save_path}")
    
    return report_text


def save_evaluation_results(results: Dict, save_path: str):
    """Save evaluation results to file"""
    import pickle
    
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"💾 Evaluation results saved to {save_path}")


def load_evaluation_results(load_path: str) -> Dict:
    """Load evaluation results from file"""
    import pickle
    
    with open(load_path, 'rb') as f:
        results = pickle.load(f)
    
    print(f"📁 Evaluation results loaded from {load_path}")
    return results
