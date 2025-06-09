import os
import time
import numpy as np
import torch
from typing import Dict, List, Optional
import logging
from collections import defaultdict, deque

from .utils import moving_average, plot_training_curves, plot_expert_evolution, save_results
from .evaluate import evaluate_agent

def train_agent(agent, env, eval_env, config: Dict, exp_dir: str, 
                logger: logging.Logger, use_wandb: bool = False, 
                eval_episodes: int = 10):
    """Train ATLAS agent online"""
    
    # Training statistics
    training_stats = defaultdict(list)
    expert_evolution = defaultdict(list)
    evaluation_results = []
    
    # Training loop variables
    total_steps = config['total_steps']
    eval_freq = config.get('eval_freq', 5000)
    save_freq = config.get('save_freq', 10000)
    
    # Episode tracking
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    
    # Start training
    logger.info(f"🚀 Starting training for {total_steps} steps")
    start_time = time.time()
    
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    
    for step in range(total_steps):
        # Select action
        action, action_info = agent.select_action(state, deterministic=False)
        
        # Environment step
        next_state, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1
        
        # Store transition
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done or truncated
        }
        
        agent.store_transition(transition)
        
        # Update agent
        if step > config.get('learning_starts', 1000):
            if step % config.get('train_freq', 1) == 0:
                update_info = agent.update()
                
                # Log training stats
                for key, value in update_info.items():
                    if isinstance(value, (int, float)):
                        training_stats[key].append(value)
        
        # Episode end
        if done or truncated:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_count += 1
            
            # Log episode stats
            training_stats['episode_reward'].append(episode_reward)
            training_stats['episode_length'].append(episode_length)
            
            if episode_count % 10 == 0:
                avg_reward = np.mean(list(episode_rewards))
                avg_length = np.mean(list(episode_lengths))
                logger.info(f"Episode {episode_count}, Step {step}: "
                           f"Reward {episode_reward:.2f} (Avg: {avg_reward:.2f}), "
                           f"Length {episode_length} (Avg: {avg_length:.1f})")
            
            # Reset for next episode
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            episode_reward = 0
            episode_length = 0
        else:
            state = next_state
        
        # Expert evolution tracking
        if step % 100 == 0:
            expert_info = agent.get_expert_info()
            expert_evolution['num_experts'].append(expert_info['num_experts'])
            expert_evolution['num_clusters'].append(expert_info['cluster_info']['num_clusters'])
            expert_evolution['total_points'].append(expert_info['cluster_info']['total_points'])
        
        # Evaluation
        if step % eval_freq == 0 and step > 0:
            logger.info(f"🎯 Evaluating at step {step}...")
            eval_results = evaluate_agent(agent, eval_env, eval_episodes)
            evaluation_results.append({
                'step': step,
                'results': eval_results
            })
            
            logger.info(f"Evaluation results: {eval_results}")
            
            # Log to W&B
            if use_wandb:
                import wandb
                wandb.log({
                    'step': step,
                    'eval/mean_reward': eval_results['mean_reward'],
                    'eval/std_reward': eval_results['std_reward'],
                    'eval/mean_length': eval_results['mean_length'],
                    'eval/num_experts': eval_results['num_experts'],
                    'eval/num_clusters': eval_results['num_clusters']
                })
        
        # Save checkpoint
        if step % save_freq == 0 and step > 0:
            checkpoint_path = os.path.join(exp_dir, 'checkpoints', f'checkpoint_step_{step}.pt')
            agent.save(checkpoint_path)
            logger.info(f"💾 Checkpoint saved at step {step}")
        
        # Log training progress
        if step % 1000 == 0 and step > 0:
            elapsed_time = time.time() - start_time
            steps_per_sec = step / elapsed_time
            eta = (total_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
            
            logger.info(f"Step {step}/{total_steps} ({step/total_steps*100:.1f}%) | "
                       f"SPS: {steps_per_sec:.1f} | ETA: {eta/3600:.1f}h")
            
            # Log to W&B
            if use_wandb:
                import wandb
                recent_stats = {}
                for key, values in training_stats.items():
                    if values and len(values) >= 10:
                        recent_stats[f'train/{key}'] = np.mean(values[-10:])
                
                recent_stats.update({
                    'step': step,
                    'steps_per_second': steps_per_sec,
                    'episode_count': episode_count,
                    'num_experts': expert_evolution['num_experts'][-1] if expert_evolution['num_experts'] else 0,
                    'num_clusters': expert_evolution['num_clusters'][-1] if expert_evolution['num_clusters'] else 0
                })
                
                wandb.log(recent_stats)
    
    # Training completed
    total_time = time.time() - start_time
    logger.info(f"✅ Training completed in {total_time/3600:.2f} hours")
    
    # Save final results
    results = {
        'training_stats': dict(training_stats),
        'expert_evolution': dict(expert_evolution),
        'evaluation_results': evaluation_results,
        'config': config,
        'total_time': total_time,
        'total_episodes': episode_count
    }
    
    save_results(results, os.path.join(exp_dir, 'training_results.pkl'))
    
    # Generate plots
    plot_training_curves(training_stats, os.path.join(exp_dir, 'plots', 'training_curves.png'))
    plot_expert_evolution(expert_evolution, os.path.join(exp_dir, 'plots', 'expert_evolution.png'))
    
    logger.info("📊 Training plots saved")
    
    return results


def train_multitask_sequential(agent, envs: List, config: Dict, exp_dir: str, 
                              logger: logging.Logger, use_wandb: bool = False):
    """Train agent on multiple tasks sequentially"""
    
    all_results = {}
    
    for task_id, env in enumerate(envs):
        logger.info(f"🎯 Training on Task {task_id + 1}/{len(envs)}")
        
        # Create task-specific directories
        task_dir = os.path.join(exp_dir, f'task_{task_id}')
        os.makedirs(task_dir, exist_ok=True)
        os.makedirs(os.path.join(task_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(task_dir, 'plots'), exist_ok=True)
        
        # Train on current task
        task_config = config.copy()
        task_config['total_steps'] = config['total_steps'] // len(envs)
        
        task_results = train_agent(
            agent=agent,
            env=env,
            eval_env=env,  # Use same env for eval in sequential setting
            config=task_config,
            exp_dir=task_dir,
            logger=logger,
            use_wandb=use_wandb
        )
        
        all_results[f'task_{task_id}'] = task_results
        
        # Save task-specific checkpoint
        task_checkpoint = os.path.join(task_dir, 'checkpoints', f'task_{task_id}_final.pt')
        agent.save(task_checkpoint)
        
        logger.info(f"✅ Task {task_id + 1} completed")
    
    # Save combined results
    save_results(all_results, os.path.join(exp_dir, 'multitask_results.pkl'))
    
    return all_results


def train_offline_batch(agent, dataset: Dict, config: Dict, exp_dir: str, 
                       logger: logging.Logger, use_wandb: bool = False):
    """Train agent on offline dataset"""
    
    training_stats = defaultdict(list)
    num_epochs = config.get('offline_epochs', 100)
    batch_size = config['batch_size']
    
    # Prepare dataset
    states = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    next_states = dataset['next_observations']
    dones = dataset['terminals']
    
    dataset_size = len(states)
    num_batches = dataset_size // batch_size
    
    logger.info(f"🎓 Starting offline training for {num_epochs} epochs")
    logger.info(f"📊 Dataset size: {dataset_size}, Batches per epoch: {num_batches}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Shuffle dataset
        indices = np.random.permutation(dataset_size)
        
        epoch_losses = defaultdict(list)
        
        for batch_idx in range(num_batches):
            # Get batch
            batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            
            batch = {
                'states': states[batch_indices],
                'actions': actions[batch_indices],
                'rewards': rewards[batch_indices],
                'next_states': next_states[batch_indices],
                'dones': dones[batch_indices]
            }
            
            # Update agent
            update_info = agent.update_offline(batch)
            
            # Collect losses
            for key, value in update_info.items():
                if isinstance(value, (int, float)):
                    epoch_losses[key].append(value)
        
        # Log epoch statistics
        epoch_stats = {}
        for key, values in epoch_losses.items():
            if values:
                epoch_stats[key] = np.mean(values)
                training_stats[key].append(epoch_stats[key])
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{num_epochs}: {epoch_stats}")
        
        # Log to W&B
        if use_wandb and epoch % 5 == 0:
            import wandb
            wandb_log = {'epoch': epoch}
            wandb_log.update({f'offline/{k}': v for k, v in epoch_stats.items()})
            wandb.log(wandb_log)
    
    total_time = time.time() - start_time
    logger.info(f"✅ Offline training completed in {total_time/60:.2f} minutes")
    
    # Save results
    results = {
        'training_stats': dict(training_stats),
        'config': config,
        'total_time': total_time,
        'num_epochs': num_epochs
    }
    
    save_results(results, os.path.join(exp_dir, 'offline_training_results.pkl'))
    
    return results