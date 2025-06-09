import numpy as np
import pickle
import os
from typing import Dict, List, Optional, Tuple, Any
import gymnasium as gym
from .env_wrapper import make_env, create_task_variants


def load_d4rl_dataset(env_name: str, quality: str = 'medium') -> Dict[str, np.ndarray]:
    """Load D4RL dataset"""
    try:
        # Add type ignore comment since d4rl is an optional dependency
        import d4rl  # type: ignore
        
        # Construct D4RL dataset name
        d4rl_name = f"{env_name.lower()}-{quality}-v2"
        
        # Create environment and get dataset
        env = gym.make(d4rl_name)
        dataset = env.get_dataset()
        
        # Standardize format
        standardized_dataset = {
            'observations': dataset['observations'],
            'actions': dataset['actions'],
            'rewards': dataset['rewards'],
            'next_observations': dataset['next_observations'],
            'terminals': dataset['terminals'],
            'timeouts': dataset.get('timeouts', np.zeros_like(dataset['terminals']))
        }
        
        env.close()
        print(f"📚 Loaded D4RL dataset: {d4rl_name}")
        print(f"   Size: {len(dataset['observations'])} transitions")
        
        return standardized_dataset
        
    except ImportError:
        print("⚠️ D4RL not available, creating synthetic dataset")
        return create_synthetic_dataset(env_name, size=10000)

def create_synthetic_dataset(env_name: str, size: int = 10000, 
                           policy_type: str = 'random') -> Dict[str, np.ndarray]:
    """Create synthetic dataset by running policy in environment"""
    
    env = make_env(env_name)
    
    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []
    timeouts = []
    
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    
    for step in range(size):
        if policy_type == 'random':
            action = env.action_space.sample()
        elif policy_type == 'zero':
            action = np.zeros(env.action_space.shape)
        else:
            action = env.action_space.sample()
        
        next_state, reward, done, truncated, info = env.step(action)
        
        observations.append(state.copy())
        actions.append(action.copy())
        rewards.append(reward)
        next_observations.append(next_state.copy())
        terminals.append(done)
        timeouts.append(truncated)
        
        if done or truncated:
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
        else:
            state = next_state
    
    env.close()
    
    dataset = {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'next_observations': np.array(next_observations),
        'terminals': np.array(terminals),
        'timeouts': np.array(timeouts)
    }
    
    print(f"📚 Created synthetic dataset for {env_name}")
    print(f"   Size: {size} transitions")
    
    return dataset


def create_multitask_dataset(env_name: str, num_tasks: int = 3, 
                            episodes_per_task: int = 100) -> Dict[str, np.ndarray]:
    """Create multi-task dataset"""
    
    # Create task variants
    envs = create_task_variants(env_name, num_tasks)
    
    all_observations = []
    all_actions = []
    all_rewards = []
    all_next_observations = []
    all_terminals = []
    all_timeouts = []
    all_task_ids = []
    
    for task_id, env in enumerate(envs):
        print(f"📊 Collecting data for task {task_id + 1}/{num_tasks}")
        
        for episode in range(episodes_per_task):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            episode_done = False
            step_count = 0
            max_steps = 1000
            
            while not episode_done and step_count < max_steps:
                action = env.action_space.sample()  # Random policy
                next_state, reward, done, truncated, info = env.step(action)
                
                all_observations.append(state.copy())
                all_actions.append(action.copy())
                all_rewards.append(reward)
                all_next_observations.append(next_state.copy())
                all_terminals.append(done)
                all_timeouts.append(truncated)
                all_task_ids.append(task_id)
                
                if done or truncated:
                    episode_done = True
                else:
                    state = next_state
                

                step_count += 1
        
        env.close()
    
    dataset = {
        'observations': np.array(all_observations),
        'actions': np.array(all_actions),
        'rewards': np.array(all_rewards),
        'next_observations': np.array(all_next_observations),
        'terminals': np.array(all_terminals),
        'timeouts': np.array(all_timeouts),
        'task_ids': np.array(all_task_ids)
    }
    
    print(f"📚 Created multi-task dataset with {num_tasks} tasks")
    print(f"   Total size: {len(all_observations)} transitions")
    print(f"   Episodes per task: {episodes_per_task}")
    
    return dataset


def create_continual_learning_dataset(env_names: List[str], 
                                     episodes_per_env: int = 100) -> Dict[str, np.ndarray]:
    """Create dataset for continual learning with different environments"""
    
    all_observations = []
    all_actions = []
    all_rewards = []
    all_next_observations = []
    all_terminals = []
    all_timeouts = []
    all_task_ids = []
    
    for task_id, env_name in enumerate(env_names):
        print(f"📊 Collecting data for {env_name} (task {task_id + 1}/{len(env_names)})")
        
        env = make_env(env_name)
        
        for episode in range(episodes_per_env):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            episode_done = False
            step_count = 0
            max_steps = 1000
            
            while not episode_done and step_count < max_steps:
                action = env.action_space.sample()  # Random policy
                next_state, reward, done, truncated, info = env.step(action)
                
                all_observations.append(state.copy())
                all_actions.append(action.copy())
                all_rewards.append(reward)
                all_next_observations.append(next_state.copy())
                all_terminals.append(done)
                all_timeouts.append(truncated)
                all_task_ids.append(task_id)
                
                if done or truncated:
                    episode_done = True
                else:
                    state = next_state
                
                step_count += 1
        
        env.close()
    
    dataset = {
        'observations': np.array(all_observations),
        'actions': np.array(all_actions),
        'rewards': np.array(all_rewards),
        'next_observations': np.array(all_next_observations),
        'terminals': np.array(all_terminals),
        'timeouts': np.array(all_timeouts),
        'task_ids': np.array(all_task_ids)
    }
    
    print(f"📚 Created continual learning dataset with {len(env_names)} environments")
    print(f"   Total size: {len(all_observations)} transitions")
    
    return dataset


def load_custom_dataset(file_path: str) -> Dict[str, np.ndarray]:
    """Load custom dataset from file"""
    
    if file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
    elif file_path.endswith('.npz'):
        dataset = dict(np.load(file_path))
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Validate dataset format
    required_keys = ['observations', 'actions', 'rewards', 'next_observations', 'terminals']
    for key in required_keys:
        if key not in dataset:
            raise ValueError(f"Dataset missing required key: {key}")
    
    print(f"📁 Loaded custom dataset from {file_path}")
    print(f"   Size: {len(dataset['observations'])} transitions")
    
    return dataset


def save_dataset(dataset: Dict[str, np.ndarray], file_path: str):
    """Save dataset to file"""
    
    if file_path.endswith('.pkl'):
        with open(file_path, 'wb') as f:
            pickle.dump(dataset, f)
    elif file_path.endswith('.npz'):
        np.savez(file_path, **dataset)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    print(f"💾 Dataset saved to {file_path}")


def split_dataset(dataset: Dict[str, np.ndarray], 
                  split_ratio: float = 0.8) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Split dataset into train and validation sets"""
    
    total_size = len(dataset['observations'])
    train_size = int(total_size * split_ratio)
    
    # Create random indices
    indices = np.random.permutation(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = {}
    val_dataset = {}
    
    for key, data in dataset.items():
        train_dataset[key] = data[train_indices]
        val_dataset[key] = data[val_indices]
    
    print(f"📊 Dataset split: {len(train_indices)} train, {len(val_indices)} validation")
    
    return train_dataset, val_dataset


def normalize_dataset(dataset: Dict[str, np.ndarray], 
                     normalize_observations: bool = True,
                     normalize_actions: bool = True,
                     normalize_rewards: bool = True) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Normalize dataset and return normalization statistics"""
    
    normalized_dataset = dataset.copy()
    normalization_stats = {}
    
    if normalize_observations:
        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0) + 1e-8
        
        normalized_dataset['observations'] = (dataset['observations'] - obs_mean) / obs_std
        normalized_dataset['next_observations'] = (dataset['next_observations'] - obs_mean) / obs_std
        
        normalization_stats['observations'] = {'mean': obs_mean, 'std': obs_std}
    
    if normalize_actions:
        action_mean = np.mean(dataset['actions'], axis=0)
        action_std = np.std(dataset['actions'], axis=0) + 1e-8
        
        normalized_dataset['actions'] = (dataset['actions'] - action_mean) / action_std
        
        normalization_stats['actions'] = {'mean': action_mean, 'std': action_std}
    
    if normalize_rewards:
        reward_mean = np.mean(dataset['rewards'])
        reward_std = np.std(dataset['rewards']) + 1e-8
        
        normalized_dataset['rewards'] = (dataset['rewards'] - reward_mean) / reward_std
        
        normalization_stats['rewards'] = {'mean': reward_mean, 'std': reward_std}
    
    print("📏 Dataset normalized")
    
    return normalized_dataset, normalization_stats


def create_replay_buffer_dataset(dataset: Dict[str, np.ndarray], 
                                buffer_size: int = 100000) -> Dict[str, np.ndarray]:
    """Create a fixed-size dataset suitable for replay buffer"""
    
    total_size = len(dataset['observations'])
    
    if total_size <= buffer_size:
        print(f"📚 Dataset size ({total_size}) <= buffer size ({buffer_size}), using full dataset")
        return dataset
    
    # Sample random subset
    indices = np.random.choice(total_size, size=buffer_size, replace=False)
    
    buffer_dataset = {}
    for key, data in dataset.items():
        buffer_dataset[key] = data[indices]
    
    print(f"📚 Created replay buffer dataset: {buffer_size} transitions")
    
    return buffer_dataset


def get_dataset_statistics(dataset: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Get comprehensive statistics about the dataset"""
    
    stats = {
        'size': len(dataset['observations']),
        'observation_dim': dataset['observations'].shape[1],
        'action_dim': dataset['actions'].shape[1],
        'episode_count': np.sum(dataset['terminals']) + np.sum(dataset.get('timeouts', [])),
        'observation_stats': {
            'mean': np.mean(dataset['observations'], axis=0),
            'std': np.std(dataset['observations'], axis=0),
            'min': np.min(dataset['observations'], axis=0),
            'max': np.max(dataset['observations'], axis=0)
        },
        'action_stats': {
            'mean': np.mean(dataset['actions'], axis=0),
            'std': np.std(dataset['actions'], axis=0),
            'min': np.min(dataset['actions'], axis=0),
            'max': np.max(dataset['actions'], axis=0)
        },
        'reward_stats': {
            'mean': np.mean(dataset['rewards']),
            'std': np.std(dataset['rewards']),
            'min': np.min(dataset['rewards']),
            'max': np.max(dataset['rewards']),
            'total': np.sum(dataset['rewards'])
        }
    }
    
    # Task-specific statistics if available
    if 'task_ids' in dataset:
        unique_tasks = np.unique(dataset['task_ids'])
        task_stats = {}
        
        for task_id in unique_tasks:
            task_mask = dataset['task_ids'] == task_id
            task_stats[f'task_{task_id}'] = {
                'size': np.sum(task_mask),
                'episodes': np.sum(dataset['terminals'][task_mask]) + np.sum(dataset.get('timeouts', [])[task_mask]),
                'mean_reward': np.mean(dataset['rewards'][task_mask]),
                'total_reward': np.sum(dataset['rewards'][task_mask])
            }
        
        stats['task_stats'] = task_stats
        stats['num_tasks'] = len(unique_tasks)
    
    return stats


def filter_dataset_by_quality(dataset: Dict[str, np.ndarray], 
                              quality_threshold: float = 0.0) -> Dict[str, np.ndarray]:
    """Filter dataset to keep only high-quality trajectories"""
    
    # Compute returns for each trajectory
    returns = []
    current_return = 0.0
    trajectory_starts = [0]
    
    for i, (reward, terminal, timeout) in enumerate(zip(
        dataset['rewards'], 
        dataset['terminals'], 
        dataset.get('timeouts', np.zeros_like(dataset['terminals']))
    )):
        current_return += reward
        
        if terminal or timeout:
            returns.append(current_return)
            current_return = 0.0
            if i + 1 < len(dataset['rewards']):
                trajectory_starts.append(i + 1)
    
    # If last trajectory didn't end, add its return
    if current_return != 0.0:
        returns.append(current_return)
    
    # Determine quality threshold
    if quality_threshold == 0.0:
        threshold = np.percentile(returns, 50)  # Median
    else:
        threshold = quality_threshold
    
    # Keep trajectories above threshold
    keep_mask = np.zeros(len(dataset['observations']), dtype=bool)
    
    for i, (start_idx, return_val) in enumerate(zip(trajectory_starts, returns)):
        if return_val >= threshold:
            # Find end of trajectory
            end_idx = trajectory_starts[i + 1] if i + 1 < len(trajectory_starts) else len(dataset['observations'])
            keep_mask[start_idx:end_idx] = True
    
    # Filter dataset
    filtered_dataset = {}
    for key, data in dataset.items():
        filtered_dataset[key] = data[keep_mask]
    
    print(f"📊 Filtered dataset: {np.sum(keep_mask)}/{len(keep_mask)} transitions kept")
    print(f"   Quality threshold: {threshold:.2f}")
    
    return filtered_dataset


class DatasetBatcher:
    """Utility class for batching dataset samples"""
    
    def __init__(self, dataset: Dict[str, np.ndarray], batch_size: int = 64, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = len(dataset['observations'])
        self.num_batches = (self.size + batch_size - 1) // batch_size
        self.reset()
    
    def reset(self):
        """Reset iterator"""
        if self.shuffle:
            self.indices = np.random.permutation(self.size)
        else:
            self.indices = np.arange(self.size)
        self.current_idx = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_idx >= self.size:
            raise StopIteration
        
        # Get batch indices
        end_idx = min(self.current_idx + self.batch_size, self.size)
        batch_indices = self.indices[self.current_idx:end_idx]
        
        # Create batch
        batch = {}
        for key, data in self.dataset.items():
            batch[key] = data[batch_indices]
        
        self.current_idx = end_idx
        return batch
    
    def __len__(self):
        return self.num_batches


def create_demonstration_dataset(env_name: str, expert_policy, 
                                num_episodes: int = 100) -> Dict[str, np.ndarray]:
    """Create dataset from expert demonstrations"""
    
    env = make_env(env_name)
    
    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []
    timeouts = []
    
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        episode_done = False
        step_count = 0
        max_steps = 1000
        
        while not episode_done and step_count < max_steps:
            # Get expert action
            action = expert_policy(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            observations.append(state.copy())
            actions.append(action.copy())
            rewards.append(reward)
            next_observations.append(next_state.copy())
            terminals.append(done)
            timeouts.append(truncated)
            
            if done or truncated:
                episode_done = True
            else:
                state = next_state
            
            step_count += 1
    
    env.close()
    
    dataset = {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'next_observations': np.array(next_observations),
        'terminals': np.array(terminals),
        'timeouts': np.array(timeouts)
    }
    
    print(f"📚 Created demonstration dataset for {env_name}")
    print(f"   Size: {len(observations)} transitions from {num_episodes} episodes")
    
    return dataset


def augment_dataset_with_noise(dataset: Dict[str, np.ndarray], 
                              noise_level: float = 0.1,
                              augmentation_factor: int = 2) -> Dict[str, np.ndarray]:
    """Augment dataset by adding noise to observations"""
    
    original_size = len(dataset['observations'])
    augmented_size = original_size * augmentation_factor
    
    augmented_dataset = {}
    
    for key, data in dataset.items():
        if key == 'observations' or key == 'next_observations':
            # Add noise to observations
            noise = np.random.normal(0, noise_level, (augmented_size,) + data.shape[1:])
            repeated_data = np.tile(data, (augmentation_factor, 1))
            augmented_dataset[key] = repeated_data + noise
        else:
            # Just repeat other data
            augmented_dataset[key] = np.tile(data, (augmentation_factor,))
    
    print(f"📈 Dataset augmented: {original_size} -> {augmented_size} transitions")
    print(f"   Noise level: {noise_level}")
    
    return augmented_dataset