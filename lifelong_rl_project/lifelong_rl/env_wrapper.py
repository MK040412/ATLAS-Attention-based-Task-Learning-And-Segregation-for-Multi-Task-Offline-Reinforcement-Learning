import gymnasium as gym
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from gymnasium.spaces import Box
from collections import deque


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations using running statistics"""
    
    def __init__(self, env, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self.running_mean = np.zeros(env.observation_space.shape)
        self.running_var = np.ones(env.observation_space.shape)
        self.count = 0
    
    def observation(self, obs):
        self.count += 1
        delta = obs - self.running_mean
        self.running_mean += delta / self.count
        delta2 = obs - self.running_mean
        self.running_var += (delta * delta2 - self.running_var) / self.count
        
        return (obs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)


class NormalizeReward(gym.RewardWrapper):
    """Normalize rewards using running statistics"""
    
    def __init__(self, env, epsilon: float = 1e-8, gamma: float = 0.99):
        super().__init__(env)
        self.epsilon = epsilon
        self.gamma = gamma
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0
        self.returns = 0.0
    
    def reward(self, reward):
        self.returns = self.returns * self.gamma + reward
        self.count += 1
        delta = self.returns - self.running_mean
        self.running_mean += delta / self.count
        delta2 = self.returns - self.running_mean
        self.running_var += (delta * delta2 - self.running_var) / self.count
        
        return reward / np.sqrt(self.running_var + self.epsilon)


class FrameStack(gym.ObservationWrapper):
    """Stack multiple frames for temporal information"""
    
    def __init__(self, env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        
        low = np.repeat(env.observation_space.low, num_stack, axis=-1)
        high = np.repeat(env.observation_space.high, num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=env.observation_space.dtype)
    
    def observation(self, obs):
        self.frames.append(obs)
        return np.concatenate(list(self.frames), axis=-1)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self.observation(obs), info


class ActionScale(gym.ActionWrapper):
    """Scale actions from [-1, 1] to action space bounds"""
    
    def __init__(self, env):
        super().__init__(env)
        self.low = env.action_space.low
        self.high = env.action_space.high
        self.action_space = Box(low=-1.0, high=1.0, shape=env.action_space.shape)
    
    def action(self, action):
        # Scale from [-1, 1] to [low, high]
        scaled_action = self.low + (action + 1.0) * 0.5 * (self.high - self.low)
        return np.clip(scaled_action, self.low, self.high)


class TimeLimit(gym.Wrapper):
    """Add time limit to environment"""
    
    def __init__(self, env, max_episode_steps: int):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.current_step += 1
        
        if self.current_step >= self.max_episode_steps:
            truncated = True
            info['TimeLimit.truncated'] = True
        
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        self.current_step = 0
        return self.env.reset(**kwargs)


class AddTaskID(gym.ObservationWrapper):
    """Add task ID to observation"""
    
    def __init__(self, env, task_id: int, num_tasks: int):
        super().__init__(env)
        self.task_id = task_id
        self.num_tasks = num_tasks
        
        # Create one-hot encoding for task ID
        self.task_encoding = np.zeros(num_tasks)
        self.task_encoding[task_id] = 1.0
        
        # Extend observation space
        orig_shape = env.observation_space.shape[0]
        new_shape = orig_shape + num_tasks
        
        self.observation_space = Box(
            low=np.concatenate([env.observation_space.low, np.zeros(num_tasks)]),
            high=np.concatenate([env.observation_space.high, np.ones(num_tasks)]),
            dtype=env.observation_space.dtype
        )
    
    def observation(self, obs):
        return np.concatenate([obs, self.task_encoding])


class AddGaussianNoise(gym.ObservationWrapper):
    """Add Gaussian noise to observations for robustness testing"""
    
    def __init__(self, env, noise_std: float = 0.1):
        super().__init__(env)
        self.noise_std = noise_std
    
    def observation(self, obs):
        noise = np.random.normal(0, self.noise_std, obs.shape)
        return obs + noise


class ClipActions(gym.ActionWrapper):
    """Clip actions to action space bounds"""
    
    def __init__(self, env):
        super().__init__(env)
    
    def action(self, action):
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)


class RecordEpisodeStatistics(gym.Wrapper):
    """Record episode statistics"""
    
    def __init__(self, env):
        super().__init__(env)
        self.episode_returns = []
        self.episode_lengths = []
        self.current_return = 0.0
        self.current_length = 0
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.current_return += reward
        self.current_length += 1
        
        if done or truncated:
            info['episode_return'] = self.current_return
            info['episode_length'] = self.current_length
            self.episode_returns.append(self.current_return)
            self.episode_lengths.append(self.current_length)
            self.current_return = 0.0
            self.current_length = 0
        
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        self.current_return = 0.0
        self.current_length = 0
        return self.env.reset(**kwargs)


class MultiTaskWrapper(gym.Wrapper):
    """Wrapper for multi-task environments"""
    
    def __init__(self, envs: List[gym.Env], task_schedule: str = 'sequential'):
        # Use first env as base
        super().__init__(envs[0])
        self.envs = envs
        self.num_tasks = len(envs)
        self.task_schedule = task_schedule
        self.current_task = 0
        self.episode_count = 0
    
    def step(self, action):
        return self.envs[self.current_task].step(action)
    
    def reset(self, **kwargs):
        self.episode_count += 1
        
        # Update task based on schedule
        if self.task_schedule == 'sequential':
            # Switch task every N episodes
            episodes_per_task = kwargs.get('episodes_per_task', 100)
            self.current_task = (self.episode_count // episodes_per_task) % self.num_tasks
        elif self.task_schedule == 'random':
            self.current_task = np.random.randint(0, self.num_tasks)
        elif self.task_schedule == 'round_robin':
            self.current_task = self.episode_count % self.num_tasks
        
        return self.envs[self.current_task].reset(**kwargs)
    
    def render(self, **kwargs):
        return self.envs[self.current_task].render(**kwargs)
    
    def close(self):
        for env in self.envs:
            env.close()


def make_env(env_name: str, seed: int = None, task_id: Optional[int] = None, 
             num_tasks: Optional[int] = None, **kwargs) -> gym.Env:
    """Create and wrap environment"""
    
    # Create base environment
    env = gym.make(env_name, **kwargs)
    
    # Set seed
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    
    # Add task ID if specified
    if task_id is not None and num_tasks is not None:
        env = AddTaskID(env, task_id, num_tasks)
    
    # Standard wrappers
    env = ClipActions(env)
    env = ActionScale(env)
    env = RecordEpisodeStatistics(env)
    
    # Optional wrappers based on environment type
    if 'Ant' in env_name or 'Humanoid' in env_name:
        # For locomotion tasks, add time limit
        env = TimeLimit(env, max_episode_steps=1000)
    
    return env


def make_multitask_env(env_names: List[str], seed: int = None, 
                      task_schedule: str = 'sequential', **kwargs) -> gym.Env:
    """Create multi-task environment"""
    
    envs = []
    for i, env_name in enumerate(env_names):
        env = make_env(
            env_name=env_name,
            seed=seed + i if seed is not None else None,
            task_id=i,
            num_tasks=len(env_names),
            **kwargs
        )
        envs.append(env)
    
    return MultiTaskWrapper(envs, task_schedule=task_schedule)


def create_modified_env(base_env_name: str, modifications: Dict[str, Any], 
                       seed: int = None) -> gym.Env:
    """Create environment with modifications (e.g., different dynamics)"""
    
    env = gym.make(base_env_name)
    
    # Apply modifications
    for param, value in modifications.items():
        if hasattr(env.unwrapped, param):
            setattr(env.unwrapped, param, value)
        elif hasattr(env.unwrapped.model, param):  # For MuJoCo envs
            setattr(env.unwrapped.model, param, value)
    
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    
    return env


def create_task_variants(base_env_name: str, num_variants: int = 3, 
                        seed: int = None) -> List[gym.Env]:
    """Create multiple task variants of the same environment"""
    
    envs = []
    
    if 'Ant' in base_env_name:
        # Ant variants with different target velocities or terrains
        modifications_list = [
            {},  # Original
            {'target_velocity': 2.0},  # Faster
            {'target_velocity': 0.5},  # Slower
        ]
    elif 'HalfCheetah' in base_env_name:
        # HalfCheetah variants
        modifications_list = [
            {},  # Original
            {'forward_reward_weight': 2.0},  # More forward reward
            {'ctrl_cost_weight': 2.0},  # More control cost
        ]
    else:
        # Default: just create copies
        modifications_list = [{} for _ in range(num_variants)]
    
    for i, modifications in enumerate(modifications_list[:num_variants]):
        env = create_modified_env(
            base_env_name, 
            modifications, 
            seed=seed + i if seed is not None else None
        )
        envs.append(env)
    
    return envs


class EpisodeMonitor(gym.Wrapper):
    """Monitor and log episode statistics"""
    
    def __init__(self, env, log_file: str = None):
        super().__init__(env)
        self.log_file = log_file
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_reward = 0.0
        self.current_length = 0
        
        if log_file:
            with open(log_file, 'w') as f:
                f.write("episode,reward,length\n")
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.current_reward += reward
        self.current_length += 1
        
        if done or truncated:
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(f"{len(self.episode_rewards)},{self.current_reward},{self.current_length}\n")
            
            self.current_reward = 0.0
            self.current_length = 0
        
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        self.current_reward = 0.0
        self.current_length = 0
        return self.env.reset(**kwargs)


def get_env_info(env_name: str) -> Dict[str, Any]:
    """Get information about environment"""
    
    env = gym.make(env_name)
    
    info = {
        'env_name': env_name,
        'observation_space': env.observation_space,
        'action_space': env.action_space,
        'observation_dim': env.observation_space.shape[0],
        'action_dim': env.action_space.shape[0],
        'max_episode_steps': getattr(env, '_max_episode_steps', None),
        'reward_threshold': getattr(env.spec, 'reward_threshold', None)
    }
    
    env.close()
    return info