import numpy as np
import torch
from typing import Tuple, Dict, List
from collections import deque
import random

class ExpertReplayBuffer:
    """Replay buffer for individual experts"""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: str = 'cpu'):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Use deque for efficient FIFO operations
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
        
        self.size = 0

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch of experiences"""
        if self.size < batch_size:
            batch_size = self.size
            
        indices = random.sample(range(self.size), batch_size)
        
        batch_states = np.array([self.states[i] for i in indices])
        batch_actions = np.array([self.actions[i] for i in indices])
        batch_rewards = np.array([self.rewards[i] for i in indices])
        batch_next_states = np.array([self.next_states[i] for i in indices])
        batch_dones = np.array([self.dones[i] for i in indices])
        
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones

    def __len__(self):
        return self.size

    def clear(self):
        """Clear all data from buffer"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.size = 0


class PrioritizedReplayBuffer:
    """Prioritized replay buffer for more important experiences"""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, 
                 alpha: float = 0.6, beta: float = 0.4, device: str = 'cpu'):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.device = device
        
        # Storage
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        # Priority storage
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        
        # Buffer state
        self.size = 0
        self.position = 0

    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool, priority: float = None):
        """Add experience with priority"""
        
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        # Set priority
        if priority is None:
            priority = self.max_priority
        self.priorities[self.position] = priority ** self.alpha
        
        # Update max priority
        self.max_priority = max(self.max_priority, priority)
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch with importance sampling"""
        if self.size == 0:
            raise ValueError("Buffer is empty")
            
        # Sample indices based on priorities
        priorities = self.priorities[:self.size]
        probabilities = priorities / np.sum(priorities)
        
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= np.max(weights)  # Normalize
        
        # Get batch data
        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices]
        batch_dones = self.dones[indices]
        
        return (batch_states, batch_actions, batch_rewards, 
                batch_next_states, batch_dones, weights, indices)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        self.priorities[indices] = priorities ** self.alpha
        self.max_priority = max(self.max_priority, np.max(priorities))

    def __len__(self):
        return self.size


class MultiTaskReplayBuffer:
    """Replay buffer that maintains separate buffers for different tasks"""
    
    def __init__(self, capacity_per_task: int, state_dim: int, action_dim: int, device: str = 'cpu'):
        self.capacity_per_task = capacity_per_task
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Task-specific buffers
        self.task_buffers: Dict[int, ExpertReplayBuffer] = {}
        self.task_ids = set()

    def add_task(self, task_id: int):
        """Add a new task buffer"""
        if task_id not in self.task_buffers:
            self.task_buffers[task_id] = ExpertReplayBuffer(
                self.capacity_per_task, self.state_dim, self.action_dim, self.device
            )
            self.task_ids.add(task_id)

    def add(self, task_id: int, state: np.ndarray, action: np.ndarray, 
            reward: float, next_state: np.ndarray, done: bool):
        """Add experience to specific task buffer"""
        if task_id not in self.task_buffers:
            self.add_task(task_id)
        
        self.task_buffers[task_id].add(state, action, reward, next_state, done)

    def sample_task(self, task_id: int, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample from specific task buffer"""
        if task_id not in self.task_buffers:
            raise ValueError(f"Task {task_id} not found in buffer")
        
        return self.task_buffers[task_id].sample(batch_size)

    def sample_mixed(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample mixed batch from all tasks"""
        if not self.task_buffers:
            raise ValueError("No task buffers available")
        
        # Determine how many samples from each task
        available_tasks = [tid for tid, buf in self.task_buffers.items() if len(buf) > 0]
        if not available_tasks:
            raise ValueError("All task buffers are empty")
        
        samples_per_task = batch_size // len(available_tasks)
        remainder = batch_size % len(available_tasks)
        
        all_states, all_actions, all_rewards = [], [], []
        all_next_states, all_dones = [], []
        
        for i, task_id in enumerate(available_tasks):
            task_batch_size = samples_per_task + (1 if i < remainder else 0)
            if task_batch_size > 0:
                task_batch = self.task_buffers[task_id].sample(task_batch_size)
                all_states.append(task_batch[0])
                all_actions.append(task_batch[1])
                all_rewards.append(task_batch[2])
                all_next_states.append(task_batch[3])
                all_dones.append(task_batch[4])
        
        # Concatenate and shuffle
        batch_states = np.concatenate(all_states, axis=0)
        batch_actions = np.concatenate(all_actions, axis=0)
        batch_rewards = np.concatenate(all_rewards, axis=0)
        batch_next_states = np.concatenate(all_next_states, axis=0)
        batch_dones = np.concatenate(all_dones, axis=0)
        
        # Shuffle the batch
        indices = np.random.permutation(len(batch_states))
        return (batch_states[indices], batch_actions[indices], batch_rewards[indices],
                batch_next_states[indices], batch_dones[indices])

    def get_task_sizes(self) -> Dict[int, int]:
        """Get size of each task buffer"""
        return {task_id: len(buffer) for task_id, buffer in self.task_buffers.items()}

    def clear_task(self, task_id: int):
        """Clear specific task buffer"""
        if task_id in self.task_buffers:
            self.task_buffers[task_id].clear()

    def remove_task(self, task_id: int):
        """Remove task buffer completely"""
        if task_id in self.task_buffers:
            del self.task_buffers[task_id]
            self.task_ids.discard(task_id)

    def __len__(self):
        return sum(len(buffer) for buffer in self.task_buffers.values())


class SequentialBuffer:
    """Buffer for storing sequential data (for context learning)"""
    
    def __init__(self, capacity: int, sequence_length: int, state_dim: int, 
                 action_dim: int, device: str = 'cpu'):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Storage for sequences
        self.sequences = deque(maxlen=capacity)
        self.size = 0

    def add_sequence(self, states: List[np.ndarray], actions: List[np.ndarray],
                    rewards: List[float], dones: List[bool]):
        """Add a complete sequence"""
        if len(states) != len(actions) or len(states) != len(rewards):
            raise ValueError("All lists must have same length")
        
        sequence = {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones)
        }
        
        self.sequences.append(sequence)
        self.size = min(self.size + 1, self.capacity)

    def sample_sequences(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch of sequences"""
        if self.size < batch_size:
            batch_size = self.size
        
        indices = random.sample(range(self.size), batch_size)
        
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        
        for idx in indices:
            seq = self.sequences[idx]
            batch_states.append(seq['states'])
            batch_actions.append(seq['actions'])
            batch_rewards.append(seq['rewards'])
            batch_dones.append(seq['dones'])
        
        return {
            'states': np.array(batch_states),
            'actions': np.array(batch_actions),
            'rewards': np.array(batch_rewards),
            'dones': np.array(batch_dones)
        }

    def extract_windows(self, states: List[np.ndarray], actions: List[np.ndarray],
                       rewards: List[float], dones: List[bool]):
        """Extract sliding windows from a long sequence and add them"""
        seq_len = len(states)
        
        for i in range(seq_len - self.sequence_length + 1):
            window_states = states[i:i + self.sequence_length]
            window_actions = actions[i:i + self.sequence_length]
            window_rewards = rewards[i:i + self.sequence_length]
            window_dones = dones[i:i + self.sequence_length]
            
            self.add_sequence(window_states, window_actions, window_rewards, window_dones)

    def __len__(self):
        return self.size


def create_expert_buffer(capacity: int, state_dim: int, action_dim: int, device: str = 'cpu') -> ExpertReplayBuffer:
    """Factory function to create expert replay buffer"""
    return ExpertReplayBuffer(capacity, state_dim, action_dim, device)

def create_prioritized_buffer(capacity: int, state_dim: int, action_dim: int, 
                            alpha: float = 0.6, beta: float = 0.4, device: str = 'cpu') -> PrioritizedReplayBuffer:
    """Factory function to create prioritized replay buffer"""
    return PrioritizedReplayBuffer(capacity, state_dim, action_dim, alpha, beta, device)

def create_multitask_buffer(capacity_per_task: int, state_dim: int, action_dim: int, 
                          device: str = 'cpu') -> MultiTaskReplayBuffer:
    """Factory function to create multi-task replay buffer"""
    return MultiTaskReplayBuffer(capacity_per_task, state_dim, action_dim, device)