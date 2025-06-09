import numpy as np
import random
from collections import deque

class SimpleReplayBuffer:
    """간단한 리플레이 버퍼"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done, instruction_emb=None):
        """경험 추가"""
        experience = {
            'state': np.array(state, dtype=np.float32),
            'action': np.array(action, dtype=np.float32),
            'reward': float(reward),
            'next_state': np.array(next_state, dtype=np.float32),
            'done': bool(done),
            'instruction_emb': np.array(instruction_emb, dtype=np.float32) if instruction_emb is not None else None
        }
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """배치 샘플링"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([exp['state'] for exp in batch])
        actions = np.array([exp['action'] for exp in batch])
        rewards = np.array([exp['reward'] for exp in batch])
        next_states = np.array([exp['next_state'] for exp in batch])
        dones = np.array([exp['done'] for exp in batch])
        
        if batch[0]['instruction_emb'] is not None:
            instructions = np.array([exp['instruction_emb'] for exp in batch])
        else:
            instructions = None
        
        return states, actions, rewards, next_states, dones, instructions
    
    def __len__(self):
        return len(self.buffer)