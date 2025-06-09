import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions), 
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # mean and log_std
        )
        self.action_dim = action_dim
    
    def forward(self, state):
        x = self.net(state)
        mean, log_std = x.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

class DimensionExactSACExpert:
    def __init__(self, state_dim, action_dim, device='cpu', hidden_dim=128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.hidden_dim = hidden_dim
        
        # 네트워크 초기화
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        
        # 타겟 네트워크 초기화
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 옵티마이저
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # SAC 파라미터
        self.log_alpha = torch.tensor(0.0, device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -action_dim
        self.tau = 0.005
        
        # 리플레이 버퍼
        self.replay_buffer = ReplayBuffer()
        
        print(f"✓ SAC Expert initialized: state_dim={state_dim}, action_dim={action_dim}")
    
    def _to_tensor(self, data):
        """데이터를 텐서로 안전하게 변환"""
        try:
            if torch.is_tensor(data):
                return data.to(self.device)
            elif isinstance(data, np.ndarray):
                return torch.FloatTensor(data).to(self.device)
            else:
                return torch.FloatTensor(np.array(data)).to(self.device)
        except Exception as e:
            print(f"Warning: Tensor conversion failed: {e}")
            return torch.zeros(1, device=self.device)
    
    def select_action(self, state, deterministic=False):
        """행동 선택"""
        try:
            # 상태를 텐서로 변환
            state_tensor = self._to_tensor(state)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            with torch.no_grad():
                mean, log_std = self.actor(state_tensor)
                
                if deterministic:
                    action = mean
                else:
                    std = log_std.exp()
                    normal = torch.distributions.Normal(mean, std)
                    action = normal.sample()
                
                # 행동을 [-1, 1] 범위로 제한
                action = torch.tanh(action)
            
            # numpy 배열로 변환하여 반환
            return action.squeeze(0).cpu().numpy()
            
        except Exception as e:
            print(f"Warning: Action selection failed: {e}")
            # 안전한 기본 행동 반환
            return np.zeros(self.action_dim, dtype=np.float32)
    
    def update(self, batch_size=32):
        """네트워크 업데이트"""
        try:
            if len(self.replay_buffer) < batch_size:
                return None
            
            # 배치 샘플링
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
            
            # 텐서로 변환
            states = self._to_tensor(states)
            actions = self._to_tensor(actions)
            rewards = self._to_tensor(rewards).unsqueeze(1)
            next_states = self._to_tensor(next_states)
            dones = self._to_tensor(dones.astype(np.float32)).unsqueeze(1)
            
            # 현재 Q값 계산
            q1, q2 = self.critic(states, actions)
            
            # 다음 상태의 행동과 Q값 계산
            with torch.no_grad():
                next_mean, next_log_std = self.actor(next_states)
                next_std = next_log_std.exp()
                next_normal = torch.distributions.Normal(next_mean, next_std)
                next_actions = torch.tanh(next_normal.sample())
                next_log_probs = next_normal.log_prob(next_actions).sum(dim=-1, keepdim=True)
                
                next_q1, next_q2 = self.target_critic(next_states, next_actions)
                next_q = torch.min(next_q1, next_q2) - self.log_alpha.exp() * next_log_probs
                target_q = rewards + (1 - dones) * 0.99 * next_q
            
            # Critic 손실
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            
            # Critic 업데이트
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Actor 업데이트
            mean, log_std = self.actor(states)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            actions_new = torch.tanh(normal.sample())
            log_probs = normal.log_prob(actions_new).sum(dim=-1, keepdim=True)
            
            q1_new, q2_new = self.critic(states, actions_new)
            q_new = torch.min(q1_new, q2_new)
            
            actor_loss = (self.log_alpha.exp() * log_probs - q_new).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Alpha 업데이트
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # 타겟 네트워크 소프트 업데이트
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            return {
                'critic_loss': critic_loss.item(),
                'actor_loss': actor_loss.item(),
                'alpha_loss': alpha_loss.item(),
                'total_loss': critic_loss.item() + actor_loss.item() + alpha_loss.item()
            }
            
        except Exception as e:
            print(f"Warning: Update failed: {e}")
            return None
    
    def save(self, filepath):
        """모델 저장"""
        try:
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'target_critic_state_dict': self.target_critic.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'log_alpha': self.log_alpha,
                'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            }, filepath)
            print(f"✓ Model saved to {filepath}")
        except Exception as e:
            print(f"Warning: Model save failed: {e}")
    
    def load(self, filepath):
        """모델 로드"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            print(f"✓ Model loaded from {filepath}")
        except Exception as e:
            print(f"Warning: Model load failed: {e}")