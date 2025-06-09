import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class SafeReplayBuffer:
    """안전한 리플레이 버퍼"""
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """경험 추가"""
        try:
            # 모든 입력을 numpy 배열로 안전하게 변환
            state = np.array(state, dtype=np.float32)
            action = np.array(action, dtype=np.float32)
            reward = float(reward)
            next_state = np.array(next_state, dtype=np.float32)
            done = bool(done)
            
            self.buffer.append((state, action, reward, next_state, done))
        except Exception as e:
            print(f"Warning: Failed to add to replay buffer: {e}")
    
    def sample(self, batch_size):
        """배치 샘플링"""
        try:
            if len(self.buffer) < batch_size:
                batch_size = len(self.buffer)
            
            batch = random.sample(self.buffer, batch_size)
            
            states = np.array([e[0] for e in batch], dtype=np.float32)
            actions = np.array([e[1] for e in batch], dtype=np.float32)
            rewards = np.array([e[2] for e in batch], dtype=np.float32)
            next_states = np.array([e[3] for e in batch], dtype=np.float32)
            dones = np.array([e[4] for e in batch], dtype=bool)
            
            return states, actions, rewards, next_states, dones
        except Exception as e:
            print(f"Warning: Failed to sample from replay buffer: {e}")
            # 안전한 기본값 반환
            dummy_state = np.zeros((batch_size, len(self.buffer[0][0])), dtype=np.float32)
            dummy_action = np.zeros((batch_size, len(self.buffer[0][1])), dtype=np.float32)
            dummy_reward = np.zeros(batch_size, dtype=np.float32)
            dummy_done = np.zeros(batch_size, dtype=bool)
            return dummy_state, dummy_action, dummy_reward, dummy_state, dummy_done
    
    def __len__(self):
        return len(self.buffer)

class SafeActor(nn.Module):
    """안전한 액터 네트워크"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # 로그 표준편차 클리핑
        self.log_std_min = -20
        self.log_std_max = 2
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state):
        """행동 샘플링"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 재매개화 트릭
        action = torch.tanh(x_t)
        
        log_prob = normal.log_prob(x_t)
        # tanh 변환에 대한 로그 확률 보정
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

class SafeCritic(nn.Module):
    """안전한 크리틱 네트워크"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Q1 네트워크
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, 1)
        
        # Q2 네트워크
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)
        
        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)
        
        return q1, q2

class DimensionExactSACExpert:
    """완전히 안전한 SAC 전문가"""
    
    def __init__(self, state_dim, action_dim, device='cuda', hidden_dim=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_dim
        
        print(f"✓ SAC Expert initialized: state_dim={state_dim}, action_dim={action_dim}")
        
        # 네트워크 초기화
        self.actor = SafeActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = SafeCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic = SafeCritic(state_dim, action_dim, hidden_dim).to(self.device)
        
        # 타겟 네트워크 초기화
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 옵티마이저
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # 엔트로피 계수
        self.alpha = 0.2
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        
        # 리플레이 버퍼
        self.replay_buffer = SafeReplayBuffer(capacity=5000)
        
        # 하이퍼파라미터
        self.gamma = 0.99
        self.tau = 0.005
    
    def _safe_to_tensor(self, data):
        """데이터를 안전하게 텐서로 변환"""
        try:
            if torch.is_tensor(data):
                return data.to(self.device).float()
            elif isinstance(data, np.ndarray):
                return torch.from_numpy(data).to(self.device).float()
            else:
                return torch.tensor(data, device=self.device, dtype=torch.float32)
        except Exception as e:
            print(f"Warning: Tensor conversion failed: {e}")
            # 안전한 기본값 반환
            if hasattr(data, '__len__'):
                return torch.zeros(len(data), device=self.device, dtype=torch.float32)
            else:
                return torch.zeros(1, device=self.device, dtype=torch.float32)
    
    def select_action(self, state, deterministic=False):
        """행동 선택"""
        try:
            # 상태를 안전하게 텐서로 변환
            state_tensor = self._safe_to_tensor(state)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            with torch.no_grad():
                if deterministic:
                    mean, _ = self.actor(state_tensor)
                    action = torch.tanh(mean)
                else:
                    action, _ = self.actor.sample(state_tensor)
            
            # numpy로 변환하여 반환
            action_np = action.cpu().numpy().squeeze()
            
            # 차원 확인
            if action_np.ndim == 0:
                action_np = np.array([action_np])
            
            return action_np
            
        except Exception as e:
            print(f"Warning: Action selection failed: {e}")
            # 안전한 랜덤 행동 반환
            return np.random.uniform(-1, 1, self.action_dim).astype(np.float32)
    
    def update(self, batch_size=32):
        """네트워크 업데이트"""
        try:
            if len(self.replay_buffer) < batch_size:
                return None
            
            # 배치 샘플링
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
            
            # 텐서로 변환
            states = self._safe_to_tensor(states)
            actions = self._safe_to_tensor(actions)
            rewards = self._safe_to_tensor(rewards).unsqueeze(1)
            next_states = self._safe_to_tensor(next_states)
            dones = self._safe_to_tensor(dones.astype(np.float32)).unsqueeze(1)
            
            # 크리틱 업데이트
            with torch.no_grad():
                next_actions, next_log_probs = self.actor.sample(next_states)
                target_q1, target_q2 = self.target_critic(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
                target_q = rewards + (1 - dones) * self.gamma * target_q
            
            current_q1, current_q2 = self.critic(states, actions)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # 액터 업데이트
            new_actions, log_probs = self.actor.sample(states)
            q1, q2 = self.critic(states, new_actions)
            q = torch.min(q1, q2)
            
            actor_loss = (self.alpha * log_probs - q).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 엔트로피 계수 업데이트
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
            
            # 타겟 네트워크 소프트 업데이트
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            return {
                'critic_loss': critic_loss.item(),
                'actor_loss': actor_loss.item(),
                'alpha_loss': alpha_loss.item(),
                'total_loss': critic_loss.item() + actor_loss.item()
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
                'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
                'log_alpha': self.log_alpha,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim
            }, filepath)
            print(f"✓ Model saved to {filepath}")
        except Exception as e:
            print(f"Warning: Failed to save model: {e}")
    
    def load(self, filepath):
        """모델 로드"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp()
            print(f"✓ Model loaded from {filepath}")
        except Exception as e:
            print(f"Warning: Failed to load model: {e}")