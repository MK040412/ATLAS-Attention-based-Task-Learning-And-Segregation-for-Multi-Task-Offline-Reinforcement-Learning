import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class SimpleReplayBuffer:
    """간단한 리플레이 버퍼"""
    
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """경험 추가 - 모든 입력을 numpy로 변환"""
        try:
            # 모든 데이터를 numpy 배열로 변환
            state = np.array(state, dtype=np.float32)
            action = np.array(action, dtype=np.float32)
            reward = float(reward)
            next_state = np.array(next_state, dtype=np.float32)
            done = bool(done)
            
            self.buffer.append((state, action, reward, next_state, done))
        except Exception as e:
            print(f"Warning: Failed to add experience: {e}")
    
    def sample(self, batch_size):
        """배치 샘플링"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([e[0] for e in batch], dtype=np.float32)
        actions = np.array([e[1] for e in batch], dtype=np.float32)
        rewards = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array([e[3] for e in batch], dtype=np.float32)
        dones = np.array([e[4] for e in batch], dtype=bool)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class SimpleActor(nn.Module):
    """간단한 액터 네트워크"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.net(state)

class SimpleCritic(nn.Module):
    """간단한 크리틱 네트워크"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class RTX4060SACExpert:
    """RTX 4060 최적화된 SAC 전문가"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, device='cuda'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 네트워크 생성
        self.actor = SimpleActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1 = SimpleCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = SimpleCritic(state_dim, action_dim, hidden_dim).to(self.device)
        
        # 타겟 네트워크
        self.target_critic1 = SimpleCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic2 = SimpleCritic(state_dim, action_dim, hidden_dim).to(self.device)
        
        # 타겟 네트워크 초기화
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # 옵티마이저
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=1e-4)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=1e-4)
        
        # 리플레이 버퍼
        self.replay_buffer = SimpleReplayBuffer(capacity=1000)
        
        # 하이퍼파라미터
        self.gamma = 0.99
        self.tau = 0.005
        self.noise_scale = 0.1
        
    def safe_to_tensor(self, data):
        """안전한 텐서 변환"""
        try:
            if torch.is_tensor(data):
                return data.to(self.device)
            else:
                # numpy 배열이나 기타 데이터를 텐서로 변환
                np_data = np.array(data, dtype=np.float32)
                return torch.FloatTensor(np_data).to(self.device)
        except Exception as e:
            print(f"Warning: Tensor conversion failed: {e}")
            # 실패시 더미 텐서 반환
            if hasattr(data, 'shape'):
                shape = data.shape
            elif hasattr(data, '__len__'):
                shape = (len(data),)
            else:
                shape = (1,)
            return torch.zeros(shape, dtype=torch.float32, device=self.device)
    
    def select_action(self, state, deterministic=False):
        """행동 선택"""
        try:
            # 상태를 안전하게 텐서로 변환
            state_tensor = self.safe_to_tensor(state)
            
            # 배치 차원 추가
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            with torch.no_grad():
                action = self.actor(state_tensor)
                
                # 탐험을 위한 노이즈 추가
                if not deterministic:
                    noise = torch.randn_like(action) * self.noise_scale
                    action = action + noise
                
                # 액션 범위 클리핑
                action = torch.clamp(action, -1.0, 1.0)
                
                # numpy로 변환하여 반환
                return action.squeeze(0).cpu().numpy()
                
        except Exception as e:
            print(f"Warning: Action selection failed: {e}")
            # 실패시 랜덤 액션 반환
            return np.random.uniform(-1, 1, self.action_dim).astype(np.float32)
    
    def update(self, batch_size=32):
        """모델 업데이트"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        try:
            # 배치 샘플링
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
            
            # 텐서로 변환
            states = self.safe_to_tensor(states)
            actions = self.safe_to_tensor(actions)
            rewards = self.safe_to_tensor(rewards).unsqueeze(1)
            next_states = self.safe_to_tensor(next_states)
            dones = self.safe_to_tensor(dones.astype(np.float32)).unsqueeze(1)
            
            # 크리틱 업데이트
            with torch.no_grad():
                next_actions = self.actor(next_states)
                target_q1 = self.target_critic1(next_states, next_actions)
                target_q2 = self.target_critic2(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2)
                target_q = rewards + self.gamma * target_q * (1 - dones)
            
            current_q1 = self.critic1(states, actions)
            current_q2 = self.critic2(states, actions)
            
            critic1_loss = F.mse_loss(current_q1, target_q)
            critic2_loss = F.mse_loss(current_q2, target_q)
            
            # 크리틱 업데이트
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()
            
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()
            
            # 액터 업데이트
            new_actions = self.actor(states)
            actor_loss = -self.critic1(states, new_actions).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 타겟 네트워크 소프트 업데이트
            self.soft_update(self.critic1, self.target_critic1)
            self.soft_update(self.critic2, self.target_critic2)
            
            return {
                'critic1_loss': critic1_loss.item(),
                'critic2_loss': critic2_loss.item(),
                'actor_loss': actor_loss.item(),
                'total_loss': critic1_loss.item() + critic2_loss.item() + actor_loss.item()
            }
            
        except Exception as e:
            print(f"Warning: Update failed: {e}")
            return None
    
    def soft_update(self, source, target):
        """소프트 업데이트"""
        try:
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + source_param.data * self.tau
                )
        except Exception as e:
            print(f"Warning: Soft update failed: {e}")
    
    def count_parameters(self):
        """파라미터 개수 계산"""
        total = 0
        for net in [self.actor, self.critic1, self.critic2]:
            total += sum(p.numel() for p in net.parameters() if p.requires_grad)
        return total
    
    def save(self, path):
        """모델 저장"""
        try:
            checkpoint = {
                'actor_state_dict': self.actor.state_dict(),
                'critic1_state_dict': self.critic1.state_dict(),
                'critic2_state_dict': self.critic2.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
                'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
                'config': {
                    'state_dim': self.state_dim,
                    'action_dim': self.action_dim
                }
            }
            torch.save(checkpoint, path)
        except Exception as e:
            print(f"Warning: Failed to save model: {e}")
    
    def load(self, path):
        """모델 로드"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
            self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        except Exception as e:
            print(f"Warning: Failed to load model: {e}")
    
    def cleanup(self):
        """메모리 정리"""
        try:
            del self.actor, self.critic1, self.critic2
            del self.target_critic1, self.target_critic2
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass