import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from config import CONFIG

class CompactReplayBuffer:
    """메모리 효율적인 리플레이 버퍼"""
    
    def __init__(self, capacity=2000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def add(self, state, action, reward, next_state, done):
        """경험 추가"""
        # numpy 배열로 변환하여 메모리 절약
        if torch.is_tensor(state):
            state = state.detach().cpu().numpy()
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        if torch.is_tensor(next_state):
            next_state = next_state.detach().cpu().numpy()
        
        experience = (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done)
        )
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.capacity
    
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
    
    def clear(self):
        """버퍼 비우기"""
        self.buffer.clear()
        self.position = 0

class CompactActor(nn.Module):
    """경량화된 액터 네트워크"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, max_action=1.0):
        super().__init__()
        self.max_action = max_action
        
        # 매우 작은 네트워크
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim * 2)  # mean과 log_std
        )
        
        # 로그 표준편차 범위 제한
        self.log_std_min = -20
        self.log_std_max = 2
        
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, state):
        """순전파"""
        x = self.net(state)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        
        # 로그 표준편차 클리핑
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state, deterministic=False):
        """행동 샘플링"""
        mean, log_std = self.forward(state)
        
        if deterministic:
            action = torch.tanh(mean) * self.max_action
            log_prob = None
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # reparameterization trick
            action = torch.tanh(x_t) * self.max_action
            
            # 로그 확률 계산 (SAC용)
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.max_action * (1 - action.pow(2)) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob

class CompactCritic(nn.Module):
    """경량화된 크리틱 네트워크"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        
        # Q1 네트워크
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Q2 네트워크
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, state, action):
        """순전파"""
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2
    
    def q1_forward(self, state, action):
        """Q1만 계산"""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)

class RTX4060SACExpert:
    """RTX 4060 최적화된 SAC 전문가"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, device="cuda"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        # 네트워크 초기화
        self.actor = CompactActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = CompactCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic = CompactCritic(state_dim, action_dim, hidden_dim).to(self.device)
        
        # 타겟 네트워크 초기화
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 온도 매개변수 (자동 조정)
        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device=self.device)
        self.target_entropy = -action_dim  # 휴리스틱
        
        # 옵티마이저 (낮은 학습률)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=CONFIG.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=CONFIG.critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=CONFIG.alpha_lr)
        
        # 리플레이 버퍼
        self.replay_buffer = CompactReplayBuffer(capacity=CONFIG.buffer_capacity)
        
        # 학습 카운터
        self.update_count = 0
        
        print(f"RTX4060 SAC Expert initialized:")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Parameters: {self.count_parameters():,}")
    
    def count_parameters(self):
        """파라미터 개수 계산"""
        total = 0
        total += sum(p.numel() for p in self.actor.parameters())
        total += sum(p.numel() for p in self.critic.parameters())
        total += sum(p.numel() for p in self.target_critic.parameters())
        return total
    
    @property
    def alpha(self):
        """온도 매개변수"""
        return self.log_alpha.exp()
    
    def select_action(self, state, deterministic=False):
        """행동 선택"""
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).to(self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            action, _ = self.actor.sample(state, deterministic=deterministic)
        
        return action.cpu().numpy().squeeze()
    
    def update(self, batch_size=None):
        """네트워크 업데이트"""
        if len(self.replay_buffer) < (batch_size or CONFIG.batch_size):
            return None
        
        batch_size = batch_size or CONFIG.batch_size
        
        try:
            # 배치 샘플링
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
            
            # 텐서로 변환
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
            
            # 크리틱 업데이트
            critic_loss = self._update_critic(states, actions, rewards, next_states, dones)
            
            # 액터 및 온도 업데이트 (덜 빈번하게)
            actor_loss = None
            alpha_loss = None
            if self.update_count % 2 == 0:  # 2번에 1번만 업데이트
                actor_loss = self._update_actor(states)
                alpha_loss = self._update_alpha(states)
            
            # 타겟 네트워크 소프트 업데이트
            if self.update_count % 2 == 0:
                self._soft_update_target()
            
            self.update_count += 1
            
            return {
                'critic_loss': critic_loss,
                'actor_loss': actor_loss,
                'alpha_loss': alpha_loss,
                'alpha': self.alpha.item(),
                'total_loss': critic_loss + (actor_loss or 0) + (alpha_loss or 0)
            }
            
        except Exception as e:
            print(f"Warning: Expert update failed: {e}")
            return None
    
    def _update_critic(self, states, actions, rewards, next_states, dones):
        """크리틱 업데이트"""
        with torch.no_grad():
            # 다음 상태에서의 행동 샘플링
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # 타겟 Q 값 계산
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + CONFIG.gamma * target_q * (~dones)
        
        # 현재 Q 값
        current_q1, current_q2 = self.critic(states, actions)
        
        # 크리틱 손실
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 크리틱 업데이트
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # 그래디언트 클리핑
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor(self, states):
        """액터 업데이트"""
        # 현재 정책으로 행동 샘플링
        actions, log_probs = self.actor.sample(states)
        
        # Q 값 계산
        q1, q2 = self.critic(states, actions)
        q = torch.min(q1, q2)
        
        # 액터 손실 (정책 개선)
        actor_loss = (self.alpha * log_probs - q).mean()
        
        # 액터 업데이트
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _update_alpha(self, states):
        """온도 매개변수 업데이트"""
        with torch.no_grad():
            _, log_probs = self.actor.sample(states)
        
        # 온도 손실
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy)).mean()
        
        # 온도 업데이트
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        return alpha_loss.item()
    
    def _soft_update_target(self):
        """타겟 네트워크 소프트 업데이트"""
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(CONFIG.tau * param.data + (1 - CONFIG.tau) * target_param.data)
    
    def save(self, path):
        """모델 저장"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'update_count': self.update_count
        }, path)
    
    def load(self, path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.log_alpha = checkpoint['log_alpha']
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.update_count = checkpoint['update_count']
    
    def cleanup(self):
        """메모리 정리"""
        self.replay_buffer.clear()
        del self.actor
        del self.critic
        del self.target_critic
        if torch.cuda.is_available():
            torch.cuda.empty_cache()