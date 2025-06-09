import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
from collections import deque
import random

class ReplayBuffer:
    """SAC용 리플레이 버퍼"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """경험 추가"""
        self.buffer.append({
            'state': np.array(state, dtype=np.float32),
            'action': np.array(action, dtype=np.float32),
            'reward': float(reward),
            'next_state': np.array(next_state, dtype=np.float32),
            'done': bool(done)
        })
    
    def sample(self, batch_size: int) -> Dict:
        """배치 샘플링"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        return {
            'states': torch.FloatTensor([e['state'] for e in batch]),
            'actions': torch.FloatTensor([e['action'] for e in batch]),
            'rewards': torch.FloatTensor([e['reward'] for e in batch]),
            'next_states': torch.FloatTensor([e['next_state'] for e in batch]),
            'dones': torch.BoolTensor([e['done'] for e in batch])
        }
    
    def __len__(self):
        return len(self.buffer)

class GaussianPolicy(nn.Module):
    """가우시안 정책 네트워크"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # 로그 표준편차 클리핑
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, state):
        """정책 출력"""
        features = self.net(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state):
        """행동 샘플링 with reparameterization trick"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # 정규분포에서 샘플링
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        
        # tanh 변환으로 [-1, 1] 범위로 제한
        action = torch.tanh(x_t)
        
        # 로그 확률 계산 (Jacobian 보정 포함)
        log_prob = normal.log_prob(x_t)
        # tanh 변환의 Jacobian: log(1 - tanh²(x))
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob

class QNetwork(nn.Module):
    """Q-함수 네트워크"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        """Q(s,a) 계산"""
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class RigorousSACExpert:
    """엄밀한 SAC 전문가 구현"""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 expert_id: int,
                 hidden_dim: int = 256,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 auto_tune_alpha: bool = True,
                 device: str = 'cuda'):
        
        self.expert_id = expert_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)
        
        # 네트워크 초기화
        self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # 타겟 네트워크
        self.target_critic1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # 타겟 네트워크 초기화
        self._hard_update(self.target_critic1, self.critic1)
        self._hard_update(self.target_critic2, self.critic2)
        
        # 옵티마이저
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        
        # 온도 파라미터 α
        self.auto_tune_alpha = auto_tune_alpha
        if auto_tune_alpha:
            self.target_entropy = -action_dim  # H(π) ≥ -dim(A)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(alpha, device=self.device)
        
        # 리플레이 버퍼
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # 통계 추적
        self.update_count = 0
        self.loss_history = {
            'critic1_loss': [],
            'critic2_loss': [],
            'actor_loss': [],
            'alpha_loss': [],
            'alpha_value': []
        }
        
        print(f"🤖 SAC Expert {expert_id} initialized:")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}")
        print(f"  Auto-tune α: {auto_tune_alpha}")
        print(f"  Target entropy: {self.target_entropy if auto_tune_alpha else 'N/A'}")
    
    def _hard_update(self, target, source):
        """하드 업데이트: θ_target ← θ_source"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def _soft_update(self, target, source):
        """소프트 업데이트: θ_target ← τθ_source + (1-τ)θ_target"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def select_action(self, state, deterministic=False):
        """행동 선택"""
        with torch.no_grad():
            if not torch.is_tensor(state):
                state = torch.FloatTensor(state).to(self.device)
            
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            if deterministic:
                # 결정적 행동 (평가용)
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
                log_prob = None
            else:
                # 확률적 행동 (탐험용)
                action, log_prob = self.actor.sample(state)
            
            return action.cpu().numpy().squeeze(), log_prob
    
    def update(self, batch_size: int = 256) -> Dict:
        """SAC 업데이트 - 엄밀한 손실 함수들"""
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # 배치 샘플링
        batch = self.replay_buffer.sample(batch_size)
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device).unsqueeze(1)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device).unsqueeze(1)
        
        # === 크리틱 업데이트 ===
        with torch.no_grad():
            # 다음 상태에서의 행동 샘플링
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # 타겟 Q 값 계산
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            
            # TD 타겟: r + γ(1-done) * (min Q(s',a') - α log π(a'|s'))
            td_target = rewards + self.gamma * (1 - dones.float()) * target_q
        
        # 현재 Q 값들
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # 크리틱 손실: L_Q = 𝔼[(Q(s,a) - (r + γ(min Q(s',a') - α log π(a'|s'))))²]
        critic1_loss = F.mse_loss(current_q1, td_target)
        critic2_loss = F.mse_loss(current_q2, td_target)
        
        # 크리틱 업데이트
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # === 액터 업데이트 ===
        # 현재 정책으로 행동 샘플링
        current_actions, current_log_probs = self.actor.sample(states)
        
        # Q 값 계산
        q1_current = self.critic1(states, current_actions)
        q2_current = self.critic2(states, current_actions)
        q_current = torch.min(q1_current, q2_current)
        
        # 액터 손실: L_π = 𝔼[α log π(a|s) - Q(s,a)]
        actor_loss = (self.alpha * current_log_probs - q_current).mean()
        
        # 액터 업데이트
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # === 온도 파라미터 α 업데이트 ===
        alpha_loss = torch.tensor(0.0)
        if self.auto_tune_alpha:
            # α 손실: L_α = -α * (log π(a|s) + H_target)
            alpha_loss = -(self.log_alpha * (current_log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # === 타겟 네트워크 소프트 업데이트 ===
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)
        
        # 통계 업데이트
        self.update_count += 1
        losses = {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha_value': self.alpha.item()
        }
        
        for key, value in losses.items():
            self.loss_history[key].append(value)
        
        return losses
    
    def get_statistics(self) -> Dict:
        """전문가 통계"""
        return {
            'expert_id': self.expert_id,
            'update_count': self.update_count,
            'buffer_size': len(self.replay_buffer),
            'current_alpha': self.alpha.item(),
            'recent_losses': {
                key: values[-10:] if values else []
                for key, values in self.loss_history.items()
            }
        }