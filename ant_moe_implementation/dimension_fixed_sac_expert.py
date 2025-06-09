import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class DimensionAdaptiveNetwork(nn.Module):
    """차원을 자동으로 맞춰주는 네트워크"""
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 적응형 입력 레이어
        self.input_adapter = nn.Linear(input_dim, hidden_dim)
        
        # 히든 레이어들
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim//2)
        
        # 출력 레이어
        self.output = nn.Linear(hidden_dim//2, output_dim)
        
        # 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 입력 차원 체크 및 조정
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # 예상 차원과 다르면 조정
        if x.size(-1) != self.input_dim:
            if x.size(-1) < self.input_dim:
                # 패딩
                padding = torch.zeros(x.size(0), self.input_dim - x.size(-1), device=x.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                # 자르기
                x = x[:, :self.input_dim]
        
        x = F.relu(self.input_adapter(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.output(x)

class SafeReplayBuffer:
    """안전한 리플레이 버퍼"""
    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = None
        self.action_dim = None
    
    def add(self, state, action, reward, next_state, done):
        try:
            # 차원 정보 저장 (첫 번째 추가 시)
            if self.state_dim is None:
                self.state_dim = len(state) if hasattr(state, '__len__') else 1
                self.action_dim = len(action) if hasattr(action, '__len__') else 1
            
            # 안전한 변환
            state = np.array(state, dtype=np.float32).flatten()
            action = np.array(action, dtype=np.float32).flatten()
            next_state = np.array(next_state, dtype=np.float32).flatten()
            
            # 차원 맞추기
            if len(state) != self.state_dim:
                if len(state) < self.state_dim:
                    state = np.pad(state, (0, self.state_dim - len(state)))
                else:
                    state = state[:self.state_dim]
            
            if len(next_state) != self.state_dim:
                if len(next_state) < self.state_dim:
                    next_state = np.pad(next_state, (0, self.state_dim - len(next_state)))
                else:
                    next_state = next_state[:self.state_dim]
            
            if len(action) != self.action_dim:
                if len(action) < self.action_dim:
                    action = np.pad(action, (0, self.action_dim - len(action)))
                else:
                    action = action[:self.action_dim]
            
            self.buffer.append((state, action, float(reward), next_state, bool(done)))
            
        except Exception as e:
            print(f"Warning: Failed to add to replay buffer: {e}")
    
    def sample(self, batch_size):
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

class DimensionFixedSACExpert:
    """차원 문제를 해결한 SAC 전문가"""
    def __init__(self, state_dim, action_dim, device='cpu', hidden_dim=128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.hidden_dim = hidden_dim
        
        print(f"Creating SAC Expert: state_dim={state_dim}, action_dim={action_dim}")
        
        # 네트워크들
        self.actor = DimensionAdaptiveNetwork(state_dim, action_dim * 2, hidden_dim).to(device)
        self.critic1 = DimensionAdaptiveNetwork(state_dim + action_dim, 1, hidden_dim).to(device)
        self.critic2 = DimensionAdaptiveNetwork(state_dim + action_dim, 1, hidden_dim).to(device)
        
        # 타겟 네트워크들
        self.target_critic1 = DimensionAdaptiveNetwork(state_dim + action_dim, 1, hidden_dim).to(device)
        self.target_critic2 = DimensionAdaptiveNetwork(state_dim + action_dim, 1, hidden_dim).to(device)
        
        # 타겟 네트워크 초기화
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # 옵티마이저들
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=1e-4)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=1e-4)
        
        # SAC 파라미터들
        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4)
        self.target_entropy = -action_dim
        
        # 리플레이 버퍼
        self.replay_buffer = SafeReplayBuffer(capacity=5000)
        
        # 학습 파라미터
        self.gamma = 0.99
        self.tau = 0.005
        
    def select_action(self, state, deterministic=False):
        """행동 선택 - 완전히 안전한 버전"""
        try:
            # 상태를 안전하게 텐서로 변환
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state).to(self.device)
            elif torch.is_tensor(state):
                if state.device != self.device:
                    state_tensor = state.to(self.device)
                else:
                    state_tensor = state
            else:
                state_tensor = torch.FloatTensor(np.array(state, dtype=np.float32)).to(self.device)
            
            # 차원 조정
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            with torch.no_grad():
                # 액터 네트워크 통과
                actor_output = self.actor(state_tensor)
                
                # 평균과 로그 표준편차 분리
                mean = actor_output[:, :self.action_dim]
                log_std = actor_output[:, self.action_dim:]
                log_std = torch.clamp(log_std, -20, 2)
                
                if deterministic:
                    action = mean
                else:
                    std = log_std.exp()
                    normal = torch.distributions.Normal(mean, std)
                    action = normal.sample()
                
                # 액션 범위 제한
                action = torch.tanh(action)
                
                # numpy로 변환하여 반환
                action_np = action.cpu().numpy().flatten()
                
                # 차원 맞추기
                if len(action_np) != self.action_dim:
                    if len(action_np) < self.action_dim:
                        action_np = np.pad(action_np, (0, self.action_dim - len(action_np)))
                    else:
                        action_np = action_np[:self.action_dim]
                
                return action_np
                
        except Exception as e:
            print(f"Warning: Action selection failed: {e}")
            # 안전한 기본 액션 반환
            return np.zeros(self.action_dim, dtype=np.float32)
    
    def update(self, batch_size=32):
        """네트워크 업데이트"""
        try:
            if len(self.replay_buffer) < batch_size:
                return None
            
            # 배치 샘플링
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
            
            # 텐서로 변환
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.BoolTensor(dones).to(self.device)
            
            # 현재 Q 값들
            current_q1 = self.critic1(torch.cat([states, actions], dim=1))
            current_q2 = self.critic2(torch.cat([states, actions], dim=1))
            
            # 다음 액션들
            with torch.no_grad():
                next_actor_output = self.actor(next_states)
                next_mean = next_actor_output[:, :self.action_dim]
                next_log_std = torch.clamp(next_actor_output[:, self.action_dim:], -20, 2)
                next_std = next_log_std.exp()
                
                next_normal = torch.distributions.Normal(next_mean, next_std)
                next_actions = next_normal.sample()
                next_actions = torch.tanh(next_actions)
                next_log_probs = next_normal.log_prob(next_actions).sum(dim=1, keepdim=True)
                
                # 타겟 Q 값들
                target_q1 = self.target_critic1(torch.cat([next_states, next_actions], dim=1))
                target_q2 = self.target_critic2(torch.cat([next_states, next_actions], dim=1))
                target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_probs
                
                target_q = rewards.unsqueeze(1) + self.gamma * target_q * (~dones).unsqueeze(1)
            
            # Critic 손실
            critic1_loss = F.mse_loss(current_q1, target_q)
            critic2_loss = F.mse_loss(current_q2, target_q)
            
            # Critic 업데이트
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()
            
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()
            
            # Actor 업데이트
            actor_output = self.actor(states)
            mean = actor_output[:, :self.action_dim]
            log_std = torch.clamp(actor_output[:, self.action_dim:], -20, 2)
            std = log_std.exp()
            
            normal = torch.distributions.Normal(mean, std)
            actions_new = normal.sample()
            actions_new = torch.tanh(actions_new)
            log_probs = normal.log_prob(actions_new).sum(dim=1, keepdim=True)
            
            q1_new = self.critic1(torch.cat([states, actions_new], dim=1))
            q2_new = self.critic2(torch.cat([states, actions_new], dim=1))
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
            
            # 타겟 네트워크 업데이트
            self._soft_update(self.target_critic1, self.critic1)
            self._soft_update(self.target_critic2, self.critic2)
            
            return {
                'critic1_loss': critic1_loss.item(),
                'critic2_loss': critic2_loss.item(),
                'actor_loss': actor_loss.item(),
                'alpha_loss': alpha_loss.item(),
                'alpha': self.log_alpha.exp().item(),
                'total_loss': critic1_loss.item() + critic2_loss.item() + actor_loss.item()
            }
            
        except Exception as e:
            print(f"Warning: Update failed: {e}")
            return None
    
    def _soft_update(self, target, source):
        """소프트 업데이트"""
        try:
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        except Exception as e:
            print(f"Warning: Soft update failed: {e}")
    
    def save(self, filepath):
        """모델 저장"""
        try:
            torch.save({
                'actor': self.actor.state_dict(),
                'critic1': self.critic1.state_dict(),
                'critic2': self.critic2.state_dict(),
                'target_critic1': self.target_critic1.state_dict(),
                'target_critic2': self.target_critic2.state_dict(),
                'log_alpha': self.log_alpha,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'hidden_dim': self.hidden_dim
            }, filepath)
            print(f"✓ Expert saved to {filepath}")
        except Exception as e:
            print(f"Warning: Failed to save expert: {e}")
    
    def load(self, filepath):
        """모델 로드"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic1.load_state_dict(checkpoint['critic1'])
            self.critic2.load_state_dict(checkpoint['critic2'])
            self.target_critic1.load_state_dict(checkpoint['target_critic1'])
            self.target_critic2.load_state_dict(checkpoint['target_critic2'])
            self.log_alpha = checkpoint['log_alpha']
            print(f"✓ Expert loaded from {filepath}")
        except Exception as e:
            print(f"Warning: Failed to load expert: {e}")
    
    def cleanup(self):
        """메모리 정리"""
        try:
            del self.actor, self.critic1, self.critic2
            del self.target_critic1, self.target_critic2
            del self.replay_buffer
            torch.cuda.empty_cache()
        except:
            pass