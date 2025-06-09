import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class ExactDimensionNetwork(nn.Module):
    """정확한 차원으로 고정된 네트워크"""
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        print(f"Creating network: {input_dim} -> {hidden_dim} -> {hidden_dim//2} -> {output_dim}")
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, output_dim)
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 입력 차원 엄격히 체크
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {x.size(-1)}")
        
        return self.layers(x)

class ExactReplayBuffer:
    """정확한 차원을 보장하는 리플레이 버퍼"""
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer = deque(maxlen=capacity)
        
        print(f"ReplayBuffer: state_dim={state_dim}, action_dim={action_dim}, capacity={capacity}")
    
    def add(self, state, action, reward, next_state, done):
        try:
            # 엄격한 차원 체크 및 변환
            state = np.array(state, dtype=np.float32).flatten()
            action = np.array(action, dtype=np.float32).flatten()
            next_state = np.array(next_state, dtype=np.float32).flatten()
            
            # 차원 검증
            if len(state) != self.state_dim:
                raise ValueError(f"State dimension mismatch: expected {self.state_dim}, got {len(state)}")
            if len(action) != self.action_dim:
                raise ValueError(f"Action dimension mismatch: expected {self.action_dim}, got {len(action)}")
            if len(next_state) != self.state_dim:
                raise ValueError(f"Next state dimension mismatch: expected {self.state_dim}, got {len(next_state)}")
            
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

class DimensionExactSACExpert:
    """차원이 정확히 맞는 SAC 전문가"""
    def __init__(self, state_dim, action_dim, device='cpu', hidden_dim=128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.hidden_dim = hidden_dim
        
        print(f"Creating ExactSAC Expert:")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Device: {device}")
        print(f"  Hidden dim: {hidden_dim}")
        
        # 네트워크들 - 정확한 차원으로
        self.actor = ExactDimensionNetwork(state_dim, action_dim * 2, hidden_dim).to(device)
        self.critic1 = ExactDimensionNetwork(state_dim + action_dim, 1, hidden_dim).to(device)
        self.critic2 = ExactDimensionNetwork(state_dim + action_dim, 1, hidden_dim).to(device)
        
        # 타겟 네트워크들
        self.target_critic1 = ExactDimensionNetwork(state_dim + action_dim, 1, hidden_dim).to(device)
        self.target_critic2 = ExactDimensionNetwork(state_dim + action_dim, 1, hidden_dim).to(device)
        
        # 타겟 네트워크 초기화
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # 옵티마이저들
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=1e-4)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=1e-4)
        
        # 엔트로피 계수
        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4)
        self.target_entropy = -action_dim
        
        # 리플레이 버퍼
        self.replay_buffer = ExactReplayBuffer(5000, state_dim, action_dim)
        
        # 하이퍼파라미터
        self.gamma = 0.99
        self.tau = 0.005
        
        print(f"✓ ExactSAC Expert initialized successfully")
    
    def _safe_tensor_convert(self, x, expected_dim=None):
        """안전한 텐서 변환 with 차원 체크"""
        try:
            if isinstance(x, np.ndarray):
                x_flat = x.flatten()
            elif torch.is_tensor(x):
                x_flat = x.detach().cpu().numpy().flatten()
            else:
                x_flat = np.array(x, dtype=np.float32).flatten()
            
            # 차원 체크
            if expected_dim is not None and len(x_flat) != expected_dim:
                raise ValueError(f"Dimension mismatch: expected {expected_dim}, got {len(x_flat)}")
            
            tensor = torch.FloatTensor(x_flat).to(self.device)
            return tensor
            
        except Exception as e:
            print(f"Warning: Tensor conversion failed: {e}")
            if expected_dim:
                return torch.zeros(expected_dim, device=self.device)
            else:
                return torch.zeros(1, device=self.device)
    
    def select_action(self, state, deterministic=False):
        """행동 선택 - 차원 보장"""
        try:
            # 상태를 정확한 차원의 텐서로 변환
            state_tensor = self._safe_tensor_convert(state, self.state_dim)
            
            with torch.no_grad():
                # 액터 네트워크 통과
                actor_output = self.actor(state_tensor.unsqueeze(0))
                
                # 평균과 로그 표준편차 분리
                mean, log_std = actor_output.chunk(2, dim=-1)
                
                # 표준편차 클리핑
                log_std = torch.clamp(log_std, -20, 2)
                std = log_std.exp()
                
                if deterministic:
                    action = mean
                else:
                    # 가우시안 분포에서 샘플링
                    normal = torch.distributions.Normal(mean, std)
                    action = normal.sample()
                
                # tanh 활성화로 [-1, 1] 범위로 제한
                action = torch.tanh(action)
                
                # numpy로 변환
                action_np = action.squeeze(0).cpu().numpy()
                
                # 차원 최종 검증
                if len(action_np) != self.action_dim:
                    print(f"Warning: Action dimension mismatch in output: expected {self.action_dim}, got {len(action_np)}")
                    if len(action_np) < self.action_dim:
                        action_np = np.pad(action_np, (0, self.action_dim - len(action_np)))
                    else:
                        action_np = action_np[:self.action_dim]
                
                return action_np.astype(np.float32)
                
        except Exception as e:
            print(f"Warning: Action selection failed: {e}")
            # 안전한 기본 행동 반환
            return np.zeros(self.action_dim, dtype=np.float32)
    
    def update(self, batch_size=32):
        """네트워크 업데이트 - 차원 안전"""
        try:
            if len(self.replay_buffer) < batch_size:
                return None
            
            # 배치 샘플링
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
            
            # 텐서 변환 with 차원 검증
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.BoolTensor(dones).to(self.device)
            
            # 차원 검증
            assert states.size(-1) == self.state_dim, f"States dim: {states.size(-1)} != {self.state_dim}"
            assert actions.size(-1) == self.action_dim, f"Actions dim: {actions.size(-1)} != {self.action_dim}"
            assert next_states.size(-1) == self.state_dim, f"Next states dim: {next_states.size(-1)} != {self.state_dim}"
            
            # 현재 Q 값들
            current_q1 = self.critic1(torch.cat([states, actions], dim=1))
            current_q2 = self.critic2(torch.cat([states, actions], dim=1))
            
            # 다음 행동 샘플링
            with torch.no_grad():
                next_actor_output = self.actor(next_states)
                next_mean, next_log_std = next_actor_output.chunk(2, dim=-1)
                next_log_std = torch.clamp(next_log_std, -20, 2)
                next_std = next_log_std.exp()
                
                next_normal = torch.distributions.Normal(next_mean, next_std)
                next_action_sample = next_normal.sample()
                next_action = torch.tanh(next_action_sample)
                next_log_prob = next_normal.log_prob(next_action_sample) - torch.log(1 - next_action.pow(2) + 1e-6)
                next_log_prob = next_log_prob.sum(dim=1, keepdim=True)
                
                # 타겟 Q 값들
                target_q1 = self.target_critic1(torch.cat([next_states, next_action], dim=1))
                target_q2 = self.target_critic2(torch.cat([next_states, next_action], dim=1))
                target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob
                
                target_q_values = rewards.unsqueeze(1) + self.gamma * target_q * (~dones).unsqueeze(1)
            
            # Critic 손실
            critic1_loss = F.mse_loss(current_q1, target_q_values)
            critic2_loss = F.mse_loss(current_q2, target_q_values)
            
            # Critic 업데이트
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
            self.critic1_optimizer.step()
            
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
            self.critic2_optimizer.step()
            
            # Actor 업데이트
            actor_output = self.actor(states)
            mean, log_std = actor_output.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)
            std = log_std.exp()
            
            normal = torch.distributions.Normal(mean, std)
            action_sample = normal.sample()
            action = torch.tanh(action_sample)
            log_prob = normal.log_prob(action_sample) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)
            
            q1_new = self.critic1(torch.cat([states, action], dim=1))
            q2_new = self.critic2(torch.cat([states, action], dim=1))
            q_new = torch.min(q1_new, q2_new)
            
            actor_loss = (self.log_alpha.exp() * log_prob - q_new).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # Alpha 업데이트
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # 타겟 네트워크 소프트 업데이트
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
            import traceback
            traceback.print_exc()
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