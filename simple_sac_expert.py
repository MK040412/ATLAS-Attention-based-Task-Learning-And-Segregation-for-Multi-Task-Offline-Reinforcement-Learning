import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from simple_replay_buffer import SimpleReplayBuffer

class SimpleActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
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
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

class SimpleCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
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

class SimpleSACExpert:
    def __init__(self, state_dim, action_dim, latent_dim=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = SimpleActor(latent_dim, action_dim).to(self.device)
        self.critic = SimpleCritic(latent_dim, action_dim).to(self.device)
        self.critic_target = SimpleCritic(latent_dim, action_dim).to(self.device)
        
        # 타겟 네트워크 초기화
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 옵티마이저
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # 엔트로피 계수
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -action_dim
        
        # 리플레이 버퍼
        self.replay_buffer = SimpleReplayBuffer(capacity=10000)
        
        self.gamma = 0.99
        self.tau = 0.005
    
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            if not torch.is_tensor(state):
                state = torch.FloatTensor(state).to(self.device)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            if deterministic:
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state)
            
            return action.cpu().numpy().squeeze(0)
    
    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return None
        
        try:
            # 배치 샘플링
            states, actions, rewards, next_states, dones, _ = self.replay_buffer.sample(batch_size)
            
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.BoolTensor(dones).to(self.device).unsqueeze(1)
            
            # 크리틱 업데이트
            with torch.no_grad():
                next_actions, next_log_probs = self.actor.sample(next_states)
                q1_next, q2_next = self.critic_target(next_states, next_actions)
                q_next = torch.min(q1_next, q2_next) - self.log_alpha.exp() * next_log_probs
                q_target = rewards + self.gamma * q_next * (~dones)
            
            q1, q2 = self.critic(states, actions)
            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # 액터 업데이트
            new_actions, log_probs = self.actor.sample(states)
            q1_new, q2_new = self.critic(states, new_actions)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.log_alpha.exp() * log_probs - q_new).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 알파 업데이트
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # 타겟 네트워크 업데이트
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            return {
                'critic_loss': critic_loss.item(),
                'actor_loss': actor_loss.item(),
                'alpha_loss': alpha_loss.item(),
                'alpha': self.log_alpha.exp().item()
            }
            
        except Exception as e:
            print(f"Warning: Expert update failed: {e}")
            return None
    
    def get_state_dict(self):
        """모델 상태 반환"""
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha.item(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """모델 상태 로드"""
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_target.load_state_dict(state_dict['critic_target'])
        self.log_alpha = torch.tensor(state_dict['log_alpha'], requires_grad=True, device=self.device)
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        self.alpha_optimizer.load_state_dict(state_dict['alpha_optimizer'])