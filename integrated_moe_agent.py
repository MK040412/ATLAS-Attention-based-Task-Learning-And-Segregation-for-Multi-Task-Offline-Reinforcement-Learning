import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any
from collections import deque
import random

class FusionEncoder(nn.Module):
    """명령어와 상태를 융합하는 인코더"""
    def __init__(self, instr_dim: int, state_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.instr_dim = instr_dim
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        # 직접 연결 후 MLP
        input_dim = instr_dim + state_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # 가중치 초기화
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, instr_emb, state_obs):
        """
        Args:
            instr_emb: [batch_size, instr_dim] 또는 [instr_dim]
            state_obs: [batch_size, state_dim] 또는 [state_dim]
        Returns:
            z: [batch_size, latent_dim] 또는 [latent_dim]
        """
        # 차원 정리
        original_batch = True
        
        if instr_emb.dim() == 3:  # [1, seq_len, dim]
            instr_emb = instr_emb.squeeze(0).mean(dim=0)  # [dim]
        elif instr_emb.dim() == 2:
            if instr_emb.size(0) == 1:  # [1, dim]
                instr_emb = instr_emb.squeeze(0)  # [dim]
            else:
                instr_emb = instr_emb.mean(dim=0)  # [dim]
        
        if state_obs.dim() == 2:
            if state_obs.size(0) == 1:  # [1, dim]
                state_obs = state_obs.squeeze(0)  # [dim]
            else:
                state_obs = state_obs[0]  # 첫 번째만 사용
        
        # 배치 차원 추가
        if instr_emb.dim() == 1:
            instr_emb = instr_emb.unsqueeze(0)  # [1, dim]
            original_batch = False
        if state_obs.dim() == 1:
            state_obs = state_obs.unsqueeze(0)  # [1, dim]
        
        # 연결 및 융합
        x = torch.cat([instr_emb, state_obs], dim=-1)  # [1, instr_dim + state_dim]
        z = self.mlp(x)  # [1, latent_dim]
        
        if not original_batch:
            z = z.squeeze(0)  # [latent_dim]
        
        return z

class SimpleDPMM:
    """간단한 DPMM 구현"""
    def __init__(self, latent_dim: int, alpha: float = 1.0, base_var: float = 1.0):
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.base_var = base_var
        
        self.clusters = []  # 각 클러스터의 중심점들
        self.cluster_counts = []  # 각 클러스터의 데이터 개수
        self.assignments = []  # 모든 데이터의 클러스터 할당
        
    def assign_and_update(self, z: np.ndarray) -> int:
        """새 데이터를 클러스터에 할당하고 업데이트"""
        z = np.array(z).flatten()
        
        if len(self.clusters) == 0:
            # 첫 번째 클러스터 생성
            self.clusters.append(z.copy())
            self.cluster_counts.append(1)
            self.assignments.append(0)
            return 0
        
        # 각 클러스터와의 거리 계산
        distances = []
        for cluster_center in self.clusters:
            dist = np.linalg.norm(z - cluster_center)
            distances.append(dist)
        
        # 가장 가까운 클러스터 찾기
        min_dist = min(distances)
        closest_cluster = distances.index(min_dist)
        
        # 새 클러스터 생성 여부 결정 (간단한 임계값 기반)
        threshold = 2.0  # 임계값
        
        if min_dist > threshold:
            # 새 클러스터 생성
            cluster_id = len(self.clusters)
            self.clusters.append(z.copy())
            self.cluster_counts.append(1)
        else:
            # 기존 클러스터에 할당
            cluster_id = closest_cluster
            self.cluster_counts[cluster_id] += 1
            
            # 클러스터 중심 업데이트 (이동 평균)
            alpha = 0.1
            self.clusters[cluster_id] = (1 - alpha) * self.clusters[cluster_id] + alpha * z
        
        self.assignments.append(cluster_id)
        return cluster_id
    
    def get_num_clusters(self) -> int:
        return len(self.clusters)
    
    def get_cluster_info(self) -> Dict[str, Any]:
        return {
            'num_clusters': len(self.clusters),
            'cluster_counts': self.cluster_counts.copy(),
            'total_assignments': len(self.assignments)
        }

class SimpleReplayBuffer:
    """간단한 리플레이 버퍼"""
    def __init__(self, capacity: int = 10000):
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
    
    def sample(self, batch_size: int):
        """배치 샘플링"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([exp['state'] for exp in batch])
        actions = np.array([exp['action'] for exp in batch])
        rewards = np.array([exp['reward'] for exp in batch])
        next_states = np.array([exp['next_state'] for exp in batch])
        dones = np.array([exp['done'] for exp in batch])
        instructions = np.array([exp['instruction_emb'] for exp in batch if exp['instruction_emb'] is not None])
        
        return states, actions, rewards, next_states, dones, instructions
    
    def __len__(self):
        return len(self.buffer)

class SimpleSACExpert:
    """간단한 SAC 전문가"""
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int, hidden_dim: int = 256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 액터 네트워크 (정책)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # mean과 log_std
        ).to(self.device)
        
        # 크리틱 네트워크 (가치 함수)
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        
        # 타겟 크리틱
        self.target_critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        
        # 타겟 네트워크 초기화
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 옵티마이저
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # SAC 파라미터
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -action_dim
        
        # 리플레이 버퍼
        self.replay_buffer = SimpleReplayBuffer(capacity=50000)
        
    def select_action(self, state, deterministic=False):
        """행동 선택"""
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).to(self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            actor_output = self.actor(state)
            mean, log_std = actor_output.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)
            
            if deterministic:
                action = mean
            else:
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                action = normal.sample()
            
            action = torch.tanh(action)  # [-1, 1] 범위로 제한
            
        return action.cpu().numpy().squeeze()
    
    def update(self, batch_size: int = 256):
        """네트워크 업데이트"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        # 배치 샘플링
        states, actions, rewards, next_states, dones, _ = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # 크리틱 업데이트
        with torch.no_grad():
            next_actor_output = self.actor(next_states)
            next_mean, next_log_std = next_actor_output.chunk(2, dim=-1)
            next_log_std = torch.clamp(next_log_std, -20, 2)
            next_std = next_log_std.exp()
            
            next_normal = torch.distributions.Normal(next_mean, next_std)
            next_actions = next_normal.sample()
            next_log_probs = next_normal.log_prob(next_actions).sum(dim=-1, keepdim=True)
            next_actions = torch.tanh(next_actions)
            
            next_q = self.target_critic(torch.cat([next_states, next_actions], dim=-1))
            target_q = rewards.unsqueeze(-1) + 0.99 * (1 - dones.unsqueeze(-1).float()) * (next_q - self.log_alpha.exp() * next_log_probs)
        
        current_q = self.critic(torch.cat([states, actions], dim=-1))
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 액터 업데이트
        actor_output = self.actor(states)
        mean, log_std = actor_output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        actions_new = normal.sample()
        log_probs = normal.log_prob(actions_new).sum(dim=-1, keepdim=True)
        actions_new = torch.tanh(actions_new)
        
        q_new = self.critic(torch.cat([states, actions_new], dim=-1))
        actor_loss = (self.log_alpha.exp() * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 알파 업데이트
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # 타겟 네트워크 소프트 업데이트
        tau = 0.005
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.log_alpha.exp().item()
        }

class IntegratedMoEAgent:
    """통합된 MoE 에이전트"""
    def __init__(self, state_dim: int, action_dim: int, instr_dim: int, latent_dim: int = 32, hidden_dim: int = 256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.instr_dim = instr_dim
        self.latent_dim = latent_dim
        
        # 융합 인코더
        self.fusion_encoder = FusionEncoder(
            instr_dim=instr_dim,
            state_dim=state_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # DPMM 클러스터링
        self.dpmm = SimpleDPMM(latent_dim=latent_dim)
        
        # 전문가들
        self.experts = []
        
        # 학습 통계
        self.step_count = 0
        self.episode_count = 0
        
    def _create_new_expert(self) -> SimpleSACExpert:
        """새로운 전문가 생성"""
        expert = SimpleSACExpert(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            latent_dim=self.latent_dim,
            hidden_dim=256
        )
        self.experts.append(expert)
        print(f"  🆕 Created new expert #{len(self.experts)}")
        return expert
    
    def select_action(self, state, instruction_emb, deterministic=False):
        """행동 선택"""
        # numpy를 tensor로 변환
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).to(self.device)
        else:
            state_tensor = state.to(self.device)
        
        if isinstance(instruction_emb, np.ndarray):
            instr_tensor = torch.FloatTensor(instruction_emb).to(self.device)
        else:
            instr_tensor = instruction_emb.to(self.device)
        
        # 융합된 특징 추출
        with torch.no_grad():
            z = self.fusion_encoder(instr_tensor, state_tensor)
            z_np = z.cpu().numpy()
        
        # 클러스터 할당
        cluster_id = self.dpmm.assign_and_update(z_np)
        
        # 전문가가 없으면 생성
        while cluster_id >= len(self.experts):
            self._create_new_expert()
        
        # 해당 전문가로부터 행동 선택
        expert = self.experts[cluster_id]
        action = expert.select_action(state, deterministic=deterministic)
        
        return action, cluster_id
    
    def store_experience(self, state, action, reward, next_state, done, instruction_emb, cluster_id):
        """경험 저장"""
        if cluster_id < len(self.experts):
            expert = self.experts[cluster_id]
            expert.replay_buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                instruction_emb=instruction_emb
            )
    
    def update_experts(self, batch_size: int = 64):
        """모든 전문가 업데이트"""
        update_results = {}
        
        for i, expert in enumerate(self.experts):
            if len(expert.replay_buffer) >= batch_size:
                result = expert.update(batch_size)
                if result:
                    update_results[f'expert_{i}'] = result
        
        self.step_count += 1
        return update_results
    
    def get_statistics(self):
        """통계 정보 반환"""
        cluster_info = self.dpmm.get_cluster_info()
        
        expert_buffer_sizes = [len(expert.replay_buffer) for expert in self.experts]
        
        return {
            'num_experts': len(self.experts),
            'num_clusters': cluster_info['num_clusters'],
            'cluster_counts': cluster_info['cluster_counts'],
            'expert_buffer_sizes': expert_buffer_sizes,
            'total_steps': self.step_count,
            'total_episodes': self.episode_count
        }
    
    def save_checkpoint(self, filepath: str):
        """체크포인트 저장"""
        checkpoint = {
            'fusion_encoder': self.fusion_encoder.state_dict(),
            'experts': [
                {
                    'actor': expert.actor.state_dict(),
                    'critic': expert.critic.state_dict(),
                    'target_critic': expert.target_critic.state_dict(),
                    'log_alpha': expert.log_alpha.item()
                }
                for expert in self.experts
            ],
            'dpmm_clusters': self.dpmm.clusters,
            'dpmm_counts': self.dpmm.cluster_counts,
            'step_count': self.step_count,
            'episode_count': self.episode_count
        }
        torch.save(checkpoint, filepath)
        print(f"✅ Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """체크포인트 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 융합 인코더 로드
        self.fusion_encoder.load_state_dict(checkpoint['fusion_encoder'])
        
        # 전문가들 로드
        self.experts = []
        for expert_data in checkpoint['experts']:
            expert = SimpleSACExpert(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                latent_dim=self.latent_dim
            )
            expert.actor.load_state_dict(expert_data['actor'])
            expert.critic.load_state_dict(expert_data['critic'])
            expert.target_critic.load_state_dict(expert_data['target_critic'])
            expert.log_alpha = torch.tensor(expert_data['log_alpha'], requires_grad=True, device=self.device)
            self.experts.append(expert)
        
        # DPMM 상태 로드
        self.dpmm.clusters = checkpoint['dpmm_clusters']
        self.dpmm.cluster_counts = checkpoint['dpmm_counts']
        
        # 통계 로드
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        
        print(f"✅ Checkpoint loaded from {filepath}")
        print(f"  Loaded {len(self.experts)} experts")
        print(f"  Step count: {self.step_count}")