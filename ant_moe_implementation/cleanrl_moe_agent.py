"""
CleanRL 기반 간단한 MoE 에이전트
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
import random
from collections import deque

class SimpleActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0):
        super().__init__()
        self.max_action = max_action
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.max_action * self.net(state)

class SimpleCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

class SimpleReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)

class SimpleSACExpert:
    """간단한 SAC 전문가"""
    def __init__(self, state_dim: int, action_dim: int, expert_id: int):
        self.expert_id = expert_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 네트워크들
        self.actor = SimpleActor(state_dim, action_dim).to(self.device)
        self.critic = SimpleCritic(state_dim, action_dim).to(self.device)
        self.target_critic = SimpleCritic(state_dim, action_dim).to(self.device)
        
        # 타겟 네트워크 초기화
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 옵티마이저
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # 리플레이 버퍼
        self.replay_buffer = SimpleReplayBuffer(10000)
        
        # 하이퍼파라미터
        self.gamma = 0.99
        self.tau = 0.005
        self.noise_std = 0.1
        
        print(f"  🧠 Created Simple SAC Expert {expert_id}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """행동 선택"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()[0]
            
            if not deterministic:
                # 탐험을 위한 노이즈 추가
                noise = np.random.normal(0, self.noise_std, size=action.shape)
                action = np.clip(action + noise, -1, 1)
            
            return action
    
    def update(self, batch_size: int = 32) -> Dict:
        """네트워크 업데이트"""
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # 배치 샘플링
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 크리틱 업데이트
        with torch.no_grad():
            next_actions = self.actor(next_states)
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones) * target_q
        
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 액터 업데이트
        new_actions = self.actor(states)
        actor_q1, actor_q2 = self.critic(states, new_actions)
        actor_loss = -torch.min(actor_q1, actor_q2).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 타겟 네트워크 소프트 업데이트
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }

class CleanRLMoEAgent:
    """CleanRL 스타일의 간단한 MoE 에이전트"""
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 간단한 클러스터링
        self.clusters = []
        self.cluster_threshold = 1.5
        self.max_clusters = 6
        
        # 전문가들
        self.experts: Dict[int, SimpleSACExpert] = {}
        
        # 언어 모델 (간단한 버전)
        from sentence_transformers import SentenceTransformer
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"🤖 CleanRL MoE Agent initialized:")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}")
    
    def _encode_instruction(self, instruction: str) -> np.ndarray:
        """지시사항 인코딩"""
        return self.sentence_model.encode(instruction)
    
    def _get_cluster_id(self, state: np.ndarray, instruction: str) -> int:
        """클러스터 ID 결정 (간단한 버전)"""
        # 지시사항과 상태의 간단한 조합
        instr_emb = self._encode_instruction(instruction)
        
        # 상태의 주요 특징 추출 (위치 정보)
        key_features = state[:6] if len(state) > 6 else state
        
        # 특징 벡터 생성
        feature_vector = np.concatenate([key_features, instr_emb[:6]])
        
        # 기존 클러스터와 비교
        if not self.clusters:
            self.clusters.append(feature_vector)
            return 0
        
        # 가장 가까운 클러스터 찾기
        distances = [np.linalg.norm(feature_vector - cluster) for cluster in self.clusters]
        min_distance = min(distances)
        closest_cluster = distances.index(min_distance)
        
        # 새 클러스터 생성 조건
        if min_distance > self.cluster_threshold and len(self.clusters) < self.max_clusters:
            self.clusters.append(feature_vector)
            return len(self.clusters) - 1
        else:
            return closest_cluster
    
    def select_action(self, state: np.ndarray, instruction: str, deterministic: bool = False):
        """행동 선택"""
        cluster_id = self._get_cluster_id(state, instruction)
        
        # 전문가 생성 (필요시)
        if cluster_id not in self.experts:
            self.experts[cluster_id] = SimpleSACExpert(
                self.state_dim, self.action_dim, cluster_id
            )
        
        # 행동 선택
        action = self.experts[cluster_id].select_action(state, deterministic)
        return action, cluster_id
    
    def store_experience(self, cluster_id: int, state, action, reward, next_state, done):
        """경험 저장"""
        if cluster_id in self.experts:
            self.experts[cluster_id].replay_buffer.add(state, action, reward, next_state, done)
    
    def update_experts(self, batch_size: int = 32) -> Dict:
        """모든 전문가 업데이트"""
        losses = {}
        for cluster_id, expert in self.experts.items():
            if len(expert.replay_buffer) >= batch_size:
                expert_losses = expert.update(batch_size)
                losses[f'expert_{cluster_id}'] = expert_losses
        return losses
    
    def get_stats(self) -> Dict:
        """통계 정보"""
        return {
            'num_clusters': len(self.clusters),
            'num_experts': len(self.experts),
            'buffer_sizes': {cid: len(expert.replay_buffer) for cid, expert in self.experts.items()}
        }