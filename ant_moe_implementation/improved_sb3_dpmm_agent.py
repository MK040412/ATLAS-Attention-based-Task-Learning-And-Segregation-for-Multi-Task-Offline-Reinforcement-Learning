#!/usr/bin/env python3
"""
개선된 SB3-DPMM 에이전트 (환경 호환성 수정)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import random
import gymnasium as gym
from gymnasium.spaces import Box

# SB3 대신 간단한 SAC 구현 사용
class SimpleSACExpert:
    """간단한 SAC 전문가 (SB3 호환성 문제 해결)"""
    
    def __init__(self, state_dim: int, action_dim: int, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 액터 네트워크
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * 2)  # mean + log_std
        ).to(self.device)
        
        # 크리틱 네트워크
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)
        
        self.critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)
        
        # 타겟 네트워크
        self.target_critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)
        
        self.target_critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)
        
        # 타겟 네트워크 초기화
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # 옵티마이저
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=3e-4)
        
        # SAC 파라미터
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -action_dim
        
        # 리플레이 버퍼
        self.replay_buffer = deque(maxlen=50000)
        
    def predict(self, obs: np.ndarray, deterministic: bool = False):
        """행동 예측 (SB3 호환)"""
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(self.device)
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        with torch.no_grad():
            actor_output = self.actor(obs)
            mean, log_std = actor_output.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)
            
            if deterministic:
                action = mean
            else:
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                action = normal.sample()
            
            action = torch.tanh(action)
        
        return action.cpu().numpy(), None
    
    def learn(self, total_timesteps: int = 1, log_interval: Optional[int] = None):
        """학습 (SB3 호환)"""
        if len(self.replay_buffer) < 256:
            return
        
        # 배치 샘플링
        batch = random.sample(self.replay_buffer, min(256, len(self.replay_buffer)))
        
        states = torch.FloatTensor([exp['state'] for exp in batch]).to(self.device)
        actions = torch.FloatTensor([exp['action'] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp['done'] for exp in batch]).to(self.device)
        
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
            
            next_q1 = self.target_critic1(torch.cat([next_states, next_actions], dim=-1))
            next_q2 = self.target_critic2(torch.cat([next_states, next_actions], dim=-1))
            next_q = torch.min(next_q1, next_q2)
            
            target_q = rewards.unsqueeze(-1) + 0.99 * (1 - dones.unsqueeze(-1).float()) * (next_q - self.log_alpha.exp() * next_log_probs)
        
        current_q1 = self.critic1(torch.cat([states, actions], dim=-1))
        current_q2 = self.critic2(torch.cat([states, actions], dim=-1))
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # 액터 업데이트
        actor_output = self.actor(states)
        mean, log_std = actor_output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        actions_new = normal.sample()
        log_probs = normal.log_prob(actions_new).sum(dim=-1, keepdim=True)
        actions_new = torch.tanh(actions_new)
        
        q1_new = self.critic1(torch.cat([states, actions_new], dim=-1))
        q2_new = self.critic2(torch.cat([states, actions_new], dim=-1))
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
        
        # 타겟 네트워크 소프트 업데이트
        tau = 0.005
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

class AdaptiveBirthDPMM:
    """적응적 Birth 제어 DPMM (이전과 동일)"""
    
    def __init__(self, latent_dim: int, initial_tau_birth: float = -2.0, 
                 tau_merge_kl: float = 0.3, initial_variance: float = 0.1,
                 max_clusters: int = 15, target_birth_rate: float = 0.12,
                 adaptive_birth: bool = True):
        
        self.latent_dim = latent_dim
        self.tau_birth = initial_tau_birth
        self.initial_tau_birth = initial_tau_birth
        self.tau_merge_kl = tau_merge_kl
        self.initial_variance = initial_variance
        self.max_clusters = max_clusters
        self.target_birth_rate = target_birth_rate
        self.adaptive_birth = adaptive_birth
        
        # 클러스터 저장
        self.clusters = {}
        self.next_cluster_id = 0
        
        # 이벤트 추적
        self.assignments = []
        self.birth_events = []
        self.merge_events = []
        
        # 적응적 제어를 위한 통계
        self.recent_assignments = deque(maxlen=50)
        self.recent_births = deque(maxlen=50)
        
        print(f"🧠 Adaptive Birth DPMM initialized:")
        print(f"  • Latent dim: {latent_dim}")
        print(f"  • Initial tau_birth: {initial_tau_birth}")
        print(f"  • Target birth rate: {target_birth_rate:.1%}")
        print(f"  • Max clusters: {max_clusters}")
    
    def _gaussian_log_likelihood(self, z: np.ndarray, cluster: Dict) -> float:
        """가우시안 로그 우도 계산"""
        diff = z - cluster['mu']
        log_likelihood = -0.5 * np.sum(diff**2) / cluster['sigma_sq']
        log_likelihood -= 0.5 * self.latent_dim * np.log(2 * np.pi * cluster['sigma_sq'])
        return log_likelihood
    
    def _update_tau_birth(self):
        """적응적 tau_birth 업데이트"""
        if not self.adaptive_birth or len(self.recent_assignments) < 20:
            return
        
        recent_birth_rate = sum(self.recent_births) / len(self.recent_births)
        
        if recent_birth_rate > self.target_birth_rate * 1.5:
            self.tau_birth -= 0.1
        elif recent_birth_rate < self.target_birth_rate * 0.5:
            self.tau_birth += 0.05
        
        self.tau_birth = max(-4.0, min(-1.0, self.tau_birth))
    
    def assign_point(self, z: np.ndarray) -> int:
        """포인트를 클러스터에 할당"""
        z = np.array(z).flatten()
        
        if len(self.clusters) == 0:
            cluster_id = self._create_new_cluster(z)
            self._record_assignment(cluster_id, is_birth=True)
            return cluster_id
        
        max_log_likelihood = -np.inf
        best_cluster_id = None
        
        for cluster_id, cluster in self.clusters.items():
            log_likelihood = self._gaussian_log_likelihood(z, cluster)
            if log_likelihood > max_log_likelihood:
                max_log_likelihood = log_likelihood
                best_cluster_id = cluster_id
        
        should_birth = (
            max_log_likelihood < self.tau_birth and 
            len(self.clusters) < self.max_clusters
        )
        
        if should_birth:
            cluster_id = self._create_new_cluster(z)
            self._record_assignment(cluster_id, is_birth=True)
            if self.adaptive_birth:
                self._update_tau_birth()
            return cluster_id
        else:
            self._update_cluster(best_cluster_id, z)
            self._record_assignment(best_cluster_id, is_birth=False)
            return best_cluster_id
    
    def _create_new_cluster(self, z: np.ndarray) -> int:
        """새 클러스터 생성"""
        cluster_id = self.next_cluster_id
        self.next_cluster_id += 1
        
        self.clusters[cluster_id] = {
            'mu': z.copy(),
            'sigma_sq': self.initial_variance,
            'N': 1
        }
        
        self.birth_events.append({
            'timestep': len(self.assignments),
            'cluster_id': cluster_id,
            'tau_birth': self.tau_birth
        })
        
        return cluster_id
    
    def _update_cluster(self, cluster_id: int, z: np.ndarray):
        """클러스터 온라인 업데이트"""
        cluster = self.clusters[cluster_id]
        cluster['N'] += 1
        alpha = 1.0 / cluster['N']
        cluster['mu'] = (1 - alpha) * cluster['mu'] + alpha * z
    
    def _record_assignment(self, cluster_id: int, is_birth: bool):
        """할당 기록"""
        self.assignments.append(cluster_id)
        self.recent_assignments.append(cluster_id)
        self.recent_births.append(1 if is_birth else 0)
    
    def merge_clusters_aggressive(self, verbose: bool = False) -> int:
        """적극적 클러스터 병합"""
        if len(self.clusters) < 2:
            return 0
        
        merged_count = 0
        cluster_ids = list(self.clusters.keys())
        
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                id1, id2 = cluster_ids[i], cluster_ids[j]
                
                if id1 not in self.clusters or id2 not in self.clusters:
                    continue
                
                cluster1 = self.clusters[id1]
                cluster2 = self.clusters[id2]
                
                distance = np.linalg.norm(cluster1['mu'] - cluster2['mu'])
                
                if distance < self.tau_merge_kl:
                    self._merge_clusters(id1, id2)
                    merged_count += 1
                    
                    if verbose:
                        print(f"    🔗 Merged clusters {id1} ↔ {id2} (distance: {distance:.3f})")
        
        return merged_count
    
    def _merge_clusters(self, id1: int, id2: int):
        """두 클러스터 병합"""
        cluster1 = self.clusters[id1]
        cluster2 = self.clusters[id2]
        
        total_N = cluster1['N'] + cluster2['N']
        new_mu = (cluster1['N'] * cluster1['mu'] + cluster2['N'] * cluster2['mu']) / total_N
        
        keep_id = min(id1, id2)
        remove_id = max(id1, id2)
        
        self.clusters[keep_id] = {
            'mu': new_mu,
            'sigma_sq': self.initial_variance,
            'N': total_N
        }
        
        del self.clusters[remove_id]
        
        self.merge_events.append({
            'timestep': len(self.assignments),
            'merged_clusters': [id1, id2],
            'new_cluster': keep_id
        })
    
    def emergency_cluster_reduction(self) -> int:
        """응급 클러스터 감소"""
        if len(self.clusters) <= 3:
            return 0
        
        cluster_items = list(self.clusters.items())
        cluster_items.sort(key=lambda x: x[1]['N'])
        
        merged_count = 0
        target_reductions = min(3, len(self.clusters) // 3)
        
        for i in range(target_reductions):
            if len(self.clusters) <= 3:
                break
                
            smallest_id = cluster_items[i][0]
            if smallest_id not in self.clusters:
                continue
            
            smallest_cluster = self.clusters[smallest_id]
            min_distance = float('inf')
            closest_id = None
            
            for other_id, other_cluster in self.clusters.items():
                if other_id == smallest_id:
                    continue
                distance = np.linalg.norm(smallest_cluster['mu'] - other_cluster['mu'])
                if distance < min_distance:
                    min_distance = distance
                    closest_id = other_id
            
            if closest_id is not None:
                self._merge_clusters(smallest_id, closest_id)
                merged_count += 1
        
        return merged_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        recent_birth_rate = sum(self.recent_births) / max(1, len(self.recent_births))
        
        return {
            'num_clusters': len(self.clusters),
            'total_assignments': len(self.assignments),
            'total_births': len(self.birth_events),
            'total_merges': len(self.merge_events),
            'recent_birth_rate': recent_birth_rate,
            'current_tau_birth': self.tau_birth,
            'cluster_sizes': {cid: cluster['N'] for cid, cluster in self.clusters.items()}
        }

class ImprovedSB3DPMMAgent:
    """개선된 SB3-DPMM 통합 에이전트 (환경 호환성 수정)"""
    
    def __init__(self, env = None, state_dim: int = None, action_dim: int = None,
                 instr_dim: int = 384, latent_dim: int = 24,
                 tau_birth: float = -2.0, tau_merge_kl: float = 0.3,
                 initial_variance: float = 0.1, max_clusters: int = 15,
                 target_birth_rate: float = 0.12, device: str = "cuda"):
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 환경에서 차원 추출 (수정됨)
        if env is not None:
            try:
                # 환경 리셋해서 실제 차원 확인
                dummy_obs, _ = env.reset()
                if isinstance(dummy_obs, np.ndarray):
                    if len(dummy_obs.shape) == 1:
                        self.state_dim = dummy_obs.shape[0]
                    else:
                        self.state_dim = dummy_obs.shape[-1]  # 마지막 차원
                else:
                    self.state_dim = len(dummy_obs)
                
                # 액션 차원
                if hasattr(env, 'action_space'):
                    if hasattr(env.action_space, 'shape'):
                        self.action_dim = env.action_space.shape[-1]
                    else:
                        self.action_dim = env.action_space.n
                else:
                    self.action_dim = 8  # 기본값
                    
            except Exception as e:
                print(f"⚠️ Environment dimension extraction failed: {e}")
                self.state_dim = state_dim or 1926  # 관측된 값 사용
                self.action_dim = action_dim or 8
        else:
            self.state_dim = state_dim or 1926
            self.action_dim = action_dim or 8
        
        self.instr_dim = instr_dim
        self.latent_dim = latent_dim
        
        print(f"🤖 ImprovedSB3DPMMAgent initializing:")
        print(f"  • State dim: {self.state_dim}")
        print(f"  • Action dim: {self.action_dim}")
        print(f"  • Instruction dim: {self.instr_dim}")
        print(f"  • Latent dim: {self.latent_dim}")
        
        # 융합 인코더
        self.fusion_encoder = self._create_fusion_encoder()
        
        # 개선된 DPMM
        self.dpmm = AdaptiveBirthDPMM(
            latent_dim=latent_dim,
            initial_tau_birth=tau_birth,
            tau_merge_kl=tau_merge_kl,
            initial_variance=initial_variance,
            max_clusters=max_clusters,
            target_birth_rate=target_birth_rate,
            adaptive_birth=True
        )
        
        # SAC 전문가들 (수정된 버전)
        self.experts = {}
        
        # 학습 통계
        self.step_count = 0
        self.episode_count = 0
        
    def _create_fusion_encoder(self) -> nn.Module:
        """융합 인코더 생성"""
        class FusionEncoder(nn.Module):
            def __init__(self, state_dim, instr_dim, latent_dim):
                super().__init__()
                
                # 상태 차원이 클 경우 압축
                if state_dim > 512:
                    self.state_proj = nn.Sequential(
                        nn.Linear(state_dim, 512),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128)
                    )
                else:
                    self.state_proj = nn.Linear(state_dim, 128)
                
                self.instr_proj = nn.Linear(instr_dim, 128)
                
                self.fusion = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, latent_dim),
                    nn.Tanh()
                )
                
            def forward(self, state, instruction):
                # 차원 정리
                if state.dim() > 1:
                    state = state.flatten()
                if instruction.dim() > 1:
                    instruction = instruction.flatten()
                
                # 배치 차원 추가
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                if instruction.dim() == 1:
                    instruction = instruction.unsqueeze(0)
                
                # 프로젝션
                state_emb = self.state_proj(state)
                if state_emb.dim() > 2:
                    state_emb = F.relu(state_emb.squeeze())
                else:
                    state_emb = F.relu(state_emb)
                    
                instr_emb = F.relu(self.instr_proj(instruction))
                
                # 차원 맞추기
                if state_emb.dim() == 1:
                    state_emb = state_emb.unsqueeze(0)
                if instr_emb.dim() == 1:
                    instr_emb = instr_emb.unsqueeze(0)
                
                # 융합
                combined = torch.cat([state_emb, instr_emb], dim=-1)
                latent = self.fusion(combined)
                
                return latent.squeeze(0) if latent.size(0) == 1 else latent
        
        encoder = FusionEncoder(self.state_dim, self.instr_dim, self.latent_dim)
        return encoder.to(self.device)
    
    def _create_expert(self, cluster_id: int) -> None:
        """새로운 SAC 전문가 생성 (수정됨)"""
        expert = SimpleSACExpert(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=str(self.device)
        )
        
        self.experts[cluster_id] = expert
        print(f"  🆕 Created SimpleSAC expert for cluster {cluster_id}")
    
    def select_action(self, state: np.ndarray, instruction_emb: np.ndarray, 
                     deterministic: bool = False) -> Tuple[np.ndarray, int]:
        """행동 선택 (수정됨)"""
        # Tensor 변환
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.FloatTensor(state).to(self.device)
        else:
            state_tensor = state.to(self.device)
        
        if not isinstance(instruction_emb, torch.Tensor):
            instr_tensor = torch.FloatTensor(instruction_emb).to(self.device)
        else:
            instr_tensor = instruction_emb.to(self.device)
        
        # 융합된 잠재 표현 생성
        try:
            with torch.no_grad():
                latent = self.fusion_encoder(state_tensor, instr_tensor)
                latent_np = latent.cpu().numpy()
        except Exception as e:
            print(f"⚠️ Fusion encoder error: {e}")
            # 폴백: 랜덤 잠재 벡터
            latent_np = np.random.randn(self.latent_dim)
        
        # DPMM 클러스터 할당
        cluster_id = self.dpmm.assign_point(latent_np)
        
        # 전문가가 없으면 생성
        if cluster_id not in self.experts:
            self._create_expert(cluster_id)
        
        # 해당 전문가로부터 행동 선택
        expert = self.experts[cluster_id]
        
        try:
            # 상태 차원 정리
            if len(state.shape) > 1:
                obs = state.flatten().reshape(1, -1)
            else:
                obs = state.reshape(1, -1)
            
            action, _ = expert.predict(obs, deterministic=deterministic)
            
            if len(action.shape) > 1:
                action = action.squeeze(0)
                
        except Exception as e:
            print(f"⚠️ Expert prediction error: {e}")
            # 폴백: 랜덤 액션
            action = np.random.uniform(-1, 1, self.action_dim)
        
        return action, cluster_id
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, 
                        done: bool, cluster_id: int):
        """경험 저장 (수정됨)"""
        if cluster_id in self.experts:
            expert = self.experts[cluster_id]
            
            # 차원 정리
            if len(state.shape) > 1:
                state = state.flatten()
            if len(next_state.shape) > 1:
                next_state = next_state.flatten()
            if len(action.shape) > 1:
                action = action.flatten()
            
            experience = {
                'state': state.astype(np.float32),
                'action': action.astype(np.float32),
                'reward': float(reward),
                'next_state': next_state.astype(np.float32),
                'done': bool(done)
            }
            
            expert.replay_buffer.append(experience)
    
    def update_experts(self, batch_size: int = 256) -> Dict[str, Any]:
        """모든 전문가 업데이트"""
        update_results = {}
        
        for cluster_id, expert in self.experts.items():
            buffer_size = len(expert.replay_buffer)
            
            if buffer_size >= batch_size:
                try:
                    expert.learn(total_timesteps=1, log_interval=None)
                    update_results[f'expert_{cluster_id}'] = {
                        'buffer_size': buffer_size,
                        'updated': True
                    }
                except Exception as e:
                    update_results[f'expert_{cluster_id}'] = {
                        'buffer_size': buffer_size,
                        'updated': False,
                        'error': str(e)
                    }
            else:
                update_results[f'expert_{cluster_id}'] = {
                    'buffer_size': buffer_size,
                    'updated': False,
                    'reason': 'insufficient_data'
                }
        
        self.step_count += 1
        return update_results
    
    def merge_clusters_aggressive(self, verbose: bool = False) -> int:
        """적극적 클러스터 병합"""
        old_cluster_count = len(self.dpmm.clusters)
        merged_count = self.dpmm.merge_clusters_aggressive(verbose=verbose)
        
        if merged_count > 0:
            # 병합된 클러스터의 전문가들 정리
            self._cleanup_merged_experts()
        
        return merged_count
    
    def emergency_cluster_reduction(self) -> int:
        """응급 클러스터 감소"""
        old_cluster_count = len(self.dpmm.clusters)
        merged_count = self.dpmm.emergency_cluster_reduction()
        
        if merged_count > 0:
            self._cleanup_merged_experts()
            print(f"  🚨 Emergency: Reduced {old_cluster_count} → {len(self.dpmm.clusters)} clusters")
        
        return merged_count
    
    def _cleanup_merged_experts(self):
        """병합된 클러스터의 전문가 정리"""
        active_clusters = set(self.dpmm.clusters.keys())
        expert_clusters = set(self.experts.keys())
        
        # 더 이상 존재하지 않는 클러스터의 전문가 제거
        for cluster_id in expert_clusters - active_clusters:
            if cluster_id in self.experts:
                del self.experts[cluster_id]
                print(f"  🗑️ Removed expert for merged cluster {cluster_id}")
    
    def get_dpmm_statistics(self) -> Dict[str, Any]:
        """DPMM 통계 반환"""
        return self.dpmm.get_statistics()
    
    def get_info(self) -> Dict[str, Any]:
        """전체 에이전트 정보 반환"""
        dpmm_stats = self.get_dpmm_statistics()
        
        buffer_sizes = {}
        total_buffer_size = 0
        for cluster_id, expert in self.experts.items():
            size = len(expert.replay_buffer)
            buffer_sizes[cluster_id] = size
            total_buffer_size += size
        
        return {
            'dpmm': dpmm_stats,
            'num_experts': len(self.experts),
            'expert_buffer_sizes': buffer_sizes,
            'total_buffer_size': total_buffer_size,
            'step_count': self.step_count,
            'episode_count': self.episode_count
        }
    
    def save(self, filepath: str):
        """모델 저장"""
        save_data = {
            'fusion_encoder': self.fusion_encoder.state_dict(),
            'dpmm_clusters': self.dpmm.clusters,
            'dpmm_stats': self.dpmm.get_statistics(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'instr_dim': self.instr_dim,
                'latent_dim': self.latent_dim
            }
        }
        
        # SAC 전문가들 저장
        expert_models = {}
        for cluster_id, expert in self.experts.items():
            expert_models[cluster_id] = {
                'actor': expert.actor.state_dict(),
                'critic1': expert.critic1.state_dict(),
                'critic2': expert.critic2.state_dict(),
                'target_critic1': expert.target_critic1.state_dict(),
                'target_critic2': expert.target_critic2.state_dict(),
                'log_alpha': expert.log_alpha.item()
            }
        save_data['experts'] = expert_models
        
        torch.save(save_data, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load(self, filepath: str):
        """모델 로드"""
        save_data = torch.load(filepath, map_location=self.device)
        
        # 융합 인코더 로드
        self.fusion_encoder.load_state_dict(save_data['fusion_encoder'])
        
        # DPMM 상태 로드
        self.dpmm.clusters = save_data['dpmm_clusters']
        
        # 통계 로드
        self.step_count = save_data['step_count']
        self.episode_count = save_data['episode_count']
        
        # 전문가들 재생성 및 로드
        expert_models = save_data.get('experts', {})
        for cluster_id, expert_data in expert_models.items():
            self._create_expert(int(cluster_id))
            expert = self.experts[int(cluster_id)]
            
            expert.actor.load_state_dict(expert_data['actor'])
            expert.critic1.load_state_dict(expert_data['critic1'])
            expert.critic2.load_state_dict(expert_data['critic2'])
            expert.target_critic1.load_state_dict(expert_data['target_critic1'])
            expert.target_critic2.load_state_dict(expert_data['target_critic2'])
            expert.log_alpha = torch.tensor(expert_data['log_alpha'], requires_grad=True, device=self.device)
        
        print(f"✅ Model loaded from {filepath}")
        print(f"  • Loaded {len(self.experts)} experts")
        print(f"  • Step count: {self.step_count}")

# 테스트 및 디버깅 함수
def test_improved_agent():
    """개선된 에이전트 테스트"""
    print("🧪 Testing ImprovedSB3DPMMAgent...")
    
    # 더미 환경 생성
    class DummyEnv:
        def __init__(self):
            self.observation_space = Box(low=-1, high=1, shape=(1926,))
            self.action_space = Box(low=-1, high=1, shape=(8,))
        
        def reset(self):
            return np.random.randn(1926), {}
        
        def step(self, action):
            return np.random.randn(1926), np.random.randn(), False, False, {}
    
    env = DummyEnv()
    
    # 에이전트 생성
    agent = ImprovedSB3DPMMAgent(
        env=env,
        instr_dim=384,
        latent_dim=16,
        tau_birth=-2.0,
        max_clusters=5
    )
    
    # 테스트 실행
    state = np.random.randn(1926)
    instruction = np.random.randn(384)
    
    print("  🎯 Testing action selection...")
    action, cluster_id = agent.select_action(state, instruction)
    print(f"    • Action shape: {action.shape}")
    print(f"    • Cluster ID: {cluster_id}")
    
    print("  💾 Testing experience storage...")
    next_state = np.random.randn(1926)
    agent.store_experience(state, action, 1.0, next_state, False, cluster_id)
    
    print("  📊 Testing statistics...")
    stats = agent.get_info()
    print(f"    • Clusters: {stats['dpmm']['num_clusters']}")
    print(f"    • Experts: {stats['num_experts']}")
    
    print("✅ Test completed successfully!")

if __name__ == "__main__":
    test_improved_agent()
