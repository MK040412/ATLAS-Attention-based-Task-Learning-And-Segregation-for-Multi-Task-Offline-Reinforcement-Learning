import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import itertools
from scipy.stats import multivariate_normal
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

class RTX4060OptimizedFusionEncoder(nn.Module):
    """RTX 4060 메모리 최적화된 Fusion Encoder"""
    
    def __init__(self, instr_dim: int, state_dim: int, latent_dim: int = 32):
        super().__init__()
        self.instr_dim = instr_dim
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        # 메모리 효율적인 작은 네트워크
        input_dim = instr_dim + state_dim
        hidden_dim = 128  # 256에서 128로 축소
        
        self.fusion_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),  # inplace로 메모리 절약
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # 가중치 초기화
        for layer in self.fusion_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)  # 작은 초기값
                nn.init.zeros_(layer.bias)
    
    def forward(self, instr_emb: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # 차원 정리
        if instr_emb.dim() > 2:
            instr_emb = instr_emb.mean(dim=-2)  # 시퀀스 차원 평균
        if instr_emb.dim() == 2 and instr_emb.size(0) == 1:
            instr_emb = instr_emb.squeeze(0)
            
        if state.dim() == 2 and state.size(0) == 1:
            state = state.squeeze(0)
        
        # 배치 차원 추가
        if instr_emb.dim() == 1:
            instr_emb = instr_emb.unsqueeze(0)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # 융합
        combined = torch.cat([instr_emb, state], dim=-1)
        z = self.fusion_net(combined)
        
        return z.squeeze(0) if z.size(0) == 1 else z

class KLDivergenceDPMM:
    """KL Divergence 기반 DPMM - 메모리 최적화"""
    
    def __init__(self, latent_dim: int = 32, alpha: float = 1.0, 
                 tau_birth: float = 1e-3, tau_merge_kl: float = 0.1):
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.tau_birth = tau_birth
        self.tau_merge_kl = tau_merge_kl
        
        self.clusters = []
        self.cluster_counts = []
        self.total_points = 0
        self.total_births = 0
        self.total_merges = 0
        
    def _create_cluster(self, z: np.ndarray) -> int:
        """새 클러스터 생성"""
        cluster_id = len(self.clusters)
        
        # 초기 공분산은 단위 행렬의 작은 배수
        initial_cov = np.eye(self.latent_dim) * 0.1
        
        cluster = {
            'mu': z.copy(),
            'cov': initial_cov,
            'count': 1,
            'sum_z': z.copy(),
            'sum_zz': np.outer(z, z)
        }
        
        self.clusters.append(cluster)
        self.cluster_counts.append(1)
        self.total_births += 1
        
        return cluster_id
    
    def _update_cluster(self, cluster_id: int, z: np.ndarray):
        """클러스터 통계 업데이트"""
        cluster = self.clusters[cluster_id]
        
        # 온라인 업데이트
        cluster['count'] += 1
        cluster['sum_z'] += z
        cluster['sum_zz'] += np.outer(z, z)
        
        # 평균 업데이트
        cluster['mu'] = cluster['sum_z'] / cluster['count']
        
        # 공분산 업데이트 (수치적 안정성 고려)
        if cluster['count'] > 1:
            mean_outer = np.outer(cluster['mu'], cluster['mu'])
            cov = (cluster['sum_zz'] / cluster['count']) - mean_outer
            
            # 정규화 항 추가 (수치적 안정성)
            cov += np.eye(self.latent_dim) * 1e-6
            cluster['cov'] = cov
        
        self.cluster_counts[cluster_id] = cluster['count']
    
    def _compute_log_likelihood(self, z: np.ndarray, cluster: dict) -> float:
        """가우시안 로그 우도 계산"""
        try:
            return multivariate_normal.logpdf(z, cluster['mu'], cluster['cov'])
        except:
            # 수치적 문제 시 낮은 우도 반환
            return -1e6
    
    def assign_and_update(self, z: np.ndarray) -> int:
        """클러스터 할당 및 업데이트"""
        z = np.array(z, dtype=np.float32)
        self.total_points += 1
        
        if len(self.clusters) == 0:
            return self._create_cluster(z)
        
        # 모든 클러스터에 대한 로그 우도 계산
        log_likelihoods = []
        for cluster in self.clusters:
            ll = self._compute_log_likelihood(z, cluster)
            log_likelihoods.append(ll)
        
        max_log_likelihood = max(log_likelihoods)
        
        # Birth criterion: 최대 우도가 임계값보다 낮으면 새 클러스터 생성
        if max_log_likelihood < np.log(self.tau_birth):
            return self._create_cluster(z)
        
        # 기존 클러스터에 할당
        best_cluster = np.argmax(log_likelihoods)
        self._update_cluster(best_cluster, z)
        
        return best_cluster
    
    def _symmetric_kl_divergence(self, mu1, cov1, mu2, cov2) -> float:
        """대칭 KL divergence 계산"""
        try:
            # KL(P||Q) + KL(Q||P) / 2
            inv_cov1 = np.linalg.inv(cov1)
            inv_cov2 = np.linalg.inv(cov2)
            
            diff = mu1 - mu2
            
            # KL(P||Q)
            kl_pq = 0.5 * (
                np.trace(inv_cov2 @ cov1) + 
                diff.T @ inv_cov2 @ diff - 
                self.latent_dim + 
                np.log(np.linalg.det(cov2) / np.linalg.det(cov1))
            )
            
            # KL(Q||P)  
            kl_qp = 0.5 * (
                np.trace(inv_cov1 @ cov2) + 
                diff.T @ inv_cov1 @ diff - 
                self.latent_dim + 
                np.log(np.linalg.det(cov1) / np.linalg.det(cov2))
            )
            
            return (kl_pq + kl_qp) / 2
        except:
            return np.inf
    
    def merge_clusters(self, verbose: bool = False) -> bool:
        """KL divergence 기반 클러스터 병합"""
        if len(self.clusters) < 2:
            return False
        
        min_kl = np.inf
        merge_pair = None
        
        # 모든 클러스터 쌍에 대해 KL divergence 계산
        for i, j in itertools.combinations(range(len(self.clusters)), 2):
            kl_div = self._symmetric_kl_divergence(
                self.clusters[i]['mu'], self.clusters[i]['cov'],
                self.clusters[j]['mu'], self.clusters[j]['cov']
            )
            
            if kl_div < min_kl:
                min_kl = kl_div
                merge_pair = (i, j)
        
        # 병합 조건 확인
        if min_kl < self.tau_merge_kl:
            i, j = merge_pair
            if verbose:
                print(f"  Merging clusters {i} and {j} (KL divergence: {min_kl:.4f})")
            
            self._merge_clusters(i, j)
            self.total_merges += 1
            return True
        
        return False
    
    def _merge_clusters(self, i: int, j: int):
        """두 클러스터 병합"""
        # j를 i로 병합 (i < j 가정)
        if i > j:
            i, j = j, i
        
        cluster_i = self.clusters[i]
        cluster_j = self.clusters[j]
        
        # 가중 평균으로 병합
        total_count = cluster_i['count'] + cluster_j['count']
        
        # 새로운 평균
        new_mu = (cluster_i['count'] * cluster_i['mu'] + 
                  cluster_j['count'] * cluster_j['mu']) / total_count
        
        # 새로운 공분산 (풀링된 공분산)
        new_cov = ((cluster_i['count'] - 1) * cluster_i['cov'] + 
                   (cluster_j['count'] - 1) * cluster_j['cov']) / (total_count - 2)
        
        # 클러스터 i 업데이트
        cluster_i['mu'] = new_mu
        cluster_i['cov'] = new_cov
        cluster_i['count'] = total_count
        cluster_i['sum_z'] = cluster_i['sum_z'] + cluster_j['sum_z']
        cluster_i['sum_zz'] = cluster_i['sum_zz'] + cluster_j['sum_zz']
        
        # 클러스터 j 제거
        del self.clusters[j]
        del self.cluster_counts[j]
    
    def get_statistics(self) -> Dict:
        """DPMM 통계 반환"""
        return {
            'num_clusters': len(self.clusters),
            'cluster_counts': self.cluster_counts.copy(),
            'total_points': self.total_points,
            'total_births': self.total_births,
            'total_merges': self.total_merges,
            'avg_cluster_size': np.mean(self.cluster_counts) if self.cluster_counts else 0
        }

class RTX4060SACExpert:
    """RTX 4060 최적화된 SAC Expert"""
    
    def __init__(self, state_dim: int, action_dim: int, expert_id: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.expert_id = expert_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 메모리 효율적인 작은 네트워크
        hidden_dim = 128  # 256에서 128로 축소
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim * 2)  # mean + log_std
        ).to(self.device)
        
        # Twin critic networks (메모리 절약을 위해 작게)
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        
        self.critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        
        # Target networks
        self.target_critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        
        self.target_critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        
        # 타겟 네트워크 초기화
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Entropy coefficient
        self.log_alpha = nn.Parameter(torch.zeros(1, device=self.device))
        self.target_entropy = -action_dim  # SAC 기본값
        
        # Optimizers (메모리 효율적인 설정)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=1e-4)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=1e-4)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4)
        
        # 작은 replay buffer (메모리 절약)
        self.replay_buffer = deque(maxlen=5000)
        self.tau = 0.005  # soft update coefficient
        
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """행동 선택"""
        with torch.no_grad():
            if not torch.is_tensor(state):
                state = torch.FloatTensor(state).to(self.device)
            
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            # Actor forward pass
            actor_output = self.actor(state)
            mean, log_std = actor_output.chunk(2, dim=-1)
            
            # 안정성을 위한 클리핑
            log_std = torch.clamp(log_std, -20, 2)
            std = log_std.exp()
            
            if deterministic:
                action = torch.tanh(mean)
            else:
                # 재매개화 트릭
                normal = torch.distributions.Normal(mean, std)
                x_t = normal.rsample()
                action = torch.tanh(x_t)
            
            return action.squeeze(0).cpu().numpy()
    
    def store_experience(self, state, action, reward, next_state, done):
        """경험 저장"""
        experience = {
            'state': np.array(state, dtype=np.float32),
            'action': np.array(action, dtype=np.float32),
            'reward': float(reward),
            'next_state': np.array(next_state, dtype=np.float32),
            'done': bool(done)
        }
        self.replay_buffer.append(experience)
    
    def sample_batch(self, batch_size: int = 32):
        """배치 샘플링"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        import random
        batch = random.sample(self.replay_buffer, batch_size)
        
        states = torch.FloatTensor([e['state'] for e in batch]).to(self.device)
        actions = torch.FloatTensor([e['action'] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e['reward'] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e['next_state'] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e['done'] for e in batch]).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def update(self, batch_size: int = 32) -> Optional[Dict]:
        """SAC 업데이트"""
        batch = self.sample_batch(batch_size)
        if batch is None:
            return None
        
        states, actions, rewards, next_states, dones = batch
        
        # Critic 업데이트
        with torch.no_grad():
            # 다음 상태에서의 행동 샘플링
            next_actor_output = self.actor(next_states)
            next_mean, next_log_std = next_actor_output.chunk(2, dim=-1)
            next_log_std = torch.clamp(next_log_std, -20, 2)
            next_std = next_log_std.exp()
            
            next_normal = torch.distributions.Normal(next_mean, next_std)
            next_x_t = next_normal.rsample()
            next_actions = torch.tanh(next_x_t)
            next_log_probs = next_normal.log_prob(next_x_t) - torch.log(1 - next_actions.pow(2) + 1e-6)
            next_log_probs = next_log_probs.sum(dim=-1, keepdim=True)
            
            # 타겟 Q 값
            target_q1 = self.target_critic1(torch.cat([next_states, next_actions], dim=-1))
            target_q2 = self.target_critic2(torch.cat([next_states, next_actions], dim=-1))
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            
            target_q = rewards.unsqueeze(-1) + 0.99 * target_q * (~dones).unsqueeze(-1)
        
        # 현재 Q 값
        current_q1 = self.critic1(torch.cat([states, actions], dim=-1))
        current_q2 = self.critic2(torch.cat([states, actions], dim=-1))
        
        # Critic 손실
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
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
        x_t = normal.rsample()
        actions_pred = torch.tanh(x_t)
        log_probs = normal.log_prob(x_t) - torch.log(1 - actions_pred.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1, keepdim=True)
        
        q1_pred = self.critic1(torch.cat([states, actions_pred], dim=-1))
        q2_pred = self.critic2(torch.cat([states, actions_pred], dim=-1))
        q_pred = torch.min(q1_pred, q2_pred)
        
        actor_loss = (self.alpha * log_probs - q_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Alpha 업데이트
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item()
        }
    
    def _soft_update(self, target, source):
        """소프트 업데이트"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

class IntegratedKLDPMMMoESAC:
    """통합된 KL-DPMM MoE-SAC 에이전트 - RTX 4060 최적화"""
    
    def __init__(self, state_dim: int, action_dim: int, instr_dim: int = 384,
                 latent_dim: int = 32, tau_birth: float = 1e-3, 
                 tau_merge_kl: float = 0.1, device: str = "cuda"):
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.instr_dim = instr_dim
        self.latent_dim = latent_dim
        
        print(f"🤖 Initializing Integrated KL-DPMM MoE-SAC Agent")
        print(f"  Device: {self.device}")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}")
        print(f"  Latent dim: {latent_dim}")
        print(f"  Birth threshold: {tau_birth}, Merge threshold: {tau_merge_kl}")
        
        # Pre-trained language model (경량화)
        print("📝 Loading pre-trained sentence transformer...")
        self.sentence_transformer = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            device=self.device
        )
        self.sentence_transformer.eval()  # 추론 모드로 고정
        
        # Fusion encoder
        self.fusion_encoder = RTX4060OptimizedFusionEncoder(
            instr_dim=instr_dim,
            state_dim=state_dim,
            latent_dim=latent_dim
        ).to(self.device)
        
        # KL-Divergence DPMM
        self.dpmm = KLDivergenceDPMM(
            latent_dim=latent_dim,
            tau_birth=tau_birth,
            tau_merge_kl=tau_merge_kl
        )
        
        # SAC Experts
        self.experts: Dict[int, RTX4060SACExpert] = {}
        self.max_experts = 8  # RTX 4060 메모리 제한
        
        # 학습 통계
        self.step_count = 0
        self.episode_count = 0
        
        # Fusion encoder optimizer
        self.fusion_optimizer = torch.optim.Adam(
            self.fusion_encoder.parameters(), lr=1e-4
        )
        
        print(f"✅ Agent initialized successfully!")
    
    def encode_instruction(self, instruction: str) -> torch.Tensor:
        """자연어 지시사항 인코딩"""
        with torch.no_grad():
            embedding = self.sentence_transformer.encode(
                instruction, 
                convert_to_tensor=True,
                device=self.device
            )
            return embedding
    
    def select_action(self, state: np.ndarray, instruction: str, 
                     deterministic: bool = False) -> Tuple[np.ndarray, int]:
        """행동 선택"""
        with torch.no_grad():
            # 지시사항 인코딩
            instr_emb = self.encode_instruction(instruction)
            
            # 상태 텐서 변환
            if not torch.is_tensor(state):
                state_tensor = torch.FloatTensor(state).to(self.device)
            else:
                state_tensor = state.to(self.device)
            
            # 융합된 잠재 표현
            latent_state = self.fusion_encoder(instr_emb, state_tensor)
            
            # DPMM 클러스터 할당
            cluster_id = self.dpmm.assign_and_update(latent_state.cpu().numpy())
            
            # 전문가 생성 (필요시)
            if cluster_id not in self.experts:
                if len(self.experts) >= self.max_experts:
                    print(f"⚠️ Maximum experts ({self.max_experts}) reached!")
                    cluster_id = min(self.experts.keys())  # 가장 오래된 전문가 사용
                else:
                    self._create_expert(cluster_id)
            
            # 행동 선택
            expert = self.experts[cluster_id]
            action = expert.select_action(state, deterministic)
            
            return action, cluster_id
    
    def _create_expert(self, expert_id: int):
        """새 전문가 생성"""
        print(f"  🧠 Creating new expert {expert_id}")
        expert = RTX4060SACExpert(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            expert_id=expert_id
        )
        self.experts[expert_id] = expert
    
    def store_experience(self, state, action, reward, next_state, done, 
                        instruction: str, cluster_id: int):
        """경험 저장"""
        if cluster_id in self.experts:
            expert = self.experts[cluster_id]
            expert.store_experience(state, action, reward, next_state, done)
    
    def update_experts(self, batch_size: int = 32) -> Dict:
        """모든 전문가 업데이트"""
        losses = {}
        updated_count = 0
        
        for expert_id, expert in self.experts.items():
            if len(expert.replay_buffer) >= batch_size:
                try:
                    expert_losses = expert.update(batch_size)
                    if expert_losses:
                        losses[f'expert_{expert_id}'] = expert_losses
                        updated_count += 1
                except Exception as e:
                    print(f"⚠️ Expert {expert_id} update failed: {e}")
        
        # Fusion encoder 업데이트 (선택적)
        if updated_count > 0 and self.step_count % 10 == 0:
            self._update_fusion_encoder()
        
        self.step_count += 1
        
        return {
            'updated_experts': updated_count,
            'total_experts': len(self.experts),
            'losses': losses
        }
    
    def _update_fusion_encoder(self):
        """Fusion encoder 업데이트 (간단한 버전)"""
        # 실제로는 더 복잡한 메타 학습 로직이 필요하지만,
        # RTX 4060 제약으로 인해 간단하게 구현
        pass  # 현재는 고정된 fusion encoder 사용
    
    def merge_clusters(self, verbose: bool = False) -> bool:
        """KL divergence 기반 클러스터 병합"""
        merged = self.dpmm.merge_clusters(verbose)
        
        if merged and verbose:
            # 전문가 병합은 복잡하므로 일단 유지
            # TODO: 실제 전문가 네트워크 병합 로직 추가
            print(f"  📊 Clusters merged, but experts maintained for stability")
        
        return merged
    
    def get_statistics(self) -> Dict:
        """에이전트 통계"""
        dpmm_stats = self.dpmm.get_statistics()
        
        expert_stats = {
            'total_experts': len(self.experts),
            'buffer_sizes': {
                expert_id: len(expert.replay_buffer) 
                for expert_id, expert in self.experts.items()
            },
            'expert_ids': list(self.experts.keys())
        }
        
        return {
            'dpmm': dpmm_stats,
            'experts': expert_stats,
            'step_count': self.step_count,
            'episode_count': self.episode_count
        }
    
    def save_checkpoint(self, filepath: str):
        """체크포인트 저장"""
        checkpoint = {
            'fusion_encoder': self.fusion_encoder.state_dict(),
            'experts': {
                expert_id: {
                    'actor': expert.actor.state_dict(),
                    'critic1': expert.critic1.state_dict(),
                    'critic2': expert.critic2.state_dict(),
                    'target_critic1': expert.target_critic1.state_dict(),
                    'target_critic2': expert.target_critic2.state_dict(),
                    'log_alpha': expert.log_alpha.item(),
                    'replay_buffer': list(expert.replay_buffer)
                }
                for expert_id, expert in self.experts.items()
            },
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
        
        torch.save(checkpoint, filepath)
        print(f"💾 Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """체크포인트 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Fusion encoder 로드
        self.fusion_encoder.load_state_dict(checkpoint['fusion_encoder'])
        
        # 전문가들 로드
        self.experts = {}
        for expert_id, expert_data in checkpoint['experts'].items():
            expert_id = int(expert_id)
            expert = RTX4060SACExpert(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                expert_id=expert_id
            )
            
            expert.actor.load_state_dict(expert_data['actor'])
            expert.critic1.load_state_dict(expert_data['critic1'])
            expert.critic2.load_state_dict(expert_data['critic2'])
            expert.target_critic1.load_state_dict(expert_data['target_critic1'])
            expert.target_critic2.load_state_dict(expert_data['target_critic2'])
            expert.log_alpha.data.fill_(expert_data['log_alpha'])
            
            # Replay buffer 복원
            expert.replay_buffer.extend(expert_data['replay_buffer'])
            
            self.experts[expert_id] = expert
        
        # 기타 상태 복원
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        
        print(f"📂 Checkpoint loaded from {filepath}")
        print(f"  Loaded {len(self.experts)} experts")
