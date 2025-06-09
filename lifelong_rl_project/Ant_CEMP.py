"""
Complete implementation of "Preserving and combining knowledge in robotic lifelong reinforcement learning"
Single self-contained module following paper requirements exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
import copy
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
import time
import math
# =======================================
#   기본 설정 함수 추가
# =======================================

class SkillMonitor:
    """Skill 사용 패턴 모니터링"""
    
    def __init__(self):
        self.skill_history = []
        self.reward_history = []
        self.last_skill = None
        self.skill_duration = 0
        
    def update(self, skill_id, reward, step):
        """Skill 변화 모니터링"""
        if self.last_skill != skill_id:
            if self.last_skill is not None:
                print(f"  🔄 Skill Switch: {self.last_skill} → {skill_id} (Duration: {self.skill_duration} steps)")
            self.last_skill = skill_id
            self.skill_duration = 0
        
        self.skill_duration += 1
        self.skill_history.append(skill_id)
        self.reward_history.append(reward)
        
        # 간헐적으로 현재 상태 출력
        if step % 50 == 0:
            recent_skills = self.skill_history[-20:] if len(self.skill_history) >= 20 else self.skill_history
            recent_rewards = self.reward_history[-20:] if len(self.reward_history) >= 20 else self.reward_history
            
            print(f"  Step {step:4d} | Current Skill: {skill_id:2d} ({self.skill_duration:2d} steps) | "
                  f"Recent Avg Reward: {np.mean(recent_rewards):6.3f}")
    
    def get_stats(self):
        """현재 통계 반환"""
        if not self.skill_history:
            return {}
        
        unique_skills = len(set(self.skill_history))
        total_switches = len([i for i in range(1, len(self.skill_history)) 
                            if self.skill_history[i] != self.skill_history[i-1]])
        
        return {
            'unique_skills': unique_skills,
            'total_switches': total_switches,
            'current_skill': self.last_skill,
            'current_duration': self.skill_duration
        }
def get_default_config():
    """Ant 환경에 최적화된 개선된 설정"""
    return {
        'env_name': 'Ant-v5',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'task_count': 5,  # 🔥 5개 태스크로 설정
        'task_names': {
            0: "Forward",
            1: "Backward", 
            2: "Crab_Work",
            3: "Recover_Flip",
            4: "Stand_Stop"
        },
        'latent_dim': 32,  # 🔥 32 → 64 (더 표현력 있는 latent space)
        'init_experts': 5,
        'pmoe_hidden_dim': 256,  # 🔥 128 → 256 (더 큰 네트워크)
        
        # ====== 학습률 개선 ======
        'learning_rate': 3e-4,  # 🔥 3e-4 → 1e-3 (더 빠른 학습)
        'actor_lr': 1e-3,
        'critic_lr': 3e-3,      # 🔥 critic은 더 빠르게
        'encoder_lr': 1e-3,
        
        # ====== 배치 및 버퍼 개선 ======
        'batch_size': 256,      # 🔥 64 → 256 (더 안정적인 학습)
        'buffer_size': 200000,  # 🔥 100000 → 200000
        'buf_per_expert': 100000,  # 🔥 50000 → 100000
        'min_buffer_size': 10000,  # 🔥 새로 추가 (최소 버퍼 크기)
        
        # ====== 학습 빈도 조정 ======
        'train_freq': 4,        # 🔥 1 → 4 (4 step마다 학습)
        'warmup_steps': 15000,  # 🔥 새로 추가 (초기 random exploration)
        'gradient_steps': 1,    # 🔥 학습 시 gradient step 수
        
        # ====== SAC 파라미터 개선 ======
        'gamma': 0.99,
        'tau': 0.01,           # 🔥 0.005 → 0.01 (더 빠른 target update)
        'target_entropy_ratio': 0.5,  # 🔥 새로 추가 (-0.5 * action_dim)
        'alpha_lr': 3e-4,      # 🔥 temperature parameter learning rate
        
        # ====== DPMM 파라미터 개선 ======
        'dp_concentration': 5.0,  # 🔥 1.0 → 5.0 (더 많은 클러스터 허용)
        'cluster_merge_threshold': 0.1,  # 🔥 1e-4 → 0.1 (덜 자주 병합)
        'cluster_min_points': 50,  # 🔥 30 → 50
        'dpmm_freq': 1000,     # 🔥 500 → 1000 (덜 자주 업데이트)
        
        # ====== 보조 손실 가중치 조정 ======
        'lambda_vb': 0.01,     # 🔥 0.1 → 0.01 (DPMM loss 비중 줄임)
        'lambda_rec': 0.1,     # 유지
        'lambda_dyn': 0.05,    # 🔥 0.1 → 0.05 (dynamics loss 비중 줄임)
        'lambda_distill': 0.01, # 🔥 0.1 → 0.01 (distillation loss 비중 줄임)
        
        # ====== 평가 및 로깅 ======
        'eval_freq': 5000,     # 🔥 1500 → 5000
        'log_freq': 1000,      # 🔥 새로 추가
        'save_freq': 10000,    # 🔥 새로 추가
        
        # ====== 학습 단계 ======
        'total_steps': 100000,  # 🔥 10000 → 100000 (충분한 학습 시간)
        'snapshot_freq': 10000, # 🔥 시각화 스냅샷 주기
        
        # ====== Exploration 개선 ======
        'exploration_noise': 0.1,  # 🔥 새로 추가
        'action_noise_clip': 0.5,  # 🔥 새로 추가
        
        'seed': 42
    }



# =======================================
#   DPMM 클래스 추가
# =======================================
class DirichletProcessMM:
    """Simplified DPMM for clustering"""
    
    def __init__(self, concentration, latent_dim, device='cpu'):
        self.concentration = concentration
        self.latent_dim = latent_dim
        self.device = device
        self.clusters = {}  # cluster_id -> {'mean': tensor, 'count': int}
        self.next_cluster_id = 0
    
    def chinese_restaurant_process(self, latent):
        """Assign cluster using Chinese Restaurant Process"""
        if isinstance(latent, torch.Tensor):
            latent = latent.detach().cpu().numpy()
        
        if len(self.clusters) == 0:
            # First point creates first cluster
            self.clusters[0] = {'mean': latent.copy(), 'count': 1}
            self.next_cluster_id = 1
            return 0
        
        # Compute distances to existing clusters
        min_dist = float('inf')
        best_cluster = 0
        
        for cluster_id, cluster_info in self.clusters.items():
            dist = np.linalg.norm(latent - cluster_info['mean'])
            if dist < min_dist:
                min_dist = dist
                best_cluster = cluster_id
        
        # Simple threshold-based assignment
        if min_dist < 1.0:  # Threshold for cluster assignment
            # Add to existing cluster
            cluster = self.clusters[best_cluster]
            old_count = cluster['count']
            new_count = old_count + 1
            cluster['mean'] = (old_count * cluster['mean'] + latent) / new_count
            cluster['count'] = new_count
            return best_cluster
        else:
            # Create new cluster
            cluster_id = self.next_cluster_id
            self.clusters[cluster_id] = {'mean': latent.copy(), 'count': 1}
            self.next_cluster_id += 1
            return cluster_id
    
    def chinese_restaurant_process_batch(self, latents):
        """Process batch of latents"""
        assignments = []
        for latent in latents:
            assignment = self.chinese_restaurant_process(latent)
            assignments.append(assignment)
        return torch.tensor(assignments)
    
    def get_num_clusters(self):
        return len(self.clusters)
# =======================================
#      2. ENVIRONMENT WRAPPER
# =======================================

class AntTaskWrapper(gym.Wrapper):
    """5가지 태스크를 위한 Ant 환경 래퍼"""
    
    def __init__(self, env, task_count: int = 5, seed: int = None):  # 🔥 task_count 파라미터 추가
        super().__init__(env)
        self.task_count = task_count  # 🔥 파라미터로 받기
        self.current_task = 0
        
        # 태스크 정의
        self.task_names = {
            0: "Forward",      # 앞으로 이동
            1: "Backward",     # 뒤로 이동  
            2: "Crab_Work",    # 옆으로 이동 (게걸음)
            3: "Recover_Flip", # 뒤집힘 복구
            4: "Stand_Stop"    # 제자리 균형
        }
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Observation space 확장 (task onehot 추가)
        orig_shape = env.observation_space.shape[0]
        new_shape = orig_shape + self.task_count
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(new_shape,), dtype=np.float32
        )
        
        # 태스크별 성능 추적
        self.task_episode_rewards = {i: [] for i in range(self.task_count)}
        self.prev_position = None
        self.stationary_steps = 0
        
        print(f"🐜 MultiTaskAntWrapper initialized with {self.task_count} tasks:")
        for task_id, name in self.task_names.items():
            if task_id < self.task_count:  # 🔥 task_count 범위 내에서만 출력
                print(f"   Task {task_id}: {name}")

    def _calculate_task_reward(self, obs, reward, info, action) -> float:
        """태스크별 맞춤 보상 함수"""
        
        # 기본 정보 추출
        x_position = info.get('x_position', obs[0] if len(obs) > 0 else 0)
        y_position = info.get('y_position', obs[1] if len(obs) > 1 else 0)
        x_velocity = info.get('x_velocity', 0)
        y_velocity = info.get('y_velocity', 0)
        
        # Ant의 높이 (z축) - 뒤집힘 감지용
        height = obs[2] if len(obs) > 2 else 0.75  # 기본 Ant 높이
        
        # 기본 구성요소들
        ctrl_cost = info.get('reward_ctrl', -0.5 * np.sum(np.square(action)))
        contact_cost = info.get('reward_contact', 0)
        survive_reward = info.get('reward_survive', 1.0)
        
        # 현재 위치
        current_position = np.array([x_position, y_position])
        
        if self.current_task == 0:  # Forward - 앞으로 이동
            forward_reward = x_velocity * 2.0  # x축 양의 방향 속도 장려
            stability_bonus = 0.1 if height > 0.5 else -0.5  # 균형 유지 보너스
            task_reward = forward_reward + ctrl_cost + contact_cost + survive_reward + stability_bonus
            
        elif self.current_task == 1:  # Backward - 뒤로 이동
            backward_reward = -x_velocity * 2.0  # x축 음의 방향 속도 장려
            stability_bonus = 0.1 if height > 0.5 else -0.5
            task_reward = backward_reward + ctrl_cost + contact_cost + survive_reward + stability_bonus
            
        elif self.current_task == 2:  # Crab_Work - 옆으로 이동 (게걸음)
            crab_reward = abs(y_velocity) * 2.0  # y축 속도 (좌우 이동) 장려
            forward_penalty = -abs(x_velocity) * 0.5  # 앞뒤 이동은 페널티
            stability_bonus = 0.1 if height > 0.5 else -0.5
            task_reward = crab_reward + forward_penalty + ctrl_cost + contact_cost + survive_reward + stability_bonus
            
        elif self.current_task == 3:  # Recover_Flip - 뒤집힘 복구
            # 높이가 높을수록 보상 (정상 자세)
            height_reward = height * 3.0  # 높이 장려
            
            # 뒤집혔을 때 복구 동작 장려
            if height < 0.3:  # 뒤집힌 상태
                recovery_reward = abs(x_velocity) + abs(y_velocity)  # 움직임 장려
            else:  # 정상 상태
                recovery_reward = 2.0  # 정상 자세 유지 보너스
            
            # 큰 동작 허용 (복구를 위해)
            ctrl_cost = ctrl_cost * 0.3  # 제어 비용 줄임
            task_reward = height_reward + recovery_reward + ctrl_cost + survive_reward
            
        elif self.current_task == 4:  # Stand_Stop - 제자리 균형
            # 위치 변화 최소화
            if self.prev_position is not None:
                movement_distance = np.linalg.norm(current_position - self.prev_position)
                position_reward = -movement_distance * 5.0  # 움직임 페널티
                
                # 정체 상태 보상
                if movement_distance < 0.01:
                    self.stationary_steps += 1
                    stillness_bonus = min(self.stationary_steps * 0.01, 1.0)  # 정체 보너스
                else:
                    self.stationary_steps = 0
                    stillness_bonus = 0
            else:
                position_reward = 0
                stillness_bonus = 0
            
            # 균형 유지 (높이 기반)
            balance_reward = height * 2.0 if height > 0.5 else -1.0
            
            # 작은 제어 입력 장려
            gentle_control = -abs(ctrl_cost) * 0.5
            
            task_reward = position_reward + balance_reward + stillness_bonus + gentle_control + survive_reward
            
        else:
            # 정의되지 않은 태스크는 기본 보상
            task_reward = reward
        
        # 이전 위치 업데이트
        self.prev_position = current_position.copy()
        
        return float(task_reward)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # 랜덤하게 새 태스크 선택 (task_count 범위 내에서)
        self.current_task = random.randint(0, self.task_count - 1)  # 🔥 self.task_count 사용
        
        # 태스크별 초기화
        self.prev_position = np.array([obs[0], obs[1]])
        self.stationary_steps = 0
        
        # 태스크 변경 알림
        task_name = self.task_names.get(self.current_task, f"Task_{self.current_task}")  # 🔥 safe get
        print(f"🎯 New Task: {self.current_task} ({task_name})")
        
        return self._augment_obs(obs), info

    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 태스크별 보상 계산
        task_reward = self._calculate_task_reward(obs, reward, info, action)
        
        # 디버깅 정보 (간헐적으로)
        if random.random() < 0.001:  # 0.1% 확률로 출력
            task_name = self.task_names[self.current_task]
            print(f"Task {self.current_task} ({task_name}): "
                  f"Original: {reward:.3f}, Task: {task_reward:.3f}, "
                  f"x_vel: {info.get('x_velocity', 0):.3f}, "
                  f"y_vel: {info.get('y_velocity', 0):.3f}, "
                  f"height: {obs[2]:.3f}")
        
        return self._augment_obs(obs), task_reward, terminated, truncated, info
    
    def _augment_obs(self, obs):
        """관찰에 태스크 원핫 인코딩 추가"""
        task_onehot = np.zeros(self.task_count, dtype=np.float32)
        task_onehot[self.current_task] = 1.0
        return np.concatenate([obs, task_onehot])
    
    def get_task_info(self):
        """현재 태스크 정보 반환"""
        return {
            'task_id': self.current_task,
            'task_name': self.task_names[self.current_task],
            'task_onehot': self.get_task_onehot()
        }
    
    def get_task_onehot(self):
        task_onehot = np.zeros(self.task_count, dtype=np.float32)
        task_onehot[self.current_task] = 1.0
        return task_onehot



# =======================================
#      3. VARIATIONAL BAYES DPMM
# =======================================

class DirichletProcessMM:
    """Variational Bayes Dirichlet Process Mixture Model (Eq. 5-7)"""
    
    def __init__(self, concentration: float, latent_dim: int, device: str = 'cpu'):
        self.alpha = concentration  # DP concentration parameter
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        
        # Cluster parameters (variational posteriors)
        self.cluster_means = []  # μ_k
        self.cluster_covs = []   # Σ_k  
        self.cluster_counts = []  # n_k
        self.cluster_weights = [] # π_k
        
        # Hyperparameters for Gaussian priors
        self.prior_mean = torch.zeros(latent_dim, device=self.device)
        self.prior_cov = torch.eye(latent_dim, device=self.device)
        
        # Memoization for efficient inference
        self.memo_assignments = {}
        self.memo_elbo = 0.0
        
        print(f"🎯 DPMM initialized: α={concentration}, latent_dim={latent_dim}")
    
    def infer_cluster(self, z: torch.Tensor) -> torch.Tensor:
        """Infer cluster assignment for single latent vector"""
        if len(z.shape) == 1:
            z = z.unsqueeze(0)
        return self.infer_batch(z)[0]
    
    def infer_batch(self, Z: torch.Tensor) -> torch.Tensor:
        """Infer cluster assignments for batch of latent vectors"""
        batch_size = Z.shape[0]
        assignments = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        for i, z in enumerate(Z):
            # Chinese Restaurant Process (CRP) assignment
            if len(self.cluster_means) == 0:
                # First customer sits at first table
                assignments[i] = self._create_new_cluster(z)
            else:
                # Compute probabilities for existing clusters + new cluster
                probs = self._compute_cluster_probabilities(z)
                cluster_id = torch.multinomial(probs, 1).item()
                
                if cluster_id == len(self.cluster_means):
                    # Create new cluster
                    cluster_id = self._create_new_cluster(z)
                else:
                    # Assign to existing cluster
                    self._update_cluster(cluster_id, z)
                
                assignments[i] = cluster_id
        
        return assignments
    
    def _compute_cluster_probabilities(self, z: torch.Tensor) -> torch.Tensor:
        """Compute cluster assignment probabilities (CRP + likelihood)"""
        num_clusters = len(self.cluster_means)
        log_probs = torch.full((num_clusters + 1,), -float('inf'), device=self.device)
        
        # Existing clusters
        total_count = sum(self.cluster_counts) if self.cluster_counts else 0
        
        for k in range(num_clusters):
            # CRP probability
            crp_prob = self.cluster_counts[k] / (total_count + self.alpha - 1)
            
            # Gaussian likelihood with numerical stability
            mean = self.cluster_means[k]
            cov = self.cluster_covs[k]
            
            # Add regularization to covariance
            cov_reg = cov + 1e-3 * torch.eye(self.latent_dim, device=self.device)
            
            try:
                likelihood = self._gaussian_logpdf(z, mean, cov_reg)
                # Clamp likelihood to prevent overflow
                likelihood = torch.clamp(likelihood, -50, 50)
                log_probs[k] = torch.log(torch.tensor(crp_prob + 1e-8, device=self.device)) + likelihood
            except:
                log_probs[k] = torch.log(torch.tensor(crp_prob + 1e-8, device=self.device)) - 50
        
        # New cluster
        crp_prob = self.alpha / (total_count + self.alpha - 1)
        try:
            prior_likelihood = self._gaussian_logpdf(z, self.prior_mean, self.prior_cov)
            prior_likelihood = torch.clamp(prior_likelihood, -50, 50)
            log_probs[num_clusters] = torch.log(torch.tensor(crp_prob + 1e-8, device=self.device)) + prior_likelihood

        except:
            log_probs[num_clusters] = torch.log(torch.tensor(crp_prob + 1e-8, device=self.device)) - 50
        
        # Convert to probabilities using log_softmax for numerical stability
        probs = torch.softmax(log_probs, dim=0)
        
        # Final safety check
        probs = torch.clamp(probs, 1e-8, 1.0)
        probs = probs / probs.sum()  # Renormalize
        
        return probs


    def _gaussian_logpdf(self, x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """Compute log probability density of multivariate Gaussian with numerical stability"""
        diff = x - mean
        
        # Add regularization to covariance
        cov_reg = cov + 1e-6 * torch.eye(self.latent_dim, device=self.device)
        
        try:
            # Use Cholesky decomposition for numerical stability
            L = torch.linalg.cholesky(cov_reg)
            solve_term = torch.linalg.solve_triangular(L, diff, upper=False)
            quad_form = torch.sum(solve_term ** 2)
            logdet = 2.0 * torch.sum(torch.log(torch.diag(L)))
            
            logpdf = -0.5 * (self.latent_dim * math.log(2 * math.pi) + logdet + quad_form)
        except:
            # Fallback to simple computation
            quad_form = torch.sum(diff ** 2) / 0.1  # Use fixed variance
            logpdf = -0.5 * (self.latent_dim * math.log(2 * math.pi) + math.log(0.1) * self.latent_dim + quad_form)
        
        # Clamp result
        return torch.clamp(logpdf, -50, 50)

    def _create_new_cluster(self, z: torch.Tensor) -> int:
        """Create new cluster with given point"""
        cluster_id = len(self.cluster_means)
        
        self.cluster_means.append(z.clone())
        self.cluster_covs.append(torch.eye(self.latent_dim, device=self.device))
        self.cluster_counts.append(1)
        self.cluster_weights.append(1.0 / (len(self.cluster_means)))
        
        self._recompute_weights()
        return cluster_id
    
    def _update_cluster(self, cluster_id: int, z: torch.Tensor):
        """Update cluster statistics with new point"""
        if cluster_id >= len(self.cluster_means):
            return
        
        # Update count
        old_count = self.cluster_counts[cluster_id]
        new_count = old_count + 1
        self.cluster_counts[cluster_id] = new_count
        
        # Update mean (online update)
        old_mean = self.cluster_means[cluster_id]
        new_mean = old_mean + (z - old_mean) / new_count
        self.cluster_means[cluster_id] = new_mean
        
        # Update covariance (simplified)
        diff = z - new_mean
        self.cluster_covs[cluster_id] += 0.1 * torch.outer(diff, diff)
        
        self._recompute_weights()
    
    def _recompute_weights(self):
        """Recompute cluster weights"""
        if not self.cluster_counts:
            return
        
        total_count = sum(self.cluster_counts)
        self.cluster_weights = [count / total_count for count in self.cluster_counts]
    
    def update_memo_vb(self, Z: torch.Tensor) -> torch.Tensor:
        """Run one full ELBO optimization step (Eq. 5-7)"""
        if Z.shape[0] == 0:
            return torch.tensor(0.0, device=self.device)
        
        # E-step: compute cluster assignments
        assignments = self.infer_batch(Z)
        
        # M-step: update cluster parameters
        unique_clusters = torch.unique(assignments)
        
        for k in unique_clusters:
            mask = assignments == k
            cluster_points = Z[mask]
            
            if len(cluster_points) > 0:
                # Update cluster parameters
                if k < len(self.cluster_means):
                    # Update existing cluster
                    self.cluster_means[k] = cluster_points.mean(dim=0)
                    if len(cluster_points) > 1:
                        centered = cluster_points - self.cluster_means[k]
                        self.cluster_covs[k] = torch.mm(centered.T, centered) / len(cluster_points)
                        self.cluster_covs[k] += 1e-6 * torch.eye(self.latent_dim, device=self.device)
        
        # Compute ELBO (Eq. 5-7)
        elbo = self._compute_elbo(Z, assignments)
        self.memo_elbo = elbo
        
        # Birth/merge moves (simplified)
        if len(self.cluster_means) > 1:
            self._maybe_merge_clusters()
        
        return elbo
    
    def _compute_elbo(self, Z: torch.Tensor, assignments: torch.Tensor) -> torch.Tensor:
        """Compute Evidence Lower BOund (Eq. 5-7)"""
        if len(self.cluster_means) == 0:
            return torch.tensor(0.0, device=self.device)
        
        elbo = 0.0
        
        # Likelihood term
        for i, z in enumerate(Z):
            cluster_id = assignments[i].item()
            if cluster_id < len(self.cluster_means):
                mean = self.cluster_means[cluster_id]
                cov = self.cluster_covs[cluster_id]
                logpdf = self._gaussian_logpdf(z, mean, cov)
                elbo += logpdf
        
        # Prior term (simplified)
        elbo += len(self.cluster_means) * math.log(self.alpha)
        
        for weight in self.cluster_weights:
            if weight > 0:
                elbo += weight * math.log(weight)
        
        return torch.detach(elbo).clone().to(self.device)
    
    def _maybe_merge_clusters(self):
        """Merge similar clusters based on KL divergence"""
        if len(self.cluster_means) < 2:
            return
        
        # Compute pairwise KL divergences
        to_merge = []
        for i in range(len(self.cluster_means)):
            for j in range(i + 1, len(self.cluster_means)):

                kl_div = self._kl_divergence(i, j)
                if kl_div < 1e-4:  # Merge threshold
                    to_merge.append((i, j, kl_div))
        
        # Merge closest clusters
        if to_merge:
            to_merge.sort(key=lambda x: x[2])  # Sort by KL divergence
            i, j, _ = to_merge[0]
            self._merge_clusters(i, j)
    
    def _kl_divergence(self, i: int, j: int) -> float:
        """Compute KL divergence between two clusters"""
        if i >= len(self.cluster_means) or j >= len(self.cluster_means):
            return float('inf')
        
        mean1, cov1 = self.cluster_means[i], self.cluster_covs[i]
        mean2, cov2 = self.cluster_means[j], self.cluster_covs[j]
        
        # KL(N1||N2) = 0.5 * [tr(Σ2^-1 Σ1) + (μ2-μ1)^T Σ2^-1 (μ2-μ1) - k + ln(det(Σ2)/det(Σ1))]
        try:
            cov2_inv = torch.inverse(cov2 + 1e-6 * torch.eye(self.latent_dim, device=self.device))
            
            trace_term = torch.trace(torch.mm(cov2_inv, cov1))
            diff = mean2 - mean1
            quad_term = torch.dot(diff, torch.mv(cov2_inv, diff))
            
            logdet1 = torch.logdet(cov1 + 1e-6 * torch.eye(self.latent_dim, device=self.device))
            logdet2 = torch.logdet(cov2 + 1e-6 * torch.eye(self.latent_dim, device=self.device))
            
            kl = 0.5 * (trace_term + quad_term - self.latent_dim + logdet2 - logdet1)
            return max(0.0, kl.item())
        except:
            return float('inf')
    
    def _merge_clusters(self, i: int, j: int):
        """Merge cluster j into cluster i"""
        if i >= len(self.cluster_means) or j >= len(self.cluster_means) or i == j:
            return
        
        # Weighted average of parameters
        count_i = self.cluster_counts[i]
        count_j = self.cluster_counts[j]
        total_count = count_i + count_j
        
        # Merge means
        new_mean = (count_i * self.cluster_means[i] + count_j * self.cluster_means[j]) / total_count
        
        # Merge covariances (simplified)
        new_cov = (count_i * self.cluster_covs[i] + count_j * self.cluster_covs[j]) / total_count
        
        # Update cluster i
        self.cluster_means[i] = new_mean
        self.cluster_covs[i] = new_cov
        self.cluster_counts[i] = total_count
        
        # Remove cluster j
        del self.cluster_means[j]
        del self.cluster_covs[j]
        del self.cluster_counts[j]
        del self.cluster_weights[j]
        
        self._recompute_weights()
    
    def get_num_clusters(self) -> int:
        return len(self.cluster_means)

# =======================================
#      4. EXPERT POLICIES & BUFFERS
# =======================================

class ReplayBuffer:
    """Experience replay buffer for each expert"""
    
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, latent_dim: int):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.latents = np.zeros((capacity, latent_dim), dtype=np.float32)
        self.task_ids = np.zeros(capacity, dtype=np.int32)
    
    def add(self, obs, action, reward, next_obs, done, latent, task_id):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.latents[self.ptr] = latent.cpu().numpy() if torch.is_tensor(latent) else latent
        self.task_ids[self.ptr] = task_id
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        indices = np.random.choice(self.size, batch_size, replace=True)
        return {
            'observations': torch.FloatTensor(self.observations[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'next_observations': torch.FloatTensor(self.next_observations[indices]),
            'dones': torch.FloatTensor(self.dones[indices]),
            'latents': torch.FloatTensor(self.latents[indices]),
            'task_ids': torch.LongTensor(self.task_ids[indices])
        }
class ExpertPolicy(nn.Module):
    """Individual expert policy with SAC components"""
    
    def __init__(self, task_obs_dim: int, action_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.task_obs_dim = task_obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # ====== 차원 확인 ======
        actor_input_dim = task_obs_dim + latent_dim
        critic_input_dim = task_obs_dim + action_dim + latent_dim
        
        print(f"🔧 Creating ExpertPolicy:")
        print(f"   Task obs dim: {task_obs_dim}")
        print(f"   Action dim: {action_dim}")
        print(f"   Latent dim: {latent_dim}")
        print(f"   Actor input: {actor_input_dim}")
        print(f"   Critic input: {critic_input_dim}")
        # =====================
        
        # Actor network (task observation + latent)
        self.actor = nn.Sequential(
            nn.Linear(actor_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # mean and log_std
        )
        
        # Critic networks (task observation + action + latent)
        self.critic1 = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.critic2 = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Target networks
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        
        # Temperature parameter for SAC
        self.log_alpha = nn.Parameter(torch.zeros(1))
    
    def get_action(self, task_obs: torch.Tensor, latent: torch.Tensor, eval_mode: bool = False):
        """Get action from policy with dimension safety"""
        # ====== 차원 안전 검사 ======
        expected_task_obs = self.task_obs_dim
        expected_latent = self.latent_dim
        actual_task_obs = task_obs.shape[-1]
        actual_latent = latent.shape[-1]
        
        if actual_task_obs != expected_task_obs:
            raise ValueError(f"Task obs dimension mismatch: expected {expected_task_obs}, got {actual_task_obs}")
        
        if actual_latent != expected_latent:
            raise ValueError(f"Latent dimension mismatch: expected {expected_latent}, got {actual_latent}")
        # ===========================
        
        obs_latent = torch.cat([task_obs, latent], dim=-1)
        actor_output = self.actor(obs_latent)
        mean, log_std = actor_output.chunk(2, dim=-1)
        
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        if eval_mode:
            action = torch.tanh(mean)
            log_prob = None
        else:
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob, mean, log_std
    
    def get_q_values(self, task_obs: torch.Tensor, action: torch.Tensor, latent: torch.Tensor, target: bool = False):
        """Get Q-values from critics with dimension safety"""
        # ====== 차원 안전 검사 ======
        if task_obs.shape[-1] != self.task_obs_dim:
            raise ValueError(f"Task obs dimension mismatch: expected {self.task_obs_dim}, got {task_obs.shape[-1]}")
        
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"Action dimension mismatch: expected {self.action_dim}, got {action.shape[-1]}")
        
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(f"Latent dimension mismatch: expected {self.latent_dim}, got {latent.shape[-1]}")
        # ===========================
        
        obs_action_latent = torch.cat([task_obs, action, latent], dim=-1)
        
        if target:
            q1 = self.target_critic1(obs_action_latent)
            q2 = self.target_critic2(obs_action_latent)
        else:
            q1 = self.critic1(obs_action_latent)
            q2 = self.critic2(obs_action_latent)
        
        return q1, q2



# =======================================
#      5. AUXILIARY MODULES
# =======================================

class EmbedReconstructor(nn.Module):
    """Reconstruct full input from latent embedding"""
    
    def __init__(self, latent_dim: int, input_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

class DynamicsPredictor(nn.Module):
    """Predict next latent from current latent and action"""
    
    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        za = torch.cat([z, a], dim=-1)
        return self.predictor(za)

# =======================================
#      6. CLUSTER-EXPERT MANAGER
# =======================================
class ClusterExpertManager:
    """Manage cluster-expert mapping and knowledge distillation"""
    
    def __init__(self, experts: List[ExpertPolicy], dpmm: DirichletProcessMM, 
                 merge_threshold: float = 0.5, min_points: int = 30):
        """
        Initialize ClusterExpertManager
        
        Args:
            experts: List of expert policies
            dpmm: Dirichlet Process Mixture Model
            merge_threshold: Threshold for cluster merging
            min_points: Minimum points required for cluster
        """
        self.experts = experts
        self.dpmm = dpmm
        self.merge_threshold = merge_threshold
        self.min_points = min_points
        
        # Cluster to expert mapping
        self.cluster_to_expert = {}
        self.expert_to_cluster = {}
        
        # Pending distillations
        self.pending_distillations = []
        
        # Initialize mapping
        for i in range(len(experts)):
            self.cluster_to_expert[i] = i
            self.expert_to_cluster[i] = i
        
        print(f"🔧 ClusterExpertManager initialized:")
        print(f"   Experts: {len(experts)}")
        print(f"   Merge threshold: {merge_threshold}")
        print(f"   Min points: {min_points}")
    
    def get_expert_for_cluster(self, cluster_id: int) -> int:
        """Get expert index for given cluster"""
        if cluster_id in self.cluster_to_expert:
            return self.cluster_to_expert[cluster_id]
        
        # If cluster doesn't have expert, assign to closest existing expert
        if self.cluster_to_expert:
            return min(self.cluster_to_expert.values())
        
        return 0
    
    def maybe_merge_clusters(self):
        """Check for cluster merges and schedule distillation with dimension safety"""
        num_clusters = self.dpmm.get_num_clusters()
        
        if num_clusters < 2:
            return
        
        # Find clusters to merge based on KL divergence
        merges = []
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                try:
                    kl_div = self.dpmm._kl_divergence(i, j)
                    if kl_div < self.merge_threshold:
                        merges.append((i, j, kl_div))
                except:
                    continue  # Skip failed KL computations
        
        # Execute merges with dimension safety
        for i, j, kl_div in merges:
            try:
                self._merge_clusters_and_experts(i, j)
                print(f"🔄 Merged clusters {i} and {j} (KL={kl_div:.3f})")
            except Exception as e:
                print(f"⚠️ Failed to merge clusters {i} and {j}: {e}")
    
    def _merge_clusters_and_experts(self, cluster_i: int, cluster_j: int):
        """Merge clusters and schedule expert distillation with dimension checks"""
        # Get corresponding experts
        expert_i = self.cluster_to_expert.get(cluster_i, 0)
        expert_j = self.cluster_to_expert.get(cluster_j, 0)
        
        if expert_i != expert_j and expert_i < len(self.experts) and expert_j < len(self.experts):
            # Dimension compatibility check
            exp_i = self.experts[expert_i]
            exp_j = self.experts[expert_j]
            
            if (exp_i.task_obs_dim == exp_j.task_obs_dim and 
                exp_i.latent_dim == exp_j.latent_dim and
                exp_i.action_dim == exp_j.action_dim):
                
                # Schedule distillation only if dimensions are compatible
                self.pending_distillations.append({
                    'source_expert': expert_j,
                    'target_expert': expert_i,
                    'steps_remaining': 50
                })
                print(f"📝 Scheduled distillation: Expert {expert_j} → {expert_i}")
            else:
                print(f"⚠️ Dimension incompatible for distillation: Expert {expert_j} → {expert_i}")
        
        # Update mappings after merge
        if cluster_j in self.cluster_to_expert:
            del self.cluster_to_expert[cluster_j]
        
        # Reassign cluster indices after merge
        new_mapping = {}
        for old_cluster, expert in self.cluster_to_expert.items():
            new_cluster = old_cluster if old_cluster < cluster_j else old_cluster - 1
            new_mapping[new_cluster] = expert
        
        self.cluster_to_expert = new_mapping
        self.expert_to_cluster = {v: k for k, v in self.cluster_to_expert.items()}
    
    def compute_distillation_loss(self, batch, device):
        """Legacy method - now handled by MultiTaskAgent._safe_compute_distillation_loss"""
        return torch.tensor(0.0, device=device)

        
        total_distill_loss = torch.tensor(0.0, device=device)
        
        # Process each pending distillation
        for distillation in self.pending_distillations[:]:  # Copy list to allow modification
            source_idx = distillation['source_expert']
            target_idx = distillation['target_expert']
            
            if (source_idx >= len(self.experts) or target_idx >= len(self.experts) or 
                source_idx == target_idx):
                continue
            
            source_expert = self.experts[source_idx]
            target_expert = self.experts[target_idx]
            
            # Move batch to device
            obs = batch['observations'].to(device)
            latents = batch['latents'].to(device)
            
            # Extract task-specific observations (remove task onehot)
            task_obs = obs[:, :-3]  # Remove task onehot (assuming 3 tasks)
            
            # Get outputs from both experts
            with torch.no_grad():
                source_action, source_log_prob, source_mean, source_log_std = source_expert.get_action(task_obs, latents)
            
            target_action, target_log_prob, target_mean, target_log_std = target_expert.get_action(task_obs, latents)
            
            # KL divergence loss between action distributions
            source_std = torch.exp(source_log_std)
            target_std = torch.exp(target_log_std)
            
            # KL(source || target)
            kl_loss = torch.distributions.kl_divergence(
                torch.distributions.Normal(source_mean, source_std),
                torch.distributions.Normal(target_mean, target_std)
            ).sum(dim=-1).mean()
            
            total_distill_loss += kl_loss
            
            # Decrease remaining steps
            distillation['steps_remaining'] -= 1
            
            # Remove completed distillations
            if distillation['steps_remaining'] <= 0:
                self.pending_distillations.remove(distillation)
                print(f"📚 Completed distillation: Expert {source_idx} → {target_idx}")
        
        return total_distill_loss

# =======================================
#      7. MULTITASK AGENT
# =======================================
class MultiTaskAgent:
    def __init__(self, config: Dict, obs_dim: int, action_dim: int):
        self.config = config
        self.device = torch.device(config['device'])
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = config['latent_dim']
        
        # ====== 차원 디버깅 ======
        print(f"🔍 Dimension Analysis:")
        print(f"   Full obs_dim: {obs_dim}")
        print(f"   Task count: {config['task_count']}")
        print(f"   Latent dim: {self.latent_dim}")
        
        task_obs_dim = obs_dim - config['task_count']
        actor_input_dim = task_obs_dim + self.latent_dim
        critic_input_dim = task_obs_dim + action_dim + self.latent_dim
        
        print(f"   Task obs dim: {task_obs_dim}")
        print(f"   Actor input dim: {actor_input_dim}")
        print(f"   Critic input dim: {critic_input_dim}")
        # ===========================
        
        # Task encoder (전체 observation 사용)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.latent_dim)
        ).to(self.device)
        
        # Expert policies with correct dimensions
        self.experts = nn.ModuleList([
            ExpertPolicy(task_obs_dim, action_dim, self.latent_dim, config['pmoe_hidden_dim'])
            for _ in range(config['init_experts'])
        ]).to(self.device)
        
        # Expert buffers
        self.expert_buffers = [
            ReplayBuffer(config['buf_per_expert'], obs_dim, action_dim, self.latent_dim)
            for _ in range(config['init_experts'])
        ]
        
        # Auxiliary modules
        self.reconstructor = EmbedReconstructor(self.latent_dim, obs_dim).to(self.device)
        self.dynamics_predictor = DynamicsPredictor(self.latent_dim, action_dim).to(self.device)
        
        # DPMM
        self.dpmm = DirichletProcessMM(
            concentration=config['dp_concentration'],
            latent_dim=self.latent_dim,
            device='cpu'
        )
        
        # ====== 수정된 ClusterExpertManager 생성 ======
        self.manager = ClusterExpertManager(
            experts=self.experts,
            dpmm=self.dpmm,
            merge_threshold=config['cluster_merge_threshold'],
            min_points=config['cluster_min_points']
        )
        # ============================================
        
        # Optimizers for each expert
        self.expert_optimizers = []
        for expert in self.experts:
            actor_params = list(expert.actor.parameters())
            critic_params = list(expert.critic1.parameters()) + list(expert.critic2.parameters())
            alpha_params = [expert.log_alpha]
            
            self.expert_optimizers.append({
                'actor': optim.Adam(actor_params, lr=config['learning_rate']),
                'critic': optim.Adam(critic_params, lr=config['learning_rate']),
                'alpha': optim.Adam(alpha_params, lr=config['learning_rate'])
            })
        
        # Auxiliary optimizers
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=config['learning_rate'])
        self.reconstructor_optimizer = optim.Adam(self.reconstructor.parameters(), lr=config['learning_rate'])
        self.dynamics_optimizer = optim.Adam(self.dynamics_predictor.parameters(), lr=config['learning_rate'])
        
        # Target entropy for SAC
        self.target_entropy = -action_dim
        
        print(f"🧠 MultiTaskAgent initialized with correct dimensions!")
        
        # 생성된 expert들의 차원 확인
        for i, expert in enumerate(self.experts):
            print(f"   Expert {i} created with task_obs_dim: {expert.task_obs_dim}, latent_dim: {expert.latent_dim}")

    
    def select_action(self, obs: np.ndarray, eval_mode: bool = False):
        """Select action using cluster-expert mechanism"""
        # Extract task-specific observation
        task_obs = obs[:-self.config['task_count']]
        task_onehot = obs[-self.config['task_count']:]
        
        # Encode latent
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent = self.encoder(obs_tensor)
        
        # Infer cluster
        cluster_id = self.dpmm.infer_cluster(latent.squeeze(0).cpu())
        
        # Get corresponding expert
        expert_id = self.manager.get_expert_for_cluster(cluster_id)
        expert = self.experts[expert_id]
        
        # Get action
        task_obs_tensor = torch.FloatTensor(task_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _, _ = expert.get_action(task_obs_tensor, latent, eval_mode)
        
        return action.squeeze(0).cpu().numpy(), latent.squeeze(0), cluster_id, expert_id
    
    def add_transition(self, obs, action, reward, next_obs, done, latent, cluster_id, expert_id, task_id):
        """Add transition to appropriate expert buffer"""
        if expert_id < len(self.expert_buffers):
            self.expert_buffers[expert_id].add(obs, action, reward, next_obs, done, latent, task_id)
    def train_step(self):
        """Main training step with dimension debugging and safe distillation"""
        total_losses = {}
        
        for expert_id, (expert, buffer, optimizers) in enumerate(
            zip(self.experts, self.expert_buffers, self.expert_optimizers)
        ):
            if buffer.size < self.config['batch_size']:
                continue
            
            batch = buffer.sample(self.config['batch_size'])
            
            obs = batch['observations'].to(self.device)
            actions = batch['actions'].to(self.device)
            rewards = batch['rewards'].to(self.device).unsqueeze(-1)
            next_obs = batch['next_observations'].to(self.device)
            dones = batch['dones'].to(self.device).unsqueeze(-1)
            latents = batch['latents'].to(self.device)
            
            # ====== 차원 디버깅 (첫 번째 expert에서만) ======
            if expert_id == 0:
                print(f"🔍 Training Step Dimensions:")
                print(f"   Full obs: {obs.shape}")
                print(f"   Task count: {self.config['task_count']}")
                print(f"   Actions: {actions.shape}")
                print(f"   Latents: {latents.shape}")
            # ===============================================
            
            # Extract task-specific observations
            task_obs = obs[:, :-self.config['task_count']]
            next_task_obs = next_obs[:, :-self.config['task_count']]
            
            # ====== 차원 확인 ======
            if expert_id == 0:
                print(f"   Task obs: {task_obs.shape}")
                print(f"   Expected by expert: {expert.task_obs_dim}")
                print(f"   Actor input will be: {task_obs.shape[-1] + latents.shape[-1]}")
            # =====================
            
            # 차원 검증
            if task_obs.shape[-1] != expert.task_obs_dim:
                print(f"❌ Expert {expert_id} dimension mismatch!")
                print(f"   Expected task_obs: {expert.task_obs_dim}")
                print(f"   Actual task_obs: {task_obs.shape[-1]}")
                continue
            
            # DPMM update
            try:
                elbo = self.dpmm.update_memo_vb(latents.cpu())
            except:
                elbo = torch.tensor(0.0)
            
            # ====== SAC Critic Update (Eq. 1-3) ======
            with torch.no_grad():
                next_latents = self.encoder(next_obs)
                next_actions, next_log_probs, _, _ = expert.get_action(next_task_obs, next_latents)
                target_q1, target_q2 = expert.get_q_values(next_task_obs, next_actions, next_latents, target=True)
                target_q = torch.min(target_q1, target_q2)
                alpha = torch.exp(expert.log_alpha)
                target_q = rewards + self.config['gamma'] * (1 - dones) * (target_q - alpha * next_log_probs)
            
            current_q1, current_q2 = expert.get_q_values(task_obs, actions, latents)
            critic_loss1 = F.mse_loss(current_q1, target_q)
            critic_loss2 = F.mse_loss(current_q2, target_q)
            
            # Update critics
            critic_loss = critic_loss1 + critic_loss2
            optimizers['critic'].zero_grad()
            critic_loss.backward()
            optimizers['critic'].step()
            
            # ====== SAC Actor Update (Eq. 4) ======
            new_actions, log_probs, _, _ = expert.get_action(task_obs, latents.detach())
            q1_new, q2_new = expert.get_q_values(task_obs, new_actions, latents.detach())
            q_new = torch.min(q1_new, q2_new)
            
            actor_loss = (alpha * log_probs - q_new).mean()
            
            # Update actor
            optimizers['actor'].zero_grad()
            actor_loss.backward()
            optimizers['actor'].step()
            
            # ====== Temperature (alpha) update ======
            alpha_loss = -(expert.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            optimizers['alpha'].zero_grad()
            alpha_loss.backward()
            optimizers['alpha'].step()
            
            # ====== Auxiliary losses ======
            L_rec = F.mse_loss(self.reconstructor(latents), obs)
            if latents.size(0) > 1:
                L_dyn = F.mse_loss(self.dynamics_predictor(latents[:-1], actions[:-1]), latents[1:])
            else:
                L_dyn = torch.tensor(0.0, device=self.device)
            
            # ====== Safe Distillation Loss ======
            try:
                L_distill = self._safe_compute_distillation_loss(batch)
            except Exception as e:
                print(f"⚠️ Distillation loss computation failed: {e}")
                L_distill = torch.tensor(0.0, device=self.device)
            # ===================================
            
            # Total auxiliary loss
            aux_loss = (self.config['lambda_vb'] * (-elbo.to(self.device)) +
                       self.config['lambda_rec'] * L_rec +
                       self.config['lambda_dyn'] * L_dyn +
                       self.config['lambda_distill'] * L_distill)
            
            # Update auxiliary modules
            self.encoder_optimizer.zero_grad()
            self.reconstructor_optimizer.zero_grad()
            self.dynamics_optimizer.zero_grad()
            
            aux_loss.backward(retain_graph=True)
            
            self.encoder_optimizer.step()
            self.reconstructor_optimizer.step()
            self.dynamics_optimizer.step()
            
            # Soft update target networks
            with torch.no_grad():
                for target_param, param in zip(expert.target_critic1.parameters(), expert.critic1.parameters()):
                    target_param.data.copy_(self.config['tau'] * param.data + (1 - self.config['tau']) * target_param.data)
                
                for target_param, param in zip(expert.target_critic2.parameters(), expert.critic2.parameters()):
                    target_param.data.copy_(self.config['tau'] * param.data + (1 - self.config['tau']) * target_param.data)
            
            # Store losses
            total_losses[f'expert_{expert_id}'] = {
                'critic_loss': critic_loss.item(),
                'actor_loss': actor_loss.item(),
                'alpha_loss': alpha_loss.item(),
                'elbo': elbo.item(),
                'L_rec': L_rec.item(),
                'L_dyn': L_dyn.item(),
                'L_distill': L_distill.item(),
                'total_loss': (critic_loss + actor_loss + alpha_loss + aux_loss).item(),
                'alpha': alpha.item()
            }
        
        # Check for cluster merges
        self.manager.maybe_merge_clusters()
        
        return total_losses
    
    def _safe_compute_distillation_loss(self, batch):
        """Safe distillation loss computation"""
        if not self.manager.pending_distillations:
            return torch.tensor(0.0, device=self.device)
        
        total_distill_loss = torch.tensor(0.0, device=self.device)
        
        obs = batch['observations'].to(self.device)
        latents = batch['latents'].to(self.device)
        
        # Extract task-specific observations
        task_obs = obs[:, :-self.config['task_count']]
        
        for distillation in self.manager.pending_distillations[:]:
            source_idx = distillation['source_expert']
            target_idx = distillation['target_expert']
            
            if (source_idx >= len(self.experts) or target_idx >= len(self.experts) or 
                source_idx == target_idx):
                distillation['steps_remaining'] = 0
                continue
            
            source_expert = self.experts[source_idx]
            target_expert = self.experts[target_idx]
            
            # ====== 차원 호환성 검사 ======
            if (task_obs.shape[-1] != source_expert.task_obs_dim or 
                task_obs.shape[-1] != target_expert.task_obs_dim or
                latents.shape[-1] != source_expert.latent_dim or
                latents.shape[-1] != target_expert.latent_dim):
                
                print(f"⚠️ Dimension mismatch in distillation {source_idx}→{target_idx}, skipping")
                distillation['steps_remaining'] = 0
                continue
            # =============================
            
            try:
                # Get outputs from both experts
                with torch.no_grad():
                    source_action, source_log_prob, source_mean, source_log_std = source_expert.get_action(task_obs, latents)
                
                target_action, target_log_prob, target_mean, target_log_std = target_expert.get_action(task_obs, latents)
                
                # KL divergence loss between action distributions
                source_std = torch.exp(source_log_std)
                target_std = torch.exp(target_log_std)
                
                # KL(source || target)
                kl_loss = torch.distributions.kl_divergence(
                    torch.distributions.Normal(source_mean, source_std),
                    torch.distributions.Normal(target_mean, target_std)
                ).sum(dim=-1).mean()
                
                total_distill_loss += kl_loss
                
            except Exception as e:
                print(f"⚠️ Error in distillation computation: {e}")
                distillation['steps_remaining'] = 0
                continue
            
            # Decrease remaining steps
            distillation['steps_remaining'] -= 1
            
            # Remove completed distillations
            if distillation['steps_remaining'] <= 0:
                self.manager.pending_distillations.remove(distillation)
                print(f"📚 Completed distillation: Expert {source_idx} → {target_idx}")
        
        return total_distill_loss

    
class ImprovedMultiTaskAgent(MultiTaskAgent):
    def __init__(self, config, obs_dim, action_dim):
        super().__init__(config, obs_dim, action_dim)
        
        # ====== Target Entropy 개선 ======
        entropy_ratio = config.get('target_entropy_ratio', 0.5)
        self.target_entropy = -entropy_ratio * action_dim  # 🔥 덜 aggressive한 entropy
        
        # ====== 개선된 Optimizer 설정 ======
        self.expert_optimizers = []
        for expert in self.experts:
            actor_params = list(expert.actor.parameters())
            critic_params = list(expert.critic1.parameters()) + list(expert.critic2.parameters())
            alpha_params = [expert.log_alpha]
            
            self.expert_optimizers.append({
                'actor': optim.Adam(actor_params, lr=config.get('actor_lr', config['learning_rate'])),
                'critic': optim.Adam(critic_params, lr=config.get('critic_lr', config['learning_rate'])),
                'alpha': optim.Adam(alpha_params, lr=config.get('alpha_lr', 3e-4))
            })
        
        # ====== 개선된 Auxiliary Optimizers ======
        encoder_lr = config.get('encoder_lr', config['learning_rate'])
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        self.reconstructor_optimizer = optim.Adam(self.reconstructor.parameters(), lr=encoder_lr)
        self.dynamics_optimizer = optim.Adam(self.dynamics_predictor.parameters(), lr=encoder_lr)
        
        print(f"🧠 Improved MultiTaskAgent initialized:")
        print(f"   Target entropy: {self.target_entropy:.2f}")
        print(f"   Actor LR: {config.get('actor_lr', config['learning_rate'])}")
        print(f"   Critic LR: {config.get('critic_lr', config['learning_rate'])}")


# =======================================
#      8. TRAINER & METRICS LOGGING
# =======================================
class Trainer:
    """Main trainer with comprehensive evaluation metrics"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Set seeds
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        random.seed(config['seed'])
        
        # Create environment
        self.env = AntTaskWrapper(
            gym.make(config['env_name'], render_mode='human',
                    terminate_when_unhealthy=True
                    ),
            task_count=config['task_count'],
            seed=config['seed']
        )
        
        # Extract dimensions
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        # Create agent
        self.agent = MultiTaskAgent(config, self.obs_dim, self.action_dim)
        
        # ====== SKILL MONITOR 추가 ======
        self.skill_monitor = SkillMonitor()
        # ===============================
        
        # 기존 metrics tracking 코드들...
        self.history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'cluster_counts': [],
            'expert_counts': [],
            'forward_transfer': [],
            'forgetting': [],
            'zero_shot': [],
            'elbo_values': []
        }
        
        # Task performance tracking
        self.task_performance_history = {task_id: [] for task_id in range(config['task_count'])}
        self.baseline_performance = {}  # Store initial performance for forgetting
        
        print(f"🚀 Trainer initialized:")
        print(f"   Environment: {config['env_name']}")
        print(f"   Obs dim: {self.obs_dim}, Action dim: {self.action_dim}")
        print(f"   Total steps: {config['total_steps']:,}")
        print(f"   📊 Skill monitoring enabled")  # 추가

    def train(self):
        """Main training loop with comprehensive evaluation"""
        step = 0
        episode_reward = 0
        episode_length = 0
        
        obs, _ = self.env.reset()
        
        print("🚀 Starting training...")
        
        while step < self.config['total_steps']:
            print(f"Step {step:,} ")
            print(f"reward: {reward}")
            # Select action
            action, latent, cluster_id, expert_id = self.agent.select_action(obs, eval_mode=False)
            
            # Environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Add transition to expert buffer
            self.agent.add_transition(
                obs, action, reward, next_obs, done, 
                latent, cluster_id, expert_id, self.env.current_task
            )
            
            episode_reward += reward
            episode_length += 1
            step += 1
            
            # Episode end
            if done:
                self.history['episode_rewards'].append(episode_reward)
                self.history['episode_lengths'].append(episode_length)
                self.task_performance_history[self.env.current_task].append(episode_reward)
                # ====== EPISODE 요약에 SKILL 정보 추가 ======
                skill_stats = self.skill_monitor.get_stats()
                if step % 1000 == 0:
                    print(f"🏁 Step {step:,}: Episode reward={episode_reward:.2f}, "
                          f"Task={self.env.current_task}, Expert={expert_id}, "
                          f"Final Skill={cluster_id}")
                    print(f"   Skill Stats: {skill_stats['unique_skills']} unique skills, "
                          f"{skill_stats['total_switches']} switches")
                # ===========================================
                
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
            
            # Training
            if step % self.config['train_freq'] == 0:
                losses = self.agent.train_step()
                
                if losses and step % 5000 == 0:
                    self._log_training_progress(step, losses)
            
            # DPMM update
            if step % self.config['dpmm_freq'] == 0:
                self._update_dpmm_statistics(step)
            
            # Evaluation
            if step % self.config['eval_freq'] == 0:
                self._comprehensive_evaluation(step)
        
        print("✅ Training completed!")
        self._final_evaluation()
    
    def _log_training_progress(self, step: int, losses: Dict):
        """Log training progress"""
        num_clusters = self.agent.dpmm.get_num_clusters()
        num_experts = len(self.agent.experts)
        
        # ====== SKILL 통계 추가 ======
        skill_stats = self.skill_monitor.get_stats()
        # ===========================
        
        print(f"🎯 Step {step:,}:")
        print(f"   Clusters: {num_clusters}, Experts: {num_experts}")
        print(f"   Pending distillations: {len(self.agent.manager.pending_distillations)}")
        
        # ====== SKILL 정보 출력 추가 ======
        print(f"   Skills: {skill_stats.get('unique_skills', 0)} unique, "
              f"{skill_stats.get('total_switches', 0)} switches, "
              f"Current: {skill_stats.get('current_skill', 'N/A')} "
              f"({skill_stats.get('current_duration', 0)} steps)")
        # ===============================
        
        # Log losses for each expert
        for expert_id, expert_losses in losses.items():
            if isinstance(expert_losses, dict):
                print(f"   {expert_id}: critic={expert_losses['critic_loss']:.3f}, "
                      f"actor={expert_losses['actor_loss']:.3f}, "
                      f"elbo={expert_losses['elbo']:.3f}")

    
    def _update_dpmm_statistics(self, step: int):
        """Update DPMM statistics and track metrics"""
        # Collect recent latents from all expert buffers
        all_latents = []
        for buffer in self.agent.expert_buffers:
            if buffer.size > 0:
                # Sample recent latents
                recent_size = min(buffer.size, 100)
                recent_indices = np.arange(max(0, buffer.size - recent_size), buffer.size)
                recent_latents = buffer.latents[recent_indices]
                all_latents.extend(recent_latents)
        
        if all_latents:
            latent_batch = torch.FloatTensor(np.array(all_latents))
            elbo = self.agent.dpmm.update_memo_vb(latent_batch)
            
            self.history['cluster_counts'].append(self.agent.dpmm.get_num_clusters())
            self.history['expert_counts'].append(len(self.agent.experts))
            self.history['elbo_values'].append(elbo.item())
            
            if step % 10000 == 0:
                print(f"📊 DPMM Update - Clusters: {self.agent.dpmm.get_num_clusters()}, "
                      f"ELBO: {elbo.item():.3f}")
    
    def _comprehensive_evaluation(self, step: int):
        """Comprehensive evaluation including forward transfer, forgetting, and zero-shot"""
        print(f"\n🏆 Comprehensive Evaluation at Step {step:,}")
        
        # 1. Forward Transfer Evaluation
        forward_transfer_scores = self._evaluate_forward_transfer()
        self.history['forward_transfer'].append({
            'step': step,
            'scores': forward_transfer_scores
        })
        
        # 2. Forgetting Evaluation
        forgetting_scores = self._evaluate_forgetting()
        self.history['forgetting'].append({
            'step': step,
            'scores': forgetting_scores
        })
        
        # 3. Zero-Shot Evaluation (on held-out task variations)
        zero_shot_scores = self._evaluate_zero_shot()
        self.history['zero_shot'].append({
            'step': step,
            'scores': zero_shot_scores
        })
        
        # Print summary
        print(f"   Forward Transfer: {np.mean(list(forward_transfer_scores.values())):.2f}")
        print(f"   Forgetting: {np.mean(list(forgetting_scores.values())):.2f}")
        print(f"   Zero-Shot: {np.mean(list(zero_shot_scores.values())):.2f}")
        print(f"   Active Clusters: {self.agent.dpmm.get_num_clusters()}")
        print(f"   Active Experts: {len(self.agent.experts)}")
    
    def _evaluate_forward_transfer(self) -> Dict[int, float]:
        """Evaluate forward transfer: performance of each expert on all tasks"""
        scores = {}
        
        for task_id in range(self.config['task_count']):
            task_scores = []
            
            for expert_id in range(len(self.agent.experts)):
                # Test expert on specific task
                score = self._evaluate_expert_on_task(expert_id, task_id, episodes=3)
                task_scores.append(score)
            
            # Forward transfer score: best expert performance on this task
            scores[task_id] = max(task_scores) if task_scores else 0.0
        
        return scores
    
    def _evaluate_forgetting(self) -> Dict[int, float]:
        """Evaluate forgetting: performance drop on previously learned tasks"""
        scores = {}
        
        for task_id in range(self.config['task_count']):
            if task_id in self.baseline_performance:
                current_score = self._evaluate_best_expert_on_task(task_id, episodes=3)
                baseline_score = self.baseline_performance[task_id]
                
                # Forgetting score: 1 - (performance_drop / baseline)
                if baseline_score > 0:
                    forgetting = max(0, 1 - (baseline_score - current_score) / baseline_score)
                else:
                    forgetting = 1.0
                
                scores[task_id] = forgetting
            else:
                # Set baseline for new task
                baseline_score = self._evaluate_best_expert_on_task(task_id, episodes=3)
                self.baseline_performance[task_id] = baseline_score
                scores[task_id] = 1.0  # No forgetting yet
        
        return scores
    
    def _evaluate_zero_shot(self) -> Dict[str, float]:
        """Evaluate zero-shot transfer on novel task variations"""
        scores = {}
        
        # Create modified environments for zero-shot evaluation
        test_variations = ['gravity_0.5', 'gravity_1.5', 'friction_0.5', 'friction_1.5']
        
        for variation in test_variations:
            # Simple evaluation without actual environment modification
            # In practice, you would modify environment parameters
            score = self._evaluate_best_expert_on_task(0, episodes=2)  # Simplified
            scores[variation] = score
        
        return scores
    
    def _evaluate_expert_on_task(self, expert_id: int, task_id: int, episodes: int = 5) -> float:
        """Evaluate specific expert on specific task"""
        if expert_id >= len(self.agent.experts):
            return 0.0
        
        total_reward = 0.0
        expert = self.agent.experts[expert_id]
        
        for _ in range(episodes):
            obs, _ = self.env.reset()
            # Force specific task
            self.env.current_task = task_id
            obs = self.env._augment_obs(obs[:-self.config['task_count']])
            
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 1000:
                # Use specific expert
                task_obs = obs[:-self.config['task_count']]
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.agent.device)
                
                with torch.no_grad():
                    latent = self.agent.encoder(obs_tensor)
                    task_obs_tensor = torch.FloatTensor(task_obs).unsqueeze(0).to(self.agent.device)
                    action, _, _, _ = expert.get_action(task_obs_tensor, latent, eval_mode=True)
                
                obs, reward, terminated, truncated, _ = self.env.step(action.squeeze(0).cpu().numpy())
                episode_reward += reward
                done = terminated or truncated
                steps += 1
            
            total_reward += episode_reward
        
        return total_reward / episodes
    
    def _evaluate_best_expert_on_task(self, task_id: int, episodes: int = 5) -> float:
        """Evaluate best expert on specific task"""
        best_score = float('-inf')
        
        for expert_id in range(len(self.agent.experts)):
            score = self._evaluate_expert_on_task(expert_id, task_id, episodes)
            best_score = max(best_score, score)
        
        return best_score if best_score != float('-inf') else 0.0
    
    def _final_evaluation(self):
        """Final comprehensive evaluation and results summary"""
        print("\n" + "="*80)
        print("🎉 FINAL EVALUATION RESULTS")
        print("="*80)
        
        # Overall statistics
        print(f"📊 Training Statistics:")
        print(f"   Total episodes: {len(self.history['episode_rewards'])}")
        print(f"   Average episode reward: {np.mean(self.history['episode_rewards']):.2f}")
        print(f"   Final clusters: {self.agent.dpmm.get_num_clusters()}")
        print(f"   Final experts: {len(self.agent.experts)}")
        
        # Task-specific performance
        print(f"\n📈 Task Performance:")
        for task_id in range(self.config['task_count']):
            if self.task_performance_history[task_id]:
                recent_performance = np.mean(self.task_performance_history[task_id][-10:])
                print(f"   Task {task_id}: {recent_performance:.2f}")
        
        # Transfer learning metrics
        if self.history['forward_transfer']:
            latest_ft = self.history['forward_transfer'][-1]['scores']
            print(f"\n🚀 Forward Transfer: {np.mean(list(latest_ft.values())):.2f}")
        
        if self.history['forgetting']:
            latest_forget = self.history['forgetting'][-1]['scores']
            print(f"🧠 Forgetting (higher=better): {np.mean(list(latest_forget.values())):.2f}")
        
        if self.history['zero_shot']:
            latest_zs = self.history['zero_shot'][-1]['scores']
            print(f"🎯 Zero-Shot: {np.mean(list(latest_zs.values())):.2f}")
        
        # Cluster evolution
        if self.history['cluster_counts']:
            print(f"\n🔄 Cluster Evolution:")
            print(f"   Initial clusters: {self.history['cluster_counts'][0]}")
            print(f"   Final clusters: {self.history['cluster_counts'][-1]}")
            print(f"   Max clusters: {max(self.history['cluster_counts'])}")
        
        print("="*80)

class ImprovedTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.warmup_steps = config.get('warmup_steps', 10000)
        self.min_buffer_size = config.get('min_buffer_size', 10000)
        self.gradient_steps = config.get('gradient_steps', 1)
        
    def train(self):
        step = 0
        episode_reward = 0
        episode_length = 0
        
        obs, _ = self.env.reset()
        
        print(f"🚀 Starting improved training...")
        print(f"   Warmup steps: {self.warmup_steps}")
        print(f"   Min buffer size: {self.min_buffer_size}")
        
        while step < self.config['total_steps']:
            # ====== Warmup 기간 동안 Random Exploration ======
            if step < self.warmup_steps:
                action = self.env.action_space.sample()
                latent = torch.randn(self.agent.latent_dim)
                cluster_id = 0
                expert_id = 0
            else:
                action, latent, cluster_id, expert_id = self.agent.select_action(obs, eval_mode=False)
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            self.agent.add_transition(
                obs, action, reward, next_obs, done,
                latent, cluster_id, expert_id, self.env.current_task
            )
            
            episode_reward += reward
            episode_length += 1
            step += 1
            
            # Episode end
            if done:
                if step % self.config.get('log_freq', 1000) == 0:
                    print(f"🏁 Step {step:,}: Episode reward={episode_reward:.2f}, "
                          f"Length={episode_length}, Task={self.env.current_task}")
                
                self.history['episode_rewards'].append(episode_reward)
                self.history['episode_lengths'].append(episode_length)
                
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
            
            # ====== 개선된 학습 조건 ======
            if (step > self.warmup_steps and 
                step % self.config['train_freq'] == 0 and
                self._check_buffer_ready()):
                
                # Multiple gradient steps
                for _ in range(self.gradient_steps):
                    losses = self.agent.train_step()
                
                if losses and step % (self.config.get('log_freq', 1000) * 5) == 0:
                    self._log_training_progress(step, losses)
            
            # 나머지 로직은 동일...
    
    def _check_buffer_ready(self):
        """버퍼가 학습할 준비가 되었는지 확인"""
        total_samples = sum(buffer.size for buffer in self.agent.expert_buffers)
        return total_samples >= self.min_buffer_size


# =======================================
#      9. ENTRY POINT
# =======================================

def main():
    """Main entry point"""
    print("🚀 Lifelong Reinforcement Learning with Knowledge Preservation")
    print("=" * 80)
    
    # Load configuration
    config = get_default_config()
    print("📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Create and run trainer
    try:
        trainer = Trainer(config)
        trainer.train()
        
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
