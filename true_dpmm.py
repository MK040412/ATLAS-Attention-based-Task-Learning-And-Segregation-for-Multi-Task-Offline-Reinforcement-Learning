import torch
import torch.distributions as dist
import numpy as np
from scipy.special import logsumexp
from typing import List, Dict, Any

class TrueDPMM:
    """진짜 DPMM 구현
    
    생성 모델:
    θ*k ~ H(λ), k=1,2,...
    π ~ GEM(α)  
    vi|π ~ Cat(π), i=1,...,N
    zi|vi,{θ*k} ~ F(θ*vi)
    """
    
    def __init__(self, latent_dim: int, alpha: float = 1.0, base_var: float = 1.0):
        self.latent_dim = latent_dim
        self.alpha = alpha  # 집중도 파라미터
        self.base_var = base_var  # 베이스 분포 H의 분산
        
        # 클러스터 파라미터들 {θ*k}
        self.cluster_means: List[torch.Tensor] = []
        self.cluster_covs: List[torch.Tensor] = []
        
        # 데이터 할당 정보
        self.cluster_counts: List[int] = []  # 각 클러스터의 데이터 개수
        self.data_assignments: List[int] = []  # vi
        self.data_points: List[torch.Tensor] = []  # zi
        
        # 베이스 분포 H(λ): 다변량 정규분포
        self.base_mean = torch.zeros(latent_dim)
        self.base_cov = base_var * torch.eye(latent_dim)
        
        print(f"DPMM initialized: latent_dim={latent_dim}, alpha={alpha}, base_var={base_var}")
    
    def _base_distribution_logpdf(self, theta: torch.Tensor) -> torch.Tensor:
        """베이스 분포 H(λ)의 로그 확률밀도"""
        return dist.MultivariateNormal(self.base_mean, self.base_cov).log_prob(theta)
    
    def _likelihood_logpdf(self, z: torch.Tensor, theta: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """클러스터 분포 F(θ*)의 로그 우도"""
        return dist.MultivariateNormal(theta, cov).log_prob(z)
    
    def _stick_breaking_weights(self, K: int) -> torch.Tensor:
        """GEM(α): Stick-breaking으로 π 생성"""
        if K == 0:
            return torch.tensor([])
        
        # β_k ~ Beta(1, α)
        betas = torch.distributions.Beta(1, self.alpha).sample((K,))
        weights = torch.zeros(K)
        remaining = 1.0
        
        for k in range(K-1):
            weights[k] = betas[k] * remaining
            remaining *= (1 - betas[k])
        weights[K-1] = remaining
        
        return weights
    
    def assign_and_update(self, z: torch.Tensor) -> int:
        """베이지안 추론으로 클러스터 할당 및 업데이트
        
        P(vi = k | z, 기존 데이터) ∝ P(z | θ*k) * P(vi = k)
        """
        if not torch.is_tensor(z):
            z = torch.tensor(z, dtype=torch.float32)
        
        if len(self.cluster_means) == 0:
            # 첫 번째 데이터: 새 클러스터 생성
            self._create_new_cluster(z)
            self.data_assignments.append(0)
            self.data_points.append(z.clone())
            return 0
        
        K = len(self.cluster_means)
        log_probs = torch.zeros(K + 1)  # 기존 K개 + 새 클러스터 1개
        
        # 기존 클러스터들에 대한 확률 계산
        for k in range(K):
            # P(z | θ*k): 우도
            likelihood = self._likelihood_logpdf(z, self.cluster_means[k], self.cluster_covs[k])
            
            # P(vi = k): 사전 확률 (Chinese Restaurant Process)
            prior = np.log(self.cluster_counts[k]) - np.log(len(self.data_points) + self.alpha)
            
            log_probs[k] = likelihood + prior
        
        # 새 클러스터에 대한 확률
        # P(z | 새 클러스터) = ∫ P(z | θ*) P(θ* | H) dθ*
        # 근사적으로 z를 새 클러스터 중심으로 사용
        new_cov = 0.1 * torch.eye(self.latent_dim)  # 작은 분산
        new_likelihood = self._likelihood_logpdf(z, z, new_cov)
        new_prior = np.log(self.alpha) - np.log(len(self.data_points) + self.alpha)
        log_probs[K] = new_likelihood + new_prior
        
        # 소프트맥스로 정규화 후 샘플링
        probs = torch.softmax(log_probs, dim=0)
        assigned_cluster = torch.multinomial(probs, 1).item()
        
        if assigned_cluster == K:
            # 새 클러스터 생성
            self._create_new_cluster(z)
        else:
            # 기존 클러스터 업데이트
            self._update_cluster(assigned_cluster, z)
        
        self.data_assignments.append(assigned_cluster)
        self.data_points.append(z.clone())
        
        return assigned_cluster
    
    def _create_new_cluster(self, z: torch.Tensor):
        """새 클러스터 생성"""
        # θ*k ~ H(λ): 베이스 분포에서 샘플링
        # 근사적으로 첫 데이터 포인트를 중심으로 사용
        self.cluster_means.append(z.clone())
        self.cluster_covs.append(0.1 * torch.eye(self.latent_dim))
        self.cluster_counts.append(1)
        
        print(f"  Created new cluster {len(self.cluster_means)-1} at {z}")
    
    def _update_cluster(self, k: int, z: torch.Tensor):
        """기존 클러스터 k 업데이트 (온라인 평균)"""
        old_count = self.cluster_counts[k]
        new_count = old_count + 1
        old_mean = self.cluster_means[k]
        
        # 온라인 평균 업데이트
        new_mean = (old_count * old_mean + z) / new_count
        
        self.cluster_means[k] = new_mean
        self.cluster_counts[k] = new_count
        
        # 공분산도 온라인 업데이트 (간단한 버전)
        # 실제로는 더 복잡한 베이지안 업데이트가 필요
    
    def merge_clusters(self, threshold: float = 2.0):
        """가까운 클러스터들 병합"""
        if len(self.cluster_means) <= 1:
            return
        
        merged = True
        while merged:
            merged = False
            for i in range(len(self.cluster_means)):
                for j in range(i+1, len(self.cluster_means)):
                    # 클러스터 간 거리 계산
                    dist_ij = torch.norm(self.cluster_means[i] - self.cluster_means[j])
                    
                    if dist_ij < threshold:
                        print(f"  Merging clusters {i} and {j} (distance: {dist_ij:.3f})")
                        
                        # 클러스터 i와 j 병합
                        count_i, count_j = self.cluster_counts[i], self.cluster_counts[j]
                        total_count = count_i + count_j
                        
                        # 가중 평균으로 새 중심 계산
                        new_mean = (count_i * self.cluster_means[i] + count_j * self.cluster_means[j]) / total_count
                        
                        # 클러스터 i 업데이트, j 제거
                        self.cluster_means[i] = new_mean
                        self.cluster_counts[i] = total_count
                        
                        del self.cluster_means[j]
                        del self.cluster_covs[j]
                        del self.cluster_counts[j]
                        
                        # 할당 업데이트
                        for idx, assignment in enumerate(self.data_assignments):
                            if assignment == j:
                                self.data_assignments[idx] = i
                            elif assignment > j:
                                self.data_assignments[idx] -= 1
                        
                        merged = True
                        break
                if merged:
                    break
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """클러스터 정보 반환"""
        return {
            'num_clusters': len(self.cluster_means),
            'cluster_counts': self.cluster_counts.copy(),
            'cluster_means': [m.clone() for m in self.cluster_means],
            'total_data_points': len(self.data_points),
            'alpha': self.alpha
        }
    
    def get_cluster_assignment_probs(self, z: torch.Tensor) -> torch.Tensor:
        """주어진 데이터 포인트의 각 클러스터 할당 확률 반환"""
        if not torch.is_tensor(z):
            z = torch.tensor(z, dtype=torch.float32)
        
        if len(self.cluster_means) == 0:
            return torch.tensor([1.0])  # 첫 번째 클러스터
        
        K = len(self.cluster_means)
        log_probs = torch.zeros(K + 1)
        
        for k in range(K):
            likelihood = self._likelihood_logpdf(z, self.cluster_means[k], self.cluster_covs[k])
            prior = np.log(self.cluster_counts[k]) - np.log(len(self.data_points) + self.alpha)
            log_probs[k] = likelihood + prior
        
        # 새 클러스터
        new_cov = 0.1 * torch.eye(self.latent_dim)
        new_likelihood = self._likelihood_logpdf(z, z, new_cov)
        new_prior = np.log(self.alpha) - np.log(len(self.data_points) + self.alpha)
        log_probs[K] = new_likelihood + new_prior
        
        return torch.softmax(log_probs, dim=0)