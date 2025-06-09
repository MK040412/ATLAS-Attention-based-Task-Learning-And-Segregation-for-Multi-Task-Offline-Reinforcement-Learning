import numpy as np
from scipy.stats import multivariate_normal
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F

class GaussianCluster:
    def __init__(self, id: int, mu: np.ndarray, sigma_sq: float, N: int = 1):
        self.id = id
        self.mu = mu.copy()
        self.sigma_sq = sigma_sq
        self.N = N
        
        # 온라인 업데이트를 위한 충분 통계량
        self.sum_x = mu * N
        self.sum_x_sq = np.outer(mu, mu) * N + np.eye(len(mu)) * sigma_sq * N
        
        # 공분산 행렬 (대각 행렬로 가정)
        self.cov = np.eye(len(mu)) * sigma_sq

def kl_divergence_gaussian(mu1, cov1, mu2, cov2):
    """
    두 다변량 가우시안 분포 간의 KL divergence 계산
    KL(P||Q) = 0.5 * [tr(Σ_Q^{-1} Σ_P) + (μ_Q - μ_P)^T Σ_Q^{-1} (μ_Q - μ_P) - k + ln(det(Σ_Q)/det(Σ_P))]
    """
    k = len(mu1)  # 차원
    
    # 수치적 안정성을 위한 정규화
    eps = 1e-6
    cov1_reg = cov1 + eps * np.eye(k)
    cov2_reg = cov2 + eps * np.eye(k)
    
    try:
        # 역행렬 계산
        cov2_inv = np.linalg.inv(cov2_reg)
        
        # 각 항 계산
        mu_diff = mu2 - mu1
        
        # tr(Σ_Q^{-1} Σ_P)
        trace_term = np.trace(cov2_inv @ cov1_reg)
        
        # (μ_Q - μ_P)^T Σ_Q^{-1} (μ_Q - μ_P)
        quad_term = mu_diff.T @ cov2_inv @ mu_diff
        
        # ln(det(Σ_Q)/det(Σ_P))
        det1 = np.linalg.det(cov1_reg)
        det2 = np.linalg.det(cov2_reg)
        log_det_term = np.log(det2 / det1)
        
        kl = 0.5 * (trace_term + quad_term - k + log_det_term)
        
        return max(0, kl)  # KL divergence는 항상 >= 0
        
    except np.linalg.LinAlgError:
        # 특이행렬인 경우 매우 큰 값 반환
        return 1e6

def symmetric_kl_divergence(mu1, cov1, mu2, cov2):
    """대칭 KL divergence: (KL(P||Q) + KL(Q||P)) / 2"""
    kl_pq = kl_divergence_gaussian(mu1, cov1, mu2, cov2)
    kl_qp = kl_divergence_gaussian(mu2, cov2, mu1, cov1)
    return (kl_pq + kl_qp) / 2.0

class KLDivergenceDPMM:
    def __init__(self, latent_dim: int, alpha: float = 1.0, sigma_sq: float = 1.0,
                 tau_birth: float = 1e-4, tau_merge_kl: float = 0.1):
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.sigma_sq = sigma_sq
        self.tau_birth = tau_birth
        self.tau_merge_kl = tau_merge_kl  # KL divergence 임계값
        
        self.clusters: Dict[int, GaussianCluster] = {}
        self.next_cluster_id = 0
        
        # 이벤트 추적
        self.birth_events = []
        self.merge_events = []
        self.assignment_history = []
        
        print(f"🧠 KL-Divergence DPMM initialized:")
        print(f"  Latent dim: {latent_dim}, α: {alpha}, σ²: {sigma_sq}")
        print(f"  Birth threshold: {tau_birth}, KL merge threshold: {tau_merge_kl}")
    
    def _gaussian_likelihood(self, z: np.ndarray, cluster: GaussianCluster) -> float:
        """가우시안 우도 계산"""
        diff = z - cluster.mu
        exp_term = -0.5 * np.sum(diff**2) / cluster.sigma_sq
        norm_term = -0.5 * self.latent_dim * np.log(2 * np.pi * cluster.sigma_sq)
        return exp_term + norm_term
    
    def assign(self, z: np.ndarray) -> int:
        """포인트를 클러스터에 할당"""
        if len(self.clusters) == 0:
            return self._create_new_cluster(z)
        
        # 기존 클러스터들과의 우도 계산
        max_likelihood = -np.inf
        best_cluster = None
        
        for cluster_id, cluster in self.clusters.items():
            likelihood = self._gaussian_likelihood(z, cluster)
            if likelihood > max_likelihood:
                max_likelihood = likelihood
                best_cluster = cluster_id
        
        # Birth heuristic 검사
        if max_likelihood < self.tau_birth:
            new_cluster_id = self._create_new_cluster(z)
            print(f"🐣 Birth triggered! New cluster {new_cluster_id} (likelihood: {max_likelihood:.2e})")
            return new_cluster_id
        else:
            self._update_cluster(best_cluster, z)
            return best_cluster
    
    def merge(self, verbose=True):
        """KL divergence 기반 클러스터 병합"""
        if len(self.clusters) < 2:
            if verbose:
                print("  🔄 KL-Merge: Not enough clusters")
            return
        
        cluster_ids = list(self.clusters.keys())
        merge_candidates = []
        
        if verbose:
            print(f"  🧮 Computing KL divergences for {len(cluster_ids)} clusters...")
        
        # 모든 쌍의 KL divergence 계산
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                id1, id2 = cluster_ids[i], cluster_ids[j]
                
                if id1 not in self.clusters or id2 not in self.clusters:
                    continue
                
                cluster1 = self.clusters[id1]
                cluster2 = self.clusters[id2]
                
                # 대칭 KL divergence 계산
                kl_div = symmetric_kl_divergence(
                    cluster1.mu, cluster1.cov,
                    cluster2.mu, cluster2.cov
                )
                
                if verbose and len(merge_candidates) < 3:  # 처음 몇 개만 출력
                    print(f"  📏 KL({id1}↔{id2}): {kl_div:.6f} (threshold: {self.tau_merge_kl})")
                
                if kl_div < self.tau_merge_kl:
                    merge_candidates.append((id1, id2, kl_div))
        
        if not merge_candidates:
            if verbose:
                print(f"  🔄 KL-Merge: No pairs within KL threshold {self.tau_merge_kl}")
            return
        
        # KL divergence 순으로 정렬하여 가장 가까운 것부터 병합
        merge_candidates.sort(key=lambda x: x[2])
        
        merged_count = 0
        for id1, id2, kl_div in merge_candidates:
            if id1 not in self.clusters or id2 not in self.clusters:
                continue
            
            if verbose:
                print(f"  🔗 KL-Merging clusters {id1} ↔ {id2} (KL: {kl_div:.6f})")
            
            self._merge_clusters(id1, id2)
            merged_count += 1
        
        if verbose:
            print(f"  ✅ KL-Merged {merged_count} pairs, {len(self.clusters)} clusters remaining")
    
    def _create_new_cluster(self, z: np.ndarray) -> int:
        """새 클러스터 생성"""
        cluster_id = self.next_cluster_id
        self.next_cluster_id += 1
        
        self.clusters[cluster_id] = GaussianCluster(
            id=cluster_id,
            mu=z.copy(),
            sigma_sq=self.sigma_sq
        )
        
        # Birth 이벤트 기록
        self.birth_events.append({
            'timestep': len(self.assignment_history),
            'cluster_id': cluster_id,
            'initial_mu': z.copy()
        })
        
        return cluster_id
    
    def _update_cluster(self, cluster_id: int, z: np.ndarray):
        """클러스터 온라인 업데이트"""
        cluster = self.clusters[cluster_id]
        
        # 온라인 평균 업데이트
        cluster.N += 1
        alpha = 1.0 / cluster.N
        cluster.mu = (1 - alpha) * cluster.mu + alpha * z
        
        # 공분산 행렬 업데이트 (대각 행렬 가정)
        cluster.cov = np.eye(self.latent_dim) * cluster.sigma_sq
    
    def _merge_clusters(self, id1: int, id2: int):
        """두 클러스터를 병합"""
        cluster1 = self.clusters[id1]
        cluster2 = self.clusters[id2]
        
        # 가중 평균으로 새 중심 계산
        total_N = cluster1.N + cluster2.N
        new_mu = (cluster1.N * cluster1.mu + cluster2.N * cluster2.mu) / total_N
        
        # 더 작은 ID를 유지
        keep_id = min(id1, id2)
        remove_id = max(id1, id2)
        
        # 클러스터 업데이트
        self.clusters[keep_id] = GaussianCluster(
            id=keep_id,
            mu=new_mu,
            sigma_sq=self.sigma_sq,
            N=total_N
        )
        
        # 제거할 클러스터 삭제
        del self.clusters[remove_id]
        
        # 병합 이벤트 기록
        self.merge_events.append({
            'timestep': len(self.assignment_history),
            'old_clusters': [id1, id2],
            'new_cluster': keep_id,
            'kl_divergence': symmetric_kl_divergence(
                cluster1.mu, cluster1.cov,
                cluster2.mu, cluster2.cov
            )
        })
    
    def get_cluster_info(self):
        """클러스터 정보 반환"""
        return {
            'num_clusters': len(self.clusters),
            'cluster_sizes': {cid: cluster.N for cid, cluster in self.clusters.items()},
            'total_births': len(self.birth_events),
            'total_merges': len(self.merge_events)
        }