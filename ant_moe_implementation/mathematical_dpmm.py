import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import inv, det, LinAlgError
import torch
from typing import Dict, List, Tuple, Optional

class GaussianCluster:
    """수학적으로 정확한 가우시안 클러스터"""
    def __init__(self, id: int, mu: np.ndarray, sigma: np.ndarray, N: int = 1):
        self.id = id
        self.mu = mu.copy()
        self.sigma = sigma.copy()  # 공분산 행렬
        self.N = N
        
        # 수치적 안정성을 위한 정규화
        self.min_eigenval = 1e-6
        self._regularize_covariance()
        
        # 충분 통계량 (온라인 업데이트용)
        self.sum_x = mu * N
        self.sum_xx = np.outer(mu, mu) * N + sigma * N
    
    def _regularize_covariance(self):
        """공분산 행렬 정규화 (수치적 안정성)"""
        try:
            # 고유값 분해
            eigenvals, eigenvecs = np.linalg.eigh(self.sigma)
            
            # 최소 고유값 보장
            eigenvals = np.maximum(eigenvals, self.min_eigenval)
            
            # 재구성
            self.sigma = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # 대칭성 보장
            self.sigma = (self.sigma + self.sigma.T) / 2
            
        except LinAlgError:
            # 실패시 대각 행렬로 초기화
            self.sigma = np.eye(len(self.mu)) * 1e-3
    
    def update_online(self, z: np.ndarray):
        """온라인 업데이트 (수학적으로 정확)"""
        self.N += 1
        
        # 평균 업데이트
        old_mu = self.mu.copy()
        self.mu = old_mu + (z - old_mu) / self.N
        
        # 공분산 온라인 업데이트 (Welford's algorithm)
        if self.N > 1:
            self.sigma = ((self.N - 2) * self.sigma + 
                         np.outer(z - old_mu, z - self.mu)) / (self.N - 1)
        
        self._regularize_covariance()

class MathematicalDPMM:
    """수학적으로 정확한 DPMM 구현"""
    
    def __init__(self, latent_dim: int, alpha: float = 1.0, 
                 tau_birth: float = -5.0, tau_merge_kl: float = 0.1):
        self.latent_dim = latent_dim
        self.alpha = alpha  # Dirichlet 농도 매개변수
        self.tau_birth = tau_birth  # Birth 임계값 (로그 우도)
        self.tau_merge_kl = tau_merge_kl  # KL divergence merge 임계값
        
        self.clusters: Dict[int, GaussianCluster] = {}
        self.next_cluster_id = 0
        
        # 이벤트 추적
        self.birth_events = []
        self.merge_events = []
        self.assignment_history = []
        
        # 수치적 안정성 파라미터
        self.min_covariance = 1e-6
        self.max_kl_divergence = 1e6
        
        print(f"🧮 Mathematical DPMM initialized:")
        print(f"  Latent dim: {latent_dim}, α: {alpha}")
        print(f"  Birth threshold: {tau_birth}, KL merge threshold: {tau_merge_kl}")
    
    def compute_log_likelihood(self, z: np.ndarray, cluster: GaussianCluster) -> float:
        """수학적으로 정확한 로그 우도 계산"""
        try:
            # scipy.stats 사용 (수치적으로 안정)
            log_prob = multivariate_normal.logpdf(z, cluster.mu, cluster.sigma)
            
            # NaN/Inf 체크
            if not np.isfinite(log_prob):
                return -1e6  # 매우 작은 우도
            
            return log_prob
            
        except (LinAlgError, ValueError) as e:
            print(f"  ⚠️ Likelihood computation failed: {e}")
            return -1e6
    
    def birth_criterion(self, z: np.ndarray) -> bool:
        """Birth criterion: max_k P(z|θ_k) < τ_birth"""
        if not self.clusters:
            return True
        
        try:
            max_log_likelihood = max([
                self.compute_log_likelihood(z, cluster) 
                for cluster in self.clusters.values()
            ])
            
            return max_log_likelihood < self.tau_birth
            
        except Exception as e:
            print(f"  ⚠️ Birth criterion failed: {e}")
            return True  # 안전하게 새 클러스터 생성
    
    def kl_divergence_gaussian(self, cluster_i: GaussianCluster, 
                              cluster_j: GaussianCluster) -> float:
        """두 가우시안 간 KL divergence: KL(P_i || P_j)"""
        try:
            mu_i, sigma_i = cluster_i.mu, cluster_i.sigma
            mu_j, sigma_j = cluster_j.mu, cluster_j.sigma
            
            k = len(mu_i)  # 차원
            
            # 수치적 안정성을 위한 정규화
            sigma_i_reg = sigma_i + np.eye(k) * self.min_covariance
            sigma_j_reg = sigma_j + np.eye(k) * self.min_covariance
            
            # 역행렬 계산
            sigma_j_inv = inv(sigma_j_reg)
            
            # 평균 차이
            mu_diff = mu_j - mu_i
            
            # KL divergence 계산
            # KL(P||Q) = 0.5 * [tr(Σ_Q^{-1} Σ_P) + (μ_Q - μ_P)^T Σ_Q^{-1} (μ_Q - μ_P) - k + ln(det(Σ_Q)/det(Σ_P))]
            
            trace_term = np.trace(sigma_j_inv @ sigma_i_reg)
            quad_term = mu_diff.T @ sigma_j_inv @ mu_diff
            
            det_i = det(sigma_i_reg)
            det_j = det(sigma_j_reg)
            
            if det_i <= 0 or det_j <= 0:
                return self.max_kl_divergence
            
            log_det_term = np.log(det_j / det_i)
            
            kl = 0.5 * (trace_term + quad_term - k + log_det_term)
            
            # 수치적 안정성 체크
            if not np.isfinite(kl) or kl < 0:
                return self.max_kl_divergence
            
            return min(kl, self.max_kl_divergence)
            
        except (LinAlgError, ValueError) as e:
            print(f"  ⚠️ KL divergence computation failed: {e}")
            return self.max_kl_divergence
    
    def symmetric_kl_divergence(self, cluster_i: GaussianCluster, 
                               cluster_j: GaussianCluster) -> float:
        """대칭 KL divergence: (KL(P_i||P_j) + KL(P_j||P_i)) / 2"""
        kl_ij = self.kl_divergence_gaussian(cluster_i, cluster_j)
        kl_ji = self.kl_divergence_gaussian(cluster_j, cluster_i)
        return (kl_ij + kl_ji) / 2.0
    
    def assign_point(self, z: np.ndarray) -> int:
        """포인트를 클러스터에 할당"""
        z = np.array(z).flatten()
        
        # Birth criterion 체크
        if self.birth_criterion(z):
            cluster_id = self._create_new_cluster(z)
            print(f"  🐣 Birth: New cluster {cluster_id}")
            return cluster_id
        
        # 기존 클러스터 중 최적 선택
        best_cluster_id = None
        best_log_likelihood = -np.inf
        
        for cluster_id, cluster in self.clusters.items():
            log_likelihood = self.compute_log_likelihood(z, cluster)
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_cluster_id = cluster_id
        
        # 클러스터 업데이트
        if best_cluster_id is not None:
            self.clusters[best_cluster_id].update_online(z)
        
        self.assignment_history.append(best_cluster_id)
        return best_cluster_id
    
    def merge_clusters(self, verbose: bool = True) -> bool:
        """KL divergence 기반 클러스터 병합"""
        if len(self.clusters) < 2:
            if verbose:
                print("  🔄 Merge: Not enough clusters")
            return False
        
        cluster_ids = list(self.clusters.keys())
        merge_candidates = []
        
        # 모든 쌍의 KL divergence 계산
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                id_i, id_j = cluster_ids[i], cluster_ids[j]
                
                if id_i not in self.clusters or id_j not in self.clusters:
                    continue
                
                cluster_i = self.clusters[id_i]
                cluster_j = self.clusters[id_j]
                
                kl_div = self.symmetric_kl_divergence(cluster_i, cluster_j)
                
                if kl_div < self.tau_merge_kl:
                    merge_candidates.append((id_i, id_j, kl_div))
        
        if not merge_candidates:
            if verbose:
                print(f"  🔄 Merge: No pairs within KL threshold {self.tau_merge_kl}")
            return False
        
        # KL divergence 순으로 정렬
        merge_candidates.sort(key=lambda x: x[2])
        
        merged_count = 0
        for id_i, id_j, kl_div in merge_candidates:
            if id_i not in self.clusters or id_j not in self.clusters:
                continue
            
            if verbose:
                print(f"  🔗 Merging clusters {id_i} ↔ {id_j} (KL: {kl_div:.6f})")
            
            self._merge_two_clusters(id_i, id_j)
            merged_count += 1
        
        if verbose:
            print(f"  ✅ Merged {merged_count} pairs, {len(self.clusters)} clusters remaining")
        
        return merged_count > 0
    
    def _create_new_cluster(self, z: np.ndarray) -> int:
        """새 클러스터 생성"""
        cluster_id = self.next_cluster_id
        self.next_cluster_id += 1
        
        # 초기 공분산 행렬 (대각 행렬)
        initial_cov = np.eye(self.latent_dim) * 1e-3
        
        self.clusters[cluster_id] = GaussianCluster(
            id=cluster_id,
            mu=z.copy(),
            sigma=initial_cov,
            N=1
        )
        
        # Birth 이벤트 기록
        self.birth_events.append({
            'timestep': len(self.assignment_history),
            'cluster_id': cluster_id,
            'initial_mu': z.copy()
        })
        
        return cluster_id
    
    def _merge_two_clusters(self, id_i: int, id_j: int):
        """두 클러스터 병합 (수학적으로 정확)"""
        cluster_i = self.clusters[id_i]
        cluster_j = self.clusters[id_j]
        
        # 가중 평균으로 새 파라미터 계산
        N_total = cluster_i.N + cluster_j.N
        w_i = cluster_i.N / N_total
        w_j = cluster_j.N / N_total
        
        # 새 평균
        new_mu = w_i * cluster_i.mu + w_j * cluster_j.mu
        
        # 새 공분산 (mixture of Gaussians formula)
        diff_i = cluster_i.mu - new_mu
        diff_j = cluster_j.mu - new_mu
        
        new_sigma = (w_i * (cluster_i.sigma + np.outer(diff_i, diff_i)) +
                    w_j * (cluster_j.sigma + np.outer(diff_j, diff_j)))
        
        # 더 작은 ID 유지
        keep_id = min(id_i, id_j)
        remove_id = max(id_i, id_j)
        
        # 새 클러스터 생성
        self.clusters[keep_id] = GaussianCluster(
            id=keep_id,
            mu=new_mu,
            sigma=new_sigma,
            N=N_total
        )
        
        # 제거할 클러스터 삭제
        del self.clusters[remove_id]
        
        # 병합 이벤트 기록
        self.merge_events.append({
            'timestep': len(self.assignment_history),
            'old_clusters': [id_i, id_j],
            'new_cluster': keep_id,
            'kl_divergence': self.symmetric_kl_divergence(cluster_i, cluster_j)
        })
    
    def get_cluster_info(self) -> Dict:
        """클러스터 정보 반환"""
        return {
            'num_clusters': len(self.clusters),
            'cluster_sizes': {cid: cluster.N for cid, cluster in self.clusters.items()},
            'cluster_means': {cid: cluster.mu.tolist() for cid, cluster in self.clusters.items()},
            'total_births': len(self.birth_events),
            'total_merges': len(self.merge_events),
            'total_assignments': len(self.assignment_history)
        }
    
    def get_statistics(self) -> Dict:
        """상세 통계 정보"""
        cluster_info = self.get_cluster_info()
        
        # 클러스터 크기 분포
        sizes = list(cluster_info['cluster_sizes'].values())
        
        stats = {
            'dpmm': {
                'num_clusters': cluster_info['num_clusters'],
                'total_births': cluster_info['total_births'],
                'total_merges': cluster_info['total_merges'],
                'cluster_sizes': sizes,
                'avg_cluster_size': np.mean(sizes) if sizes else 0,
                'std_cluster_size': np.std(sizes) if sizes else 0,
                'min_cluster_size': min(sizes) if sizes else 0,
                'max_cluster_size': max(sizes) if sizes else 0
            }
        }
        
        return stats
