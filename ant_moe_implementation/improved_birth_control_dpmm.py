import numpy as np
from typing import Dict, List, Tuple
import warnings

class AdaptiveBirthDPMM:
    """Birth 문제를 해결한 적응적 DPMM"""
    
    def __init__(self, latent_dim: int, 
                 initial_tau_birth: float = -2.0,
                 tau_merge_kl: float = 0.3,
                 initial_variance: float = 0.2,
                 adaptive_birth: bool = True,
                 max_clusters: int = 20):
        
        self.latent_dim = latent_dim
        self.tau_birth = initial_tau_birth
        self.initial_tau_birth = initial_tau_birth
        self.tau_merge_kl = tau_merge_kl
        self.initial_variance = initial_variance
        self.adaptive_birth = adaptive_birth
        self.max_clusters = max_clusters
        
        self.clusters = {}
        self.next_cluster_id = 0
        
        # 적응적 제어를 위한 통계
        self.birth_history = []
        self.assignment_count = 0
        self.recent_birth_rate = 0.0
        self.target_birth_rate = 0.15  # 목표: 15% birth rate
        
        # 이벤트 추적
        self.birth_events = []
        self.merge_events = []
        
        print(f"🧠 Adaptive Birth DPMM initialized:")
        print(f"  • Initial tau_birth: {initial_tau_birth}")
        print(f"  • Target birth rate: {self.target_birth_rate * 100:.1f}%")
        print(f"  • Max clusters: {max_clusters}")
    
    def _update_adaptive_threshold(self):
        """적응적 임계값 업데이트"""
        if not self.adaptive_birth or len(self.birth_history) < 20:
            return
        
        # 최근 birth rate 계산
        recent_window = min(50, len(self.birth_history))
        recent_births = sum(self.birth_history[-recent_window:])
        self.recent_birth_rate = recent_births / recent_window
        
        # 임계값 조정
        if self.recent_birth_rate > self.target_birth_rate * 1.5:
            # Birth rate가 너무 높으면 더 엄격하게
            self.tau_birth -= 0.1
            print(f"  📉 Decreasing tau_birth to {self.tau_birth:.2f} (birth rate: {self.recent_birth_rate:.2%})")
        elif self.recent_birth_rate < self.target_birth_rate * 0.5:
            # Birth rate가 너무 낮으면 더 관대하게
            self.tau_birth += 0.05
            print(f"  📈 Increasing tau_birth to {self.tau_birth:.2f} (birth rate: {self.recent_birth_rate:.2%})")
        
        # 임계값 범위 제한
        self.tau_birth = np.clip(self.tau_birth, -6.0, 0.0)
    
    def assign_point(self, z: np.ndarray) -> int:
        """개선된 포인트 할당"""
        z = np.array(z).flatten()
        self.assignment_count += 1
        
        # 클러스터 수 제한 체크
        if len(self.clusters) >= self.max_clusters:
            # 최대 클러스터 수에 도달하면 강제로 기존 클러스터에 할당
            return self._assign_to_existing_cluster(z)
        
        # Birth 결정
        should_birth = self._should_create_new_cluster(z)
        self.birth_history.append(should_birth)
        
        if should_birth:
            cluster_id = self._create_new_cluster(z)
            
            # 주기적으로 적응적 임계값 업데이트
            if self.assignment_count % 20 == 0:
                self._update_adaptive_threshold()
            
            return cluster_id
        else:
            return self._assign_to_existing_cluster(z)
    
    def _should_create_new_cluster(self, z: np.ndarray) -> bool:
        """새 클러스터 생성 여부 결정"""
        if len(self.clusters) == 0:
            return True
        
        # 모든 클러스터와의 우도 계산
        max_log_likelihood = -np.inf
        for cluster in self.clusters.values():
            log_likelihood = self._compute_log_likelihood(z, cluster)
            max_log_likelihood = max(max_log_likelihood, log_likelihood)
        
        # Birth criterion
        return max_log_likelihood < self.tau_birth
    
    def _assign_to_existing_cluster(self, z: np.ndarray) -> int:
        """기존 클러스터 중 최적에 할당"""
        if not self.clusters:
            return self._create_new_cluster(z)
        
        best_cluster_id = None
        best_log_likelihood = -np.inf
        
        for cluster_id, cluster in self.clusters.items():
            log_likelihood = self._compute_log_likelihood(z, cluster)
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_cluster_id = cluster_id
        
        # 클러스터 업데이트
        self._update_cluster(best_cluster_id, z)
        return best_cluster_id
    
    def _create_new_cluster(self, z: np.ndarray) -> int:
        """새 클러스터 생성"""
        cluster_id = self.next_cluster_id
        self.next_cluster_id += 1
        
        # 더 큰 초기 공분산으로 클러스터 생성
        initial_cov = np.eye(self.latent_dim) * self.initial_variance
        
        self.clusters[cluster_id] = {
            'id': cluster_id,
            'mu': z.copy(),
            'sigma': initial_cov,
            'N': 1,
            'sum_x': z.copy(),
            'sum_xx': np.outer(z, z)
        }
        
        self.birth_events.append({
            'timestep': self.assignment_count,
            'cluster_id': cluster_id,
            'tau_birth_used': self.tau_birth
        })
        
        return cluster_id
    
    def _update_cluster(self, cluster_id: int, z: np.ndarray):
        """클러스터 온라인 업데이트 (Welford's algorithm 개선)"""
        cluster = self.clusters[cluster_id]
        
        # 이전 값들
        old_N = cluster['N']
        old_mu = cluster['mu'].copy()
        
        # 새로운 값들
        new_N = old_N + 1
        new_mu = old_mu + (z - old_mu) / new_N
        
        # 공분산 업데이트 (더 적극적)
        if old_N == 1:
            # 두 번째 포인트일 때
            diff = z - old_mu
            cluster['sigma'] = np.outer(diff, diff) / 2 + np.eye(self.latent_dim) * self.initial_variance
        else:
            # Welford's algorithm for covariance
            diff_old = z - old_mu
            diff_new = z - new_mu
            
            # 더 큰 학습률 적용
            learning_rate = min(0.1, 2.0 / new_N)
            cov_update = np.outer(diff_old, diff_new) * learning_rate
            
            cluster['sigma'] = (1 - learning_rate) * cluster['sigma'] + cov_update
            
            # 최소 분산 보장
            cluster['sigma'] += np.eye(self.latent_dim) * 1e-4
        
        # 업데이트
        cluster['N'] = new_N
        cluster['mu'] = new_mu
        cluster['sum_x'] += z
        cluster['sum_xx'] += np.outer(z, z)
    
    def _compute_log_likelihood(self, z: np.ndarray, cluster: dict) -> float:
        """로그 우도 계산 (수치적 안정성 개선)"""
        try:
            mu = cluster['mu']
            sigma = cluster['sigma']
            
            # 정규화 및 안정성 체크
            sigma_reg = sigma + np.eye(len(mu)) * 1e-6
            
            # 역행렬 계산
            sigma_inv = np.linalg.inv(sigma_reg)
            det_sigma = np.linalg.det(sigma_reg)
            
            if det_sigma <= 0:
                return -np.inf
            
            # 로그 우도 계산
            diff = z - mu
            quad_term = -0.5 * diff.T @ sigma_inv @ diff
            norm_term = -0.5 * (len(mu) * np.log(2 * np.pi) + np.log(det_sigma))
            
            return quad_term + norm_term
            
        except (np.linalg.LinAlgError, ValueError):
            return -np.inf
    
    def merge_clusters_aggressive(self, verbose: bool = True):
        """더 적극적인 클러스터 병합"""
        if len(self.clusters) < 2:
            return
        
        cluster_ids = list(self.clusters.keys())
        merge_candidates = []
        
        # KL divergence 계산
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                id1, id2 = cluster_ids[i], cluster_ids[j]
                
                cluster1 = self.clusters[id1]
                cluster2 = self.clusters[id2]
                
                # 대칭 KL divergence
                kl_div = self._symmetric_kl_divergence(
                    cluster1['mu'], cluster1['sigma'],
                    cluster2['mu'], cluster2['sigma']
                )
                
                if kl_div < self.tau_merge_kl:
                    merge_candidates.append((id1, id2, kl_div))
        
        # 병합 실행
        merged_count = 0
        for id1, id2, kl_div in sorted(merge_candidates, key=lambda x: x[2]):
            if id1 in self.clusters and id2 in self.clusters:
                self._merge_two_clusters(id1, id2)
                merged_count += 1
                
                if verbose:
                    print(f"  🔗 Merged clusters {id1}↔{id2} (KL: {kl_div:.4f})")
        
        if verbose and merged_count > 0:
            print(f"  ✅ Merged {merged_count} pairs, {len(self.clusters)} clusters remaining")
    
    def _symmetric_kl_divergence(self, mu1, sigma1, mu2, sigma2):
        """대칭 KL divergence 계산"""
        try:
            kl1 = self._kl_divergence_gaussian(mu1, sigma1, mu2, sigma2)
            kl2 = self._kl_divergence_gaussian(mu2, sigma2, mu1, sigma1)
            return (kl1 + kl2) / 2.0
        except:
            return np.inf
    
    def _kl_divergence_gaussian(self, mu1, sigma1, mu2, sigma2):
        """가우시안 KL divergence"""
        k = len(mu1)
        
        # 정규화
        sigma1_reg = sigma1 + np.eye(k) * 1e-6
        sigma2_reg = sigma2 + np.eye(k) * 1e-6
        
        sigma2_inv = np.linalg.inv(sigma2_reg)
        
        mu_diff = mu2 - mu1
        trace_term = np.trace(sigma2_inv @ sigma1_reg)
        quad_term = mu_diff.T @ sigma2_inv @ mu_diff
        log_det_term = np.log(np.linalg.det(sigma2_reg) / np.linalg.det(sigma1_reg))
        
        return 0.5 * (trace_term + quad_term - k + log_det_term)
    
    def _merge_two_clusters(self, id1: int, id2: int):
        """두 클러스터 병합"""
        cluster1 = self.clusters[id1]
        cluster2 = self.clusters[id2]
        
        # 가중 평균으로 새 파라미터 계산
        N1, N2 = cluster1['N'], cluster2['N']
        total_N = N1 + N2
        
        new_mu = (N1 * cluster1['mu'] + N2 * cluster2['mu']) / total_N
        
        # 공분산 병합 (더 정확한 공식)
        diff1 = cluster1['mu'] - new_mu
        diff2 = cluster2['mu'] - new_mu
        
        new_sigma = (N1 * (cluster1['sigma'] + np.outer(diff1, diff1)) + 
                     N2 * (cluster2['sigma'] + np.outer(diff2, diff2))) / total_N
        
        # 더 작은 ID 유지
        keep_id = min(id1, id2)
        remove_id = max(id1, id2)
        
        # 업데이트
        self.clusters[keep_id] = {
            'id': keep_id,
            'mu': new_mu,
            'sigma': new_sigma,
            'N': total_N,
            'sum_x': cluster1['sum_x'] + cluster2['sum_x'],
            'sum_xx': cluster1['sum_xx'] + cluster2['sum_xx']
        }
        
        del self.clusters[remove_id]
        
        self.merge_events.append({
            'timestep': self.assignment_count,
            'merged_clusters': [id1, id2],
            'result_cluster': keep_id
        })
    
    def get_statistics(self):
        """통계 정보 반환"""
        return {
            'num_clusters': len(self.clusters),
            'total_assignments': self.assignment_count,
            'total_births': len(self.birth_events),
            'total_merges': len(self.merge_events),
            'current_tau_birth': self.tau_birth,
            'recent_birth_rate': self.recent_birth_rate,
            'cluster_sizes': [cluster['N'] for cluster in self.clusters.values()]
        }
