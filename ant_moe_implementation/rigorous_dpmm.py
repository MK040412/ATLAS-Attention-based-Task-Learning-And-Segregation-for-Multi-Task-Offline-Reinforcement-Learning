import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class GaussianCluster:
    """가우시안 클러스터 정보"""
    id: int
    mu: np.ndarray          # 평균 벡터
    sigma_sq: float         # 분산 (σ²I)
    N: int                  # 데이터 포인트 수
    sum_x: np.ndarray       # Σx_i (온라인 업데이트용)
    sum_x_sq: float         # Σ||x_i||² (온라인 업데이트용)

class RigorousDPMM:
    """엄밀한 DPMM 구현 - Gaussian likelihood 기반"""
    
    def __init__(self, 
                 latent_dim: int,
                 alpha: float = 1.0,           # DP concentration parameter
                 sigma_sq: float = 1.0,        # 기본 분산
                 tau_birth: float = 1e-4,      # Birth 임계값
                 tau_merge: float = 0.5):      # Merge 임계값
        
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.sigma_sq = sigma_sq
        self.tau_birth = tau_birth
        self.tau_merge = tau_merge
        
        # 클러스터 저장소
        self.clusters: Dict[int, GaussianCluster] = {}
        self.next_cluster_id = 0
        
        # 통계 추적
        self.birth_events = []
        self.merge_events = []
        self.assignment_history = []
        
        print(f"🧠 Rigorous DPMM initialized:")
        print(f"  Latent dim: {latent_dim}")
        print(f"  α: {alpha}, σ²: {sigma_sq}")
        print(f"  τ_birth: {tau_birth}, τ_merge: {tau_merge}")
    
    def _gaussian_likelihood(self, z: np.ndarray, cluster: GaussianCluster) -> float:
        """가우시안 우도 계산: ℓ_k = 𝒩(z_t; μ_k, σ²I)"""
        diff = z - cluster.mu
        # 다변량 가우시안: (2πσ²)^(-d/2) * exp(-||z-μ||²/(2σ²))
        log_likelihood = (
            -0.5 * self.latent_dim * np.log(2 * np.pi * self.sigma_sq) -
            0.5 * np.sum(diff**2) / self.sigma_sq
        )
        return np.exp(log_likelihood)
    
    def _create_new_cluster(self, z: np.ndarray) -> int:
        """새 클러스터 생성"""
        cluster_id = self.next_cluster_id
        self.next_cluster_id += 1
        
        cluster = GaussianCluster(
            id=cluster_id,
            mu=z.copy(),
            sigma_sq=self.sigma_sq,
            N=1,
            sum_x=z.copy(),
            sum_x_sq=np.sum(z**2)
        )
        
        self.clusters[cluster_id] = cluster
        self.birth_events.append({
            'timestep': len(self.assignment_history),
            'cluster_id': cluster_id,
            'initial_point': z.copy()
        })
        
        print(f"🐣 New cluster {cluster_id} born at μ={z}")
        return cluster_id
    
    def _update_cluster(self, cluster_id: int, z: np.ndarray):
        """클러스터 온라인 업데이트"""
        cluster = self.clusters[cluster_id]
        
        # 온라인 평균 업데이트: μ_new = ((N-1)μ_old + z_t) / N
        cluster.N += 1
        cluster.sum_x += z
        cluster.mu = cluster.sum_x / cluster.N
        cluster.sum_x_sq += np.sum(z**2)
        
        # 분산 업데이트 (선택적)
        if cluster.N > 1:
            # 샘플 분산: σ² = (Σ||x_i||² - N||μ||²) / (N-1)
            cluster.sigma_sq = max(
                (cluster.sum_x_sq - cluster.N * np.sum(cluster.mu**2)) / (cluster.N - 1),
                0.01  # 최소 분산
            )
    
    def assign(self, z: np.ndarray) -> int:
        """DPMM 클러스터 할당 - Birth Heuristic 포함"""
        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy()
        
        z = z.flatten()  # 1D로 변환
        
        # 빈 클러스터면 첫 번째 클러스터 생성
        if not self.clusters:
            cluster_id = self._create_new_cluster(z)
            self.assignment_history.append(cluster_id)
            return cluster_id
        
        # 모든 클러스터에 대해 우도 계산
        likelihoods = {}
        for cluster_id, cluster in self.clusters.items():
            likelihoods[cluster_id] = self._gaussian_likelihood(z, cluster)
        
        max_likelihood = max(likelihoods.values())
        best_cluster_id = max(likelihoods.keys(), key=lambda k: likelihoods[k])
        
        print(f"📊 Likelihoods: {[(k, f'{v:.6f}') for k, v in likelihoods.items()]}")
        print(f"   Max likelihood: {max_likelihood:.6f}, threshold: {self.tau_birth}")
        
        # Birth Heuristic: max_k ℓ_k < τ_birth
        if max_likelihood < self.tau_birth:
            cluster_id = self._create_new_cluster(z)
            print(f"🐣 Birth triggered! New cluster {cluster_id}")
        else:
            cluster_id = best_cluster_id
            self._update_cluster(cluster_id, z)
            print(f"📍 Assigned to existing cluster {cluster_id}")
        
        self.assignment_history.append(cluster_id)
        return cluster_id
    
    def merge(self):
        """Merge Heuristic: d_{ij} = ∥μ_i - μ_j∥ < τ_merge"""
        if len(self.clusters) < 2:
            return
        
        cluster_ids = list(self.clusters.keys())
        merge_pairs = []
        
        # 모든 클러스터 쌍에 대해 거리 계산
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                id_i, id_j = cluster_ids[i], cluster_ids[j]
                mu_i = self.clusters[id_i].mu
                mu_j = self.clusters[id_j].mu
                
                distance = np.linalg.norm(mu_i - mu_j)
                
                if distance < self.tau_merge:
                    merge_pairs.append((id_i, id_j, distance))
        
        # 가장 가까운 쌍부터 병합
        merge_pairs.sort(key=lambda x: x[2])
        
        for id_i, id_j, distance in merge_pairs:
            if id_i in self.clusters and id_j in self.clusters:
                self._merge_clusters(id_i, id_j)
                print(f"🔗 Merged clusters {id_i} and {id_j} (distance: {distance:.4f})")
    
    def _merge_clusters(self, id_i: int, id_j: int) -> int:
        """두 클러스터 병합"""
        cluster_i = self.clusters[id_i]
        cluster_j = self.clusters[id_j]
        
        # 가중 평균으로 새 평균 계산
        N_new = cluster_i.N + cluster_j.N
        mu_new = (cluster_i.N * cluster_i.mu + cluster_j.N * cluster_j.mu) / N_new
        
        # 새 클러스터 생성
        new_id = self.next_cluster_id
        self.next_cluster_id += 1
        
        merged_cluster = GaussianCluster(
            id=new_id,
            mu=mu_new,
            sigma_sq=(cluster_i.sigma_sq + cluster_j.sigma_sq) / 2,  # 평균 분산
            N=N_new,
            sum_x=cluster_i.sum_x + cluster_j.sum_x,
            sum_x_sq=cluster_i.sum_x_sq + cluster_j.sum_x_sq
        )
        
        # 병합 이벤트 기록
        self.merge_events.append({
            'timestep': len(self.assignment_history),
            'old_clusters': [id_i, id_j],
            'new_cluster': new_id,
            'old_mus': [cluster_i.mu.copy(), cluster_j.mu.copy()],
            'new_mu': mu_new.copy()
        })
        
        # 클러스터 업데이트
        self.clusters[new_id] = merged_cluster
        del self.clusters[id_i]
        del self.clusters[id_j]
        
        return new_id
    
    def get_cluster_count(self) -> int:
        """현재 클러스터 수 K_t"""
        return len(self.clusters)
    
    def get_statistics(self) -> Dict:
        """DPMM 통계"""
        return {
            'num_clusters': len(self.clusters),
            'cluster_sizes': {k: c.N for k, c in self.clusters.items()},
            'cluster_means': {k: c.mu.tolist() for k, c in self.clusters.items()},
            'cluster_variances': {k: c.sigma_sq for k, c in self.clusters.items()},
            'total_births': len(self.birth_events),
            'total_merges': len(self.merge_events),
            'assignment_history_length': len(self.assignment_history)
        }
    
    def get_cluster_evolution(self) -> Tuple[List[int], List[int], List[int]]:
        """K_t 진화 추적"""
        timesteps = []
        cluster_counts = []
        birth_indicators = []
        
        K_t = 0
        for t, assignment in enumerate(self.assignment_history):
            # Birth 체크
            birth_at_t = any(event['timestep'] == t for event in self.birth_events)
            if birth_at_t:
                K_t += 1
            
            # Merge 체크 (단순화: merge는 즉시 -1)
            merge_at_t = any(event['timestep'] == t for event in self.merge_events)
            if merge_at_t:
                K_t -= 1  # 단순화 (실제로는 더 복잡)
            
            timesteps.append(t)
            cluster_counts.append(K_t)
            birth_indicators.append(int(birth_at_t))
        
        return timesteps, cluster_counts, birth_indicators