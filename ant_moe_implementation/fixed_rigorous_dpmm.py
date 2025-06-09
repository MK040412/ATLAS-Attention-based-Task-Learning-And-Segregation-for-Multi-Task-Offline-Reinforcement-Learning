class RigorousDPMM:
    def __init__(self, latent_dim: int, alpha: float = 1.0, sigma_sq: float = 1.0,
                 tau_birth: float = 1e-4, tau_merge: float = 0.5):
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.sigma_sq = sigma_sq
        self.tau_birth = tau_birth
        self.tau_merge = tau_merge
        
        self.clusters: Dict[int, GaussianCluster] = {}
        self.next_cluster_id = 0
        self.assignment_cache = {}  # 캐시 추가
        
        # 이벤트 추적
        self.birth_events = []
        self.merge_events = []
        self.assignment_history = []
        
        print(f"🧠 Rigorous DPMM initialized:")
        print(f"  Latent dim: {latent_dim}, α: {alpha}, σ²: {sigma_sq}")
        print(f"  Birth threshold: {tau_birth}, Merge threshold: {tau_merge}")
    
    def assign(self, z: np.ndarray) -> int:
        """포인트를 클러스터에 할당 (캐시 사용)"""
        # 캐시 키 생성 (부동소수점 정밀도 문제 해결)
        cache_key = tuple(np.round(z, decimals=6))
        
        # 캐시에서 확인
        if cache_key in self.assignment_cache:
            cached_cluster = self.assignment_cache[cache_key]
            # 클러스터가 여전히 존재하는지 확인
            if cached_cluster in self.clusters:
                return cached_cluster
            else:
                # 캐시에서 제거
                del self.assignment_cache[cache_key]
        
        if len(self.clusters) == 0:
            # 첫 번째 클러스터 생성
            cluster_id = self._create_new_cluster(z)
            self.assignment_cache[cache_key] = cluster_id
            return cluster_id
        
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
            # 새 클러스터 생성
            new_cluster_id = self._create_new_cluster(z)
            self.assignment_cache[cache_key] = new_cluster_id
            print(f"🐣 Birth triggered! New cluster {new_cluster_id} (likelihood: {max_likelihood:.2e})")
            return new_cluster_id
        else:
            # 기존 클러스터에 할당
            self._update_cluster(best_cluster, z)
            self.assignment_cache[cache_key] = best_cluster
            return best_cluster
    
    def merge(self):
        """클러스터 병합 (캐시 업데이트 포함)"""
        if len(self.clusters) < 2:
            return
        
        cluster_ids = list(self.clusters.keys())
        merged_pairs = []
        
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                id1, id2 = cluster_ids[i], cluster_ids[j]
                
                if id1 not in self.clusters or id2 not in self.clusters:
                    continue
                
                cluster1 = self.clusters[id1]
                cluster2 = self.clusters[id2]
                
                # 거리 계산
                distance = np.linalg.norm(cluster1.mu - cluster2.mu)
                
                if distance < self.tau_merge:
                    merged_pairs.append((id1, id2, distance))
        
        # 거리 순으로 정렬하여 가장 가까운 것부터 병합
        merged_pairs.sort(key=lambda x: x[2])
        
        for id1, id2, distance in merged_pairs:
            if id1 in self.clusters and id2 in self.clusters:
                self._merge_clusters(id1, id2)
                
                # 캐시 업데이트
                self._update_cache_after_merge(id1, id2)
    
    def _update_cache_after_merge(self, old_id1: int, old_id2: int):
        """병합 후 캐시 업데이트"""
        # 병합된 클러스터들을 참조하는 캐시 엔트리들을 찾아서 업데이트
        keys_to_update = []
        for cache_key, cluster_id in self.assignment_cache.items():
            if cluster_id == old_id1 or cluster_id == old_id2:
                keys_to_update.append(cache_key)
        
        # 새로운 클러스터 ID 찾기 (병합 후 남은 클러스터)
        remaining_clusters = set(self.clusters.keys())
        new_cluster_id = None
        
        # 가장 최근에 생성된 클러스터를 새 ID로 사용
        if remaining_clusters:
            new_cluster_id = max(remaining_clusters)
        
        # 캐시 업데이트
        for cache_key in keys_to_update:
            if new_cluster_id is not None:
                self.assignment_cache[cache_key] = new_cluster_id
            else:
                del self.assignment_cache[cache_key]