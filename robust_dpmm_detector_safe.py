import numpy as np
import pickle
from sklearn.mixture import BayesianGaussianMixture
from collections import defaultdict, Counter

class RobustDPMMOODDetector:
    """안전한 DPMM OOD 검출기"""
    
    def __init__(self, latent_dim=32, alpha=1.0, max_components=10):
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.max_components = max_components
        
        # 클러스터 데이터 저장
        self.cluster_data = defaultdict(list)
        self.cluster_centers = {}
        self.cluster_counts = Counter()
        self.next_cluster_id = 0
        
        # 베이지안 GMM 모델
        self.bgmm = BayesianGaussianMixture(
            n_components=max_components,
            covariance_type='full',
            weight_concentration_prior=alpha,
            random_state=42
        )
        self.is_fitted = False
        
        print(f"✓ RobustDPMMOODDetector initialized: latent_dim={latent_dim}")
    
    def _safe_array_conversion(self, data):
        """데이터를 안전하게 numpy 배열로 변환"""
        try:
            if isinstance(data, np.ndarray):
                return data.astype(np.float32)
            elif hasattr(data, 'numpy'):  # torch tensor
                return data.detach().cpu().numpy().astype(np.float32)
            else:
                return np.array(data, dtype=np.float32)
        except Exception as e:
            print(f"Warning: Array conversion failed: {e}")
            return np.zeros(self.latent_dim, dtype=np.float32)
    
    def assign(self, latent_vector, threshold=2.0):
        """잠재 벡터를 클러스터에 할당"""
        try:
            # 안전한 배열 변환
            z = self._safe_array_conversion(latent_vector)
            if z.ndim > 1:
                z = z.flatten()
            
            # 차원 확인 및 조정
            if len(z) != self.latent_dim:
                if len(z) > self.latent_dim:
                    z = z[:self.latent_dim]
                else:
                    # 패딩
                    padded = np.zeros(self.latent_dim, dtype=np.float32)
                    padded[:len(z)] = z
                    z = padded
            
            # 첫 번째 데이터포인트인 경우
            if self.next_cluster_id == 0:
                cluster_id = 0
                self.cluster_data[cluster_id].append(z)
                self.cluster_centers[cluster_id] = z.copy()
                self.cluster_counts[cluster_id] += 1
                self.next_cluster_id = 1
                return cluster_id
            
            # 기존 클러스터와의 거리 계산
            min_distance = float('inf')
            best_cluster = -1
            
            for cluster_id, center in self.cluster_centers.items():
                try:
                    distance = np.linalg.norm(z - center)
                    if distance < min_distance:
                        min_distance = distance
                        best_cluster = cluster_id
                except Exception as dist_error:
                    print(f"Warning: Distance calculation failed: {dist_error}")
                    continue
            
            # 임계값 기반 할당
            if min_distance < threshold and best_cluster >= 0:
                # 기존 클러스터에 할당
                cluster_id = best_cluster
                self.cluster_data[cluster_id].append(z)
                self.cluster_counts[cluster_id] += 1
                
                # 클러스터 중심 업데이트 (이동 평균)
                alpha = 0.1
                self.cluster_centers[cluster_id] = (
                    (1 - alpha) * self.cluster_centers[cluster_id] + alpha * z
                )
            else:
                # 새로운 클러스터 생성
                cluster_id = self.next_cluster_id
                self.cluster_data[cluster_id].append(z)
                self.cluster_centers[cluster_id] = z.copy()
                self.cluster_counts[cluster_id] = 1
                self.next_cluster_id += 1
            
            return cluster_id
            
        except Exception as e:
            print(f"Warning: Cluster assignment failed: {e}")
            # 안전한 기본값 반환
            return 0
    
    def merge(self, threshold=1.5):
        """유사한 클러스터들을 병합"""
        try:
            if len(self.cluster_centers) < 2:
                return
            
            # 병합할 클러스터 쌍 찾기
            clusters_to_merge = []
            cluster_ids = list(self.cluster_centers.keys())
            
            for i in range(len(cluster_ids)):
                for j in range(i + 1, len(cluster_ids)):
                    id1, id2 = cluster_ids[i], cluster_ids[j]
                    try:
                        distance = np.linalg.norm(
                            self.cluster_centers[id1] - self.cluster_centers[id2]
                        )
                        if distance < threshold:
                            clusters_to_merge.append((id1, id2, distance))
                    except Exception as merge_error:
                        print(f"Warning: Merge distance calculation failed: {merge_error}")
                        continue
            
            # 거리 순으로 정렬하여 가장 가까운 것부터 병합
            clusters_to_merge.sort(key=lambda x: x[2])
            
            merged_count = 0
            for id1, id2, distance in clusters_to_merge:
                if id1 in self.cluster_centers and id2 in self.cluster_centers:
                    # id2를 id1으로 병합
                    self.cluster_data[id1].extend(self.cluster_data[id2])
                    self.cluster_counts[id1] += self.cluster_counts[id2]
                    
                    # 새로운 중심 계산
                    all_data = np.array(self.cluster_data[id1])
                    self.cluster_centers[id1] = np.mean(all_data, axis=0)
                    
                    # id2 제거
                    del self.cluster_data[id2]
                    del self.cluster_centers[id2]
                    del self.cluster_counts[id2]
                    
                    merged_count += 1
            
            if merged_count > 0:
                print(f"  🔄 Merged {merged_count} cluster pairs")
                
        except Exception as e:
            print(f"Warning: Cluster merge failed: {e}")
    
    def get_cluster_info(self):
        """클러스터 정보 반환"""
        try:
            return {
                'num_clusters': len(self.cluster_centers),
                'cluster_counts': dict(self.cluster_counts),
                'total_points': sum(self.cluster_counts.values())
            }
        except Exception as e:
            print(f"Warning: Failed to get cluster info: {e}")
            return {
                'num_clusters': 0,
                'cluster_counts': {},
                'total_points': 0
            }
    
    def save_state(self, filepath):
        """DPMM 상태 저장"""
        try:
            state = {
                'cluster_data': dict(self.cluster_data),
                'cluster_centers': self.cluster_centers,
                'cluster_counts': dict(self.cluster_counts),
                'next_cluster_id': self.next_cluster_id,
                'latent_dim': self.latent_dim,
                'alpha': self.alpha,
                'max_components': self.max_components
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            print(f"✓ DPMM state saved to {filepath}")
            
        except Exception as e:
            print(f"Warning: Failed to save DPMM state: {e}")
    
    def load_state(self, filepath):
        """DPMM 상태 로드"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.cluster_data = defaultdict(list, state['cluster_data'])
            self.cluster_centers = state['cluster_centers']
            self.cluster_counts = Counter(state['cluster_counts'])
            self.next_cluster_id = state['next_cluster_id']
            self.latent_dim = state['latent_dim']
            self.alpha = state['alpha']
            self.max_components = state['max_components']
            
            print(f"✓ DPMM state loaded from {filepath}")
            
        except Exception as e:
            print(f"Warning: Failed to load DPMM state: {e}")