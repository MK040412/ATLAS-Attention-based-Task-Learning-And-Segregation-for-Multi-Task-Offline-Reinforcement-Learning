import numpy as np
import pickle
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class RobustDPMMOODDetector:
    """강건한 DPMM 기반 OOD 탐지기"""
    
    def __init__(self, latent_dim, max_components=10, alpha=1.0, random_state=42):
        self.latent_dim = latent_dim
        self.max_components = max_components
        self.alpha = alpha
        self.random_state = random_state
        
        # 데이터 저장
        self.data_points = []
        self.assignments = []
        self.assignment_history = []
        
        # 스케일러
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        
        # DPMM 모델
        self.dpmm = BayesianGaussianMixture(
            n_components=max_components,
            covariance_type='full',
            weight_concentration_prior=alpha,
            random_state=random_state,
            max_iter=100,
            tol=1e-3,
            warm_start=True
        )
        self.dpmm_fitted = False
        
        # 클러스터 통계
        self.cluster_counts = Counter()
        self.cluster_centroids = {}
        self.merge_history = []
        
        print(f"RobustDPMMOODDetector initialized:")
        print(f"  Latent dim: {latent_dim}")
        print(f"  Max components: {max_components}")
        print(f"  Alpha: {alpha}")
    
    def _safe_array_convert(self, data):
        """안전한 배열 변환"""
        try:
            if isinstance(data, (list, tuple)):
                arr = np.array(data, dtype=np.float32)
            elif isinstance(data, np.ndarray):
                arr = data.astype(np.float32)
            else:
                arr = np.array([data], dtype=np.float32)
            
            # 1차원으로 평탄화
            arr = arr.flatten()
            
            # 차원 검증
            if len(arr) != self.latent_dim:
                if len(arr) < self.latent_dim:
                    # 패딩
                    arr = np.pad(arr, (0, self.latent_dim - len(arr)), mode='constant')
                else:
                    # 자르기
                    arr = arr[:self.latent_dim]
            
            # NaN/Inf 처리
            if not np.isfinite(arr).all():
                arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return arr
            
        except Exception as e:
            print(f"Warning: Array conversion failed: {e}")
            return np.zeros(self.latent_dim, dtype=np.float32)
    
    def assign(self, latent_vector):
        """클러스터 할당"""
        try:
            # 안전한 변환
            z = self._safe_array_convert(latent_vector)
            
            # 데이터 저장
            self.data_points.append(z.copy())
            
            # 첫 번째 데이터 포인트
            if len(self.data_points) == 1:
                self.assignments.append(0)
                self.assignment_history.append(0)
                self.cluster_counts[0] = 1
                self.cluster_centroids[0] = z.copy()
                return 0
            
            # 스케일러 피팅 (충분한 데이터가 있을 때)
            if len(self.data_points) >= 3 and not self.scaler_fitted:
                try:
                    data_matrix = np.array(self.data_points)
                    self.scaler.fit(data_matrix)
                    self.scaler_fitted = True
                except Exception as scaler_error:
                    print(f"Warning: Scaler fitting failed: {scaler_error}")
            
            # 데이터 정규화
            if self.scaler_fitted:
                try:
                    z_scaled = self.scaler.transform(z.reshape(1, -1)).flatten()
                except Exception as scale_error:
                    print(f"Warning: Scaling failed: {scale_error}")
                    z_scaled = z
            else:
                z_scaled = z
            
            # DPMM 피팅 (충분한 데이터가 있을 때)
            if len(self.data_points) >= 5:
                try:
                    if self.scaler_fitted:
                        data_matrix = self.scaler.transform(np.array(self.data_points))
                    else:
                        data_matrix = np.array(self.data_points)
                    
                    self.dpmm.fit(data_matrix)
                    self.dpmm_fitted = True
                    
                    # 클러스터 할당
                    cluster_id = self.dpmm.predict(z_scaled.reshape(1, -1))[0]
                    
                    # 유효한 클러스터인지 확인 (가중치 기반)
                    if hasattr(self.dpmm, 'weights_') and len(self.dpmm.weights_) > cluster_id:
                        if self.dpmm.weights_[cluster_id] < 0.01:  # 매우 작은 가중치
                            # 가장 가까운 유효한 클러스터 찾기
                            valid_clusters = [i for i, w in enumerate(self.dpmm.weights_) if w >= 0.01]
                            if valid_clusters:
                                distances = []
                                for valid_id in valid_clusters:
                                    if hasattr(self.dpmm, 'means_'):
                                        dist = np.linalg.norm(z_scaled - self.dpmm.means_[valid_id])
                                        distances.append((dist, valid_id))
                                if distances:
                                    cluster_id = min(distances)[1]
                    
                except Exception as dpmm_error:
                    print(f"Warning: DPMM fitting/prediction failed: {dpmm_error}")
                    # 거리 기반 할당으로 폴백
                    cluster_id = self._distance_based_assignment(z)
            else:
                # 거리 기반 할당
                cluster_id = self._distance_based_assignment(z)
            
            # 할당 기록
            self.assignments.append(cluster_id)
            self.assignment_history.append(cluster_id)
            self.cluster_counts[cluster_id] += 1
            
            # 중심점 업데이트
            if cluster_id in self.cluster_centroids:
                # 이동 평균으로 업데이트
                alpha = 0.1
                self.cluster_centroids[cluster_id] = (
                    (1 - alpha) * self.cluster_centroids[cluster_id] + alpha * z
                )
            else:
                self.cluster_centroids[cluster_id] = z.copy()
            
            return int(cluster_id)
            
        except Exception as e:
            print(f"Warning: Cluster assignment failed: {e}")
            # 안전한 기본값
            return 0
    
    def _distance_based_assignment(self, z, threshold=2.0):
        """거리 기반 클러스터 할당"""
        try:
            if not self.cluster_centroids:
                return 0
            
            min_distance = float('inf')
            closest_cluster = 0
            
            for cluster_id, centroid in self.cluster_centroids.items():
                distance = np.linalg.norm(z - centroid)
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = cluster_id
            
            # 새 클러스터 생성 조건
            if min_distance > threshold and len(self.cluster_centroids) < self.max_components:
                new_cluster_id = max(self.cluster_centroids.keys()) + 1
                return new_cluster_id
            else:
                return closest_cluster
                
        except Exception as e:
            print(f"Warning: Distance-based assignment failed: {e}")
            return 0
    
    def merge(self, threshold=1.5):
        """클러스터 병합"""
        try:
            if len(self.cluster_centroids) < 2:
                return
            
            merged_pairs = []
            centroids_list = list(self.cluster_centroids.items())
            
            for i in range(len(centroids_list)):
                for j in range(i + 1, len(centroids_list)):
                    id1, centroid1 = centroids_list[i]
                    id2, centroid2 = centroids_list[j]
                    
                    distance = np.linalg.norm(centroid1 - centroid2)
                    
                    if distance < threshold:
                        # 더 많은 데이터를 가진 클러스터로 병합
                        if self.cluster_counts[id1] >= self.cluster_counts[id2]:
                            keep_id, merge_id = id1, id2
                        else:
                            keep_id, merge_id = id2, id1
                        
                        merged_pairs.append((keep_id, merge_id))
            
            # 병합 실행
            for keep_id, merge_id in merged_pairs:
                if merge_id in self.cluster_centroids:
                    # 가중 평균으로 중심점 업데이트
                    w1 = self.cluster_counts[keep_id]
                    w2 = self.cluster_counts[merge_id]
                    total_weight = w1 + w2
                    
                    self.cluster_centroids[keep_id] = (
                        (w1 * self.cluster_centroids[keep_id] + w2 * self.cluster_centroids[merge_id]) / total_weight
                    )
                    
                    # 카운트 업데이트
                    self.cluster_counts[keep_id] += self.cluster_counts[merge_id]
                    
                    # 병합된 클러스터 제거
                    del self.cluster_centroids[merge_id]
                    del self.cluster_counts[merge_id]
                    
                    # 할당 기록 업데이트
                    self.assignments = [keep_id if x == merge_id else x for x in self.assignments]
                    
                    self.merge_history.append((keep_id, merge_id))
                    
                    print(f"  Merged cluster {merge_id} into {keep_id}")
            
            if merged_pairs:
                print(f"  Total clusters after merge: {len(self.cluster_centroids)}")
                
        except Exception as e:
            print(f"Warning: Cluster merge failed: {e}")
    
    def get_cluster_info(self):
        """클러스터 정보 반환"""
        try:
            return {
                'num_clusters': len(self.cluster_centroids),
                'cluster_counts': dict(self.cluster_counts),
                'total_data_points': len(self.data_points),
                'merge_count': len(self.merge_history),
                'dpmm_fitted': self.dpmm_fitted,
                'scaler_fitted': self.scaler_fitted
            }
        except Exception as e:
            print(f"Warning: Failed to get cluster info: {e}")
            return {'num_clusters': 0, 'cluster_counts': {}}
    
    def save(self, filepath):
        """상태 저장"""
        try:
            state = {
                'latent_dim': self.latent_dim,
                'max_components': self.max_components,
                'alpha': self.alpha,
                'data_points': self.data_points,
                'assignments': self.assignments,
                'assignment_history': self.assignment_history,
                'cluster_counts': dict(self.cluster_counts),
                'cluster_centroids': self.cluster_centroids,
                'merge_history': self.merge_history,
                'scaler_fitted': self.scaler_fitted,
                'dpmm_fitted': self.dpmm_fitted
            }
            
            if self.scaler_fitted:
                state['scaler_mean'] = self.scaler.mean_
                state['scaler_scale'] = self.scaler.scale_
            
            if self.dpmm_fitted:
                state['dpmm_weights'] = self.dpmm.weights_
                state['dpmm_means'] = self.dpmm.means_
                state['dpmm_covariances'] = self.dpmm.covariances_
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            print(f"✓ DPMM state saved to {filepath}")
            
        except Exception as e:
            print(f"Warning: Failed to save DPMM state: {e}")
    
    def load(self, filepath):
        """상태 로드"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.latent_dim = state['latent_dim']
            self.max_components = state['max_components']
            self.alpha = state['alpha']
            self.data_points = state['data_points']
            self.assignments = state['assignments']
            self.assignment_history = state['assignment_history']
            self.cluster_counts = Counter(state['cluster_counts'])
            self.cluster_centroids = state['cluster_centroids']
            self.merge_history = state['merge_history']
            self.scaler_fitted = state['scaler_fitted']
            self.dpmm_fitted = state['dpmm_fitted']
            
            if self.scaler_fitted:
                self.scaler.mean_ = state['scaler_mean']
                self.scaler.scale_ = state['scaler_scale']
            
            if self.dpmm_fitted:
                self.dpmm.weights_ = state['dpmm_weights']
                self.dpmm.means_ = state['dpmm_means']
                self.dpmm.covariances_ = state['dpmm_covariances']
            
            print(f"✓ DPMM state loaded from {filepath}")
            
        except Exception as e:
            print(f"Warning: Failed to load DPMM state: {e}")
    
    def cleanup(self):
        """메모리 정리"""
        try:
            self.data_points.clear()
            self.assignments.clear()
            self.assignment_history.clear()
            self.cluster_counts.clear()
            self.cluster_centroids.clear()
            self.merge_history.clear()
        except:
            pass