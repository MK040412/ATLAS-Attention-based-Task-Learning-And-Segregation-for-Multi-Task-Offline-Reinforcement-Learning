import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans
import pickle

class DPMMOODDetector:
    """DPMM 기반 Out-of-Distribution 탐지기"""
    
    def __init__(self, latent_dim=32, max_components=10, alpha=1.0):
        self.latent_dim = latent_dim
        self.max_components = max_components
        self.alpha = alpha
        
        # 베이지안 가우시안 혼합 모델
        self.bgm = BayesianGaussianMixture(
            n_components=max_components,
            covariance_type='full',
            weight_concentration_prior=alpha,
            random_state=42
        )
        
        # 데이터 저장
        self.data_points = []
        self.cluster_assignments = []
        self.is_fitted = False
        
        print(f"DPMM OOD Detector initialized:")
        print(f"  Latent dim: {latent_dim}, Max components: {max_components}")
    
    def assign(self, latent_vector):
        """잠재 벡터를 클러스터에 할당"""
        if not isinstance(latent_vector, np.ndarray):
            latent_vector = np.array(latent_vector, dtype=np.float32)
        
        # 차원 맞추기
        if latent_vector.shape[-1] != self.latent_dim:
            if latent_vector.size >= self.latent_dim:
                latent_vector = latent_vector.flatten()[:self.latent_dim]
            else:
                # 패딩
                padded = np.zeros(self.latent_dim, dtype=np.float32)
                padded[:latent_vector.size] = latent_vector.flatten()
                latent_vector = padded
        
        latent_vector = latent_vector.reshape(1, -1)
        
        # 데이터 저장
        self.data_points.append(latent_vector.squeeze())
        
        # 충분한 데이터가 쌓이면 모델 학습
        if len(self.data_points) >= 10 and not self.is_fitted:
            self._fit_model()
        elif len(self.data_points) % 50 == 0 and self.is_fitted:
            # 주기적으로 모델 업데이트
            self._update_model()
        
        # 클러스터 할당
        if self.is_fitted:
            try:
                cluster_id = self.bgm.predict(latent_vector)[0]
                
                # 유효한 클러스터인지 확인 (가중치가 충분히 큰지)
                weights = self.bgm.weights_
                if cluster_id < len(weights) and weights[cluster_id] > 0.01:
                    self.cluster_assignments.append(cluster_id)
                    return int(cluster_id)
                else:
                    # 새로운 클러스터 생성
                    new_cluster_id = len([w for w in weights if w > 0.01])
                    self.cluster_assignments.append(new_cluster_id)
                    return int(new_cluster_id)
            except Exception as e:
                print(f"Warning: Cluster assignment failed: {e}")
                # 기본 클러스터 반환
                cluster_id = len(self.cluster_assignments) % self.max_components
                self.cluster_assignments.append(cluster_id)
                return int(cluster_id)
        else:
            # 모델이 학습되지 않은 경우 간단한 할당
            cluster_id = len(self.cluster_assignments) % 3  # 최대 3개 클러스터로 시작
            self.cluster_assignments.append(cluster_id)
            return int(cluster_id)
    
    def _fit_model(self):
        """모델 학습"""
        try:
            if len(self.data_points) < 5:
                return
            
            data_matrix = np.array(self.data_points)
            self.bgm.fit(data_matrix)
            self.is_fitted = True
            
            # 유효한 컴포넌트 수 계산
            active_components = np.sum(self.bgm.weights_ > 0.01)
            print(f"  DPMM fitted with {active_components} active components")
            
        except Exception as e:
            print(f"Warning: DPMM fitting failed: {e}")
            # 폴백: KMeans 사용
            self._fallback_kmeans()
    
    def _update_model(self):
        """모델 업데이트"""
        try:
            if len(self.data_points) < 10:
                return
            
            # 최근 데이터만 사용 (메모리 절약)
            recent_data = self.data_points[-1000:] if len(self.data_points) > 1000 else self.data_points
            data_matrix = np.array(recent_data)
            
            self.bgm.fit(data_matrix)
            
        except Exception as e:
            print(f"Warning: DPMM update failed: {e}")
    
    def _fallback_kmeans(self):
        """폴백: KMeans 클러스터링"""
        try:
            n_clusters = min(5, len(self.data_points))
            if n_clusters < 2:
                return
            
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            data_matrix = np.array(self.data_points)
            self.kmeans.fit(data_matrix)
            self.is_fitted = True
            self.use_kmeans = True
            
            print(f"  Using KMeans fallback with {n_clusters} clusters")
            
        except Exception as e:
            print(f"Warning: KMeans fallback failed: {e}")
    
    def merge(self, threshold=2.0):
        """클러스터 병합"""
        if not self.is_fitted or len(self.data_points) < 20:
            return
        
        try:
            # 가중치가 낮은 컴포넌트들을 병합
            weights = self.bgm.weights_
            active_mask = weights > 0.01
            
            if np.sum(active_mask) <= 2:
                return  # 이미 충분히 적은 클러스터
            
            # 가중치 기반 병합 (간단한 버전)
            low_weight_indices = np.where((weights > 0.001) & (weights < 0.05))[0]
            if len(low_weight_indices) > 1:
                print(f"  Merging {len(low_weight_indices)} low-weight clusters")
            
        except Exception as e:
            print(f"Warning: Cluster merging failed: {e}")
    
    def get_cluster_info(self):
        """클러스터 정보 반환"""
        try:
            if self.is_fitted and hasattr(self.bgm, 'weights_'):
                weights = self.bgm.weights_
                active_components = np.sum(weights > 0.01)
                
                # 클러스터별 데이터 개수 계산
                cluster_counts = {}
                for cluster_id in self.cluster_assignments:
                    cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
                
                return {
                    'num_clusters': int(active_components),
                    'cluster_counts': list(cluster_counts.values()),
                    'total_data_points': len(self.data_points)
                }
            else:
                unique_clusters = len(set(self.cluster_assignments)) if self.cluster_assignments else 0
                return {
                    'num_clusters': unique_clusters,
                    'cluster_counts': [],
                    'total_data_points': len(self.data_points)
                }
        except Exception as e:
            print(f"Warning: Failed to get cluster info: {e}")
            return {
                'num_clusters': 0,
                'cluster_counts': [],
                'total_data_points': len(self.data_points)
            }
    
    def get_state(self):
        """상태 저장용 딕셔너리 반환"""
        try:
            state = {
                'data_points': self.data_points[-100:],  # 최근 100개만 저장
                'cluster_assignments': self.cluster_assignments[-100:],
                'is_fitted': self.is_fitted,
                'latent_dim': self.latent_dim
            }
            
            if self.is_fitted and hasattr(self.bgm, 'weights_'):
                state['bgm_weights'] = self.bgm.weights_
                state['bgm_means'] = self.bgm.means_
            
            return state
        except Exception as e:
            print(f"Warning: Failed to get DPMM state: {e}")
            return {'latent_dim': self.latent_dim, 'is_fitted': False}
    
    def load_state(self, state):
        """상태 로드"""
        try:
            self.data_points = state.get('data_points', [])
            self.cluster_assignments = state.get('cluster_assignments', [])
            self.is_fitted = state.get('is_fitted', False)
            self.latent_dim = state.get('latent_dim', self.latent_dim)
            
            if self.is_fitted and len(self.data_points) >= 10:
                self._fit_model()
            
        except Exception as e:
            print(f"Warning: Failed to load DPMM state: {e}")
    
    def cleanup(self):
        """메모리 정리"""
        self.data_points = []
        self.cluster_assignments = []
        if hasattr(self, 'bgm'):
            del self.bgm
        if hasattr(self, 'kmeans'):
            del self.kmeans
