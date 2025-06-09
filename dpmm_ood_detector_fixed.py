import numpy as np
import pickle
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans
import torch

class DPMMOODDetector:
    def __init__(self, latent_dim=32, alpha=1.0, max_components=10):
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.max_components = max_components
        
        # 데이터 저장
        self.data_points = []
        self.cluster_assignments = []
        
        # 베이지안 가우시안 혼합 모델
        self.bgm = BayesianGaussianMixture(
            n_components=max_components,
            covariance_type='full',
            weight_concentration_prior=alpha,
            random_state=42
        )
        
        self.is_fitted = False
        self.cluster_centers = {}
        self.cluster_counts = {}
        
    def assign(self, z):
        """클러스터 할당"""
        try:
            z = np.array(z, dtype=np.float32).reshape(1, -1)
            
            # 첫 번째 데이터 포인트
            if len(self.data_points) == 0:
                self.data_points.append(z.flatten())
                self.cluster_assignments.append(0)
                self.cluster_centers[0] = z.flatten()
                self.cluster_counts[0] = 1
                return 0
            
            # 데이터 추가
            self.data_points.append(z.flatten())
            
            # 충분한 데이터가 있으면 BGM 피팅
            if len(self.data_points) >= 5 and len(self.data_points) % 5 == 0:
                self._fit_bgm()
            
            # 클러스터 할당
            if self.is_fitted:
                cluster_id = self.bgm.predict(z)[0]
            else:
                # BGM이 피팅되지 않았으면 거리 기반 할당
                cluster_id = self._distance_based_assign(z.flatten())
            
            self.cluster_assignments.append(cluster_id)
            
            # 클러스터 정보 업데이트
            if cluster_id not in self.cluster_counts:
                self.cluster_counts[cluster_id] = 0
                self.cluster_centers[cluster_id] = z.flatten()
            
            self.cluster_counts[cluster_id] += 1
            
            # 클러스터 중심 업데이트 (이동 평균)
            alpha = 0.1
            self.cluster_centers[cluster_id] = (
                (1 - alpha) * self.cluster_centers[cluster_id] + 
                alpha * z.flatten()
            )
            
            return int(cluster_id)
            
        except Exception as e:
            print(f"Warning: Cluster assignment failed: {e}")
            return 0
    
    def _fit_bgm(self):
        """베이지안 가우시안 혼합 모델 피팅"""
        try:
            if len(self.data_points) < 2:
                return
            
            X = np.array(self.data_points)
            self.bgm.fit(X)
            self.is_fitted = True
            
        except Exception as e:
            print(f"Warning: BGM fitting failed: {e}")
            self.is_fitted = False
    
    def _distance_based_assign(self, z, threshold=2.0):
        """거리 기반 클러스터 할당"""
        min_dist = float('inf')
        best_cluster = 0
        
        for cluster_id, center in self.cluster_centers.items():
            dist = np.linalg.norm(z - center)
            if dist < min_dist:
                min_dist = dist
                best_cluster = cluster_id
        
        # 새로운 클러스터 생성 조건
        if min_dist > threshold:
            new_cluster_id = max(self.cluster_centers.keys()) + 1
            return new_cluster_id
        
        return best_cluster
    
    def merge(self, threshold=1.5):
        """클러스터 병합"""
        try:
            if len(self.cluster_centers) < 2:
                return
            
            centers = list(self.cluster_centers.keys())
            merged = []
            
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    c1, c2 = centers[i], centers[j]
                    if c1 in merged or c2 in merged:
                        continue
                    
                    center1 = self.cluster_centers[c1]
                    center2 = self.cluster_centers[c2]
                    
                    dist = np.linalg.norm(center1 - center2)
                    
                    if dist < threshold:
                        # c2를 c1에 병합
                        count1 = self.cluster_counts[c1]
                        count2 = self.cluster_counts[c2]
                        total_count = count1 + count2
                        
                        # 가중 평균으로 중심 업데이트
                        self.cluster_centers[c1] = (
                            (count1 * center1 + count2 * center2) / total_count
                        )
                        self.cluster_counts[c1] = total_count
                        
                        # c2 제거
                        del self.cluster_centers[c2]
                        del self.cluster_counts[c2]
                        merged.append(c2)
            
        except Exception as e:
            print(f"Warning: Cluster merge failed: {e}")
    
    def get_cluster_info(self):
        """클러스터 정보 반환"""
        return {
            'num_clusters': len(self.cluster_centers),
            'cluster_counts': dict(self.cluster_counts),
            'total_points': len(self.data_points),
            'is_fitted': self.is_fitted
        }
    
    def save(self, filepath):
        """상태 저장"""
        try:
            state = {
                'latent_dim': self.latent_dim,
                'alpha': self.alpha,
                'max_components': self.max_components,
                'data_points': self.data_points,
                'cluster_assignments': self.cluster_assignments,
                'cluster_centers': self.cluster_centers,
                'cluster_counts': self.cluster_counts,
                'is_fitted': self.is_fitted
            }
            
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
            self.alpha = state['alpha']
            self.max_components = state['max_components']
            self.data_points = state['data_points']
            self.cluster_assignments = state['cluster_assignments']
            self.cluster_centers = state['cluster_centers']
            self.cluster_counts = state['cluster_counts']
            self.is_fitted = state['is_fitted']
            
            # BGM 재구성
            if self.is_fitted and len(self.data_points) > 0:
                self._fit_bgm()
            
            print(f"✓ DPMM state loaded from {filepath}")
            
        except Exception as e:
            print(f"Warning: Failed to load DPMM state: {e}")
    
    def cleanup(self):
        """메모리 정리"""
        try:
            self.data_points.clear()
            self.cluster_assignments.clear()
            self.cluster_centers.clear()
            self.cluster_counts.clear()
            self.bgm = None
            torch.cuda.empty_cache()
        except:
            pass