import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

class SimpleDPMM:
    """간단한 DPMM 구현 (실제로는 적응적 K-means)"""
    def __init__(self, latent_dim, alpha=1.0, base_var=1.0):
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.base_var = base_var
        
        self.clusters = []  # 클러스터 중심들
        self.cluster_counts = []  # 각 클러스터의 데이터 개수
        self.assignments = []  # 모든 데이터의 클러스터 할당
        self.data_points = []  # 저장된 데이터 포인트들
        
        self.distance_threshold = 2.0  # 새 클러스터 생성 임계값
    
    def assign_and_update(self, data_point):
        """데이터 포인트를 클러스터에 할당하고 업데이트"""
        data_point = np.array(data_point).flatten()
        self.data_points.append(data_point)
        
        if len(self.clusters) == 0:
            # 첫 번째 클러스터 생성
            self.clusters.append(data_point.copy())
            self.cluster_counts.append(1)
            self.assignments.append(0)
            return 0
        
        # 기존 클러스터들과의 거리 계산
        distances = []
        for cluster_center in self.clusters:
            dist = np.linalg.norm(data_point - cluster_center)
            distances.append(dist)
        
        min_distance = min(distances)
        closest_cluster = np.argmin(distances)
        
        # 거리 기반 클러스터 할당
        if min_distance < self.distance_threshold:
            # 기존 클러스터에 할당
            cluster_id = closest_cluster
            self.cluster_counts[cluster_id] += 1
            
            # 클러스터 중심 업데이트 (이동 평균)
            alpha = 0.1  # 학습률
            self.clusters[cluster_id] = (1 - alpha) * self.clusters[cluster_id] + alpha * data_point
        else:
            # 새 클러스터 생성
            cluster_id = len(self.clusters)
            self.clusters.append(data_point.copy())
            self.cluster_counts.append(1)
        
        self.assignments.append(cluster_id)
        return cluster_id
    
    def merge_clusters(self, threshold=1.5):
        """가까운 클러스터들 병합"""
        if len(self.clusters) <= 1:
            return
        
        merged = True
        while merged and len(self.clusters) > 1:
            merged = False
            
            # 모든 클러스터 쌍의 거리 계산
            min_dist = float('inf')
            merge_pair = None
            
            for i in range(len(self.clusters)):
                for j in range(i + 1, len(self.clusters)):
                    dist = np.linalg.norm(self.clusters[i] - self.clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        merge_pair = (i, j)
            
            # 임계값보다 가까우면 병합
            if min_dist < threshold and merge_pair is not None:
                i, j = merge_pair
                
                # 가중 평균으로 새 중심 계산
                count_i = self.cluster_counts[i]
                count_j = self.cluster_counts[j]
                total_count = count_i + count_j
                
                new_center = (count_i * self.clusters[i] + count_j * self.clusters[j]) / total_count
                
                # 클러스터 i를 새 중심으로 업데이트, j 제거
                self.clusters[i] = new_center
                self.cluster_counts[i] = total_count
                
                del self.clusters[j]
                del self.cluster_counts[j]
                
                # 할당 업데이트 (j를 i로, j보다 큰 인덱스들은 1씩 감소)
                for k in range(len(self.assignments)):
                    if self.assignments[k] == j:
                        self.assignments[k] = i
                    elif self.assignments[k] > j:
                        self.assignments[k] -= 1
                
                merged = True
                print(f"  Merged clusters {i} and {j}, new count: {total_count}")
    
    def get_cluster_info(self):
        """클러스터 정보 반환"""
        return {
            'num_clusters': len(self.clusters),
            'cluster_counts': self.cluster_counts.copy(),
            'total_points': len(self.data_points)
        }
    
    def get_state(self):
        """DPMM 상태 반환"""
        return {
            'clusters': [c.copy() for c in self.clusters],
            'cluster_counts': self.cluster_counts.copy(),
            'assignments': self.assignments.copy(),
            'data_points': [p.copy() for p in self.data_points]
        }