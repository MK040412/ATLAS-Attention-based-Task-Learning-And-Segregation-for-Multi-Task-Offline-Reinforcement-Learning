"""
UMAP 시각화를 위한 클러스터링 분석 도구
"""

import numpy as np
import matplotlib.pyplot as plt
import umap
import seaborn as sns
from collections import defaultdict
import torch
import pickle
from typing import Dict, List, Tuple
import os

# 기존 모듈 import
from Ant_CEMP import *

class ClusteringVisualizer:
    """클러스터링 과정을 시각화하는 클래스"""
    
    def __init__(self, save_dir: str = "clustering_analysis"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 시각화 데이터 저장
        self.latent_snapshots = []  # [(step, latents, clusters, tasks)]
        self.clustering_history = []
        
        # UMAP 설정
        self.umap_reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )
        
        print(f"🎨 ClusteringVisualizer initialized, saving to: {save_dir}")
    
    def add_snapshot(self, step: int, latents: np.ndarray, clusters: np.ndarray, tasks: np.ndarray):
        """클러스터링 스냅샷 추가"""
        self.latent_snapshots.append({
            'step': step,
            'latents': latents.copy(),
            'clusters': clusters.copy(), 
            'tasks': tasks.copy(),
            'num_clusters': len(np.unique(clusters))
        })
        
        print(f"📸 Snapshot saved at step {step}: {len(latents)} points, {len(np.unique(clusters))} clusters")
    
    def create_umap_visualization(self, snapshot_indices: List[int] = None):
        """UMAP으로 클러스터링 시각화"""
        if not self.latent_snapshots:
            print("❌ No snapshots available for visualization")
            return
        
        if snapshot_indices is None:
            # 전체 스냅샷 중에서 균등하게 선택
            total_snapshots = len(self.latent_snapshots)
            snapshot_indices = [0, total_snapshots//3, 2*total_snapshots//3, total_snapshots-1]
            snapshot_indices = [i for i in snapshot_indices if i < total_snapshots]
        
        # 서브플롯 설정
        n_snapshots = len(snapshot_indices)
        fig, axes = plt.subplots(2, n_snapshots, figsize=(5*n_snapshots, 10))
        if n_snapshots == 1:
            axes = axes.reshape(2, 1)
        
        plt.suptitle('UMAP Clustering Visualization During Training', fontsize=16, fontweight='bold')
        
        for i, snapshot_idx in enumerate(snapshot_indices):
            snapshot = self.latent_snapshots[snapshot_idx]
            
            # UMAP 차원 축소
            latents_2d = self.umap_reducer.fit_transform(snapshot['latents'])
            
            # 상단: 클러스터별 색상
            ax1 = axes[0, i]
            scatter1 = ax1.scatter(
                latents_2d[:, 0], latents_2d[:, 1], 
                c=snapshot['clusters'], 
                cmap='tab10', 
                alpha=0.7,
                s=50
            )
            ax1.set_title(f'Step {snapshot["step"]:,}\nClusters: {snapshot["num_clusters"]}')
            ax1.set_xlabel('UMAP 1')
            ax1.set_ylabel('UMAP 2')
            
            # 클러스터 중심점 표시
            unique_clusters = np.unique(snapshot['clusters'])
            for cluster_id in unique_clusters:
                mask = snapshot['clusters'] == cluster_id
                if np.any(mask):
                    center = latents_2d[mask].mean(axis=0)
                    ax1.scatter(center[0], center[1], c='black', s=200, marker='x', linewidths=3)
                    ax1.annotate(f'C{cluster_id}', (center[0], center[1]), 
                               xytext=(5, 5), textcoords='offset points', fontweight='bold')
            
            # 하단: 태스크별 색상
            ax2 = axes[1, i]
            scatter2 = ax2.scatter(
                latents_2d[:, 0], latents_2d[:, 1], 
                c=snapshot['tasks'], 
                cmap='viridis', 
                alpha=0.7,
                s=50
            )
            ax2.set_title(f'Tasks Distribution')
            ax2.set_xlabel('UMAP 1')
            ax2.set_ylabel('UMAP 2')
            
            # 컬러바 추가
            if i == n_snapshots - 1:
                plt.colorbar(scatter1, ax=ax1, label='Cluster ID')
                plt.colorbar(scatter2, ax=ax2, label='Task ID')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/umap_clustering_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_cluster_evolution_plot(self):
        """클러스터 수 변화 시각화"""
        if not self.latent_snapshots:
            return
        
        steps = [s['step'] for s in self.latent_snapshots]
        num_clusters = [s['num_clusters'] for s in self.latent_snapshots]
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, num_clusters, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Training Steps')
        plt.ylabel('Number of Clusters')
        plt.title('Cluster Count Evolution During Training')
        plt.grid(True, alpha=0.3)
        
        # 주요 변화점 표시
        for i in range(1, len(num_clusters)):
            if num_clusters[i] != num_clusters[i-1]:
                plt.axvline(x=steps[i], color='red', linestyle='--', alpha=0.5)
                plt.annotate(f'{num_clusters[i]}', 
                           (steps[i], num_clusters[i]), 
                           xytext=(10, 10), textcoords='offset points')
        
        plt.savefig(f'{self.save_dir}/cluster_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_task_cluster_heatmap(self):
        """태스크-클러스터 관계 히트맵"""
        if not self.latent_snapshots:
            return
        
        # 최종 스냅샷 사용
        final_snapshot = self.latent_snapshots[-1]
        
        # 태스크-클러스터 매트릭스 생성
        unique_tasks = np.unique(final_snapshot['tasks'])
        unique_clusters = np.unique(final_snapshot['clusters'])
        
        matrix = np.zeros((len(unique_tasks), len(unique_clusters)))
        
        for i, task in enumerate(unique_tasks):
            for j, cluster in enumerate(unique_clusters):
                mask = (final_snapshot['tasks'] == task) & (final_snapshot['clusters'] == cluster)
                matrix[i, j] = np.sum(mask)
        
        # 히트맵 생성
        plt.figure(figsize=(10, 6))
        sns.heatmap(matrix, 
                   xticklabels=[f'Cluster {c}' for c in unique_clusters],
                   yticklabels=[f'Task {t}' for t in unique_tasks],
                   annot=True, fmt='g', cmap='Blues')
        plt.title('Task-Cluster Assignment Matrix (Final State)')
        plt.xlabel('Clusters')
        plt.ylabel('Tasks')
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/task_cluster_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_data(self, filename: str = "clustering_data.pkl"):
        """시각화 데이터 저장"""
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.latent_snapshots, f)
        print(f"💾 Data saved to: {filepath}")
    
    def load_data(self, filename: str = "clustering_data.pkl"):
        """시각화 데이터 로드"""
        filepath = os.path.join(self.save_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.latent_snapshots = pickle.load(f)
            print(f"📂 Data loaded from: {filepath}")
        else:
            print(f"❌ File not found: {filepath}")


class ModifiedTrainer(Trainer):
    """시각화 기능이 추가된 트레이너"""
    
    def __init__(self, config: Dict, visualizer: ClusteringVisualizer = None):
        super().__init__(config)
        self.visualizer = visualizer or ClusteringVisualizer()
        
        # 스냅샷 저장 간격
        self.snapshot_freq = config.get('snapshot_freq', 2000)
        
        print(f"📊 Modified trainer with visualization enabled")
    
    def _collect_latent_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """모든 expert buffer에서 latent 데이터 수집"""
        all_latents = []
        all_clusters = []
        all_tasks = []
        
        for buffer in self.agent.expert_buffers:
            if buffer.size > 0:
                # 최근 데이터 샘플링
                sample_size = min(buffer.size, 200)
                indices = np.random.choice(buffer.size, sample_size, replace=False)
                
                latents = buffer.latents[indices]
                tasks = buffer.task_ids[indices]
                
                # 클러스터 할당
                clusters = []
                for latent in latents:
                    latent_tensor = torch.FloatTensor(latent)
                    cluster_id = self.agent.dpmm.infer_cluster(latent_tensor)
                    clusters.append(cluster_id)
                
                all_latents.extend(latents)
                all_clusters.extend(clusters)
                all_tasks.extend(tasks)
        
        return (np.array(all_latents), 
                np.array(all_clusters), 
                np.array(all_tasks))
    
    def train(self):
        """시각화 기능이 추가된 훈련 루프"""
        step = 0
        episode_reward = 0
        episode_length = 0
        
        obs, _ = self.env.reset()
        
        print("🚀 Starting training with visualization...")
        
        while step < self.config['total_steps']:
            # 기존 훈련 로직
            action, latent, cluster_id, expert_id = self.agent.select_action(obs, eval_mode=False)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            self.agent.add_transition(
                obs, action, reward, next_obs, done, 
                latent, cluster_id, expert_id, self.env.current_task
            )
            
            episode_reward += reward
            episode_length += 1
            step += 1
            
            # Episode end
            if done:
                self.history['episode_rewards'].append(episode_reward)
                self.history['episode_lengths'].append(episode_length)
                self.task_performance_history[self.env.current_task].append(episode_reward)
                
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
            
            # Training
            if step % self.config['train_freq'] == 0:
                losses = self.agent.train_step()
                
                if losses and step % 5000 == 0:
                    self._log_training_progress(step, losses)
            
            # DPMM update
            if step % self.config['dpmm_freq'] == 0:
                self._update_dpmm_statistics(step)
            
            # 스냅샷 저장
            if step % self.snapshot_freq == 0 and step > 0:
                try:
                    latents, clusters, tasks = self._collect_latent_data()
                    if len(latents) > 0:
                        self.visualizer.add_snapshot(step, latents, clusters, tasks)
                except Exception as e:
                    print(f"⚠️ Snapshot collection failed at step {step}: {e}")
            
            # Evaluation
            if step % self.config['eval_freq'] == 0:
                self._comprehensive_evaluation(step)
        
        # 최종 스냅샷
        try:
            latents, clusters, tasks = self._collect_latent_data()
            if len(latents) > 0:
                self.visualizer.add_snapshot(step, latents, clusters, tasks)
        except Exception as e:
            print(f"⚠️ Final snapshot failed: {e}")
        
        print("✅ Training completed!")
        self._final_evaluation()
        
        # 시각화 생성
        self.create_visualizations()
    
    def create_visualizations(self):
        """모든 시각화 생성"""
        print("\n🎨 Creating visualizations...")
        
        try:
            # 1. UMAP 클러스터링 진화
            self.visualizer.create_umap_visualization()
            
            # 2. 클러스터 수 변화
            self.visualizer.create_cluster_evolution_plot()
            
            # 3. 태스크-클러스터 히트맵
            self.visualizer.create_task_cluster_heatmap()
            
            # 4. 데이터 저장
            self.visualizer.save_data()
            
            print("✅ All visualizations created successfully!")
            
        except Exception as e:
            print(f"❌ Visualization creation failed: {e}")
            import traceback
            traceback.print_exc()


def create_animated_clustering(visualizer: ClusteringVisualizer, output_file: str = "clustering_animation.gif"):
    """클러스터링 과정의 애니메이션 생성"""
    import matplotlib.animation as animation
    
    if not visualizer.latent_snapshots:
        print("❌ No snapshots available for animation")
        return
    
    # 모든 latent 데이터를 합쳐서 전체 UMAP 공간 생성
    all_latents = np.vstack([s['latents'] for s in visualizer.latent_snapshots])
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    all_latents_2d = umap_reducer.fit_transform(all_latents)
    
    # 각 스냅샷의 시작 인덱스 계산
    start_indices = [0]
    for snapshot in visualizer.latent_snapshots[:-1]:
        start_indices.append(start_indices[-1] + len(snapshot['latents']))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    def animate(frame):
        if frame >= len(visualizer.latent_snapshots):
            return
        
        ax1.clear()
        ax2.clear()
        
        snapshot = visualizer.latent_snapshots[frame]
        start_idx = start_indices[frame]
        end_idx = start_idx + len(snapshot['latents'])
        
        latents_2d = all_latents_2d[start_idx:end_idx]
        
        # 클러스터별 시각화
        scatter1 = ax1.scatter(
            latents_2d[:, 0], latents_2d[:, 1],
            c=snapshot['clusters'],
            cmap='tab10',
            alpha=0.7,
            s=50
        )
        ax1.set_title(f'Step {snapshot["step"]:,} - Clusters: {snapshot["num_clusters"]}')
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        
        # 태스크별 시각화
        scatter2 = ax2.scatter(
            latents_2d[:, 0], latents_2d[:, 1],
            c=snapshot['tasks'],
            cmap='viridis',
            alpha=0.7,
            s=50
        )
        ax2.set_title(f'Tasks Distribution')
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        
        # 클러스터 중심점 표시
        unique_clusters = np.unique(snapshot['clusters'])
        for cluster_id in unique_clusters:
            mask = snapshot['clusters'] == cluster_id
            if np.any(mask):
                center = latents_2d[mask].mean(axis=0)
                ax1.scatter(center[0], center[1], c='black', s=200, marker='x', linewidths=3)
    
    anim = animation.FuncAnimation(
        fig, animate, frames=len(visualizer.latent_snapshots),
        interval=1000, repeat=True
    )
    
    # GIF로 저장
    output_path = os.path.join(visualizer.save_dir, output_file)
    anim.save(output_path, writer='pillow', fps=1)
    print(f"🎬 Animation saved to: {output_path}")
    
    plt.show()


def analyze_cluster_stability(visualizer: ClusteringVisualizer):
    """클러스터 안정성 분석"""
    if len(visualizer.latent_snapshots) < 2:
        print("❌ Need at least 2 snapshots for stability analysis")
        return
    
    print("\n📊 Cluster Stability Analysis")
    print("=" * 50)
    
    stability_scores = []
    
    for i in range(1, len(visualizer.latent_snapshots)):
        prev_snapshot = visualizer.latent_snapshots[i-1]
        curr_snapshot = visualizer.latent_snapshots[i]
        
        # 동일한 latent points에 대한 클러스터 할당 비교 (간단화된 버전)
        stability = np.mean(prev_snapshot['clusters'] == curr_snapshot['clusters'][:len(prev_snapshot['clusters'])])
        stability_scores.append(stability)
        
        print(f"Step {prev_snapshot['step']} → {curr_snapshot['step']}: {stability:.3f}")
    
    avg_stability = np.mean(stability_scores)
    print(f"\nAverage Stability: {avg_stability:.3f}")
    
    # 안정성 그래프
    plt.figure(figsize=(10, 6))
    steps = [visualizer.latent_snapshots[i]['step'] for i in range(1, len(visualizer.latent_snapshots))]
    plt.plot(steps, stability_scores, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Training Steps')
    plt.ylabel('Cluster Stability')
    plt.title('Cluster Assignment Stability Over Time')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=avg_stability, color='red', linestyle='--', label=f'Average: {avg_stability:.3f}')
    plt.legend()
    plt.savefig(f'{visualizer.save_dir}/cluster_stability.png', dpi=300, bbox_inches='tight')
    plt.show()


def main_with_visualization():
    """시각화가 포함된 메인 실행 함수"""
    print("🚀 Lifelong RL with Clustering Visualization")
    print("=" * 80)
    
    # 설정
    config = get_default_config()
    config['total_steps'] = 50000  # 🔥 5개 태스크이므로 더 길게
    config['snapshot_freq'] = 5000  # 🔥 스냅샷 주기
    config['task_count'] = 5        # 🔥 명시적으로 5개 태스크 설정
    
    print("📋 Configuration (with 5 tasks):")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # 시각화기 생성
    visualizer = ClusteringVisualizer(save_dir="clustering_results_5tasks")
    
    try:
        # 수정된 트레이너로 훈련
        trainer = ModifiedTrainer(config, visualizer)
        trainer.train()
        
        print("\n✅ Training completed successfully!")
        
        # 추가 분석
        print("\n🔍 Additional Analysis...")
        
        # 1. 애니메이션 생성
        try:
            create_animated_clustering(visualizer)
        except Exception as e:
            print(f"⚠️ Animation creation failed: {e}")
        
        # 2. 클러스터 안정성 분석
        try:
            analyze_cluster_stability(visualizer)
        except Exception as e:
            print(f"⚠️ Stability analysis failed: {e}")
        
        # 3. 상세 통계
        print(f"\n📈 Final Statistics:")
        print(f"   Total snapshots: {len(visualizer.latent_snapshots)}")
        if visualizer.latent_snapshots:
            final_snapshot = visualizer.latent_snapshots[-1]
            print(f"   Final clusters: {final_snapshot['num_clusters']}")
            print(f"   Final data points: {len(final_snapshot['latents'])}")
        
        print(f"\n💾 All results saved to: {visualizer.save_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        # 중간 결과라도 시각화
        if len(visualizer.latent_snapshots) > 0:
            print("🎨 Creating visualizations with available data...")
            visualizer.create_umap_visualization()
            visualizer.create_cluster_evolution_plot()
            
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


def load_and_visualize(data_path: str = "clustering_results/clustering_data.pkl"):
    """저장된 데이터로부터 시각화 생성"""
    print(f"📂 Loading data from: {data_path}")
    
    visualizer = ClusteringVisualizer()
    visualizer.load_data(data_path)
    
    if visualizer.latent_snapshots:
        print("🎨 Creating visualizations from loaded data...")
        visualizer.create_umap_visualization()
        visualizer.create_cluster_evolution_plot()
        visualizer.create_task_cluster_heatmap()
        
        # 추가 분석
        analyze_cluster_stability(visualizer)
        
        try:
            create_animated_clustering(visualizer)
        except Exception as e:
            print(f"⚠️ Animation creation failed: {e}")
    else:
        print("❌ No data found in the file")


if __name__ == "__main__":
    # 실행 옵션 선택
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "load":
        # 저장된 데이터로부터 시각화
        data_path = sys.argv[2] if len(sys.argv) > 2 else "clustering_results/clustering_data.pkl"
        load_and_visualize(data_path)
    else:
        # 새로운 훈련 + 시각화
        main_with_visualization()
