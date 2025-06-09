import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List
import pandas as pd

class ExperimentVisualizer:
    """실험 결과 시각화 클래스"""
    
    def __init__(self, save_dir: str = "./experiment_plots"):
        self.save_dir = save_dir
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 스타일 설정
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_training_curves(self, training_stats: Dict, title: str = "Training Progress"):
        """학습 곡선 플롯"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Episode Rewards
        rewards = training_stats.get('episode_rewards', [])
        if rewards:
            axes[0, 0].plot(rewards, alpha=0.6, color='blue', linewidth=1)
            # 이동 평균
            window = min(20, len(rewards) // 10)
            if window > 1:
                moving_avg = pd.Series(rewards).rolling(window=window).mean()
                axes[0, 0].plot(moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window})')
            
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Cluster Evolution
        cluster_counts = training_stats.get('cluster_evolution', [])
        if cluster_counts:
            axes[0, 1].plot(cluster_counts, color='green', linewidth=2, marker='o', markersize=3)
            axes[0, 1].set_title('DPMM Cluster Evolution')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Number of Clusters')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Expert Evolution
        expert_counts = training_stats.get('expert_evolution', [])
        if expert_counts:
            axes[1, 0].plot(expert_counts, color='orange', linewidth=2, marker='s', markersize=3)
            axes[1, 0].set_title('Expert Network Evolution')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Number of Experts')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. DPMM Events
        dpmm_events = training_stats.get('dpmm_events', [])
        if dpmm_events:
            birth_episodes = [e['episode'] for e in dpmm_events if e.get('type') == 'birth']
            merge_episodes = [e['episode'] for e in dpmm_events if e.get('type') == 'merge']
            
            if birth_episodes:
                axes[1, 1].scatter(birth_episodes, [1]*len(birth_episodes), 
                                 color='red', alpha=0.7, s=50, label='Births')
            if merge_episodes:
                axes[1, 1].scatter(merge_episodes, [0]*len(merge_episodes), 
                                 color='blue', alpha=0.7, s=50, label='Merges')
            
            axes[1, 1].set_title('DPMM Events')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Event Type')
            axes[1, 1].set_yticks([0, 1])
            axes[1, 1].set_yticklabels(['Merge', 'Birth'])
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/training_curves.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_task_comparison(self, all_stats: Dict, task_names: List[str]):
        """태스크별 성능 비교"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Multi-Task Performance Comparison', fontsize=16, fontweight='bold')
        
        task_rewards = []
        task_final_clusters = []
        task_final_experts = []
        
        # 태스크별 데이터 추출
        for i, task_data in enumerate(all_stats.get('tasks', [])):
            rewards = task_data.get('episode_rewards', [])
            clusters = task_data.get('cluster_counts', [])
            experts = task_data.get('expert_counts', [])
            
            if rewards:
                task_rewards.append({
                    'task': task_names[i] if i < len(task_names) else f'Task {i+1}',
                    'avg_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'final_reward': np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
                })
            
            if clusters:
                task_final_clusters.append(clusters[-1])
            if experts:
                task_final_experts.append(experts[-1])
        
        # 1. 태스크별 평균 보상
        if task_rewards:
            task_names_plot = [t['task'] for t in task_rewards]
            avg_rewards = [t['avg_reward'] for t in task_rewards]
            std_rewards = [t['std_reward'] for t in task_rewards]
            
            bars = axes[0, 0].bar(task_names_plot, avg_rewards, yerr=std_rewards, 
                                 capsize=5, alpha=0.7, color='skyblue')
            axes[0, 0].set_title('Average Reward per Task')
            axes[0, 0].set_ylabel('Average Reward')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 값 표시
            for bar, reward in zip(bars, avg_rewards):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               f'{reward:.2f}', ha='center', va='bottom')
        
        # 2. 태스크별 최종 보상 (학습 진행도)
        if task_rewards:
            final_rewards = [t['final_reward'] for t in task_rewards]
            bars = axes[0, 1].bar(task_names_plot, final_rewards, alpha=0.7, color='lightgreen')
            axes[0, 1].set_title('Final Performance per Task')
            axes[0, 1].set_ylabel('Final Reward (Last 10 episodes)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            for bar, reward in zip(bars, final_rewards):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               f'{reward:.2f}', ha='center', va='bottom')
        
        # 3. 태스크별 최종 클러스터 수
        if task_final_clusters and task_names:
            axes[1, 0].bar(task_names[:len(task_final_clusters)], task_final_clusters, 
                          alpha=0.7, color='coral')
            axes[1, 0].set_title('Final Number of Clusters per Task')
            axes[1, 0].set_ylabel('Number of Clusters')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            for i, count in enumerate(task_final_clusters):
                axes[1, 0].text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        # 4. 태스크별 최종 전문가 수
        if task_final_experts and task_names:
            axes[1, 1].bar(task_names[:len(task_final_experts)], task_final_experts, 
                          alpha=0.7, color='gold')
            axes[1, 1].set_title('Final Number of Experts per Task')
            axes[1, 1].set_ylabel('Number of Experts')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            for i, count in enumerate(task_final_experts):
                axes[1, 1].text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/task_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_dpmm_analysis(self, training_stats: Dict):
        """DPMM 상세 분석 플롯"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DPMM Detailed Analysis', fontsize=16, fontweight='bold')
        
        # 1. Birth vs Merge Events Timeline
        dpmm_events = training_stats.get('dpmm_events', [])
        if dpmm_events:
            birth_episodes = [e['episode'] for e in dpmm_events if e.get('type') == 'birth']
            merge_episodes = [e['episode'] for e in dpmm_events if e.get('type') == 'merge']
            
            # Birth events
            if birth_episodes:
                birth_hist, birth_bins = np.histogram(birth_episodes, bins=20)
                axes[0, 0].bar(birth_bins[:-1], birth_hist, width=np.diff(birth_bins), 
                              alpha=0.7, color='red', label='Births')
            
            # Merge events
            if merge_episodes:
                merge_hist, merge_bins = np.histogram(merge_episodes, bins=20)
                axes[0, 0].bar(merge_bins[:-1], -merge_hist, width=np.diff(merge_bins), 
                              alpha=0.7, color='blue', label='Merges')
            
            axes[0, 0].set_title('Birth vs Merge Events Distribution')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Event Count')
            axes[0, 0].legend()
            axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 2. Cluster Count Distribution
        cluster_counts = training_stats.get('cluster_evolution', [])
        if cluster_counts:
            axes[0, 1].hist(cluster_counts, bins=15, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].set_title('Cluster Count Distribution')
            axes[0, 1].set_xlabel('Number of Clusters')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(np.mean(cluster_counts), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(cluster_counts):.1f}')
            axes[0, 1].legend()
        
        # 3. Expert vs Cluster Relationship
        expert_counts = training_stats.get('expert_evolution', [])
        if cluster_counts and expert_counts:
            min_len = min(len(cluster_counts), len(expert_counts))
            axes[0, 2].scatter(cluster_counts[:min_len], expert_counts[:min_len], 
                              alpha=0.6, s=30)
            axes[0, 2].plot([0, max(cluster_counts)], [0, max(cluster_counts)], 
                           'r--', alpha=0.5, label='Perfect Correlation')
            axes[0, 2].set_title('Experts vs Clusters Relationship')
            axes[0, 2].set_xlabel('Number of Clusters')
            axes[0, 2].set_ylabel('Number of Experts')
            axes[0, 2].legend()
        
        # 4. Reward vs Cluster Count Correlation
        rewards = training_stats.get('episode_rewards', [])
        if rewards and cluster_counts:
            min_len = min(len(rewards), len(cluster_counts))
            
            # 이동 평균으로 스무딩
            window = max(1, min_len // 20)
            if window > 1:
                smooth_rewards = pd.Series(rewards[:min_len]).rolling(window=window).mean()
                smooth_clusters = pd.Series(cluster_counts[:min_len]).rolling(window=window).mean()
                
                axes[1, 0].scatter(smooth_clusters, smooth_rewards, alpha=0.6, s=20)
                
                # 상관계수 계산
                correlation = np.corrcoef(smooth_clusters.dropna(), smooth_rewards.dropna())[0, 1]
                axes[1, 0].set_title(f'Reward vs Clusters (Corr: {correlation:.3f})')
                axes[1, 0].set_xlabel('Number of Clusters (Smoothed)')
                axes[1, 0].set_ylabel('Episode Reward (Smoothed)')
        
        # 5. Birth Rate Over Time
        if dpmm_events:
            episodes = [e['episode'] for e in dpmm_events]
            if episodes:
                max_episode = max(episodes)
                window_size = max(10, max_episode // 20)
                
                birth_rates = []
                episode_windows = []
                
                for i in range(0, max_episode, window_size):
                    window_births = len([e for e in birth_episodes 
                                       if i <= e < i + window_size])
                    birth_rate = window_births / window_size
                    birth_rates.append(birth_rate)
                    episode_windows.append(i + window_size // 2)
                
                axes[1, 1].plot(episode_windows, birth_rates, marker='o', linewidth=2)
                axes[1, 1].set_title('Birth Rate Over Time')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Births per Episode')
                axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Cluster Stability Analysis
        if cluster_counts:
            # 클러스터 수 변화율
            cluster_changes = np.diff(cluster_counts)
            stability_score = 1.0 / (1.0 + np.std(cluster_changes))
            
            axes[1, 2].plot(cluster_changes, alpha=0.7, color='purple')
            axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 2].set_title(f'Cluster Stability (Score: {stability_score:.3f})')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Cluster Count Change')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/dpmm_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_hyperparameter_sensitivity(self, results_dict: Dict[str, Dict]):
        """하이퍼파라미터 민감도 분석"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Hyperparameter Sensitivity Analysis', fontsize=16, fontweight='bold')
        
        # tau_birth 민감도
        tau_birth_values = []
        tau_birth_rewards = []
        tau_birth_clusters = []
        
        for config_name, results in results_dict.items():
            if 'tau_birth' in config_name:
                # 설정에서 tau_birth 값 추출
                tau_val = float(config_name.split('tau_birth_')[1].split('_')[0])
                tau_birth_values.append(tau_val)
                
                rewards = results.get('episode_rewards', [])
                clusters = results.get('cluster_evolution', [])
                
                tau_birth_rewards.append(np.mean(rewards) if rewards else 0)
                tau_birth_clusters.append(np.mean(clusters) if clusters else 0)
        
        if tau_birth_values:
            # 정렬
            sorted_indices = np.argsort(tau_birth_values)
            tau_birth_values = np.array(tau_birth_values)[sorted_indices]
            tau_birth_rewards = np.array(tau_birth_rewards)[sorted_indices]
            tau_birth_clusters = np.array(tau_birth_clusters)[sorted_indices]
            
            axes[0, 0].plot(tau_birth_values, tau_birth_rewards, 'o-', color='blue', linewidth=2)
            axes[0, 0].set_title('tau_birth vs Average Reward')
            axes[0, 0].set_xlabel('tau_birth')
            axes[0, 0].set_ylabel('Average Reward')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(tau_birth_values, tau_birth_clusters, 'o-', color='red', linewidth=2)
            axes[0, 1].set_title('tau_birth vs Average Clusters')
            axes[0, 1].set_xlabel('tau_birth')
            axes[0, 1].set_ylabel('Average Number of Clusters')
            axes[0, 1].grid(True, alpha=0.3)
        
        # tau_merge_kl 민감도 (유사하게 구현)
        tau_merge_values = []
        tau_merge_rewards = []
        tau_merge_merges = []
        
        for config_name, results in results_dict.items():
            if 'tau_merge' in config_name:
                tau_val = float(config_name.split('tau_merge_')[1].split('_')[0])
                tau_merge_values.append(tau_val)
                
                rewards = results.get('episode_rewards', [])
                dpmm_events = results.get('dpmm_events', [])
                
                tau_merge_rewards.append(np.mean(rewards) if rewards else 0)
                merge_count = len([e for e in dpmm_events if e.get('type') == 'merge'])
                tau_merge_merges.append(merge_count)
        
        if tau_merge_values:
            sorted_indices = np.argsort(tau_merge_values)
            tau_merge_values = np.array(tau_merge_values)[sorted_indices]
            tau_merge_rewards = np.array(tau_merge_rewards)[sorted_indices]
            tau_merge_merges = np.array(tau_merge_merges)[sorted_indices]
            
            axes[1, 0].plot(tau_merge_values, tau_merge_rewards, 'o-', color='green', linewidth=2)
            axes[1, 0].set_title('tau_merge_kl vs Average Reward')
            axes[1, 0].set_xlabel('tau_merge_kl')
            axes[1, 0].set_ylabel('Average Reward')
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(tau_merge_values, tau_merge_merges, 'o-', color='orange', linewidth=2)
            axes[1, 1].set_title('tau_merge_kl vs Merge Events')
            axes[1, 1].set_xlabel('tau_merge_kl')
            axes[1, 1].set_ylabel('Number of Merge Events')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/hyperparameter_sensitivity.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_report(self, training_stats: Dict, task_names: List[str]):
        """종합 실험 보고서 생성"""
        print("📊 Generating Comprehensive Experiment Report...")
        
        # 모든 그래프 생성
        self.plot_training_curves(training_stats, "Complete SB3-DPMM Training Results")
        self.plot_task_comparison(training_stats, task_names)
        self.plot_dpmm_analysis(training_stats)
        
        # 수치 요약 생성
        self.generate_numerical_summary(training_stats, task_names)
        
        print(f"✅ All plots saved to {self.save_dir}/")
    
    def generate_numerical_summary(self, training_stats: Dict, task_names: List[str]):
        """수치 요약 생성"""
        summary = {
            'Overall Performance': {},
            'DPMM Statistics': {},
            'Task-wise Performance': {}
        }
        
        # 전체 성능
        rewards = training_stats.get('episode_rewards', [])
        if rewards:
            summary['Overall Performance'] = {
                'Total Episodes': len(rewards),
                'Average Reward': f"{np.mean(rewards):.3f}",
                'Best Reward': f"{max(rewards):.3f}",
                'Final Performance': f"{np.mean(rewards[-20:]):.3f}" if len(rewards) >= 20 else "N/A",
                'Reward Std': f"{np.std(rewards):.3f}"
            }
        
        # DPMM 통계
        cluster_counts = training_stats.get('cluster_evolution', [])
        expert_counts = training_stats.get('expert_evolution', [])
        dpmm_events = training_stats.get('dpmm_events', [])
        
        if cluster_counts:
            birth_events = [e for e in dpmm_events if e.get('type') == 'birth']
            merge_events = [e for e in dpmm_events if e.get('type') == 'merge']
            
            summary['DPMM Statistics'] = {
                'Final Clusters': cluster_counts[-1] if cluster_counts else 0,
                'Final Experts': expert_counts[-1] if expert_counts else 0,
                'Total Births': len(birth_events),
                'Total Merges': len(merge_events),
                'Average Clusters': f"{np.mean(cluster_counts):.2f}",
                'Cluster Stability': f"{1.0 / (1.0 + np.std(np.diff(cluster_counts))):.3f}",
                'Birth Rate': f"{len(birth_events) / len(rewards) * 100:.2f}%" if rewards else "0%"
            }
        
        # 태스크별 성능
        tasks_data = training_stats.get('tasks', [])
        for i, task_data in enumerate(tasks_data):
            task_name = task_names[i] if i < len(task_names) else f'Task_{i+1}'
            task_rewards = task_data.get('episode_rewards', [])
            
            if task_rewards:
                summary['Task-wise Performance'][task_name] = {
                    'Episodes': len(task_rewards),
                    'Average Reward': f"{np.mean(task_rewards):.3f}",
                    'Best Reward': f"{max(task_rewards):.3f}",
                    'Learning Progress': f"{(np.mean(task_rewards[-10:]) - np.mean(task_rewards[:10])):.3f}" if len(task_rewards) >= 20 else "N/A"
                }
        
        # 요약을 파일로 저장
        import json
        with open(f"{self.save_dir}/experiment_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 콘솔 출력
        print("\n" + "="*60)
        print("📈 EXPERIMENT SUMMARY")
        print("="*60)
        
        for category, data in summary.items():
            print(f"\n🔸 {category}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict):
                        print(f"  📌 {key}:")
                        for sub_key, sub_value in value.items():
                            print(f"    • {sub_key}: {sub_value}")
                    else:
                        print(f"  • {key}: {value}")
        
        print(f"\n💾 Detailed summary saved to: {self.save_dir}/experiment_summary.json")

# 실험 결과 로드 및 시각화 함수
def load_and_visualize_results(stats_path: str, task_names: List[str] = None):
    """실험 결과 로드 및 시각화"""
    try:
        # NumPy 파일 로드
        data = np.load(stats_path, allow_pickle=True)
        
        # 딕셔너리로 변환
        training_stats = {}
        for key in data.files:
            training_stats[key] = data[key].item() if data[key].ndim == 0 else data[key]
        
        print(f"✅ Loaded training stats from {stats_path}")
        print(f"📊 Available keys: {list(training_stats.keys())}")
        
        # 기본 태스크 이름
        if task_names is None:
            task_names = [
                'Push Cube Basic',
                'Push Cube Far',
                'Push Cube Left',
                'Push Cube Right',
                'Push Cube Diagonal'
            ]
        
        # 시각화 생성
        visualizer = ExperimentVisualizer()
        visualizer.create_comprehensive_report(training_stats, task_names)
        
        return training_stats
        
    except Exception as e:
        print(f"❌ Failed to load results: {e}")
        return None

# Birth 문제 분석 함수
def analyze_birth_problem(training_stats: Dict):
    """Birth 문제 상세 분석"""
    print("\n" + "="*60)
    print("🔍 BIRTH PROBLEM ANALYSIS")
    print("="*60)
    
    dpmm_events = training_stats.get('dpmm_events', [])
    cluster_evolution = training_stats.get('cluster_evolution', [])
    episode_rewards = training_stats.get('episode_rewards', [])
    
    if not dpmm_events:
        print("❌ No DPMM events data available")
        return
    
    birth_events = [e for e in dpmm_events if e.get('type') == 'birth']
    merge_events = [e for e in dpmm_events if e.get('type') == 'merge']
    
    total_episodes = len(episode_rewards) if episode_rewards else 0
    
    print(f"📊 Birth Problem Statistics:")
    print(f"  • Total Episodes: {total_episodes}")
    print(f"  • Total Birth Events: {len(birth_events)}")
    print(f"  • Total Merge Events: {len(merge_events)}")
    print(f"  • Birth Rate: {len(birth_events) / max(1, total_episodes) * 100:.2f}% per episode")
    print(f"  • Merge Rate: {len(merge_events) / max(1, total_episodes) * 100:.2f}% per episode")
    
    if cluster_evolution:
        print(f"  • Final Clusters: {cluster_evolution[-1]}")
        print(f"  • Max Clusters: {max(cluster_evolution)}")
        print(f"  • Average Clusters: {np.mean(cluster_evolution):.2f}")
    
    # Birth 패턴 분석
    if birth_events:
        birth_episodes = [e['episode'] for e in birth_events]
        birth_intervals = np.diff(birth_episodes) if len(birth_episodes) > 1 else []
        
        print(f"\n🔍 Birth Pattern Analysis:")
        print(f"  • First Birth: Episode {min(birth_episodes)}")
        print(f"  • Last Birth: Episode {max(birth_episodes)}")
        if birth_intervals:
            print(f"  • Average Birth Interval: {np.mean(birth_intervals):.2f} episodes")
            print(f"  • Min Birth Interval: {min(birth_intervals)} episodes")
            print(f"  • Max Birth Interval: {max(birth_intervals)} episodes")
    
    # 권장 해결책
    print(f"\n💡 RECOMMENDED SOLUTIONS:")
    print(f"  1. Increase tau_birth threshold (current seems too strict)")
    print(f"     • Try tau_birth = -2.0 instead of -5.0")
    print(f"  2. Increase initial cluster variance")
    print(f"     • Try initial_variance = 0.1 instead of 1e-3")
    print(f"  3. More aggressive merging")
    print(f"     • Try tau_merge_kl = 0.5 instead of 0.1")
    print(f"  4. Implement adaptive birth threshold")
    print(f"     • Adjust threshold based on recent birth rate")

# 하이퍼파라미터 스위프 실험 함수
def run_hyperparameter_sweep():
    """하이퍼파라미터 스위프 실험"""
    
    sweep_script = '''
#!/usr/bin/env python3
"""
하이퍼파라미터 스위프 실험
"""

import subprocess
import itertools
import os

# 실험할 하이퍼파라미터
tau_birth_values = [-1.0, -2.0, -3.0, -4.0, -5.0]
tau_merge_values = [0.05, 0.1, 0.2, 0.5, 1.0]
latent_dims = [16, 24, 32]

# 실험 설정
base_cmd = [
    "python", "main_complete_sb3_dpmm_training.py",
    "--episodes", "30",  # 빠른 실험을 위해 적은 에피소드
    "--max_steps", "200"
]

results_dir = "./hyperparameter_sweep_results"
os.makedirs(results_dir, exist_ok=True)

experiment_count = 0
total_experiments = len(tau_birth_values) * len(tau_merge_values) * len(latent_dims)

print(f"🚀 Starting hyperparameter sweep: {total_experiments} experiments")

for tau_birth, tau_merge, latent_dim in itertools.product(tau_birth_values, tau_merge_values, latent_dims):
    experiment_count += 1
    
    # 실험 이름
    exp_name = f"tau_birth_{tau_birth}_tau_merge_{tau_merge}_latent_{latent_dim}"
    exp_dir = os.path.join(results_dir, exp_name)
    
    print(f"\\n[{experiment_count}/{total_experiments}] Running: {exp_name}")
    
    # 명령어 구성
    cmd = base_cmd + [
        "--tau_birth", str(tau_birth),
        "--tau_merge_kl", str(tau_merge),
        "--latent_dim", str(latent_dim),
        "--save_dir", exp_dir
    ]
    
    try:
        # 실험 실행
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30분 타임아웃
        
        if result.returncode == 0:
            print(f"  ✅ Success: {exp_name}")
        else:
            print(f"  ❌ Failed: {exp_name}")
            print(f"     Error: {result.stderr[:200]}...")
            
    except subprocess.TimeoutExpired:
        print(f"  ⏰ Timeout: {exp_name}")
    except Exception as e:
        print(f"  💥 Exception: {exp_name} - {e}")

print(f"\\n🎉 Hyperparameter sweep completed!")
print(f"📊 Results saved in: {results_dir}")
'''
    
    # 스위프 스크립트 저장
    with open("hyperparameter_sweep.py", "w") as f:
        f.write(sweep_script)
    
    print("✅ Hyperparameter sweep script created: hyperparameter_sweep.py")
    print("🚀 Run with: python hyperparameter_sweep.py")

# 메인 시각화 실행 함수
def main_visualization_demo():
    """시각화 데모 실행"""
    
    # 더미 데이터 생성 (실제 데이터가 없을 때)
    dummy_stats = {
        'episode_rewards': np.random.normal(10, 5, 200) + np.linspace(0, 20, 200),  # 학습 진행
        'cluster_evolution': np.random.poisson(8, 200) + np.random.randint(-2, 3, 200).cumsum(),
        'expert_evolution': np.random.poisson(6, 200) + np.random.randint(-1, 2, 200).cumsum(),
        'dpmm_events': [
            {'episode': i, 'type': 'birth'} for i in np.random.choice(200, 30, replace=False)
        ] + [
            {'episode': i, 'type': 'merge'} for i in np.random.choice(200, 15, replace=False)
        ],
        'tasks': [
            {
                'episode_rewards': np.random.normal(8, 3, 40) + np.linspace(0, 15, 40),
                'cluster_counts': np.random.poisson(5, 40),
                'expert_counts': np.random.poisson(4, 40)
            }
            for _ in range(5)
        ]
    }
    
    task_names = [
        'Push Cube Basic',
        'Push Cube Far', 
        'Push Cube Left',
        'Push Cube Right',
        'Push Cube Diagonal'
    ]
    
    print("📊 Running visualization demo with dummy data...")
    
    visualizer = ExperimentVisualizer(save_dir="./demo_plots")
    visualizer.create_comprehensive_report(dummy_stats, task_names)
    
    # Birth 문제 분석
    analyze_birth_problem(dummy_stats)
    
    print("✅ Demo completed! Check ./demo_plots/ for results")

if __name__ == "__main__":
    # 데모 실행
    main_visualization_demo()
    
    # 하이퍼파라미터 스위프 스크립트 생성
    run_hyperparameter_sweep()
