#!/usr/bin/env python3
"""
개선된 DPMM 테스트 스크립트
"""

import numpy as np
import matplotlib.pyplot as plt
from improved_birth_control_dpmm import AdaptiveBirthDPMM

def test_birth_control():
    """Birth 제어 테스트"""
    print("🧪 Testing Birth Control...")
    
    # 테스트 데이터 생성 (3개 클러스터)
    np.random.seed(42)
    
    cluster1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100)
    cluster2 = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], 100)
    cluster3 = np.random.multivariate_normal([0, 5], [[1, 0], [0, 1]], 100)
    
    test_data = np.vstack([cluster1, cluster2, cluster3])
    np.random.shuffle(test_data)
    
    # 다양한 설정으로 테스트
    configs = [
        {"name": "Default", "tau_birth": -2.0, "adaptive": True},
        {"name": "Strict", "tau_birth": -4.0, "adaptive": True},
        {"name": "Lenient", "tau_birth": -1.0, "adaptive": True},
        {"name": "Non-adaptive", "tau_birth": -2.0, "adaptive": False}
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n--- Testing {config['name']} ---")
        
        dpmm = AdaptiveBirthDPMM(
            latent_dim=2,
            initial_tau_birth=config['tau_birth'],
            adaptive_birth=config['adaptive'],
            max_clusters=10
        )
        
        assignments = []
        for i, point in enumerate(test_data):
            cluster_id = dpmm.assign_point(point)
            assignments.append(cluster_id)
            
            # 주기적 병합
            if i % 50 == 0 and i > 0:
                dpmm.merge_clusters_aggressive(verbose=False)
        
        stats = dpmm.get_statistics()
        results[config['name']] = {
            'assignments': assignments,
            'stats': stats,
            'dpmm': dpmm
        }
        print(f"  Total births: {stats['total_births']}")
        print(f"  Total merges: {stats['total_merges']}")
        print(f"  Birth rate: {stats['total_births']/len(test_data)*100:.1f}%")
        print(f"  Final tau_birth: {stats['current_tau_birth']:.3f}")
        print(f"  Final clusters: {stats['num_clusters']}")
        print(f"  Total births: {stats['total_births']}")
        print(f"  Total merges: {stats['total_merges']}")
        print(f"  Birth rate: {stats['total_births']/len(test_data)*100:.1f}%")
        print(f"  Final tau_birth: {stats['current_tau_birth']:.3f}")
    
    # 결과 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Birth Control Test Results', fontsize=16)
    
    for i, (name, result) in enumerate(results.items()):
        ax = axes[i//2, i%2]
        
        # 클러스터별 색상으로 포인트 플롯
        assignments = result['assignments']
        unique_clusters = list(set(assignments))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
        
        for cluster_id, color in zip(unique_clusters, colors):
            mask = np.array(assignments) == cluster_id
            points = test_data[mask]
            ax.scatter(points[:, 0], points[:, 1], c=[color], alpha=0.6, s=20)
        
        ax.set_title(f'{name}\nClusters: {result["stats"]["num_clusters"]}, '
                    f'Births: {result["stats"]["total_births"]}')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./birth_control_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def compare_birth_rates():
    """Birth rate 비교 실험"""
    print("\n🔬 Birth Rate Comparison Experiment...")
    
    # 시뮬레이션 데이터
    np.random.seed(123)
    n_points = 500
    
    # 점진적으로 변화하는 데이터 (concept drift)
    data_points = []
    for i in range(n_points):
        # 시간에 따라 클러스터 중심이 이동
        t = i / n_points
        center1 = [t * 3, 0]
        center2 = [0, t * 3]
        
        if np.random.random() < 0.5:
            point = np.random.multivariate_normal(center1, [[0.5, 0], [0, 0.5]])
        else:
            point = np.random.multivariate_normal(center2, [[0.5, 0], [0, 0.5]])
        
        data_points.append(point)
    
    data_points = np.array(data_points)
    
    # 다양한 설정 테스트
    test_configs = [
        {"tau_birth": -1.5, "adaptive": True, "name": "Adaptive Lenient"},
        {"tau_birth": -2.5, "adaptive": True, "name": "Adaptive Moderate"},
        {"tau_birth": -3.5, "adaptive": True, "name": "Adaptive Strict"},
        {"tau_birth": -2.5, "adaptive": False, "name": "Fixed Moderate"},
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n--- {config['name']} ---")
        
        dpmm = AdaptiveBirthDPMM(
            latent_dim=2,
            initial_tau_birth=config['tau_birth'],
            adaptive_birth=config['adaptive'],
            target_birth_rate=0.12,  # 12% 목표
            max_clusters=15
        )
        
        cluster_evolution = []
        birth_events = []
        tau_evolution = []
        
        for i, point in enumerate(data_points):
            cluster_id = dpmm.assign_point(point)
            
            stats = dpmm.get_statistics()
            cluster_evolution.append(stats['num_clusters'])
            tau_evolution.append(stats['current_tau_birth'])
            
            if len(dpmm.birth_events) > len(birth_events):
                birth_events.append(i)
            
            # 주기적 병합
            if i % 30 == 0 and i > 0:
                dpmm.merge_clusters_aggressive(verbose=False)
        
        final_stats = dpmm.get_statistics()
        results[config['name']] = {
            'cluster_evolution': cluster_evolution,
            'birth_events': birth_events,
            'tau_evolution': tau_evolution,
            'final_stats': final_stats
        }
        
        print(f"  Final clusters: {final_stats['num_clusters']}")
        print(f"  Birth rate: {final_stats['total_births']/n_points*100:.1f}%")
        print(f"  Final tau_birth: {final_stats['current_tau_birth']:.3f}")
    
    # 결과 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Birth Rate Comparison Results', fontsize=16)
    
    # 1. 클러스터 진화
    for name, result in results.items():
        axes[0, 0].plot(result['cluster_evolution'], label=name, linewidth=2)
    axes[0, 0].set_title('Cluster Count Evolution')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Number of Clusters')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Birth events
    for name, result in results.items():
        birth_times = result['birth_events']
        axes[0, 1].scatter(birth_times, [name]*len(birth_times), alpha=0.7, s=30)
    axes[0, 1].set_title('Birth Events Timeline')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Configuration')
    
    # 3. Tau evolution (adaptive only)
    for name, result in results.items():
        if 'Adaptive' in name:
            axes[1, 0].plot(result['tau_evolution'], label=name, linewidth=2)
    axes[1, 0].set_title('Adaptive Tau Birth Evolution')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Tau Birth')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Final statistics comparison
    names = list(results.keys())
    birth_rates = [results[name]['final_stats']['total_births']/n_points*100 for name in names]
    final_clusters = [results[name]['final_stats']['num_clusters'] for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = axes[1, 1].bar(x - width/2, birth_rates, width, label='Birth Rate (%)', alpha=0.7)
    bars2 = axes[1, 1].bar(x + width/2, final_clusters, width, label='Final Clusters', alpha=0.7)
    
    axes[1, 1].set_title('Final Statistics Comparison')
    axes[1, 1].set_xlabel('Configuration')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 1].legend()
    
    # 값 표시
    for bar, value in zip(bars1, birth_rates):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    
    for bar, value in zip(bars2, final_clusters):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{value}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('./birth_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def generate_final_recommendations():
    """최종 권장사항 생성"""
    print("\n" + "="*60)
    print("💡 FINAL RECOMMENDATIONS FOR BIRTH PROBLEM")
    print("="*60)
    
    recommendations = {
        "Immediate Fixes": [
            "Change tau_birth from -5.0 to -2.0 (more lenient threshold)",
            "Increase initial_variance from 1e-3 to 0.1 (larger initial clusters)",
            "Implement max_clusters limit (e.g., 15-20 clusters)",
            "Use more aggressive merging (tau_merge_kl = 0.3 instead of 0.1)"
        ],
        
        "Adaptive Solutions": [
            "Implement adaptive birth threshold based on recent birth rate",
            "Target birth rate of 10-15% (not 50%+)",
            "Adjust threshold every 20-50 assignments",
            "Use learning rate decay for cluster updates"
        ],
        
        "Code Changes": [
            "Replace MathematicalDPMM with AdaptiveBirthDPMM",
            "Add birth rate monitoring and logging",
            "Implement cluster count limits in main training loop",
            "Add early stopping if birth rate > 30%"
        ],
        
        "Hyperparameter Tuning": [
            "Run grid search on tau_birth: [-1.0, -1.5, -2.0, -2.5, -3.0]",
            "Test tau_merge_kl: [0.2, 0.3, 0.5, 0.8]",
            "Experiment with initial_variance: [0.05, 0.1, 0.2]",
            "Try different target_birth_rates: [0.08, 0.12, 0.15]"
        ]
    }
    
    for category, items in recommendations.items():
        print(f"\n🔸 {category}:")
        for i, item in enumerate(items, 1):
            print(f"  {i}. {item}")
    
    # 코드 예시
    print(f"\n📝 EXAMPLE CODE INTEGRATION:")
    print(f"""
# Replace in your main training script:
from improved_birth_control_dpmm import AdaptiveBirthDPMM

# Instead of:
# dpmm = MathematicalDPMM(latent_dim=32, tau_birth=-5.0, tau_merge_kl=0.1)

# Use:
dpmm = AdaptiveBirthDPMM(
    latent_dim=32,
    initial_tau_birth=-2.0,        # More lenient
    tau_merge_kl=0.3,              # More aggressive merging  
    initial_variance=0.1,          # Larger initial clusters
    adaptive_birth=True,           # Enable adaptation
    max_clusters=20,               # Limit cluster explosion
    target_birth_rate=0.12         # Target 12% birth rate
)

# Add monitoring in training loop:
if episode % 10 == 0:
    stats = dpmm.get_statistics()
    if stats['recent_birth_rate'] > 0.3:  # 30% threshold
        print(f"⚠️  High birth rate detected: {{stats['recent_birth_rate']:.1%}}")
        # Consider early intervention
""")

if __name__ == "__main__":
    print("🚀 Running Birth Control Analysis...")
    
    # 테스트 실행
    test_results = test_birth_control()
    comparison_results = compare_birth_rates()
    
    # 최종 권장사항
    generate_final_recommendations()
    
    print("\n✅ Analysis completed!")
    print("📊 Check generated plots: birth_control_test.png, birth_rate_comparison.png")
