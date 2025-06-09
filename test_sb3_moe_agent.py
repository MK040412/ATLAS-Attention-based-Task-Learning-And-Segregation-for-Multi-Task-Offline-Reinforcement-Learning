#!/usr/bin/env python3
"""
SB3 MoE 에이전트 테스트
"""

import argparse
import os
import yaml
import numpy as np
import torch
import time
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Test SB3 MoE Agent")
    parser.add_argument("--model_dir", type=str, default="./sb3_moe_outputs/final_sb3_moe_model")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--test_instructions", type=str, default="test_instructions.yaml")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    parser.add_argument("--detailed", action="store_true", help="Detailed output")  # 변경: verbose -> detailed
    parser.add_argument("--save_results", type=str, default="./test_results")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(vars(args))
    sim = app.app

    try:
        from fixed_simple_gym_wrapper import SimpleAntPushWrapper
        from sb3_moe_agent import SB3MoEAgent
        from language_processor import LanguageProcessor
        from config import CONFIG

        print(f"🧪 SB3 MoE Agent Testing")
        print(f"  Model directory: {args.model_dir}")
        print(f"  Episodes per test: {args.episodes}")
        print(f"  Max steps per episode: {args.max_steps}")
        print(f"  Deterministic actions: {args.deterministic}")
        print(f"  Detailed output: {args.detailed}")  # 변경
        
        # 결과 저장 디렉토리 생성
        if args.save_results:
            os.makedirs(args.save_results, exist_ok=True)
        
        # 테스트 지시사항 로드
        test_instructions = []
        if os.path.exists(args.test_instructions):
            test_instructions = yaml.safe_load(open(args.test_instructions))
            print(f"  Loaded {len(test_instructions)} test instructions")
        else:
            # 기본 테스트 지시사항
            test_instructions = [
                {
                    'name': 'basic_push_test',
                    'instruction': 'Push the blue cube to the goal position',
                    'env_cfg': {
                        'cube_position': [2.0, 0.0, 0.1],
                        'goal_position': [3.0, 0.0, 0.1]
                    }
                },
                {
                    'name': 'left_push_test',
                    'instruction': 'Push the cube to the left side',
                    'env_cfg': {
                        'cube_position': [2.0, 0.0, 0.1],
                        'goal_position': [2.0, 2.0, 0.1]
                    }
                },
                {
                    'name': 'ood_test_1',
                    'instruction': 'Move the object to the target location',  # OOD: 다른 표현
                    'env_cfg': {
                        'cube_position': [1.5, 0.5, 0.1],
                        'goal_position': [3.5, 1.5, 0.1]
                    }
                },
                {
                    'name': 'ood_test_2',
                    'instruction': 'Shove the box forward',  # OOD: 새로운 동사
                    'env_cfg': {
                        'cube_position': [1.0, 0.0, 0.1],
                        'goal_position': [4.0, 0.0, 0.1]
                    }
                }
            ]
            print(f"  Using {len(test_instructions)} default test instructions")

        # 언어 처리기 초기화
        lang_processor = LanguageProcessor(device=CONFIG.device)
        
        # 첫 번째 환경으로 에이전트 초기화
        first_env = SimpleAntPushWrapper(custom_cfg=test_instructions[0]['env_cfg'])
        first_instr_emb = lang_processor.encode(test_instructions[0]['instruction'])
        instr_dim = first_instr_emb.shape[-1]
        
        # MoE 에이전트 생성 (일단 빈 상태)
        agent = SB3MoEAgent(
            env=first_env,
            instr_dim=instr_dim,
            latent_dim=32,  # 로드시 덮어씌워짐
            cluster_threshold=2.0  # 로드시 덮어씌워짐
        )
        
        # 모델 로드
        if os.path.exists(args.model_dir):
            print(f"📂 Loading model from {args.model_dir}")
            agent.load(args.model_dir)
            print(f"  ✅ Model loaded successfully")
            agent_info = agent.get_info()
            print(f"  📊 Loaded model info:")
            print(f"    Clusters: {agent_info['dpmm']['num_clusters']}")
            print(f"    Experts: {agent_info['num_experts']}")
            print(f"    Total buffer size: {agent_info['total_buffer_size']}")
        else:
            print(f"❌ Model directory not found: {args.model_dir}")
            return

        # 테스트 결과 수집
        all_test_results = {
            'instruction_results': {},
            'cluster_usage': {},
            'performance_summary': {},
            'ood_analysis': {}
        }

        # 각 테스트 지시사항별 평가
        for test_idx, test_case in enumerate(test_instructions):
            print(f"\n{'='*60}")
            print(f"🧪 Test {test_idx + 1}/{len(test_instructions)}: {test_case['name']}")
            print(f"📝 Instruction: {test_case['instruction']}")
            print(f"{'='*60}")
            
            # 환경 설정
            if test_idx > 0:
                first_env.close()
                first_env = SimpleAntPushWrapper(custom_cfg=test_case['env_cfg'])
                agent.env = first_env
            
            # 지시사항 임베딩
            instr_emb = lang_processor.encode(test_case['instruction'])
            
            # 테스트 통계
            test_rewards = []
            test_success_rates = []
            test_step_counts = []
            cluster_assignments = []
            episode_details = []
            
            # 에피소드별 테스트
            for episode in range(args.episodes):
                if args.detailed:  # 변경: verbose -> detailed
                    print(f"\n--- Episode {episode + 1}/{args.episodes} ---")
                
                # 환경 리셋
                state, _ = first_env.reset()
                episode_reward = 0
                step_count = 0
                episode_clusters = []
                
                episode_start_time = time.time()
                
                while step_count < args.max_steps:
                    # 행동 선택
                    action, cluster_id = agent.select_action(
                        state, 
                        instr_emb.squeeze(0).detach().cpu().numpy(),
                        deterministic=args.deterministic
                    )
                    
                    episode_clusters.append(cluster_id)
                    
                    if step_count == 0 and args.detailed:  # 변경
                        print(f"  Assigned to cluster/expert: {cluster_id}")
                    
                    # 환경 스텝
                    next_state, reward, done, truncated, info = first_env.step(action)
                    episode_reward += reward
                    
                    state = next_state
                    step_count += 1
                    
                    if done or truncated:
                        break
                
                episode_time = time.time() - episode_start_time
                
                # 성공 여부 판정 (박스가 목표에 충분히 가까운지)
                cube_pos = first_env.get_cube_position()
                goal_pos = first_env.get_goal_position()
                distance_to_goal = np.linalg.norm(cube_pos - goal_pos)
                success = distance_to_goal < 0.5  # 50cm 이내면 성공
                
                # 통계 기록
                test_rewards.append(episode_reward)
                test_success_rates.append(1.0 if success else 0.0)
                test_step_counts.append(step_count)
                cluster_assignments.extend(episode_clusters)
                
                episode_details.append({
                    'episode': episode + 1,
                    'reward': episode_reward,
                    'success': success,
                    'steps': step_count,
                    'time': episode_time,
                    'distance_to_goal': distance_to_goal,
                    'clusters_used': list(set(episode_clusters)),
                    'primary_cluster': max(set(episode_clusters), key=episode_clusters.count)
                })
                
                if args.detailed:  # 변경
                    print(f"  📊 Episode {episode + 1} Result:")
                    print(f"    Reward: {episode_reward:.2f}")
                    print(f"    Success: {'✅' if success else '❌'}")
                    print(f"    Steps: {step_count}")
                    print(f"    Distance to goal: {distance_to_goal:.3f}m")
                    print(f"    Time: {episode_time:.2f}s")
                    print(f"    Primary cluster: {max(set(episode_clusters), key=episode_clusters.count)}")
            
            # 테스트 케이스 결과 요약
            avg_reward = np.mean(test_rewards)
            success_rate = np.mean(test_success_rates)
            avg_steps = np.mean(test_step_counts)
            cluster_usage = {cluster: cluster_assignments.count(cluster) 
                           for cluster in set(cluster_assignments)}
            
            test_result = {
                'instruction': test_case['instruction'],
                'episodes': args.episodes,
                'avg_reward': avg_reward,
                'success_rate': success_rate,
                'avg_steps': avg_steps,
                'best_reward': max(test_rewards),
                'worst_reward': min(test_rewards),
                'cluster_usage': cluster_usage,
                'episode_details': episode_details
            }
            
            all_test_results['instruction_results'][test_case['name']] = test_result
            
            print(f"\n🏁 Test '{test_case['name']}' Results:")
            print(f"  📈 Average reward: {avg_reward:.2f}")
            print(f"  🎯 Success rate: {success_rate*100:.1f}%")
            print(f"  👟 Average steps: {avg_steps:.1f}")
            print(f"  ⚡ Best reward: {max(test_rewards):.2f}")
            print(f"  🎭 Clusters used: {sorted(cluster_usage.keys())}")
            if cluster_usage:
                print(f"  📊 Primary cluster: {max(cluster_usage, key=cluster_usage.get)} ({max(cluster_usage.values())} steps)")

        # 전체 테스트 요약
        print(f"\n{'='*60}")
        print(f"📊 OVERALL TEST SUMMARY")
        print(f"{'='*60}")
        
        all_rewards = []
        all_success_rates = []
        all_cluster_usage = {}
        ood_results = {}
        
        for test_name, result in all_test_results['instruction_results'].items():
            all_rewards.append(result['avg_reward'])
            all_success_rates.append(result['success_rate'])
            
            # 클러스터 사용량 집계
            for cluster, count in result['cluster_usage'].items():
                all_cluster_usage[cluster] = all_cluster_usage.get(cluster, 0) + count
            
            # OOD 분석 (test name에 'ood'가 포함된 경우)
            if 'ood' in test_name.lower():
                ood_results[test_name] = result
        
        overall_avg_reward = np.mean(all_rewards)
        overall_success_rate = np.mean(all_success_rates)
        
        print(f"🎯 Overall Performance:")
        print(f"  Average reward across all tests: {overall_avg_reward:.2f}")
        print(f"  Average success rate: {overall_success_rate*100:.1f}%")
        print(f"  Total episodes tested: {len(test_instructions) * args.episodes}")
        
        print(f"\n🎭 Cluster Usage Analysis:")
        if all_cluster_usage:
            sorted_clusters = sorted(all_cluster_usage.items(), key=lambda x: x[1], reverse=True)
            for cluster, count in sorted_clusters:
                percentage = (count / sum(all_cluster_usage.values())) * 100
                print(f"  Cluster {cluster}: {count} steps ({percentage:.1f}%)")
        else:
            print("  No cluster usage data available")
        
        if ood_results:
            print(f"\n🔮 OOD (Out-of-Distribution) Analysis:")
            ood_avg_reward = np.mean([r['avg_reward'] for r in ood_results.values()])
            ood_avg_success = np.mean([r['success_rate'] for r in ood_results.values()])
            in_dist_results = {k: v for k, v in all_test_results['instruction_results'].items() 
                             if 'ood' not in k.lower()}
            if in_dist_results:
                id_avg_reward = np.mean([r['avg_reward'] for r in in_dist_results.values()])
                id_avg_success = np.mean([r['success_rate'] for r in in_dist_results.values()])
                
                print(f"  OOD average reward: {ood_avg_reward:.2f}")
                print(f"  OOD average success rate: {ood_avg_success*100:.1f}%")
                print(f"  In-distribution average reward: {id_avg_reward:.2f}")
                print(f"  In-distribution average success rate: {id_avg_success*100:.1f}%")
                print(f"  Performance retention: {(ood_avg_reward/id_avg_reward*100):.1f}% reward, {(ood_avg_success/id_avg_success*100):.1f}% success")
        
        # 전문가별 성능 분석
        print(f"\n🧠 Expert Specialization Analysis:")
        expert_performance = {}
        for test_name, result in all_test_results['instruction_results'].items():
            for detail in result['episode_details']:
                if 'primary_cluster' in detail:
                    cluster = detail['primary_cluster']
                    if cluster not in expert_performance:
                        expert_performance[cluster] = {'rewards': [], 'successes': [], 'instructions': set()}
                    expert_performance[cluster]['rewards'].append(detail['reward'])
                    expert_performance[cluster]['successes'].append(detail['success'])
                    expert_performance[cluster]['instructions'].add(result['instruction'])
        
        for cluster in sorted(expert_performance.keys()):
            perf = expert_performance[cluster]
            avg_reward = np.mean(perf['rewards'])
            success_rate = np.mean(perf['successes'])
            print(f"  Expert {cluster}:")
            print(f"    Average reward: {avg_reward:.2f}")
            print(f"    Success rate: {success_rate*100:.1f}%")
            print(f"    Episodes handled: {len(perf['rewards'])}")
            print(f"    Instructions: {len(perf['instructions'])}")
        
        # 결과 저장
        if args.save_results:
            results_file = os.path.join(args.save_results, "test_results.yaml")
            with open(results_file, 'w') as f:
                # numpy 타입을 Python 타입으로 변환
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {key: convert_numpy(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(item) for item in obj]
                    else:
                        return obj
                
                yaml.dump(convert_numpy(all_test_results), f, default_flow_style=False)
            
            print(f"\n💾 Test results saved to: {results_file}")
        
        # 최종 평가
        print(f"\n🏆 Final Assessment:")
        if overall_success_rate >= 0.7:
            print(f"  🌟 EXCELLENT: Success rate > 70%")
        elif overall_success_rate >= 0.5:
            print(f"  ✅ GOOD: Success rate > 50%")
        elif overall_success_rate >= 0.3:
            print(f"  ⚠️ FAIR: Success rate > 30%")
        else:
            print(f"  ❌ POOR: Success rate < 30%")
        
        if all_cluster_usage and len(all_cluster_usage) > 1:
            print(f"  🎭 Multi-expert system active: {len(all_cluster_usage)} clusters used")
        else:
            print(f"  ⚠️ Single cluster dominance detected")
        
        if ood_results and in_dist_results and (ood_avg_reward/id_avg_reward) > 0.7:
            print(f"  🔮 Good OOD generalization: {(ood_avg_reward/id_avg_reward*100):.1f}% performance retention")
        elif ood_results and in_dist_results:
            print(f"  🔮 Limited OOD generalization: {(ood_avg_reward/id_avg_reward*100):.1f}% performance retention")
        
        first_env.close()
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        sim.close()

if __name__ == "__main__":
    main()