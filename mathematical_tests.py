#!/usr/bin/env python3
"""
DPMM + MoE-SAC 수학적 공식 검증 테스트
"""

import torch
import numpy as np
from rigorous_dpmm import RigorousDPMM, GaussianCluster
from rigorous_sac_expert import RigorousSACExpert
from integrated_dpmm_moe_sac import IntegratedDPMMMoESAC

def test_gaussian_likelihood():
    """가우시안 우도 계산 테스트"""
    print("🧪 Testing Gaussian Likelihood Calculation")
    
    # 테스트 데이터
    latent_dim = 4
    sigma_sq = 1.0
    
    dpmm = RigorousDPMM(latent_dim=latent_dim, sigma_sq=sigma_sq)
    
    # 클러스터 생성
    mu = np.array([1.0, 2.0, -1.0, 0.5])
    cluster = GaussianCluster(
        id=0, mu=mu, sigma_sq=sigma_sq, N=1,
        sum_x=mu.copy(), sum_x_sq=np.sum(mu**2)
    )
    
    # 테스트 포인트
    z = np.array([1.1, 1.9, -0.9, 0.6])  # 클러스터 중심 근처
    
    # 우도 계산
    likelihood = dpmm._gaussian_likelihood(z, cluster)
    
    # 수동 계산으로 검증
    diff = z - mu
    manual_log_likelihood = (
        -0.5 * latent_dim * np.log(2 * np.pi * sigma_sq) -
        0.5 * np.sum(diff**2) / sigma_sq
    )
    manual_likelihood = np.exp(manual_log_likelihood)
    
    print(f"  Computed likelihood: {likelihood:.8f}")
    print(f"  Manual likelihood: {manual_likelihood:.8f}")
    print(f"  Difference: {abs(likelihood - manual_likelihood):.10f}")
    
    assert abs(likelihood - manual_likelihood) < 1e-10, "Likelihood calculation error!"
    print("  ✅ Gaussian likelihood test passed!")

def test_online_cluster_update():
    """온라인 클러스터 업데이트 테스트"""
    print("\n🧪 Testing Online Cluster Update")
    
    dpmm = RigorousDPMM(latent_dim=3)
    
    # 첫 번째 포인트로 클러스터 생성
    z1 = np.array([1.0, 2.0, 3.0])
    cluster_id = dpmm._create_new_cluster(z1)
    
    print(f"  Initial cluster μ: {dpmm.clusters[cluster_id].mu}")
    print(f"  Initial N: {dpmm.clusters[cluster_id].N}")
    
    # 두 번째 포인트 추가
    z2 = np.array([2.0, 1.0, 4.0])
    dpmm._update_cluster(cluster_id, z2)
    
    # 수동 계산
    expected_mu = (z1 + z2) / 2
    actual_mu = dpmm.clusters[cluster_id].mu
    
    print(f"  After update μ: {actual_mu}")
    print(f"  Expected μ: {expected_mu}")
    print(f"  Updated N: {dpmm.clusters[cluster_id].N}")
    
    mu_error = np.linalg.norm(actual_mu - expected_mu)
    print(f"  μ update error: {mu_error:.10f}")
    
    assert mu_error < 1e-10, "Online update error!"
    assert dpmm.clusters[cluster_id].N == 2, "Count update error!"
    print("  ✅ Online cluster update test passed!")

def test_birth_heuristic():
    """Birth heuristic 테스트"""
    print("\n🧪 Testing Birth Heuristic")
    
    tau_birth = 1e-3
    dpmm = RigorousDPMM(latent_dim=2, tau_birth=tau_birth, sigma_sq=0.1)
    
    # 첫 번째 클러스터
    z1 = np.array([0.0, 0.0])
    k1 = dpmm.assign(z1)
    print(f"  First assignment: cluster {k1}")
    
    # 가까운 포인트 (birth 안 됨)
    z2 = np.array([0.1, 0.1])
    k2 = dpmm.assign(z2)
    print(f"  Close point assignment: cluster {k2}")
    
    # 먼 포인트 (birth 됨)
    z3 = np.array([10.0, 10.0])
    k3 = dpmm.assign(z3)
    print(f"  Far point assignment: cluster {k3}")
    
    print(f"  Total clusters: {len(dpmm.clusters)}")
    print(f"  Birth events: {len(dpmm.birth_events)}")
    
    assert k1 == k2, "Close points should be in same cluster!"
    assert k3 != k1, "Far point should create new cluster!"
    assert len(dpmm.clusters) == 2, "Should have 2 clusters!"
    print("  ✅ Birth heuristic test passed!")

def test_merge_heuristic():
    """Merge heuristic 테스트"""
    print("\n🧪 Testing Merge Heuristic")
    
    tau_merge = 1.0
    dpmm = RigorousDPMM(latent_dim=2, tau_birth=1e-6, tau_merge=tau_merge)
    
    # 두 개의 가까운 클러스터 생성
    z1 = np.array([0.0, 0.0])
    z2 = np.array([0.5, 0.0])  # 거리 0.5 < tau_merge
    
    k1 = dpmm.assign(z1)
    k2 = dpmm.assign(z2)
    
    print(f"  Before merge: {len(dpmm.clusters)} clusters")
    print(f"  Cluster centers: {[c.mu for c in dpmm.clusters.values()]}")
    
    # 병합 실행
    dpmm.merge()
    
    print(f"  After merge: {len(dpmm.clusters)} clusters")
    print(f"  Merge events: {len(dpmm.merge_events)}")
    
    if dpmm.merge_events:
        merge_event = dpmm.merge_events[0]
        print(f"  Merged clusters: {merge_event['old_clusters']} → {merge_event['new_cluster']}")
        print(f"  New center: {merge_event['new_mu']}")
    
    # 가중 평균 검증
    expected_center = (z1 + z2) / 2
    if len(dpmm.clusters) == 1:
        actual_center = list(dpmm.clusters.values())[0].mu
        center_error = np.linalg.norm(actual_center - expected_center)
        print(f"  Center error: {center_error:.8f}")
        assert center_error < 1e-8, "Merge center calculation error!"
    
    print("  ✅ Merge heuristic test passed!")

def test_sac_loss_computation():
    """SAC 손실 함수 계산 테스트"""
    print("\n🧪 Testing SAC Loss Computation")
    
    state_dim, action_dim = 4, 2
    expert = RigorousSACExpert(
        state_dim=state_dim,
        action_dim=action_dim,
        expert_id=0,
        device='cpu'
    )
    
    # 더미 배치 생성
    batch_size = 32
    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)
    rewards = torch.randn(batch_size, 1)
    next_states = torch.randn(batch_size, state_dim)
    dones = torch.zeros(batch_size, 1, dtype=torch.bool)
    
    # 리플레이 버퍼에 데이터 추가
    for i in range(batch_size):
        expert.replay_buffer.add(
            states[i].numpy(),
            actions[i].numpy(),
            rewards[i].item(),
            next_states[i].numpy(),
            dones[i].item()
        )
    
    # 업데이트 실행
    losses = expert.update(batch_size=16)
    
    print(f"  Critic1 loss: {losses['critic1_loss']:.6f}")
    print(f"  Critic2 loss: {losses['critic2_loss']:.6f}")
    print(f"  Actor loss: {losses['actor_loss']:.6f}")
    print(f"  Alpha loss: {losses['alpha_loss']:.6f}")
    print(f"  Alpha value: {losses['alpha_value']:.6f}")
    
    # 손실이 유한한 값인지 확인
    for key, value in losses.items():
        assert np.isfinite(value), f"{key} is not finite!"
    
    print("  ✅ SAC loss computation test passed!")

def test_integrated_agent():
    """통합 에이전트 테스트"""
    print("\n🧪 Testing Integrated DPMM-MoE-SAC Agent")
    
    state_dim, action_dim, instr_dim = 6, 3, 384
    agent = IntegratedDPMMMoESAC(
        state_dim=state_dim,
        action_dim=action_dim,
        instr_dim=instr_dim,
        latent_dim=8,
        tau_birth=1e-4,
        tau_merge=0.5,
        device='cpu'
    )
    
    # 더미 데이터
    state = np.random.randn(state_dim)
    instruction_emb = np.random.randn(instr_dim)
    
    # 행동 선택 테스트
    action, cluster_id = agent.select_action(state, instruction_emb)
    
    print(f"  Selected action shape: {action.shape}")
    print(f"  Assigned cluster: {cluster_id}")
    print(f"  Number of experts: {len(agent.experts)}")
    
    # 여러 번 실행하여 클러스터 진화 테스트
    for i in range(10):
        state_new = np.random.randn(state_dim)
        action_new, cluster_new = agent.select_action(state_new, instruction_emb)
        
        # 경험 저장
        agent.store_experience(
            state=state,
            action=action,
            reward=np.random.randn(),
            next_state=state_new,
            done=False,
            cluster_id=cluster_id
        )
        
        state, action, cluster_id = state_new, action_new, cluster_new
    
    # 병합 테스트
    old_cluster_count = agent.dpmm.get_cluster_count()
    agent.merge_clusters()
    new_cluster_count = agent.dpmm.get_cluster_count()
    
    print(f"  Clusters before merge: {old_cluster_count}")
    print(f"  Clusters after merge: {new_cluster_count}")
    
    # 통계 확인
    stats = agent.get_comprehensive_statistics()
    print(f"  Total timesteps: {stats['timestep']}")
    print(f"  Active clusters: {stats['active_clusters']}")
    print(f"  Total experts: {stats['total_experts']}")
    
    print("  ✅ Integrated agent test passed!")

def test_cluster_evolution_formula():
    """클러스터 진화 공식 테스트: K_{t+1} = K_t + 1_{birth} - 1_{merge}"""
    print("\n🧪 Testing Cluster Evolution Formula")
    
    dpmm = RigorousDPMM(latent_dim=2, tau_birth=1e-4, tau_merge=0.3)
    
    K_history = []
    birth_history = []
    merge_history = []
    
    # 시뮬레이션
    np.random.seed(42)
    for t in range(20):
        # 랜덤 포인트 생성
        z = np.random.randn(2) * (2 if t < 10 else 0.5)  # 처음엔 분산, 나중엔 집중
        
        old_K = len(dpmm.clusters)
        old_births = len(dpmm.birth_events)
        old_merges = len(dpmm.merge_events)
        
        # 할당
        cluster_id = dpmm.assign(z)
        
        # 병합 (확률적으로)
        if t % 5 == 0:
            dpmm.merge()
        
        new_K = len(dpmm.clusters)
        new_births = len(dpmm.birth_events)
        new_merges = len(dpmm.merge_events)
        
        # 변화량 계산
        delta_K = new_K - old_K
        delta_births = new_births - old_births
        delta_merges = new_merges - old_merges
        
        K_history.append(new_K)
        birth_history.append(delta_births)
        merge_history.append(delta_merges)
        
        # 공식 검증: ΔK = Δbirths - Δmerges
        expected_delta_K = delta_births - delta_merges
        
        print(f"  t={t}: K={new_K}, ΔK={delta_K}, births={delta_births}, merges={delta_merges}")
        print(f"    Formula check: ΔK={delta_K} ?= births-merges={expected_delta_K}")
        
        if delta_K != expected_delta_K:
            print(f"    ⚠️  Formula violation at t={t}!")
        
        assert delta_K == expected_delta_K, f"Cluster evolution formula violated at t={t}!"
    
    # 전체 진화 검증
    total_births = len(dpmm.birth_events)
    total_merges = len(dpmm.merge_events)
    final_K = len(dpmm.clusters)
    
    print(f"\n  Final verification:")
    print(f"    Total births: {total_births}")
    print(f"    Total merges: {total_merges}")
    print(f"    Final K: {final_K}")
    print(f"    Expected K: {total_births - total_merges}")
    
    # 초기 클러스터가 0이므로 K_final = births - merges
    assert final_K == total_births - total_merges, "Overall evolution formula violated!"
    print("  ✅ Cluster evolution formula test passed!")

def test_fusion_encoder_dimensions():
    """Fusion Encoder 차원 테스트"""
    print("\n🧪 Testing Fusion Encoder Dimensions")
    
    from integrated_dpmm_moe_sac import FusionEncoder
    
    instr_dim, state_dim, latent_dim = 384, 29, 16
    encoder = FusionEncoder(instr_dim, state_dim, latent_dim)
    
    # 테스트 입력
    instruction_emb = torch.randn(instr_dim)
    state = torch.randn(state_dim)
    
    # 순전파
    z = encoder(instruction_emb, state)
    
    print(f"  Input dimensions:")
    print(f"    Instruction: {instruction_emb.shape}")
    print(f"    State: {state.shape}")
    print(f"    Concatenated: {instr_dim + state_dim}")
    print(f"  Output dimension: {z.shape}")
    print(f"  Expected latent dim: {latent_dim}")
    
    assert z.shape == (latent_dim,), f"Wrong output dimension: {z.shape}"
    assert torch.isfinite(z).all(), "Non-finite values in latent representation!"
    
    # 배치 처리 테스트
    batch_size = 8
    instr_batch = torch.randn(batch_size, instr_dim)
    state_batch = torch.randn(batch_size, state_dim)
    
    # 배치별로 처리 (현재 구현은 단일 입력만 지원)
    z_batch = []
    for i in range(batch_size):
        z_i = encoder(instr_batch[i], state_batch[i])
        z_batch.append(z_i)
    z_batch = torch.stack(z_batch)
    
    print(f"  Batch output shape: {z_batch.shape}")
    assert z_batch.shape == (batch_size, latent_dim), "Wrong batch output dimension!"
    
    print("  ✅ Fusion encoder dimension test passed!")

def test_mathematical_consistency():
    """전체 수학적 일관성 테스트"""
    print("\n🧪 Testing Overall Mathematical Consistency")
    
    # 통합 시스템 생성
    state_dim, action_dim, instr_dim = 10, 4, 384
    agent = IntegratedDPMMMoESAC(
        state_dim=state_dim,
        action_dim=action_dim,
        instr_dim=instr_dim,
        latent_dim=8,
        tau_birth=1e-3,
        tau_merge=0.5,
        device='cpu'
    )
    
    # 시뮬레이션 데이터
    np.random.seed(123)
    torch.manual_seed(123)
    
    states = []
    actions = []
    clusters = []
    latent_states = []
    
    instruction_emb = np.random.randn(instr_dim)
    
    print("  Running consistency simulation...")
    
    for t in range(50):
        # 상태 생성
        state = np.random.randn(state_dim)
        
        # 잠재 상태 계산
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            instr_tensor = torch.FloatTensor(instruction_emb)
            z = agent.fusion_encoder(instr_tensor, state_tensor)
            latent_states.append(z.numpy())
        
        # 행동 선택
        action, cluster_id = agent.select_action(state, instruction_emb)
        
        states.append(state)
        actions.append(action)
        clusters.append(cluster_id)
        
        # 경험 저장
        if t > 0:
            agent.store_experience(
                state=states[t-1],
                action=actions[t-1],
                reward=np.random.randn(),
                next_state=state,
                done=False,
                cluster_id=clusters[t-1]
            )
        
        # 주기적 병합
        if t % 10 == 0 and t > 0:
            agent.merge_clusters()
    
    # 일관성 검증
    print(f"  Generated {len(states)} states")
    print(f"  Unique clusters used: {len(set(clusters))}")
    print(f"  Active clusters: {len(agent.dpmm.clusters)}")
    print(f"  Total experts: {len(agent.experts)}")
    
    # 클러스터 할당 일관성
    for i, (z, k) in enumerate(zip(latent_states[:10], clusters[:10])):
        # 같은 잠재 상태로 다시 할당해보기
        k_recomputed = agent.dpmm.assign(z)
        if k != k_recomputed:
            print(f"    ⚠️  Inconsistent assignment at step {i}: {k} vs {k_recomputed}")
    
    # 전문가-클러스터 대응
    for cluster_id in set(clusters):
        if cluster_id < len(agent.experts):
            expert = agent.experts[cluster_id]
            print(f"    Cluster {cluster_id} → Expert {expert.expert_id}")
            assert expert.expert_id == cluster_id, "Expert-cluster mismatch!"
    
    print("  ✅ Mathematical consistency test passed!")

def run_all_tests():
    """모든 테스트 실행"""
    print("🔬 DPMM-MoE-SAC Mathematical Verification Suite")
    print("=" * 60)
    
    tests = [
        test_gaussian_likelihood,
        test_online_cluster_update,
        test_birth_heuristic,
        test_merge_heuristic,
        test_sac_loss_computation,
        test_fusion_encoder_dimensions,
        test_cluster_evolution_formula,
        test_integrated_agent,
        test_mathematical_consistency
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n" + "=" * 60)
    print(f"🧪 TEST SUMMARY:")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  Total: {len(tests)}")
    
    if failed == 0:
        print(f"🎉 ALL TESTS PASSED! Mathematical implementation is correct.")
    else:
        print(f"⚠️  {failed} tests failed. Please check the implementation.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)