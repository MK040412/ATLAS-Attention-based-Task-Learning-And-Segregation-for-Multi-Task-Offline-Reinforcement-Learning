#!/usr/bin/env python3
"""
RTX 4060 최적화 컴포넌트 테스트 스크립트
"""

import torch
import numpy as np
from config import CONFIG

def test_memory_usage():
    """메모리 사용량 테스트"""
    print("=== Memory Usage Test ===")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Current Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    else:
        print("CUDA not available")

def test_language_processor():
    """언어 처리기 테스트"""
    print("\n=== Language Processor Test ===")
    
    try:
        from language_processor import LanguageProcessor
        
        lang = LanguageProcessor(device=CONFIG.device)
        
        # 테스트 문장들
        test_instructions = [
            "Push the blue cube to the goal position",
            "Move the cube closer to the target",
            "Push the cube sideways to reach the goal"
        ]
        
        for instruction in test_instructions:
            emb = lang.encode(instruction)
            print(f"Instruction: '{instruction}'")
            print(f"Embedding shape: {emb.shape}")
            print(f"Embedding norm: {torch.norm(emb).item():.3f}")
            print()
        
        # 유사도 테스트
        sim = lang.compute_similarity(test_instructions[0], test_instructions[1])
        print(f"Similarity between first two instructions: {sim:.3f}")
        
        lang.cleanup()
        print("Language processor test passed!")
        
    except Exception as e:
        print(f"Language processor test failed: {e}")

def test_fusion_encoder():
    """융합 인코더 테스트"""
    print("\n=== Fusion Encoder Test ===")
    
    try:
        from memory_efficient_fusion import create_fusion_encoder
        
        # 테스트 파라미터
        instr_dim = CONFIG.instr_emb_dim
        state_dim = 35  # 예상 상태 차원
        latent_dim = CONFIG.latent_dim
        
        fusion = create_fusion_encoder(instr_dim, state_dim, latent_dim, CONFIG.hidden_dim)
        fusion = fusion.to(CONFIG.device)
        
        print(f"Fusion encoder created:")
        print(f"  Input dims: instr={instr_dim}, state={state_dim}")
        print(f"  Output dim: {latent_dim}")
        
        memory_info = fusion.get_memory_usage()
        print(f"  Memory usage: {memory_info['memory_mb']:.2f} MB")
        
        # 테스트 입력
        test_instr = torch.randn(instr_dim, device=CONFIG.device)
        test_state = torch.randn(state_dim, device=CONFIG.device)
        
        # 순전파 테스트
        with torch.no_grad():
            latent = fusion(test_instr, test_state)
            print(f"  Output shape: {latent.shape}")
            print(f"  Output range: [{latent.min().item():.3f}, {latent.max().item():.3f}]")
        
        print("Fusion encoder test passed!")
        
    except Exception as e:
        print(f"Fusion encoder test failed: {e}")

def test_dpmm_detector():
    """DPMM 탐지기 테스트"""
    print("\n=== DPMM Detector Test ===")
    
    try:
        from dpmm_ood_detector import DPMMOODDetector
        
        dpmm = DPMMOODDetector(latent_dim=CONFIG.latent_dim)
        
        # 테스트 데이터 생성
        test_data = []
        for i in range(50):
            # 3개의 클러스터를 시뮬레이션
            if i < 20:
                data = np.random.normal([1, 1], 0.5, CONFIG.latent_dim)
            elif i < 35:
                data = np.random.normal([-1, -1], 0.5, CONFIG.latent_dim)
            else:
                data = np.random.normal([0, 2], 0.5, CONFIG.latent_dim)
            
            test_data.append(data)
        
        # 클러스터 할당 테스트
        assignments = []
        for data in test_data:
            cluster_id = dpmm.assign(data)
            assignments.append(cluster_id)
        
        print(f"Cluster assignments: {set(assignments)}")
        
        # 클러스터 정보
        info = dpmm.get_cluster_info()
        print(f"Cluster info: {info}")
        
        # 병합 테스트
        dpmm.merge()
        final_info = dpmm.get_cluster_info()
        print(f"After merge: {final_info}")
        
        print("DPMM detector test passed!")
        
    except Exception as e:
        print(f"DPMM detector test failed: {e}")

def test_sac_expert():
    """SAC 전문가 테스트"""
    print("\n=== SAC Expert Test ===")
    
    try:
        from rtx4060_sac_expert import RTX4060SACExpert
        
        state_dim = 35
        action_dim = 8
        
        expert = RTX4060SACExpert(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=CONFIG.hidden_dim,
            device=CONFIG.device
        )
        
        print(f"SAC Expert created with {expert.count_parameters():,} parameters")
        
        # 행동 선택 테스트
        test_state = np.random.randn(state_dim).astype(np.float32)
        action = expert.select_action(test_state, deterministic=False)
        print(f"Action shape: {action.shape}")
        print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")
        
        # 경험 추가 테스트
        for _ in range(20):
            state = np.random.randn(state_dim).astype(np.float32)
            action = np.random.randn(action_dim).astype(np.float32)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim).astype(np.float32)
            done = np.random.choice([True, False])
            
            expert.replay_buffer.add(state, action, reward, next_state, done)
        
        print(f"Replay buffer size: {len(expert.replay_buffer)}")
        
        # 업데이트 테스트 (충분한 데이터가 있을 때)
        if len(expert.replay_buffer) >= 16:
            loss_info = expert.update(batch_size=16)
            if loss_info:
                print(f"Update successful: {loss_info}")
        
        expert.cleanup()
        print("SAC expert test passed!")
        
    except Exception as e:
        print(f"SAC expert test failed: {e}")

def test_gpu_memory():
    """GPU 메모리 테스트"""
    print("\n=== GPU Memory Test ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU memory test")
        return
    
    try:
        # 초기 메모리
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        print(f"Initial GPU memory: {initial_memory / 1024**2:.1f} MB")
        
        # 큰 텐서 생성
        test_tensors = []
        for i in range(5):
            tensor = torch.randn(1000, 1000, device=CONFIG.device)
            test_tensors.append(tensor)
            current_memory = torch.cuda.memory_allocated()
            print(f"After tensor {i+1}: {current_memory / 1024**2:.1f} MB")
        
        # 메모리 정리
        del test_tensors
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        print(f"After cleanup: {final_memory / 1024**2:.1f} MB")
        
        print("GPU memory test passed!")
        
    except Exception as e:
        print(f"GPU memory test failed: {e}")

def main():
    """메인 테스트 함수"""
    print("RTX 4060 Component Testing")
    print("=" * 50)
    
    test_memory_usage()
    test_language_processor()
    test_fusion_encoder()
    test_dpmm_detector()
    test_sac_expert()
    test_gpu_memory()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    
    if torch.cuda.is_available():
        print(f"Final GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

if __name__ == "__main__":
    main()