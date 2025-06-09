#!/usr/bin/env python3
"""
융합 모델만 단독 테스트
"""

import torch
import numpy as np
from config import CONFIG
from language_processor import LanguageProcessor
from memory_efficient_fusion_fixed import create_fusion_encoder
from dpmm_ood_detector import DPMMOODDetector

def test_fusion_pipeline():
    """전체 융합 파이프라인 테스트"""
    
    print("=== Testing Fusion Pipeline ===")
    
    # 파라미터
    state_dim = 66
    instr_dim = CONFIG.instr_emb_dim
    latent_dim = CONFIG.latent_dim
    hidden_dim = CONFIG.hidden_dim
    
    print(f"State dim: {state_dim}")
    print(f"Instruction dim: {instr_dim}")
    print(f"Latent dim: {latent_dim}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Device: {CONFIG.device}")
    
    # 컴포넌트 생성
    try:
        lang = LanguageProcessor(device=CONFIG.device)
        fusion = create_fusion_encoder(instr_dim, state_dim, latent_dim, hidden_dim).to(CONFIG.device)
        dpmm = DPMMOODDetector(latent_dim=latent_dim)
        
        print(f"✓ Components created")
        print(f"✓ Fusion parameters: {fusion.count_parameters():,}")
        
    except Exception as e:
        print(f"✗ Component creation failed: {e}")
        return
    
    # 지시사항 임베딩 테스트
    print("\n--- Instruction Embedding Test ---")
    test_instructions = [
        "Push the blue cube to the goal position",
        "Move the cube forward to reach the target",
        "Navigate to the cube and push it away",
        "Approach the object and move it to the destination"
    ]
    
    instruction_embeddings = []
    for i, instr in enumerate(test_instructions):
        try:
            emb = lang.encode(instr)
            instruction_embeddings.append(emb)
            print(f"✓ Instruction {i+1}: {emb.shape}, device: {emb.device}")
        except Exception as e:
            print(f"✗ Instruction {i+1} failed: {e}")
    
    # 융합 테스트
    print("\n--- Fusion Test ---")
    for i in range(len(instruction_embeddings)):
        try:
            instr_emb = instruction_embeddings[i]
            
            # 다양한 상태로 테스트
            for j in range(3):
                state = np.random.randn(state_dim).astype(np.float32)
                
                # 융합 실행
                with torch.no_grad():
                    z = fusion(instr_emb, state)
                    z_np = z.detach().cpu().numpy()
                
                print(f"✓ Fusion {i+1}-{j+1}: input_state={state.shape}, output_z={z.shape}")
                print(f"  Z stats: mean={z_np.mean():.3f}, std={z_np.std():.3f}")
                
                # DPMM 클러스터링
                cluster_id = dpmm.assign(z_np)
                print(f"  Cluster: {cluster_id}")
                
        except Exception as e:
            print(f"✗ Fusion test {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 클러스터 정보
    print("\n--- Cluster Information ---")
    try:
        cluster_info = dpmm.get_cluster_info()
        print(f"✓ Total clusters: {cluster_info['num_clusters']}")
        print(f"✓ Cluster counts: {cluster_info['cluster_counts']}")
        
        # 병합 테스트
        dpmm.merge()
        cluster_info_after = dpmm.get_cluster_info()
        print(f"✓ After merge: {cluster_info_after['num_clusters']} clusters")
        
    except Exception as e:
        print(f"✗ Cluster test failed: {e}")
    
    # 저장/로드 테스트
    print("\n--- Save/Load Test ---")
    try:
        save_path = "./test_fusion.pth"
        fusion.save_checkpoint(save_path)
        print(f"✓ Fusion model saved")
        
        # 새로운 모델 생성 후 로드
        fusion2 = create_fusion_encoder(instr_dim, state_dim, latent_dim, hidden_dim).to(CONFIG.device)
        fusion2.load_checkpoint(save_path)
        print(f"✓ Fusion model loaded")
        
        # 동일한 입력에 대해 동일한 출력 확인
        test_instr = instruction_embeddings[0]
        test_state = np.random.randn(state_dim).astype(np.float32)
        
        with torch.no_grad():
            z1 = fusion(test_instr, test_state)
            z2 = fusion2(test_instr, test_state)
        
        diff = torch.abs(z1 - z2).mean().item()
        print(f"✓ Output difference after load: {diff:.8f}")
        
    except Exception as e:
        print(f"✗ Save/Load test failed: {e}")
    
    # 정리
    try:
        lang.cleanup()
        print("✓ Cleanup completed")
    except:
        pass
    
    print("\n✓ All fusion tests completed!")

if __name__ == "__main__":
    test_fusion_pipeline()