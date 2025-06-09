#!/usr/bin/env python3
"""
텐서 변환 문제 디버깅 스크립트
"""

import torch
import numpy as np
from config import CONFIG

def debug_tensor_conversion():
    """텐서 변환 문제 디버깅"""
    
    print("=== Debugging Tensor Conversion Issues ===")
    
    # 테스트 데이터 생성
    test_state_np = np.random.randn(35).astype(np.float32)
    test_instr_np = np.random.randn(384).astype(np.float32)
    
    print(f"Test state shape: {test_state_np.shape}, dtype: {test_state_np.dtype}")
    print(f"Test instruction shape: {test_instr_np.shape}, dtype: {test_instr_np.dtype}")
    
    # 다양한 변환 방법 테스트
    print("\n1. Testing torch.FloatTensor conversion:")
    try:
        state_tensor1 = torch.FloatTensor(test_state_np).to(CONFIG.device)
        print(f"  ✓ Success: {state_tensor1.shape}, device: {state_tensor1.device}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    print("\n2. Testing torch.from_numpy conversion:")
    try:
        state_tensor2 = torch.from_numpy(test_state_np.copy()).float().to(CONFIG.device)
        print(f"  ✓ Success: {state_tensor2.shape}, device: {state_tensor2.device}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    print("\n3. Testing torch.tensor conversion:")
    try:
        state_tensor3 = torch.tensor(test_state_np, dtype=torch.float32, device=CONFIG.device)
        print(f"  ✓ Success: {state_tensor3.shape}, device: {state_tensor3.device}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # 융합 모델 테스트
    print("\n4. Testing fusion model:")
    try:
        from memory_efficient_fusion import create_fusion_encoder
        
        fusion = create_fusion_encoder(
            CONFIG.instr_emb_dim,
            35,
            CONFIG.latent_dim,
            CONFIG.hidden_dim
        ).to(CONFIG.device)
        
        # 텐서 변환
        instr_tensor = torch.tensor(test_instr_np, dtype=torch.float32, device=CONFIG.device)
        state_tensor = torch.tensor(test_state_np, dtype=torch.float32, device=CONFIG.device)
        
        print(f"  Input tensors created successfully")
        print(f"  Instruction tensor: {instr_tensor.shape}, device: {instr_tensor.device}")
        print(f"  State tensor: {state_tensor.shape}, device: {state_tensor.device}")
        
        # 융합 실행
        with torch.no_grad():
            result = fusion(instr_tensor, state_tensor)
            print(f"  ✓ Fusion successful: {result.shape}, device: {result.device}")
        
    except Exception as e:
        print(f"  ✗ Fusion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_tensor_conversion()