#!/usr/bin/env python3
"""
매우 간단한 테스트 스크립트
"""

import torch
import numpy as np
from rtx4060_sac_expert_fixed import RTX4060SACExpert

def test_sac_expert():
    """SAC 전문가 단독 테스트"""
    
    print("=== Testing SAC Expert ===")
    
    # 테스트 파라미터
    state_dim = 66
    action_dim = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # 전문가 생성
    try:
        expert = RTX4060SACExpert(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,
            device=device
        )
        print(f"✓ Expert created: {expert.count_parameters():,} parameters")
    except Exception as e:
        print(f"✗ Expert creation failed: {e}")
        return
    
    # 액션 선택 테스트
    print("\n--- Action Selection Test ---")
    for i in range(5):
        try:
            # 다양한 형태의 상태로 테스트
            if i == 0:
                state = np.random.randn(state_dim).astype(np.float32)
                print(f"Test {i+1}: numpy array")
            elif i == 1:
                state = torch.randn(state_dim, dtype=torch.float32)
                print(f"Test {i+1}: torch tensor (CPU)")
            elif i == 2:
                state = torch.randn(state_dim, dtype=torch.float32).to(device)
                print(f"Test {i+1}: torch tensor (GPU)")
            elif i == 3:
                state = np.random.randn(state_dim).astype(np.float64)  # 다른 dtype
                print(f"Test {i+1}: numpy array (float64)")
            else:
                state = list(np.random.randn(state_dim))  # 리스트
                print(f"Test {i+1}: python list")
            
            action = expert.select_action(state, deterministic=False)
            print(f"  ✓ Action shape: {action.shape}, dtype: {action.dtype}")
            print(f"  ✓ Action range: [{action.min():.3f}, {action.max():.3f}]")
            
        except Exception as e:
            print(f"  ✗ Test {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 경험 저장 및 업데이트 테스트
    print("\n--- Experience and Update Test ---")
    try:
        # 더미 경험 생성
        for _ in range(50):  # 충분한 경험 생성
            state = np.random.randn(state_dim).astype(np.float32)
            action = np.random.uniform(-1, 1, action_dim).astype(np.float32)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim).astype(np.float32)
            done = np.random.choice([True, False])
            
            expert.replay_buffer.add(state, action, reward, next_state, done)
        
        print(f"✓ Added {len(expert.replay_buffer)} experiences")
        
        # 업데이트 테스트
        loss_info = expert.update(batch_size=16)
        if loss_info:
            print(f"✓ Update successful:")
            for key, value in loss_info.items():
                print(f"  {key}: {value:.6f}")
        else:
            print("✗ Update returned None")
            
    except Exception as e:
        print(f"✗ Experience/Update test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 저장/로드 테스트
    print("\n--- Save/Load Test ---")
    try:
        save_path = "./test_expert.pth"
        expert.save(save_path)
        print(f"✓ Model saved to {save_path}")
        
        # 새로운 전문가 생성 후 로드
        expert2 = RTX4060SACExpert(state_dim, action_dim, device=device)
        expert2.load(save_path)
        print(f"✓ Model loaded successfully")
        
        # 동일한 입력에 대해 비슷한 출력 확인
        test_state = np.random.randn(state_dim).astype(np.float32)
        action1 = expert.select_action(test_state, deterministic=True)
        action2 = expert2.select_action(test_state, deterministic=True)
        
        diff = np.abs(action1 - action2).mean()
        print(f"✓ Action difference after load: {diff:.6f}")
        
        expert2.cleanup()
        
    except Exception as e:
        print(f"✗ Save/Load test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 정리
    expert.cleanup()
    print("\n✓ All tests completed!")

if __name__ == "__main__":
    test_sac_expert()