#!/usr/bin/env python3
"""
학습 결과 분석 스크립트
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_training_results(output_dir="./outputs"):
    """학습 결과 분석"""
    
    print("=== Training Results Analysis ===")
    
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} not found!")
        return
    
    # 태스크별 결과 분석
    task_dirs = [d for d in os.listdir(output_dir) if d.startswith("task_")]
    task_dirs.sort()
    
    print(f"Found {len(task_dirs)} completed tasks:")
    
    total_experts = 0
    task_results = []
    
    for task_dir in task_dirs:
        task_path = os.path.join(output_dir, task_dir)
        
        # 전문가 모델 개수 확인
        expert_files = [f for f in os.listdir(task_path) if f.startswith("expert_") and f.endswith(".pth")]
        num_experts = len(expert_files)
        total_experts = max(total_experts, num_experts)
        
        # 융합 모델 확인
        fusion_exists = os.path.exists(os.path.join(task_path, "fusion_encoder.pth"))
        
        task_info = {
            'name': task_dir,
            'experts': num_experts,
            'fusion_model': fusion_exists,
            'path': task_path
        }
        task_results.append(task_info)
        
        print(f"\n{task_dir}:")
        print(f"  Experts: {num_experts}")
        print(f"  Fusion model: {'✓' if fusion_exists else '✗'}")
        
        # 모델 크기 확인
        total_size = 0
        for expert_file in expert_files:
            file_path = os.path.join(task_path, expert_file)
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            total_size += size
        
        if fusion_exists:
            fusion_path = os.path.join(task_path, "fusion_encoder.pth")
            fusion_size = os.path.getsize(fusion_path) / (1024 * 1024)
            total_size += fusion_size
            print(f"  Fusion model size: {fusion_size:.2f} MB")
        
        print(f"  Total size: {total_size:.2f} MB")
    
    print(f"\n=== Summary ===")
    print(f"Total tasks completed: {len(task_results)}")
    print(f"Maximum experts created: {total_experts}")
    print(f"Training completed successfully!")
    
    return task_results

def test_saved_models(output_dir="./outputs"):
    """저장된 모델 테스트"""
    
    print("\n=== Testing Saved Models ===")
    
    try:
        from rtx4060_sac_expert import RTX4060SACExpert
        from memory_efficient_fusion import create_fusion_encoder
        from config import CONFIG
        
        # 마지막 태스크의 모델 로드 테스트
        task_dirs = [d for d in os.listdir(output_dir) if d.startswith("task_")]
        if not task_dirs:
            print("No task directories found!")
            return
        
        last_task = sorted(task_dirs)[-1]
        task_path = os.path.join(output_dir, last_task)
        
        print(f"Testing models from {last_task}...")
        
        # 전문가 모델 로드 테스트
        expert_files = [f for f in os.listdir(task_path) if f.startswith("expert_") and f.endswith(".pth")]
        
        if expert_files:
            expert_path = os.path.join(task_path, expert_files[0])
            print(f"Loading expert model: {expert_files[0]}")
            
            # 더미 차원으로 전문가 생성
            test_expert = RTX4060SACExpert(
                state_dim=35,  # 예상 상태 차원
                action_dim=8,  # 예상 액션 차원
                hidden_dim=CONFIG.hidden_dim,
                device=CONFIG.device
            )
            
            try:
                test_expert.load(expert_path)
                print("  ✓ Expert model loaded successfully")
                
                # 테스트 추론
                test_state = np.random.randn(35).astype(np.float32)
                action = test_expert.select_action(test_state, deterministic=True)
                print(f"  ✓ Test inference successful, action shape: {action.shape}")
                
                test_expert.cleanup()
                
            except Exception as e:
                print(f"  ✗ Expert model test failed: {e}")
        
        # 융합 모델 로드 테스트
        fusion_path = os.path.join(task_path, "fusion_encoder.pth")
        if os.path.exists(fusion_path):
            print(f"Testing fusion model...")
            
            try:
                fusion = create_fusion_encoder(
                    CONFIG.instr_emb_dim,
                    35,  # 상태 차원
                    CONFIG.latent_dim,
                    CONFIG.hidden_dim
                ).to(CONFIG.device)
                
                fusion.load_checkpoint(fusion_path)
                print("  ✓ Fusion model loaded successfully")
                
                # 테스트 추론
                test_instr = torch.randn(CONFIG.instr_emb_dim, device=CONFIG.device)
                test_state = torch.randn(35, device=CONFIG.device)
                
                with torch.no_grad():
                    latent = fusion(test_instr, test_state)
                    print(f"  ✓ Test inference successful, latent shape: {latent.shape}")
                
            except Exception as e:
                print(f"  ✗ Fusion model test failed: {e}")
        
        print("Model testing completed!")
        
    except Exception as e:
        print(f"Model testing failed: {e}")

def create_training_report(output_dir="./outputs"):
    """학습 보고서 생성"""
    
    print("\n=== Creating Training Report ===")
    
    report_path = os.path.join(output_dir, "training_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("RTX 4060 Optimized Ant MoE Training Report\n")
        f.write("=" * 50 + "\n\n")
        
        # 시스템 정보
        f.write("System Information:\n")
        f.write(f"  Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"  PyTorch Version: {torch.__version__}\n")
        f.write(f"  CUDA Available: {torch.cuda.is_available()}\n\n")
        
        # 설정 정보
        from config import CONFIG
        f.write("Configuration:\n")
        f.write(f"  Hidden Dim: {CONFIG.hidden_dim}\n")
        f.write(f"  Latent Dim: {CONFIG.latent_dim}\n")
        f.write(f"  Batch Size: {CONFIG.batch_size}\n")
        f.write(f"  Buffer Capacity: {CONFIG.buffer_capacity}\n")
        f.write(f"  Learning Rates: Actor={CONFIG.actor_lr}, Critic={CONFIG.critic_lr}\n\n")
        
        # 태스크 결과
        task_results = analyze_training_results(output_dir)
        f.write("Task Results:\n")
        for result in task_results:
            f.write(f"  {result['name']}: {result['experts']} experts\n")
        
        f.write(f"\nTotal Experts Created: {max([r['experts'] for r in task_results])}\n")
        f.write("Training Status: COMPLETED\n")
    
    print(f"Training report saved to: {report_path}")

def main():
    """메인 함수"""
    
    output_dir = "./outputs"
    
    # 결과 분석
    analyze_training_results(output_dir)
    
    # 모델 테스트
    test_saved_models(output_dir)
    
    # 보고서 생성
    create_training_report(output_dir)
    
    print("\n" + "=" * 50)
    print("Analysis completed!")
    print("Check ./outputs/training_report.txt for detailed report")

if __name__ == "__main__":
    main()