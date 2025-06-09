import torch
import os

class CONFIG:
    # RTX 4060 최적화 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 메모리 최적화
    use_mixed_precision = True
    max_batch_size = 32
    gradient_accumulation_steps = 2
    
    # Pre-trained 모델 설정
    text_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 22MB
    instr_emb_dim = 384
    
    # 네트워크 크기 (메모리 절약)
    latent_dim = 32
    hidden_dim = 128
    
    # DPMM 파라미터
    dpmm_alpha = 1.0
    dpmm_base_var = 1.0
    tau_birth = 1e-3
    tau_merge_kl = 0.1
    
    # SAC 파라미터
    gamma = 0.99
    tau = 0.005
    actor_lr = 1e-4
    critic_lr = 1e-4
    alpha_lr = 1e-4
    
    # 학습 파라미터 (RTX 4060 최적화)
    batch_size = 32
    buffer_capacity = 5000
    update_frequency = 10
    merge_frequency = 15
    
    # 메모리 관리
    max_experts = 8
    cleanup_frequency = 10
    
    # 환경 설정
    max_episode_length = 300
    num_envs = 64  # RTX 4060에 맞게 축소
    
    @classmethod
    def setup_rtx4060_optimization(cls):
        """RTX 4060 특화 최적화 설정"""
        if torch.cuda.is_available():
            # CUDA 메모리 최적화
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
            # 메모리 분할 최적화
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            # 메모리 정리
            torch.cuda.empty_cache()
            
            print(f"🔧 RTX 4060 optimizations applied")
            print(f"  Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
