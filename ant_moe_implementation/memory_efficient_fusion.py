import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import CONFIG

class MemoryEfficientFusionEncoder(nn.Module):
    """메모리 효율적인 융합 인코더"""
    
    def __init__(self, instr_dim, state_dim, latent_dim, hidden_dim=64):
        super().__init__()
        self.instr_dim = instr_dim
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # 지시사항 프로젝션 (경량화)
        self.instr_proj = nn.Sequential(
            nn.Linear(instr_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 상태 프로젝션 (경량화)
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 융합 네트워크 (매우 경량)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.Tanh()  # 출력 범위 제한
        )
        
        # 가중치 초기화
        self._init_weights()
        
        print(f"MemoryEfficientFusionEncoder initialized:")
        print(f"  Input dims: instr={instr_dim}, state={state_dim}")
        print(f"  Hidden dim: {hidden_dim}, Latent dim: {latent_dim}")
        print(f"  Parameters: {self.count_parameters():,}")
    
    def _init_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def count_parameters(self):
        """파라미터 개수 계산"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_memory_usage(self):
        """메모리 사용량 추정"""
        param_count = self.count_parameters()
        # 4 bytes per float32 parameter
        memory_mb = (param_count * 4) / (1024 * 1024)
        return {
            'parameters': param_count,
            'memory_mb': memory_mb
        }
    
    def forward(self, instr_emb, state_obs):
        """순전파"""
        # 입력 차원 정규화
        instr_emb = self._normalize_input(instr_emb, expected_dim=self.instr_dim)
        state_obs = self._normalize_input(state_obs, expected_dim=self.state_dim)
        
        # 배치 차원 확인
        if instr_emb.dim() == 1:
            instr_emb = instr_emb.unsqueeze(0)
        if state_obs.dim() == 1:
            state_obs = state_obs.unsqueeze(0)
        
        # 프로젝션
        instr_features = self.instr_proj(instr_emb)  # (batch, hidden_dim//2)
        state_features = self.state_proj(state_obs)  # (batch, hidden_dim//2)
        
        # 융합
        combined = torch.cat([instr_features, state_features], dim=-1)  # (batch, hidden_dim)
        latent = self.fusion(combined)  # (batch, latent_dim)
        
        # 단일 배치인 경우 차원 제거
        if latent.size(0) == 1:
            latent = latent.squeeze(0)
        
        return latent
    
    def _normalize_input(self, tensor, expected_dim):
        """입력 텐서 정규화"""
        if not torch.is_tensor(tensor):
            tensor = torch.FloatTensor(tensor).to(next(self.parameters()).device)
        
        # 차원 맞추기
        if tensor.dim() == 0:  # 스칼라
            tensor = tensor.unsqueeze(0).repeat(expected_dim)
        elif tensor.dim() == 1:
            if tensor.size(0) != expected_dim:
                if tensor.size(0) > expected_dim:
                    tensor = tensor[:expected_dim]
                else:
                    # 패딩
                    padding = torch.zeros(expected_dim - tensor.size(0), 
                                        device=tensor.device, dtype=tensor.dtype)
                    tensor = torch.cat([tensor, padding])
        elif tensor.dim() == 2:
            # 배치 차원이 있는 경우
            if tensor.size(-1) != expected_dim:
                if tensor.size(-1) > expected_dim:
                    tensor = tensor[..., :expected_dim]
                else:
                    batch_size = tensor.size(0)
                    padding = torch.zeros(batch_size, expected_dim - tensor.size(-1),
                                        device=tensor.device, dtype=tensor.dtype)
                    tensor = torch.cat([tensor, padding], dim=-1)
        elif tensor.dim() == 3:
            # 시퀀스 차원이 있는 경우 (예: BERT 출력)
            tensor = tensor.mean(dim=1)  # 시퀀스 차원에서 평균
            tensor = self._normalize_input(tensor, expected_dim)
        
        return tensor
    
    def encode_instruction(self, instr_emb):
        """지시사항만 인코딩"""
        instr_emb = self._normalize_input(instr_emb, self.instr_dim)
        if instr_emb.dim() == 1:
            instr_emb = instr_emb.unsqueeze(0)
        return self.instr_proj(instr_emb)
    
    def encode_state(self, state_obs):
        """상태만 인코딩"""
        state_obs = self._normalize_input(state_obs, self.state_dim)
        if state_obs.dim() == 1:
            state_obs = state_obs.unsqueeze(0)
        return self.state_proj(state_obs)
    
    def save_checkpoint(self, path):
        """체크포인트 저장"""
        checkpoint = {
            'state_dict': self.state_dict(),
            'config': {
                'instr_dim': self.instr_dim,
                'state_dim': self.state_dim,
                'latent_dim': self.latent_dim,
                'hidden_dim': self.hidden_dim
            }
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """체크포인트 로드"""
        checkpoint = torch.load(path, map_location=next(self.parameters()).device)
        self.load_state_dict(checkpoint['state_dict'])
        return checkpoint['config']

class AdaptiveFusionEncoder(MemoryEfficientFusionEncoder):
    """적응적 융합 인코더 (더 고급 버전)"""
    
    def __init__(self, instr_dim, state_dim, latent_dim, hidden_dim=64):
        super().__init__(instr_dim, state_dim, latent_dim, hidden_dim)
        
        # 어텐션 메커니즘 (경량)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=2,
            dropout=0.1,
            batch_first=True
        )
        
        # 게이팅 메커니즘
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, instr_emb, state_obs):
        """적응적 순전파"""
        # 기본 인코딩
        instr_emb = self._normalize_input(instr_emb, expected_dim=self.instr_dim)
        state_obs = self._normalize_input(state_obs, expected_dim=self.state_dim)
        
        if instr_emb.dim() == 1:
            instr_emb = instr_emb.unsqueeze(0)
        if state_obs.dim() == 1:
            state_obs = state_obs.unsqueeze(0)
        
        # 프로젝션
        instr_features = self.instr_proj(instr_emb)
        state_features = self.state_proj(state_obs)
        
        # 어텐션 적용 (선택적)
        if CONFIG.use_attention:
            try:
                # 셀프 어텐션
                features_stack = torch.stack([instr_features, state_features], dim=1)
                attended_features, _ = self.attention(
                    features_stack, features_stack, features_stack
                )
                instr_features = attended_features[:, 0]
                state_features = attended_features[:, 1]
            except Exception as e:
                print(f"Warning: Attention failed, using basic fusion: {e}")
        
        # 게이팅
        combined = torch.cat([instr_features, state_features], dim=-1)
        gate_weights = self.gate(combined)
        
        # 가중 결합
        weighted_features = (gate_weights[:, 0:1] * instr_features + 
                           gate_weights[:, 1:2] * state_features)
        
        # 최종 융합
        final_combined = torch.cat([weighted_features, combined], dim=-1)
        latent = self.fusion(final_combined)
        
        if latent.size(0) == 1:
            latent = latent.squeeze(0)
        
        return latent

def create_fusion_encoder(instr_dim, state_dim, latent_dim, hidden_dim=64, adaptive=False):
    """융합 인코더 팩토리 함수"""
    if adaptive and hasattr(CONFIG, 'use_adaptive_fusion') and CONFIG.use_adaptive_fusion:
        return AdaptiveFusionEncoder(instr_dim, state_dim, latent_dim, hidden_dim)
    else:
        return MemoryEfficientFusionEncoder(instr_dim, state_dim, latent_dim, hidden_dim)