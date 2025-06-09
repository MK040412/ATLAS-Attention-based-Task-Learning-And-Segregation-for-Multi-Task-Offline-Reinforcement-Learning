import torch
import torch.nn as nn
import numpy as np

class DimensionAwareFusion(nn.Module):
    """차원을 자동으로 인식하고 처리하는 융합 모듈"""
    
    def __init__(self, instr_dim, state_dim, latent_dim):
        super().__init__()
        self.instr_dim = instr_dim
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        print(f"🔗 DimensionAwareFusion init:")
        print(f"  Instruction dim: {instr_dim}")
        print(f"  State dim: {state_dim}")
        print(f"  Latent dim: {latent_dim}")
        
        # 지시사항 프로젝션
        self.instr_proj = nn.Sequential(
            nn.Linear(instr_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim // 2)
        )
        
        # 상태 프로젝션
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim // 2)
        )
        
        # 융합 레이어
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
    def _safe_reshape(self, tensor, target_dim, name="tensor"):
        """텐서를 안전하게 리셰이프"""
        try:
            if tensor.dim() == 1:
                # 1D 텐서인 경우 배치 차원 추가
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() == 3:
                # 3D 텐서인 경우 시퀀스 차원에서 평균
                if tensor.size(0) == 1:
                    tensor = tensor.squeeze(0)  # (1, seq, dim) -> (seq, dim)
                tensor = tensor.mean(dim=0, keepdim=True)  # (seq, dim) -> (1, dim)
            elif tensor.dim() == 2:
                # 2D 텐서는 그대로 사용
                pass
            else:
                print(f"Warning: Unexpected {name} dimension: {tensor.shape}")
                tensor = tensor.view(1, -1)
            
            # 마지막 차원이 target_dim과 맞는지 확인
            if tensor.size(-1) != target_dim:
                print(f"Warning: {name} last dim mismatch. Expected {target_dim}, got {tensor.size(-1)}")
                # 적응적 처리
                if tensor.size(-1) > target_dim:
                    tensor = tensor[..., :target_dim]  # 잘라내기
                else:
                    # 패딩
                    pad_size = target_dim - tensor.size(-1)
                    padding = torch.zeros(*tensor.shape[:-1], pad_size, device=tensor.device)
                    tensor = torch.cat([tensor, padding], dim=-1)
            
            return tensor
            
        except Exception as e:
            print(f"Error in reshaping {name}: {e}")
            # 안전한 기본값 반환
            return torch.zeros(1, target_dim, device=tensor.device if torch.is_tensor(tensor) else 'cpu')
    
    def forward(self, instr_emb, state_obs):
        """순전파"""
        try:
            # 입력 차원 확인
            if torch.is_tensor(instr_emb):
                instr_tensor = instr_emb
            else:
                instr_tensor = torch.FloatTensor(instr_emb).to(next(self.parameters()).device)
            
            if torch.is_tensor(state_obs):
                state_tensor = state_obs
            else:
                state_tensor = torch.FloatTensor(state_obs).to(next(self.parameters()).device)
            
            # 안전한 리셰이프
            instr_tensor = self._safe_reshape(instr_tensor, self.instr_dim, "instruction")
            state_tensor = self._safe_reshape(state_tensor, self.state_dim, "state")
            
            # 프로젝션
            instr_proj = self.instr_proj(instr_tensor)  # (batch, latent_dim//2)
            state_proj = self.state_proj(state_tensor)  # (batch, latent_dim//2)
            
            # 연결
            combined = torch.cat([instr_proj, state_proj], dim=-1)  # (batch, latent_dim)
            
            # 융합
            fused = self.fusion(combined)  # (batch, latent_dim)
            
            return fused.squeeze(0) if fused.size(0) == 1 else fused
            
        except Exception as e:
            print(f"Error in DimensionAwareFusion forward: {e}")
            # 안전한 기본값 반환
            device = next(self.parameters()).device
            return torch.zeros(self.latent_dim, device=device)
    
    def save_checkpoint(self, filepath):
        """체크포인트 저장"""
        try:
            torch.save({
                'state_dict': self.state_dict(),
                'instr_dim': self.instr_dim,
                'state_dim': self.state_dim,
                'latent_dim': self.latent_dim,
                'hidden_dim': self.hidden_dim
            }, filepath)
            print(f"✓ Fusion model saved to {filepath}")
        except Exception as e:
            print(f"Warning: Failed to save fusion model: {e}")
    
    def load_checkpoint(self, filepath):
        """체크포인트 로드"""
        try:
            checkpoint = torch.load(filepath, map_location=self.instr_proj[0].weight.device)
            
            # 차원 검증
            if (checkpoint['instr_dim'] != self.instr_dim or 
                checkpoint['state_dim'] != self.state_dim or
                checkpoint['latent_dim'] != self.latent_dim):
                raise ValueError("Dimension mismatch in checkpoint")
            
            self.load_state_dict(checkpoint['state_dict'])
            print(f"✓ Fusion model loaded from {filepath}")
        except Exception as e:
            print(f"Warning: Failed to load fusion model: {e}")

def create_dimension_aware_fusion(instr_dim, state_dim, latent_dim, hidden_dim=128):
    """차원 인식 융합 모델 생성"""
    return DimensionAwareFusion(instr_dim, state_dim, latent_dim, hidden_dim)