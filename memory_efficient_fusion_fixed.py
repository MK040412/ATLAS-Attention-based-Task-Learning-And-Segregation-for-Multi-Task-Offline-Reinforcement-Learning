import torch
import torch.nn as nn
import numpy as np

class MemoryEfficientFusion(nn.Module):
    """메모리 효율적인 융합 모델"""
    
    def __init__(self, instr_dim, state_dim, latent_dim, hidden_dim):
        super().__init__()
        self.instr_dim = instr_dim
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # 더 작은 네트워크
        self.instr_proj = nn.Linear(instr_dim, hidden_dim // 4)
        self.state_proj = nn.Linear(state_dim, hidden_dim // 4)
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, latent_dim)
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, instr_emb, state_obs):
        """순전파"""
        try:
            # 입력 검증 및 변환
            if not torch.is_tensor(instr_emb):
                instr_emb = torch.tensor(instr_emb, dtype=torch.float32, device=self.instr_proj.weight.device)
            if not torch.is_tensor(state_obs):
                state_obs = torch.tensor(state_obs, dtype=torch.float32, device=self.state_proj.weight.device)
            
            # 디바이스 맞추기
            device = self.instr_proj.weight.device
            instr_emb = instr_emb.to(device)
            state_obs = state_obs.to(device)
            
            # 차원 조정
            if instr_emb.dim() > 1:
                instr_emb = instr_emb.flatten()
            if state_obs.dim() > 1:
                state_obs = state_obs.flatten()
            
            # 크기 검증
            if instr_emb.size(0) != self.instr_dim:
                if instr_emb.size(0) > self.instr_dim:
                    instr_emb = instr_emb[:self.instr_dim]
                else:
                    # 패딩
                    padding = torch.zeros(self.instr_dim - instr_emb.size(0), device=device)
                    instr_emb = torch.cat([instr_emb, padding])
            
            if state_obs.size(0) != self.state_dim:
                if state_obs.size(0) > self.state_dim:
                    state_obs = state_obs[:self.state_dim]
                else:
                    # 패딩
                    padding = torch.zeros(self.state_dim - state_obs.size(0), device=device)
                    state_obs = torch.cat([state_obs, padding])
            
            # 프로젝션
            instr_proj = self.instr_proj(instr_emb)
            state_proj = self.state_proj(state_obs)
            
            # 융합
            combined = torch.cat([instr_proj, state_proj])
            result = self.fusion(combined)
            
            return result
            
        except Exception as e:
            print(f"Fusion forward error: {e}")
            # 오류 시 더미 출력 반환
            device = self.instr_proj.weight.device
            return torch.zeros(self.latent_dim, device=device)
    
    def count_parameters(self):
        """파라미터 개수 계산"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_checkpoint(self, path):
        """체크포인트 저장"""
        try:
            checkpoint = {
                'model_state_dict': self.state_dict(),
                'config': {
                    'instr_dim': self.instr_dim,
                    'state_dim': self.state_dim,
                    'latent_dim': self.latent_dim,
                    'hidden_dim': self.hidden_dim
                }
            }
            torch.save(checkpoint, path)
        except Exception as e:
            print(f"Failed to save fusion checkpoint: {e}")
    
    def load_checkpoint(self, path):
        """체크포인트 로드"""
        try:
            checkpoint = torch.load(path, map_location=self.instr_proj.weight.device)
            self.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Failed to load fusion checkpoint: {e}")

def create_fusion_encoder(instr_dim, state_dim, latent_dim, hidden_dim):
    """융합 인코더 생성"""
    return MemoryEfficientFusion(instr_dim, state_dim, latent_dim, hidden_dim)