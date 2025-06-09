import torch
import torch.nn as nn

class FusionEncoder(nn.Module):
    """수식을 정확히 따르는 Fusion Encoder - 비전 기반 상태 포함
    
    x = [e; s] ∈ R^(de+ds)
    z = fφ(x) = MLP(x)
    
    여기서 s는 이제 [proprioception; vision_cube_pos; goal_pos]를 포함
    """
    def __init__(self, instr_dim, state_dim, latent_dim, hidden_dim=256):
        super().__init__()
        self.instr_dim = instr_dim
        self.state_dim = state_dim  # 업데이트된 상태 차원 (기존 + 6)
        self.latent_dim = latent_dim
        
        print(f"FusionEncoder init: instr_dim={instr_dim}, state_dim={state_dim}, latent_dim={latent_dim}")
        print(f"  State includes: proprioception + vision_cube_pos(3) + goal_pos(3)")
        
        # 수식대로: x = [e; s] 직접 연결 후 MLP
        input_dim = instr_dim + state_dim
        
        # z = MLP(x) - 더 깊은 네트워크로 비전 정보 처리
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # 가중치 초기화
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, instr_emb, state_obs):
        """
        Args:
            instr_emb: 명령어 임베딩 [batch_size, instr_dim] 또는 [instr_dim]
            state_obs: 상태 관측 [batch_size, state_dim] 또는 [state_dim]
                      state_dim = proprioception_dim + 3(vision_cube) + 3(goal)
        
        Returns:
            z: 잠재 표현 [batch_size, latent_dim] 또는 [latent_dim]
        """
        # 차원 정리
        original_batch = True
        
        # 명령어 임베딩 처리
        if instr_emb.dim() == 3:  # [1, seq_len, dim]
            instr_emb = instr_emb.squeeze(0).mean(dim=0)  # [dim]
        elif instr_emb.dim() == 2:
            if instr_emb.size(0) == 1:  # [1, dim]
                instr_emb = instr_emb.squeeze(0)  # [dim]
            else:  # [seq_len, dim]
                instr_emb = instr_emb.mean(dim=0)  # [dim]
        
        # 상태 관측 처리
        if state_obs.dim() == 2:
            if state_obs.size(0) == 1:  # [1, dim]
                state_obs = state_obs.squeeze(0)  # [dim]
                original_batch = False
            # else: 진짜 배치 [batch_size, dim]
        elif state_obs.dim() == 1:  # [dim]
            original_batch = False
        
        # 배치 차원 맞추기
        if instr_emb.dim() == 1 and state_obs.dim() == 1:
            # 둘 다 1차원이면 배치 차원 추가
            instr_emb = instr_emb.unsqueeze(0)  # [1, instr_dim]
            state_obs = state_obs.unsqueeze(0)  # [1, state_dim]
        elif instr_emb.dim() == 1 and state_obs.dim() == 2:
            # 명령어만 1차원이면 상태의 배치 크기에 맞춤
            instr_emb = instr_emb.unsqueeze(0).repeat(state_obs.size(0), 1)
        elif instr_emb.dim() == 2 and state_obs.dim() == 1:
            # 상태만 1차원이면 명령어의 배치 크기에 맞춤
            state_obs = state_obs.unsqueeze(0).repeat(instr_emb.size(0), 1)
        
        # 상태 차원 검증
        expected_state_dim = self.state_dim
        actual_state_dim = state_obs.size(-1)
        
        if actual_state_dim != expected_state_dim:
            print(f"Warning: State dimension mismatch. Expected: {expected_state_dim}, Got: {actual_state_dim}")
            # 차원이 맞지 않으면 패딩 또는 자르기
            if actual_state_dim < expected_state_dim:
                # 패딩
                padding = torch.zeros(state_obs.size(0), expected_state_dim - actual_state_dim, 
                                    device=state_obs.device, dtype=state_obs.dtype)
                state_obs = torch.cat([state_obs, padding], dim=-1)
            else:
                # 자르기
                state_obs = state_obs[:, :expected_state_dim]
        
        # 수식대로: x = [e; s]
        x = torch.cat([instr_emb, state_obs], dim=-1)
        
        # z = MLP(x)
        z = self.mlp(x)
        
        # 원래 배치가 없었으면 배치 차원 제거
        if not original_batch and z.size(0) == 1:
            z = z.squeeze(0)
        
        return z