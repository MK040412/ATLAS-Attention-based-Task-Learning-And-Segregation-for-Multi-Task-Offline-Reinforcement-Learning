import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class InstructionFusion(nn.Module):
    def __init__(self, instr_dim, state_dim, hidden_dim=128):
        super().__init__()
        self.instr_dim = instr_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        print(f"InstructionFusion init: instr_dim={instr_dim}, state_dim={state_dim}, hidden_dim={hidden_dim}")
        
        # 지시사항 임베딩 처리
        self.instr_proj = nn.Linear(instr_dim, hidden_dim)
        # 상태 임베딩 처리
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        # 융합 레이어
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, instr_emb, state_obs):
        # 지시사항 임베딩 처리
        if instr_emb.dim() == 3:
            if instr_emb.size(0) == 1:
                instr_emb = instr_emb.squeeze(0)  # (1, seq_len, dim) -> (seq_len, dim)
            instr_emb = instr_emb.mean(dim=0)  # 시퀀스 차원에서 평균 -> (dim,)
        elif instr_emb.dim() == 2:
            instr_emb = instr_emb.mean(dim=0)  # (seq_len, dim) -> (dim,)
        
        # 상태 관측값 처리 - 1차원으로 유지
        if state_obs.dim() == 2:
            if state_obs.size(0) == 1:
                state_obs = state_obs.squeeze(0)  # (1, dim) -> (dim,)
            else:
                state_obs = state_obs[0]  # 첫 번째 배치만 사용
        
        # 배치 차원 추가
        if instr_emb.dim() == 1:
            instr_emb = instr_emb.unsqueeze(0)  # (dim,) -> (1, dim)
        if state_obs.dim() == 1:
            state_obs = state_obs.unsqueeze(0)  # (dim,) -> (1, dim)
            
        # 프로젝션
        instr_proj = self.instr_proj(instr_emb)  # (1, hidden_dim)
        state_proj = self.state_proj(state_obs)  # (1, hidden_dim)
        
        # 연결 및 융합
        combined = torch.cat([instr_proj, state_proj], dim=-1)  # (1, hidden_dim * 2)
        fused = self.fusion(combined)  # (1, hidden_dim)
        
        return fused.squeeze(0)  # (hidden_dim,)

class MoESACAgent:
    def __init__(self, state_dim, action_dim, instr_dim, num_experts=4, hidden_dim=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        print(f"MoESACAgent init: state_dim={state_dim}, action_dim={action_dim}")
        
        # 지시사항 융합 모듈
        self.fusion = InstructionFusion(instr_dim, state_dim, hidden_dim//2).to(self.device)
        
        # MoE 게이팅 네트워크
        self.gating = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, num_experts),
            nn.Softmax(dim=-1)
        ).to(self.device)
        
        # 전문가 네트워크들
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim//2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ) for _ in range(num_experts)
        ]).to(self.device)
        
        # 가치 함수
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        
        # 옵티마이저
        all_params = list(self.fusion.parameters()) + \
                    list(self.gating.parameters()) + \
                    list(self.experts.parameters()) + \
                    list(self.value_net.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=3e-4)
        
        # 리플레이 버퍼는 외부에서 설정
        self.replay_buffer = None
        
    def store_experience(self, state, action, reward, next_state, done):
        """경험 저장"""
        if self.replay_buffer is not None:
            self.replay_buffer.add(state, action, reward, next_state, done)
        
    def select_action(self, state, instruction_emb):
        """행동 선택"""
        with torch.no_grad():
            # numpy를 torch tensor로 변환
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(self.device)
            if isinstance(instruction_emb, np.ndarray):
                instruction_emb = torch.FloatTensor(instruction_emb).to(self.device)
                
            # 융합된 특징 추출
            fused_features = self.fusion(instruction_emb, state)
            
            # 전문가 가중치 계산
            expert_weights = self.gating(fused_features)
            
            # 각 전문가의 출력 계산
            expert_outputs = []
            for expert in self.experts:
                output = expert(fused_features)
                expert_outputs.append(output)
            expert_outputs = torch.stack(expert_outputs)  # (num_experts, action_dim)
            
            # 가중 평균으로 최종 행동 계산
            action = torch.sum(expert_weights.unsqueeze(-1) * expert_outputs, dim=0)
            
            # 더 큰 탐험 노이즈
            noise = torch.randn_like(action) * 0.5  # 0.1에서 0.5로 증가
            action = action + noise
            
            # 액션 스케일링 (더 큰 움직임)
            action = torch.tanh(action) * 3.0  # [-3, 3] 범위로 확장
            
            return action.cpu().numpy()
    
    def update(self, batch):
        """배치 업데이트"""
        try:
            states, actions, rewards, next_states, dones, instructions = batch
            
            # numpy를 tensor로 변환
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.BoolTensor(dones).to(self.device)
            instructions = torch.FloatTensor(instructions).to(self.device)
            
            batch_size = states.shape[0]
            
            # 현재 상태의 가치 계산
            current_features_list = []
            for i in range(batch_size):
                features = self.fusion(instructions[i], states[i])
                current_features_list.append(features)
            current_features = torch.stack(current_features_list)
            
            current_values = self.value_net(current_features).squeeze()
            
            # 다음 상태의 가치 계산
            with torch.no_grad():
                next_features_list = []
                for i in range(batch_size):
                    features = self.fusion(instructions[i], next_states[i])
                    next_features_list.append(features)
                next_features = torch.stack(next_features_list)
                
                next_values = self.value_net(next_features).squeeze()
                target_values = rewards + 0.99 * next_values * (~dones)
            
            # 가치 함수 손실
            value_loss = F.mse_loss(current_values, target_values)
            
            # 정책 손실 (간단한 버전)
            expert_weights = self.gating(current_features)
            expert_outputs = []
            for expert in self.experts:
                output = expert(current_features)
                expert_outputs.append(output)
            expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch, num_experts, action_dim)
            
            predicted_actions = torch.sum(expert_weights.unsqueeze(-1) * expert_outputs, dim=1)
            policy_loss = F.mse_loss(predicted_actions, actions)
            
            # 전체 손실
            total_loss = value_loss + policy_loss
            
            # 업데이트
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.get_all_parameters(), max_norm=1.0)
            self.optimizer.step()
            
            return {
                'value_loss': value_loss.item(),
                'policy_loss': policy_loss.item(),
                'total_loss': total_loss.item()
            }
            
        except Exception as e:
            print(f"Update error: {e}")
            return None
    
    def get_all_parameters(self):
        """모든 파라미터 반환"""
        params = []
        params.extend(self.fusion.parameters())
        params.extend(self.gating.parameters())
        params.extend(self.experts.parameters())
        params.extend(self.value_net.parameters())
        return params
    
    def save(self, filepath):
        """모델 저장"""
        torch.save({
            'fusion_state_dict': self.fusion.state_dict(),
            'gating_state_dict': self.gating.state_dict(),
            'experts_state_dict': self.experts.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.fusion.load_state_dict(checkpoint['fusion_state_dict'])
        self.gating.load_state_dict(checkpoint['gating_state_dict'])
        self.experts.load_state_dict(checkpoint['experts_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
