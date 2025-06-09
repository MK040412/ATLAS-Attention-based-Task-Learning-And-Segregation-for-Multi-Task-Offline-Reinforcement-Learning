import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple
from simple_sac_expert import SimpleSACExpert
from simple_dpmm import SimpleDPMM
from corrected_fusion_encoder import FusionEncoder

class SimplifiedMoEAgent:
    """간단화된 MoE SAC 에이전트"""
    def __init__(self, state_dim: int, action_dim: int, instr_dim: int, 
                 latent_dim: int = 32, fusion_hidden_dim: int = 256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.instr_dim = instr_dim
        self.latent_dim = latent_dim
        
        # Fusion Encoder
        self.fusion_encoder = FusionEncoder(
            instr_dim=instr_dim,
            state_dim=state_dim,
            latent_dim=latent_dim,
            hidden_dim=fusion_hidden_dim
        ).to(self.device)
        
        # DPMM
        self.dpmm = SimpleDPMM(latent_dim=latent_dim, alpha=1.0, base_var=1.0)
        
        # 전문가들
        self.experts: List[SimpleSACExpert] = []
        
        print(f"Simplified MoE Agent initialized:")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Latent dim: {latent_dim}")
    
    def _create_expert(self) -> SimpleSACExpert:
        """새 전문가 생성"""
        expert = SimpleSACExpert(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            latent_dim=self.latent_dim
        )
        self.experts.append(expert)
        print(f"  Created expert {len(self.experts)-1}")
        return expert
    
    def select_action(self, state, instruction_emb, deterministic=False):
        """행동 선택"""
        with torch.no_grad():
            # 텐서 변환
            if not torch.is_tensor(state):
                state = torch.FloatTensor(state).to(self.device)
            if not torch.is_tensor(instruction_emb):
                instruction_emb = torch.FloatTensor(instruction_emb).to(self.device)
            
            # 융합된 특징 추출
            latent_state = self.fusion_encoder(instruction_emb, state)
            
            # DPMM으로 클러스터 할당
            cluster_id = self.dpmm.assign_and_update(latent_state.cpu().numpy())
            
            # 전문가가 없으면 생성
            while cluster_id >= len(self.experts):
                self._create_expert()
            
            # 해당 전문가로부터 행동 선택
            expert = self.experts[cluster_id]
            action = expert.select_action(latent_state, deterministic)
            
            return action, cluster_id, latent_state.cpu().numpy()
    
    def store_experience(self, state, action, reward, next_state, done, 
                        instruction_emb, cluster_id, latent_state):
        """경험 저장"""
        if cluster_id < len(self.experts):
            expert = self.experts[cluster_id]
            
            # 다음 상태의 잠재 표현 계산
            next_latent_state = self._get_latent_state(next_state, instruction_emb)
            
            expert.replay_buffer.add(
                latent_state, action, reward, 
                next_latent_state, done, instruction_emb
            )
    
    def _get_latent_state(self, state, instruction_emb):
        """상태를 잠재 공간으로 변환"""
        with torch.no_grad():
            if not torch.is_tensor(state):
                state = torch.FloatTensor(state).to(self.device)
            if not torch.is_tensor(instruction_emb):
                instruction_emb = torch.FloatTensor(instruction_emb).to(self.device)
            
            latent_state = self.fusion_encoder(instruction_emb, state)
            return latent_state.cpu().numpy()
    
    def update_experts(self, batch_size: int = 64):
        """모든 전문가 업데이트"""
        losses = {}
        for i, expert in enumerate(self.experts):
            if len(expert.replay_buffer) >= batch_size:
                try:
                    expert_losses = expert.update(batch_size)
                    losses[f'expert_{i}'] = expert_losses
                except Exception as e:
                    print(f"Warning: Failed to update expert {i}: {e}")
                    losses[f'expert_{i}'] = None
        
        return losses
    
    def merge_clusters(self, threshold: float = 1.5):
        """클러스터 병합"""
        old_num_clusters = len(self.experts)
        
        try:
            self.dpmm.merge_clusters(threshold)
            new_num_clusters = self.dpmm.get_cluster_info()['num_clusters']
            
            if new_num_clusters < old_num_clusters:
                print(f"  Clusters merged: {old_num_clusters} -> {new_num_clusters}")
        except Exception as e:
            print(f"Warning: Cluster merging failed: {e}")
    
    def get_info(self):
        """에이전트 정보 반환"""
        try:
            cluster_info = self.dpmm.get_cluster_info()
        except:
            cluster_info = {'num_clusters': len(self.experts), 'cluster_counts': []}
        
        return {
            'num_experts': len(self.experts),
            'cluster_info': cluster_info,
            'buffer_sizes': [len(expert.replay_buffer) for expert in self.experts],
            'state_dim': self.state_dim
        }