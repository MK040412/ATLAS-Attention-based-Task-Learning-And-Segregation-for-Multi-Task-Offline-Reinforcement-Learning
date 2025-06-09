import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import deque
import random

# 기존 ReplayBuffer, PolicyNetwork, ValueNetwork, QNetwork, MoESACExpert 클래스들은 동일...
# (이전에 제공한 코드와 동일하므로 생략)

class DPMMMoESACAgent:
    """DPMM-MoE SAC 에이전트 - 비전 기반 상태 지원"""
    def __init__(self, state_dim: int, action_dim: int, instr_dim: int, 
                 latent_dim: int = 32, fusion_hidden_dim: int = 256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = state_dim  # 업데이트된 상태 차원 (proprioception + vision_cube + goal)
        self.action_dim = action_dim
        self.instr_dim = instr_dim
        self.latent_dim = latent_dim
        
        # Fusion Encoder는 외부에서 주입됨 (동적 상태 차원 처리를 위해)
        self.fusion_encoder = None
        
        # 진짜 DPMM
        from true_dpmm import TrueDPMM
        self.dpmm = TrueDPMM(latent_dim=latent_dim, alpha=1.0, base_var=1.0)
        
        # 전문가들
        self.experts: List[MoESACExpert] = []
        
        print(f"DPMM-MoE SAC Agent initialized (Vision-enhanced):")
        print(f"  State dim: {state_dim} (includes vision-based cube detection)")
        print(f"  Action dim: {action_dim}")
        print(f"  Instruction dim: {instr_dim}")
        print(f"  Latent dim: {latent_dim}")
    
    def _create_expert(self) -> MoESACExpert:
        """새 전문가 생성"""
        expert = MoESACExpert(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            latent_dim=self.latent_dim
        )
        self.experts.append(expert)
        print(f"  Created expert {len(self.experts)-1} for vision-enhanced states")
        return expert
    
    def select_action(self, state, instruction_emb, deterministic=False):
        """행동 선택 - 비전 기반 상태 처리"""
        if self.fusion_encoder is None:
            raise ValueError("FusionEncoder not initialized. Please set agent.fusion_encoder.")
        
        with torch.no_grad():
            # 텐서 변환
            if not torch.is_tensor(state):
                state = torch.FloatTensor(state).to(self.device)
            if not torch.is_tensor(instruction_emb):
                instruction_emb = torch.FloatTensor(instruction_emb).to(self.device)
            
            # 상태 차원 검증
            expected_dim = self.state_dim
            actual_dim = state.shape[-1] if state.dim() > 0 else len(state)
            
            if actual_dim != expected_dim:
                print(f"Warning: State dimension mismatch in action selection. Expected: {expected_dim}, Got: {actual_dim}")
                # 차원 맞추기
                if actual_dim < expected_dim:
                    padding = torch.zeros(expected_dim - actual_dim, device=self.device)
                    state = torch.cat([state, padding])
                else:
                    state = state[:expected_dim]
            
            # 수식대로 융합: z = fφ([e; s])
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
        """경험 저장 - 비전 기반 상태 처리"""
        if cluster_id < len(self.experts):
            expert = self.experts[cluster_id]
            
            # 다음 상태의 잠재 표현 계산
            next_latent_state = self._get_latent_state(next_state, instruction_emb)
            
            expert.replay_buffer.add(
                latent_state, action, reward, 
                next_latent_state, done, instruction_emb
            )
    
    def _get_latent_state(self, state, instruction_emb):
        """상태를 잠재 공간으로 변환 - 비전 정보 포함"""
        if self.fusion_encoder is None:
            raise ValueError("FusionEncoder not initialized.")
        
        with torch.no_grad():
            if not torch.is_tensor(state):
                state = torch.FloatTensor(state).to(self.device)
            if not torch.is_tensor(instruction_emb):
                instruction_emb = torch.FloatTensor(instruction_emb).to(self.device)
            
            # 상태 차원 검증 및 조정
            expected_dim = self.state_dim
            actual_dim = state.shape[-1] if state.dim() > 0 else len(state)
            
            if actual_dim != expected_dim:
                if actual_dim < expected_dim:
                    padding = torch.zeros(expected_dim - actual_dim, device=self.device)
                    state = torch.cat([state, padding])
                else:
                    state = state[:expected_dim]
            
            latent_state = self.fusion_encoder(instruction_emb, state)
            return latent_state.cpu().numpy()
    
    def update_experts(self, batch_size: int = 256):
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
    
    def merge_clusters(self, threshold: float = 2.0):
        """클러스터 병합"""
        old_num_clusters = len(self.experts)
        
        try:
            self.dpmm.merge_clusters(threshold)
            new_num_clusters = self.dpmm.get_cluster_info()['num_clusters']
            
            if new_num_clusters < old_num_clusters:
                print(f"  Clusters merged: {old_num_clusters} -> {new_num_clusters}")
                # 실제 전문가 병합은 복잡하므로 일단 유지
                # TODO: 전문가 네트워크도 병합하는 로직 추가
        except Exception as e:
            print(f"Warning: Cluster merging failed: {e}")
    
    def get_info(self):
        """에이전트 정보 반환 - 비전 기반 상태 정보 포함"""
        try:
            cluster_info = self.dpmm.get_cluster_info()
        except:
            cluster_info = {'num_clusters': len(self.experts), 'cluster_counts': []}
        
        return {
            'num_experts': len(self.experts),
            'cluster_info': cluster_info,
            'buffer_sizes': [len(expert.replay_buffer) for expert in self.experts],
            'state_dim': self.state_dim,
            'vision_enhanced': True
        }
    
    def save(self, filepath: str):
        """모델 저장"""
        save_dict = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'instr_dim': self.instr_dim,
            'latent_dim': self.latent_dim,
            'num_experts': len(self.experts),
            'experts': [expert.get_state_dict() for expert in self.experts],
            'dpmm_state': self.dpmm.get_state() if hasattr(self.dpmm, 'get_state') else None,
            'fusion_encoder': self.fusion_encoder.state_dict() if self.fusion_encoder else None
        }
        torch.save(save_dict, filepath)
        print(f"Vision-enhanced agent saved to {filepath}")
    
    def load(self, filepath: str):
        """모델 로드"""
        save_dict = torch.load(filepath, map_location=self.device)
        
        # 차원 검증
        if save_dict['state_dim'] != self.state_dim:
            print(f"Warning: Loaded state_dim ({save_dict['state_dim']}) != current state_dim ({self.state_dim})")
        
        # 전문가들 로드
        self.experts = []
        for expert_dict in save_dict['experts']:
            expert = MoESACExpert(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                latent_dim=self.latent_dim
            )
            expert.load_state_dict(expert_dict)
            self.experts.append(expert)
        
        # Fusion encoder 로드
        if save_dict['fusion_encoder'] and self.fusion_encoder:
            self.fusion_encoder.load_state_dict(save_dict['fusion_encoder'])
        
        print(f"Vision-enhanced agent loaded from {filepath}")
        print(f"  Loaded {len(self.experts)} experts")