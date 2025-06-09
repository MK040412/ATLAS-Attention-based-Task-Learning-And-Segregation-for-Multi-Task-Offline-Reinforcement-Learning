import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any
from kl_divergence_dpmm import KLDivergenceDPMM
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from gymnasium.spaces import Box

class FusionEncoder(nn.Module):
    def __init__(self, instr_dim, state_dim, latent_dim, hidden_dim=256):
        super().__init__()
        self.instr_dim = instr_dim
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        input_dim = instr_dim + state_dim
        
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
        # 차원 정리
        if instr_emb.dim() == 3:
            instr_emb = instr_emb.squeeze(0).mean(dim=0)
        elif instr_emb.dim() == 2:
            if instr_emb.size(0) == 1:
                instr_emb = instr_emb.squeeze(0)
            else:
                instr_emb = instr_emb.mean(dim=0)
        
        if state_obs.dim() == 2:
            if state_obs.size(0) == 1:
                state_obs = state_obs.squeeze(0)
            else:
                state_obs = state_obs[0]
        
        # 배치 차원 추가
        if instr_emb.dim() == 1:
            instr_emb = instr_emb.unsqueeze(0)
        if state_obs.dim() == 1:
            state_obs = state_obs.unsqueeze(0)
        
        # 연결 및 융합
        combined = torch.cat([instr_emb, state_obs], dim=-1)
        z = self.mlp(combined)
        
        return z.squeeze(0)

class DummyEnvWrapper(gym.Env):
    """SAC를 위한 더미 환경 래퍼"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def reset(self, seed=None, options=None):
        state = np.zeros(self.state_dim, dtype=np.float32)
        return state, {}
    
    def step(self, action):
        next_state = np.zeros(self.state_dim, dtype=np.float32)
        reward = 0.0
        done = False
        truncated = False
        info = {}
        return next_state, reward, done, truncated, info

class StableBaselines3SACExpert:
    """Stable Baselines3 SAC를 사용한 전문가"""
    def __init__(self, state_dim, action_dim, expert_id=0):
        self.expert_id = expert_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 더미 환경 생성 (SAC 초기화용)
        dummy_env = DummyEnvWrapper(state_dim, action_dim)
        
        # SAC 에이전트 초기화
        self.sac = SAC(
            "MlpPolicy",
            dummy_env,
            learning_rate=3e-4,
            buffer_size=10000,
            learning_starts=100,
            batch_size=64,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            target_update_interval=1,
            target_entropy='auto',
            use_sde=False,
            sde_sample_freq=-1,
            use_sde_at_warmup=False,
            tensorboard_log=None,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0,
            seed=None,
            device='auto',
            _init_setup_model=True
        )
        
        # 경험 저장을 위한 버퍼
        self.experience_buffer = []
        
        print(f"🤖 SAC Expert {expert_id} initialized")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}")
    
    def select_action(self, state, deterministic=False):
        """행동 선택"""
        if isinstance(state, np.ndarray):
            state_tensor = state
        else:
            state_tensor = np.array(state, dtype=np.float32)
        
        # SAC로 행동 예측
        action, _ = self.sac.predict(state_tensor, deterministic=deterministic)
        return action
    
    def store_experience(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.experience_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
        
        # 버퍼가 너무 크면 오래된 경험 제거
        if len(self.experience_buffer) > 10000:
            self.experience_buffer.pop(0)
    
    def update(self):
        """SAC 업데이트"""
        if len(self.experience_buffer) < 100:  # 최소 경험 수
            return {'loss': 0.0}
        
        # 최근 경험들을 SAC replay buffer에 추가
        for exp in self.experience_buffer[-64:]:  # 최근 64개만
            self.sac.replay_buffer.add(
                exp['state'],
                exp['next_state'],
                exp['action'],
                exp['reward'],
                exp['done'],
                [{}]  # infos
            )
        
        # SAC 학습
        if self.sac.replay_buffer.size() >= self.sac.learning_starts:
            self.sac.train(gradient_steps=1, batch_size=self.sac.batch_size)
        
        return {'loss': 0.0}  # 실제 loss는 SAC 내부에서 처리

class IntegratedKLDPMMMoESAC:
    """KL Divergence DPMM + MoE-SAC 통합 에이전트"""
    def __init__(self, state_dim, action_dim, instr_dim, latent_dim=32,
                 tau_birth=1e-4, tau_merge_kl=0.1, device='cpu'):
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.instr_dim = instr_dim
        
        # KL-Divergence 기반 DPMM
        self.dpmm = KLDivergenceDPMM(
            latent_dim=latent_dim,
            tau_birth=tau_birth,
            tau_merge_kl=tau_merge_kl
        )
        
        # Fusion Encoder
        self.fusion_encoder = FusionEncoder(
            instr_dim, state_dim, latent_dim
        ).to(self.device)
        
        # SAC 전문가들
        self.experts: List[StableBaselines3SACExpert] = []
        
        # 통계 추적
        self.assignment_history = []
        self.expert_usage_count = {}
        
        print(f"🤖 Integrated KL-DPMM-MoE-SAC Agent initialized")
        print(f"  State: {state_dim}, Action: {action_dim}, Instruction: {instr_dim}")
        print(f"  Latent: {latent_dim}, Device: {device}")
    
    def _create_new_expert(self) -> int:
        """새로운 SAC 전문가 생성"""
        expert_id = len(self.experts)
        expert = StableBaselines3SACExpert(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            expert_id=expert_id
        )
        self.experts.append(expert)
        self.expert_usage_count[expert_id] = 0
        
        print(f"🐣 Created new SAC expert {expert_id}")
        return expert_id
    
    def select_action(self, state, instruction_emb, deterministic=False):
        """행동 선택 및 클러스터 할당"""
        # 상태와 지시사항을 latent space로 매핑
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state).to(self.device)
            else:
                state_tensor = state.to(self.device)
            
            if isinstance(instruction_emb, np.ndarray):
                instr_tensor = torch.FloatTensor(instruction_emb).to(self.device)
            else:
                instr_tensor = instruction_emb.to(self.device)
            
            # Fusion encoding
            z = self.fusion_encoder(instr_tensor, state_tensor)
            z_np = z.detach().cpu().numpy()
        
        # DPMM 클러스터 할당
        cluster_id = self.dpmm.assign(z_np)
        
        # 해당 클러스터에 대응하는 전문가가 없으면 생성
        if cluster_id >= len(self.experts):
            while len(self.experts) <= cluster_id:
                self._create_new_expert()
        
        # 전문가로부터 행동 선택
        expert = self.experts[cluster_id]
        action = expert.select_action(state, deterministic=deterministic)
        
        # 사용 통계 업데이트
        self.expert_usage_count[cluster_id] += 1
        self.assignment_history.append({
            'cluster_id': cluster_id,
            'expert_id': cluster_id,
            'z': z_np.copy()
        })
        
        return action, cluster_id
    
    def store_experience(self, state, action, reward, next_state, done, cluster_id):
        """경험 저장"""
        if cluster_id < len(self.experts):
            expert = self.experts[cluster_id]
            expert.store_experience(state, action, reward, next_state, done)
    
    def update_experts(self):
        """모든 전문가 업데이트"""
        total_loss = 0.0
        updated_count = 0
        
        for i, expert in enumerate(self.experts):
            if len(expert.experience_buffer) >= 10:  # 최소 경험 수
                loss_info = expert.update()
                total_loss += loss_info.get('loss', 0.0)
                updated_count += 1
        
        return {
            'total_loss': total_loss,
            'updated_experts': updated_count,
            'total_experts': len(self.experts)
        }
    
    def merge_clusters(self, verbose=False):
        """KL divergence 기반 클러스터 병합"""
        old_count = len(self.dpmm.clusters)
        self.dpmm.merge(verbose=verbose)
        new_count = len(self.dpmm.clusters)
        
        if new_count < old_count:
            if verbose:
                print(f"  🔗 KL-Merge: {old_count} → {new_count} clusters")
            return True
        return False
    
    def get_statistics(self):
        """통계 정보 반환"""
        cluster_info = self.dpmm.get_cluster_info()
        
        return {
            'dpmm': cluster_info,
            'experts': {
                'total_experts': len(self.experts),
                'expert_usage': self.expert_usage_count.copy(),
                'total_assignments': len(self.assignment_history)
            },
            'recent_assignments': self.assignment_history[-10:] if self.assignment_history else []
        }
    
    def save_checkpoint(self, filepath):
        """체크포인트 저장"""
        checkpoint = {
            'fusion_encoder': self.fusion_encoder.state_dict(),
            'dpmm_clusters': {
                'clusters': self.dpmm.clusters,
                'next_cluster_id': self.dpmm.next_cluster_id,
                'birth_events': self.dpmm.birth_events,
                'merge_events': self.dpmm.merge_events
            },
            'expert_count': len(self.experts),
            'assignment_history': self.assignment_history,
            'expert_usage_count': self.expert_usage_count,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'instr_dim': self.instr_dim,
                'latent_dim': self.dpmm.latent_dim,
                'tau_birth': self.dpmm.tau_birth,
                'tau_merge_kl': self.dpmm.tau_merge_kl
            }
        }
        
        # SAC 전문가들 저장
        expert_models = {}
        for i, expert in enumerate(self.experts):
            expert_models[f'expert_{i}'] = {
                'policy': expert.sac.policy.state_dict(),
                'experience_count': len(expert.experience_buffer)
            }
        checkpoint['experts'] = expert_models
        
        torch.save(checkpoint, filepath)
        print(f"💾 Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """체크포인트 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Fusion encoder 로드
        self.fusion_encoder.load_state_dict(checkpoint['fusion_encoder'])
        
        # DPMM 상태 복원
        dpmm_data = checkpoint['dpmm_clusters']
        self.dpmm.clusters = dpmm_data['clusters']
        self.dpmm.next_cluster_id = dpmm_data['next_cluster_id']
        self.dpmm.birth_events = dpmm_data['birth_events']
        self.dpmm.merge_events = dpmm_data['merge_events']
        
        # 전문가들 복원
        expert_count = checkpoint['expert_count']
        self.experts = []
        expert_models = checkpoint['experts']
        
        for i in range(expert_count):
            expert = StableBaselines3SACExpert(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                expert_id=i
            )
            
            # SAC 정책 로드
            if f'expert_{i}' in expert_models:
                expert.sac.policy.load_state_dict(expert_models[f'expert_{i}']['policy'])
            
            self.experts.append(expert)
        
        # 기타 상태 복원
        self.assignment_history = checkpoint['assignment_history']
        self.expert_usage_count = checkpoint['expert_usage_count']
        
        print(f"📂 Checkpoint loaded from {filepath}")
        print(f"  Restored {len(self.experts)} experts and {len(self.dpmm.clusters)} clusters")