import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv
import gym

from mathematical_dpmm import MathematicalDPMM

class SB3DPMMExpert:
    """SB3 기반 SAC 전문가 (DPMM과 완전 통합)"""
    
    def __init__(self, env: GymEnv, expert_id: int, 
                 learning_rate: float = 3e-4, buffer_size: int = 50000):
        self.expert_id = expert_id
        self.env = env
        
        # SB3 SAC 에이전트
        self.sac_agent = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            target_update_interval=1,
            verbose=0,
            device='auto'
        )
        
        # 개별 리플레이 버퍼
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            env.observation_space,
            env.action_space,
            device=self.sac_agent.device,
            handle_timeout_termination=False
        )
        
        # 학습 통계
        self.training_steps = 0
        self.total_samples = 0
        
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """행동 선택"""
        action, _ = self.sac_agent.predict(observation, deterministic=deterministic)
        return action
    
    def store_experience(self, obs: np.ndarray, action: np.ndarray, 
                        reward: float, next_obs: np.ndarray, done: bool):
        """경험 저장"""
        self.replay_buffer.add(obs, next_obs, action, reward, done, [{}])
        self.total_samples += 1
    
    def update(self, batch_size: int = 256) -> Optional[Dict]:
        """전문가 업데이트"""
        if self.replay_buffer.size() < batch_size:
            return None
        
        try:
            # SB3 SAC 학습
            self.sac_agent.replay_buffer = self.replay_buffer
            self.sac_agent.learn(total_timesteps=batch_size, log_interval=None)
            self.training_steps += 1
            
            return {
                'expert_id': self.expert_id,
                'training_steps': self.training_steps,
                'buffer_size': self.replay_buffer.size()
            }
        except Exception as e:
            print(f"  ⚠️ Expert {self.expert_id} update failed: {e}")
            return None
    
    def get_info(self) -> Dict:
        """전문가 정보"""
        return {
            'expert_id': self.expert_id,
            'training_steps': self.training_steps,
            'buffer_size': self.replay_buffer.size(),
            'total_samples': self.total_samples
        }

class FusionEncoder(nn.Module):
    """명령어-상태 융합 인코더 (수치적 안정성 개선)"""
    
    def __init__(self, instr_dim: int, state_dim: int, latent_dim: int):
        super().__init__()
        self.instr_dim = instr_dim
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        # 레이어 정규화 추가
        self.instr_norm = nn.LayerNorm(instr_dim)
        self.state_norm = nn.LayerNorm(state_dim)
        
        # 융합 네트워크
        self.fusion_net = nn.Sequential(
            nn.Linear(instr_dim + state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, latent_dim),
            nn.Tanh()  # 출력 범위 제한
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, instr_emb: torch.Tensor, state_obs: torch.Tensor) -> torch.Tensor:
        """순전파 (안전한 차원 처리)"""
        # 차원 정규화
        if instr_emb.dim() > 2:
            instr_emb = instr_emb.mean(dim=-2)  # 시퀀스 차원 평균
        if instr_emb.dim() == 1:
            instr_emb = instr_emb.unsqueeze(0)
        
        if state_obs.dim() == 1:
            state_obs = state_obs.unsqueeze(0)
        
        # 정규화
        instr_emb = self.instr_norm(instr_emb)
        state_obs = self.state_norm(state_obs)
        
        # 융합
        combined = torch.cat([instr_emb, state_obs], dim=-1)
        latent = self.fusion_net(combined)
        
        return latent

class CompleteSB3DPMMAgent:
    """완전한 SB3-DPMM 통합 에이전트"""
    
    def __init__(self, env: GymEnv, instr_dim: int, latent_dim: int = 32,
                 tau_birth: float = -5.0, tau_merge_kl: float = 0.1):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 차원 정보
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.instr_dim = instr_dim
        self.latent_dim = latent_dim
        
        # 융합 인코더
        self.fusion_encoder = FusionEncoder(
            instr_dim=instr_dim,
            state_dim=self.state_dim,
            latent_dim=latent_dim
        ).to(self.device)
        
        # 수학적 DPMM
        self.dpmm = MathematicalDPMM(
            latent_dim=latent_dim,
            alpha=1.0,
            tau_birth=tau_birth,
            tau_merge_kl=tau_merge_kl
        )
        
        # SB3 전문가들
        self.experts: Dict[int, SB3DPMMExpert] = {}
        
        # 학습 통계
        self.total_steps = 0
        self.episode_count = 0
        
        print(f"🤖 Complete SB3-DPMM Agent initialized:")
        print(f"  State dim: {self.state_dim}, Action dim: {self.action_dim}")
        print(f"  Instruction dim: {instr_dim}, Latent dim: {latent_dim}")
        print(f"  DPMM params: tau_birth={tau_birth}, tau_merge_kl={tau_merge_kl}")
    
    def _create_expert(self, cluster_id: int) -> SB3DPMMExpert:
        """새 전문가 생성"""
        expert = SB3DPMMExpert(
            env=self.env,
            expert_id=cluster_id,
            learning_rate=3e-4,
            buffer_size=50000
        )
        self.experts[cluster_id] = expert
        print(f"  🆕 Created SB3 expert #{cluster_id}")
        return expert
    
    def select_action(self, state: np.ndarray, instruction_emb: np.ndarray, 
                     deterministic: bool = False) -> Tuple[np.ndarray, int]:
        """행동 선택 (DPMM 클러스터링 + SB3 전문가)"""
        try:
            # Tensor 변환
            state_tensor = torch.FloatTensor(state).to(self.device)
            instr_tensor = torch.FloatTensor(instruction_emb).to(self.device)
            
            # 융합된 잠재 표현
            with torch.no_grad():
                latent = self.fusion_encoder(instr_tensor, state_tensor)
                latent_np = latent.cpu().numpy().flatten()
            
            # DPMM 클러스터 할당
            cluster_id = self.dpmm.assign_point(latent_np)
            
            # 전문가 생성 (필요시)
            if cluster_id not in self.experts:
                self._create_expert(cluster_id)
            
            # 전문가로부터 행동 선택
            expert = self.experts[cluster_id]
            action = expert.select_action(state, deterministic=deterministic)
            
            return action, cluster_id
            
        except Exception as e:
            print(f"  ⚠️ Action selection failed: {e}")
            # 안전한 기본 행동
            action = self.env.action_space.sample()
            return action, 0
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, 
                        done: bool, cluster_id: int):
        """경험 저장"""
        try:
            if cluster_id in self.experts:
                self.experts[cluster_id].store_experience(
                    state, action, reward, next_state, done
                )
                self.total_steps += 1
        except Exception as e:
            print(f"  ⚠️ Experience storage failed: {e}")
    
    def update_experts(self, batch_size: int = 256) -> Dict:
        """모든 전문가 업데이트"""
        update_results = {}
        
        for cluster_id, expert in self.experts.items():
            try:
                result = expert.update(batch_size=batch_size)
                if result:
                    update_results[f'expert_{cluster_id}'] = result
            except Exception as e:
                print(f"  ⚠️ Expert {cluster_id} update failed: {e}")
        
        return update_results
    
    def merge_clusters(self, verbose: bool = True) -> bool:
        """DPMM 클러스터 병합"""
        try:
            return self.dpmm.merge_clusters(verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"  ⚠️ Cluster merge failed: {e}")
            return False
    
    def get_info(self) -> Dict:
        """에이전트 정보"""
        try:
            dpmm_stats = self.dpmm.get_statistics()
            
            expert_info = {}
            total_buffer_size = 0
            for cluster_id, expert in self.experts.items():
                expert_info[cluster_id] = expert.get_info()
                total_buffer_size += expert.replay_buffer.size()
            
            return {
                'num_experts': len(self.experts),
                'total_buffer_size': total_buffer_size,
                'total_steps': self.total_steps,
                'episode_count': self.episode_count,
                'dpmm': dpmm_stats['dpmm'],
                'experts': expert_info
            }
            
        except Exception as e:
            print(f"  ⚠️ Info gathering failed: {e}")
            return {
                'num_experts': len(self.experts),
                'total_buffer_size': 0,
                'total_steps': self.total_steps,
                'episode_count': self.episode_count,
                'dpmm': {'num_clusters': 0, 'total_births': 0, 'total_merges': 0},
                'experts': {}
            }
    
    def save(self, filepath: str):
        """모델 저장"""
        try:
            checkpoint = {
                'fusion_encoder': self.fusion_encoder.state_dict(),
                'dpmm_clusters': {
                    cid: {
                        'mu': cluster.mu.tolist(),
                        'sigma': cluster.sigma.tolist(),
                        'N': cluster.N
                    }
                    for cid, cluster in self.dpmm.clusters.items()
                },
                'dpmm_params': {
                    'tau_birth': self.dpmm.tau_birth,
                    'tau_merge_kl': self.dpmm.tau_merge_kl,
                    'next_cluster_id': self.dpmm.next_cluster_id
                },
                'total_steps': self.total_steps,
                'episode_count': self.episode_count
            }
            
            # SB3 전문가들 저장
            expert_models = {}
            for cluster_id, expert in self.experts.items():
                model_path = f"{filepath}_expert_{cluster_id}"
                expert.sac_agent.save(model_path)
                expert_models[cluster_id] = {
                    'model_path': model_path,
                    'expert_info': expert.get_info()
                }
            
            checkpoint['experts'] = expert_models
            
            torch.save(checkpoint, f"{filepath}_main.pt")
            print(f"✅ Complete model saved to {filepath}")
            
        except Exception as e:
            print(f"⚠️ Model save failed: {e}")
    
    def load(self, filepath: str):
        """모델 로드"""
        try:
            checkpoint = torch.load(f"{filepath}_main.pt", map_location=self.device)
            
            # 융합 인코더 로드
            self.fusion_encoder.load_state_dict(checkpoint['fusion_encoder'])
            
            # DPMM 상태 복원
            self.dpmm.clusters = {}
            for cid, cluster_data in checkpoint['dpmm_clusters'].items():
                from mathematical_dpmm import GaussianCluster
                cluster = GaussianCluster(
                    id=int(cid),
                    mu=np.array(cluster_data['mu']),
                    sigma=np.array(cluster_data['sigma']),
                    N=cluster_data['N']
                )
                self.dpmm.clusters[int(cid)] = cluster
            
            # DPMM 파라미터 복원
            dpmm_params = checkpoint['dpmm_params']
            self.dpmm.tau_birth = dpmm_params['tau_birth']
            self.dpmm.tau_merge_kl = dpmm_params['tau_merge_kl']
            self.dpmm.next_cluster_id = dpmm_params['next_cluster_id']
            
            # SB3 전문가들 로드
            self.experts = {}
            for cluster_id, expert_data in checkpoint['experts'].items():
                expert = self._create_expert(int(cluster_id))
                expert.sac_agent.load(expert_data['model_path'])
            
            # 통계 복원
            self.total_steps = checkpoint['total_steps']
            self.episode_count = checkpoint['episode_count']
            
            print(f"✅ Complete model loaded from {filepath}")
            print(f"  Loaded {len(self.experts)} experts")
            print(f"  DPMM clusters: {len(self.dpmm.clusters)}")
            
        except Exception as e:
            print(f"⚠️ Model load failed: {e}")
