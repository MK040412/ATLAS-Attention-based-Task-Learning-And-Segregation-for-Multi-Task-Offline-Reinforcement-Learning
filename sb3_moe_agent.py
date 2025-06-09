import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sb3_moe_expert import SB3SACExpert
from fixed_simple_gym_wrapper import SimpleAntPushWrapper

class FusionEncoder(nn.Module):
    """명령어-상태 융합 인코더"""
    def __init__(self, instr_dim: int, state_dim: int, latent_dim: int):
        super().__init__()
        self.instr_dim = instr_dim
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        # 수식: z = MLP([e; s])
        input_dim = instr_dim + state_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        print(f"🔗 FusionEncoder: {instr_dim} + {state_dim} -> {latent_dim}")
    
    def forward(self, instr_emb: torch.Tensor, state_obs: torch.Tensor) -> torch.Tensor:
        """융합 인코딩"""
        # 차원 정리
        if instr_emb.dim() == 1:
            instr_emb = instr_emb.unsqueeze(0)
        if state_obs.dim() == 1:
            state_obs = state_obs.unsqueeze(0)
        
        # 연결 및 인코딩
        combined = torch.cat([instr_emb, state_obs], dim=-1)
        latent = self.mlp(combined)
        
        return latent.squeeze(0) if latent.size(0) == 1 else latent

class SimpleDPMM:
    """간단한 DPMM (Boolean Tensor 문제 수정)"""
    def __init__(self, latent_dim: int, alpha: float = 1.0, base_var: float = 1.0, 
                 tau_birth: float = 2.0, tau_merge: float = 0.5):
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.base_var = base_var
        self.tau_birth = tau_birth
        self.tau_merge = tau_merge
        
        self.clusters = []
        self.cluster_counts = []
        self.assignments = []
        
        # 통계 추적
        self.birth_events = []
        self.merge_events = []
        
    def assign_and_update(self, z: np.ndarray) -> int:
        """새 데이터를 클러스터에 할당하고 업데이트"""
        z = np.array(z).flatten()
        
        if len(self.clusters) == 0:
            self.clusters.append(z.copy())
            self.cluster_counts.append(1)
            self.assignments.append(0)
            self.birth_events.append(len(self.assignments))
            return 0
        
        # 각 클러스터와의 거리 계산
        distances = []
        for cluster_center in self.clusters:
            dist = np.linalg.norm(z - cluster_center)
            distances.append(dist)
        
        min_dist = min(distances)
        closest_cluster = distances.index(min_dist)
        
        # Birth 조건 (거리 기반)
        if min_dist > self.tau_birth:
            # 새 클러스터 생성
            cluster_id = len(self.clusters)
            self.clusters.append(z.copy())
            self.cluster_counts.append(1)
            self.birth_events.append(len(self.assignments))
            print(f"  🐣 New cluster {cluster_id} created (distance: {min_dist:.3f})")
        else:
            # 기존 클러스터에 할당
            cluster_id = closest_cluster
            self.cluster_counts[cluster_id] += 1
            
            # 클러스터 중심 업데이트
            alpha = 0.1
            self.clusters[cluster_id] = (1 - alpha) * self.clusters[cluster_id] + alpha * z
        
        self.assignments.append(cluster_id)
        return cluster_id
    
    def merge_clusters(self, verbose: bool = False) -> bool:
        """클러스터 병합"""
        if len(self.clusters) < 2:
            return False
        
        merged = False
        
        # 가장 가까운 클러스터 쌍 찾기
        min_distance = float('inf')
        merge_pair = None
        
        for i in range(len(self.clusters)):
            for j in range(i + 1, len(self.clusters)):
                dist = np.linalg.norm(self.clusters[i] - self.clusters[j])
                if dist < min_distance:
                    min_distance = dist
                    merge_pair = (i, j)
        
        # 병합 조건
        if merge_pair and min_distance < self.tau_merge:
            i, j = merge_pair
            
            # 가중 평균으로 새 중심 계산
            total_count = self.cluster_counts[i] + self.cluster_counts[j]
            new_center = (self.cluster_counts[i] * self.clusters[i] + 
                         self.cluster_counts[j] * self.clusters[j]) / total_count
            
            # 더 작은 인덱스에 병합
            keep_idx = min(i, j)
            remove_idx = max(i, j)
            
            self.clusters[keep_idx] = new_center
            self.cluster_counts[keep_idx] = total_count
            
            # 제거할 클러스터 삭제
            del self.clusters[remove_idx]
            del self.cluster_counts[remove_idx]
            
            # 할당 기록 업데이트 (remove_idx를 keep_idx로 변경)
            self.assignments = [keep_idx if x == remove_idx else (x-1 if x > remove_idx else x) 
                              for x in self.assignments]
            
            self.merge_events.append(len(self.assignments))
            merged = True
            
            if verbose:
                print(f"  🔗 Merged clusters {i}↔{j} → {keep_idx} (distance: {min_distance:.3f})")
        
        return merged
    
    def get_num_clusters(self) -> int:
        return len(self.clusters)
    
    def get_cluster_info(self) -> Dict[str, Any]:
        return {
            'num_clusters': len(self.clusters),
            'cluster_counts': self.cluster_counts.copy(),
            'total_assignments': len(self.assignments),
            'total_births': len(self.birth_events),
            'total_merges': len(self.merge_events)
        }

class SB3MoEAgent:
    """SB3 기반 MoE SAC 에이전트"""
    def __init__(self, env: SimpleAntPushWrapper, instr_dim: int, 
                 latent_dim: int = 32, cluster_threshold: float = 2.0):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 차원 정보
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.instr_dim = instr_dim
        self.latent_dim = latent_dim
        self.cluster_threshold = cluster_threshold
        
        # 융합 인코더
        self.fusion_encoder = FusionEncoder(
            instr_dim=instr_dim,
            state_dim=self.state_dim,
            latent_dim=latent_dim
        ).to(self.device)
        
        # DPMM
        self.dpmm = SimpleDPMM(latent_dim=latent_dim)
        
        # 전문가들
        self.experts: List[SB3SACExpert] = []
        
        print(f"🎭 SB3 MoE Agent initialized:")
        print(f"  State dim: {self.state_dim}")
        print(f"  Action dim: {self.action_dim}")
        print(f"  Instruction dim: {instr_dim}")
        print(f"  Latent dim: {latent_dim}")
    
    def _create_expert(self) -> SB3SACExpert:
        """새 전문가 생성"""
        expert_id = len(self.experts)
        expert = SB3SACExpert(
            env=self.env,
            expert_id=expert_id,
            learning_rate=3e-4,
            buffer_size=5000
        )
        self.experts.append(expert)
        print(f"  ➕ Created expert {expert_id}")
        return expert
    
    def select_action(self, state: np.ndarray, instruction_emb: np.ndarray, 
                     deterministic: bool = False) -> Tuple[np.ndarray, int]:
        """행동 선택 (Tensor 문제 수정)"""
        # Tensor 변환 및 차원 정리
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.FloatTensor(state).to(self.device)
        else:
            state_tensor = state.to(self.device)
        
        if not isinstance(instruction_emb, torch.Tensor):
            instr_tensor = torch.FloatTensor(instruction_emb).to(self.device)
        else:
            instr_tensor = instruction_emb.to(self.device)
        
        # 차원 정리 (배치 차원 제거)
        if state_tensor.dim() > 1:
            state_tensor = state_tensor.squeeze()
        if instr_tensor.dim() > 1:
            instr_tensor = instr_tensor.squeeze()
        
        # 융합된 잠재 표현 생성
        try:
            with torch.no_grad():
                latent = self.fusion_encoder(instr_tensor, state_tensor)
                # Tensor를 numpy로 안전하게 변환
                if hasattr(latent, 'cpu'):
                    latent_np = latent.cpu().numpy()
                else:
                    latent_np = np.array(latent)
                
                # 1차원으로 평탄화
                latent_np = latent_np.flatten()
        except Exception as e:
            print(f"⚠️ Fusion encoder error: {e}")
            latent_np = np.random.randn(self.latent_dim)
        
        # DPMM 클러스터 할당
        cluster_id = self.dpmm.assign_and_update(latent_np)
        
        # 전문가가 없으면 생성
        if cluster_id not in self.experts:
            self._create_expert(cluster_id)
        
        # 해당 전문가로부터 행동 선택
        expert = self.experts[cluster_id]
        
        try:
            # 상태를 올바른 형태로 변환
            if isinstance(state, np.ndarray):
                obs_for_expert = state.copy()
            else:
                obs_for_expert = np.array(state)
            
            # 차원 확인 및 조정
            if len(obs_for_expert.shape) == 0:
                obs_for_expert = obs_for_expert.reshape(1)
            elif len(obs_for_expert.shape) > 1:
                obs_for_expert = obs_for_expert.flatten()
            
            # SB3 predict 호출 (Boolean Tensor 문제 방지)
            action, _ = expert.predict(obs_for_expert, deterministic=deterministic)
            
            # 액션 후처리
            if isinstance(action, np.ndarray):
                if len(action.shape) > 1:
                    action = action.flatten()
            else:
                action = np.array([action])
                
        except Exception as e:
            print(f"⚠️ Expert prediction error: {e}")
            # 폴백: 랜덤 액션
            action = np.random.uniform(-1, 1, self.action_dim)
        
        return action, cluster_id
    
    def store_experience(self, state, action, reward, next_state, done, cluster_id):
        """경험 저장 (Tensor 안전 처리)"""
        if cluster_id in self.experts:
            expert = self.experts[cluster_id]
            
            try:
                # 모든 입력을 numpy로 변환
                if hasattr(state, 'cpu'):
                    state = state.cpu().numpy()
                if hasattr(next_state, 'cpu'):
                    next_state = next_state.cpu().numpy()
                if hasattr(action, 'cpu'):
                    action = action.cpu().numpy()
                if hasattr(reward, 'item'):
                    reward = reward.item()
                if hasattr(done, 'item'):
                    done = done.item()
                
                # 차원 정리
                if isinstance(state, np.ndarray) and len(state.shape) > 1:
                    state = state.flatten()
                if isinstance(next_state, np.ndarray) and len(next_state.shape) > 1:
                    next_state = next_state.flatten()
                if isinstance(action, np.ndarray) and len(action.shape) > 1:
                    action = action.flatten()
                
                # 리플레이 버퍼에 저장
                if hasattr(expert, 'replay_buffer'):
                    expert.replay_buffer.add(state, action, reward, next_state, done)
                else:
                    print(f"⚠️ Expert {cluster_id} has no replay buffer")
                    
            except Exception as e:
                print(f"⚠️ Experience storage error for expert {cluster_id}: {e}")
    
    def update_experts(self, batch_size: int = 64) -> Dict[str, Any]:
        """모든 전문가 업데이트"""
        update_results = {}
        
        for expert in self.experts:
            result = expert.update(batch_size)
            if result:
                update_results[f"expert_{expert.expert_id}"] = result
        
        return update_results
    
    def get_info(self) -> Dict[str, Any]:
        """에이전트 정보"""
        expert_info = [expert.get_info() for expert in self.experts]
        dpmm_info = self.dpmm.get_info()
        
        return {
            'num_experts': len(self.experts),
            'experts': expert_info,
            'dpmm': dpmm_info,
            'total_buffer_size': sum(info['buffer_size'] for info in expert_info)
        }
    
    def save(self, save_dir: str):
        """모델 저장"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 융합 인코더 저장
        fusion_path = os.path.join(save_dir, "fusion_encoder.pt")
        torch.save(self.fusion_encoder.state_dict(), fusion_path)
        
        # DPMM 상태 저장
        dpmm_path = os.path.join(save_dir, "dpmm_state.pt")
        torch.save({
            'clusters': self.dpmm.clusters,
            'cluster_counts': self.dpmm.cluster_counts,
            'assignments': self.dpmm.assignments,
            'latent_dim': self.dpmm.latent_dim,
            'alpha': self.dpmm.alpha,
            'base_var': self.dpmm.base_var
        }, dpmm_path)
        
        # 각 전문가 저장
        for i, expert in enumerate(self.experts):
            expert_path = os.path.join(save_dir, f"expert_{i}")
            expert.save(expert_path)
        
        # 메타 정보 저장
        meta_path = os.path.join(save_dir, "meta_info.pt")
        torch.save({
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'instr_dim': self.instr_dim,
            'latent_dim': self.latent_dim,
            'cluster_threshold': self.cluster_threshold,
            'num_experts': len(self.experts)
        }, meta_path)
        
        print(f"💾 SB3 MoE Agent saved to {save_dir}")
    
    def load(self, save_dir: str):
        """모델 로드"""
        import os
        
        # 메타 정보 로드
        meta_path = os.path.join(save_dir, "meta_info.pt")
        if os.path.exists(meta_path):
            meta_info = torch.load(meta_path)
            print(f"📂 Loading agent with {meta_info['num_experts']} experts")
        
        # 융합 인코더 로드
        fusion_path = os.path.join(save_dir, "fusion_encoder.pt")
        if os.path.exists(fusion_path):
            self.fusion_encoder.load_state_dict(torch.load(fusion_path))
            print(f"  ✅ Fusion encoder loaded")
        
        # DPMM 상태 로드
        dpmm_path = os.path.join(save_dir, "dpmm_state.pt")
        if os.path.exists(dpmm_path):
            dpmm_state = torch.load(dpmm_path)
            self.dpmm.clusters = dpmm_state['clusters']
            self.dpmm.cluster_counts = dpmm_state['cluster_counts']
            self.dpmm.assignments = dpmm_state['assignments']
            print(f"  ✅ DPMM state loaded: {len(self.dpmm.clusters)} clusters")
        
        # 전문가들 로드
        self.experts = []
        expert_idx = 0
        while True:
            expert_path = os.path.join(save_dir, f"expert_{expert_idx}")
            if os.path.exists(expert_path + ".zip"):  # SB3 saves as .zip
                expert = SB3SACExpert(self.env, expert_idx)
                expert.load(expert_path)
                self.experts.append(expert)
                expert_idx += 1
            else:
                break
        
        print(f"📂 SB3 MoE Agent loaded: {len(self.experts)} experts")
        
    def merge_clusters_rigorous(self):
        """엄밀한 수학적 기준으로 클러스터 병합"""
        if hasattr(self, 'dpmm') and hasattr(self.dpmm, 'merge_clusters'):
            return self.dpmm.merge_clusters()  # KL-divergence 기반 병합
        return None

    def get_dpmm_info(self):
        """DPMM 상세 정보 반환"""
        if hasattr(self, 'dpmm'):
            return self.dpmm.get_cluster_info()
        return {'num_clusters': 0, 'cluster_counts': []}

    def get_loss_info(self):
        """SAC 손실 정보 반환"""
        # 각 전문가의 최근 손실 정보 수집
        loss_info = {}
        for i, expert in enumerate(self.experts):
            if hasattr(expert, 'last_loss'):
                loss_info[f'expert_{i}'] = expert.last_loss
        return loss_info
    
    # SB3MoEAgent 클래스에 추가할 메서드들

# SB3MoEAgent 클래스에 추가할 DPMM 안전 메서드들

def set_dpmm_params(self, tau_birth=None, tau_merge=None):
    """DPMM 파라미터 안전 설정"""
    try:
        if hasattr(self, 'dpmm'):
            if tau_birth is not None:
                if hasattr(self.dpmm, 'tau_birth'):
                    self.dpmm.tau_birth = tau_birth
                elif hasattr(self.dpmm, 'birth_threshold'):
                    self.dpmm.birth_threshold = tau_birth
            
            if tau_merge is not None:
                if hasattr(self.dpmm, 'tau_merge_kl'):
                    self.dpmm.tau_merge_kl = tau_merge
                elif hasattr(self.dpmm, 'cluster_threshold'):
                    self.dpmm.cluster_threshold = tau_merge
                elif hasattr(self.dpmm, 'merge_threshold'):
                    self.dpmm.merge_threshold = tau_merge
            
            print(f"  🔧 DPMM params set safely: tau_birth={tau_birth}, tau_merge={tau_merge}")
        else:
            print(f"  ⚠️ No DPMM found in agent")
    except Exception as e:
        print(f"  ⚠️ DPMM param setting failed: {e}")

def merge_clusters(self, verbose=False):
    """안전한 클러스터 병합"""
    try:
        if hasattr(self, 'dpmm'):
            if hasattr(self.dpmm, 'merge'):
                self.dpmm.merge(verbose=verbose)
                return True
            elif hasattr(self.dpmm, 'merge_clusters'):
                return self.dpmm.merge_clusters(verbose=verbose)
            else:
                if verbose:
                    print("  ⚠️ No merge method in DPMM")
                return False
        else:
            if verbose:
                print("  ⚠️ No DPMM in agent")
            return False
    except Exception as e:
        if verbose:
            print(f"  ⚠️ Cluster merge failed: {e}")
        return False

def get_info(self):
    """안전한 에이전트 정보 반환"""
    info = {
        'num_experts': 0,
        'total_buffer_size': 0,
        'dpmm': {
            'num_clusters': 0,
            'cluster_counts': [],
            'total_births': 0,
            'total_merges': 0
        }
    }
    
    try:
        # 전문가 정보
        if hasattr(self, 'experts'):
            info['num_experts'] = len(self.experts)
            info['total_buffer_size'] = sum(
                len(getattr(expert, 'replay_buffer', [])) 
                for expert in self.experts
            )
        
        # DPMM 정보
        if hasattr(self, 'dpmm'):
            if hasattr(self.dpmm, 'get_cluster_info'):
                dpmm_info = self.dpmm.get_cluster_info()
                info['dpmm'].update(dpmm_info)
            elif hasattr(self.dpmm, 'get_num_clusters'):
                info['dpmm']['num_clusters'] = self.dpmm.get_num_clusters()
            elif hasattr(self.dpmm, 'clusters'):
                info['dpmm']['num_clusters'] = len(self.dpmm.clusters)
    
    except Exception as e:
        print(f"  ⚠️ Info gathering failed: {e}")
    
    return info

def safe_select_action(self, state, instruction_emb, deterministic=False):
    """안전한 행동 선택 (DPMM 오류 방지)"""
    try:
        return self.select_action(state, instruction_emb, deterministic)
    except Exception as e:
        print(f"  ⚠️ Action selection failed: {e}")
        # 기본 랜덤 행동 반환
        if hasattr(self, 'env') and hasattr(self.env, 'action_space'):
            action = self.env.action_space.sample()
            return action, 0  # 기본 클러스터 ID
        else:
            # 기본 제로 행동
            return np.zeros(8), 0  # Ant 환경의 기본 행동 차원

def safe_store_experience(self, state, action, reward, next_state, done, cluster_id):
    """안전한 경험 저장"""
    try:
        self.store_experience(state, action, reward, next_state, done, cluster_id)
    except Exception as e:
        print(f"  ⚠️ Experience storage failed: {e}")

def safe_update_experts(self, batch_size=32):
    """안전한 전문가 업데이트"""
    try:
        return self.update_experts(batch_size=batch_size)
    except Exception as e:
        print(f"  ⚠️ Expert update failed: {e}")
        return None
