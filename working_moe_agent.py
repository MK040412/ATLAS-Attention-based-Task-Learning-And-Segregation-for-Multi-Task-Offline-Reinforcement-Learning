import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, Counter
from working_sac_expert import WorkingSACExpert

class WorkingFusionEncoder(nn.Module):
    def __init__(self, instr_dim, state_dim, latent_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(instr_dim + state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, instr_emb, state_obs):
        # 안전한 차원 처리
        if instr_emb.dim() > 1:
            instr_emb = instr_emb.mean(dim=0) if instr_emb.dim() == 2 else instr_emb.squeeze()
        if state_obs.dim() > 1:
            state_obs = state_obs.squeeze()
        
        # 배치 차원 추가
        if instr_emb.dim() == 1:
            instr_emb = instr_emb.unsqueeze(0)
        if state_obs.dim() == 1:
            state_obs = state_obs.unsqueeze(0)
        
        combined = torch.cat([instr_emb, state_obs], dim=-1)
        return self.mlp(combined).squeeze(0)

class WorkingDPMM:
    def __init__(self, latent_dim, alpha=1.0):
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.cluster_data = defaultdict(list)
        self.cluster_centers = {}
        self.cluster_counts = Counter()
        self.next_cluster_id = 0
    
    def assign_and_update(self, point):
        point = np.array(point).flatten()
        
        if len(self.cluster_centers) == 0:
            # 첫 번째 클러스터 생성
            cluster_id = self.next_cluster_id
            self.cluster_centers[cluster_id] = point.copy()
            self.cluster_data[cluster_id].append(point)
            self.cluster_counts[cluster_id] = 1
            self.next_cluster_id += 1
            return cluster_id
        
        # 기존 클러스터와의 거리 계산
        distances = {}
        for cid, center in self.cluster_centers.items():
            distances[cid] = np.linalg.norm(point - center)
        
        # 가장 가까운 클러스터 찾기
        closest_cluster = min(distances, key=distances.get)
        min_distance = distances[closest_cluster]
        
        # 임계값 기반 할당
        threshold = 2.0
        if min_distance < threshold:
            # 기존 클러스터에 할당
            self.cluster_data[closest_cluster].append(point)
            self.cluster_counts[closest_cluster] += 1
            # 중심 업데이트
            self.cluster_centers[closest_cluster] = np.mean(self.cluster_data[closest_cluster], axis=0)
            return closest_cluster
        else:
            # 새 클러스터 생성
            cluster_id = self.next_cluster_id
            self.cluster_centers[cluster_id] = point.copy()
            self.cluster_data[cluster_id].append(point)
            self.cluster_counts[cluster_id] = 1
            self.next_cluster_id += 1
            return cluster_id
    
    def get_cluster_info(self):
        return {
            'num_clusters': len(self.cluster_centers),
            'cluster_counts': dict(self.cluster_counts)
        }

class WorkingLanguageProcessor:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print("✅ Language processor initialized")
        except:
            print("⚠️ SentenceTransformer not available, using dummy embeddings")
            self.model = None
    
    def encode(self, text):
        if self.model is not None:
            embedding = self.model.encode(text)
            return torch.FloatTensor(embedding).to(self.device)
        else:
            # 더미 임베딩
            return torch.randn(384).to(self.device)

class WorkingMoEAgent:
    def __init__(self, state_dim, action_dim, instr_dim=384, latent_dim=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.instr_dim = instr_dim
        self.latent_dim = latent_dim
        
        # 컴포넌트 초기화
        self.fusion_encoder = WorkingFusionEncoder(
            instr_dim, state_dim, latent_dim
        ).to(self.device)
        
        self.dpmm = WorkingDPMM(latent_dim)
        self.experts = []
        self.expert_usage = Counter()
        
        print(f"✅ Working MoE Agent initialized:")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Instruction dim: {instr_dim}")
        print(f"  Latent dim: {latent_dim}")
    
    def _create_expert(self):
        expert = WorkingSACExpert(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        self.experts.append(expert)
        print(f"🧠 Created expert {len(self.experts)-1}")
        return expert
    
    def select_action(self, state, instruction_emb, deterministic=False):
        # 텐서 변환
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).to(self.device)
        if not torch.is_tensor(instruction_emb):
            instruction_emb = torch.FloatTensor(instruction_emb).to(self.device)
        
        # 융합된 특징 추출
        with torch.no_grad():
            latent_state = self.fusion_encoder(instruction_emb, state)
        
        # DPMM으로 클러스터 할당
        cluster_id = self.dpmm.assign_and_update(latent_state.cpu().numpy())
        
        # 전문가가 없으면 생성
        while cluster_id >= len(self.experts):
            self._create_expert()
        
        # 전문가로부터 행동 선택
        expert = self.experts[cluster_id]
        action = expert.select_action(state.cpu().numpy(), deterministic)
        
        # 사용량 기록
        self.expert_usage[cluster_id] += 1
        
        return action, cluster_id
    
    def update_expert(self, expert_id, state, action, reward, next_state, done):
        if expert_id < len(self.experts):
            expert = self.experts[expert_id]
            expert.replay_buffer.add(state, action, reward, next_state, done)
            return expert.update()
        return None
    
    def get_stats(self):
        cluster_info = self.dpmm.get_cluster_info()
        return {
            'num_experts': len(self.experts),
            'num_clusters': cluster_info['num_clusters'],
            'cluster_counts': cluster_info['cluster_counts'],
            'expert_usage': dict(self.expert_usage)
        }