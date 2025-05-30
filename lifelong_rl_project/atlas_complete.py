"""Complete implementation of ATLAS: Attention-based Task Learning And Segregation for Multi-Task Offline Reinforcement Learning"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
import copy
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
import time
import math

# =======================================
# 기본 설정
# =======================================
def get_default_config():
    """Get default configuration"""
    return {
        'env_name': 'Ant-v5',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'task_count': 3,
        'latent_dim': 32,
        'init_experts': 3,
        'pmoe_hidden_dim': 128,
        'dp_concentration': 1.0,
        'cluster_merge_threshold': 1e-4,
        'cluster_min_points': 30,
        'distillation_steps': 50,
        'learning_rate': 3e-4,
        'batch_size': 64,
        'gamma': 0.99,
        'tau': 0.005,
        'buffer_size': 100000,
        'buf_per_expert': 50000,
        'train_freq': 1,
        'dpmm_freq': 500,
        'eval_freq': 1500,
        'total_steps': 10000,
        'lambda_vb': 0.1,
        'lambda_rec': 0.1,
        'lambda_dyn': 0.1,
        'lambda_distill': 0.1,
        'lambda_attention': 0.05,  # 추가: attention loss weight
        'seed': 42
    }

# =======================================
# 1. ENVIRONMENT WRAPPER
# =======================================
class AntTaskWrapper(gym.Wrapper):
    """Ant environment with discrete task sampling and task-specific rewards"""
    
    def __init__(self, env, task_count: int, seed: Optional[int] = None):
        super().__init__(env)
        self.task_count = task_count
        self.current_task = 0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Augment observation space with task one-hot
        orig_shape = env.observation_space.shape[0]
        new_shape = orig_shape + task_count
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_shape,), dtype=np.float32
        )
        
        print(f"🎯 AntTaskWrapper: {task_count} tasks, obs_dim: {orig_shape} → {new_shape}")
    
    def _calculate_task_reward(self, obs, reward, info) -> float:
        """Calculate reward based on the current task"""
        x_position = info.get('x_position', 0)
        y_position = info.get('y_position', 0)
        x_velocity = info.get('x_velocity', 0)
        
        forward_reward = info.get('reward_forward', x_velocity)
        ctrl_cost = info.get('reward_ctrl', 0)
        contact_cost = info.get('reward_contact', 0)
        survive_reward = info.get('reward_survive', 0)
        
        if self.current_task == 0:
            # Task 0: Forward movement
            task_specific_reward = forward_reward + ctrl_cost + contact_cost + survive_reward
        elif self.current_task == 1:
            # Task 1: Backward movement
            backward_reward = -x_velocity
            task_specific_reward = backward_reward + ctrl_cost + contact_cost + survive_reward
        elif self.current_task == 2:
            # Task 2: Stay at origin
            position_target = 0.0
            position_error = abs(x_position - position_target)
            position_reward = 1.0 if position_error < 0.5 else -position_error
            task_specific_reward = position_reward + ctrl_cost + contact_cost + survive_reward
        else:
            task_specific_reward = reward
            
        return float(task_specific_reward)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_task = random.randint(0, self.task_count - 1)
        return self._augment_obs(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        task_reward = self._calculate_task_reward(obs, reward, info)
        return self._augment_obs(obs), task_reward, terminated, truncated, info
    
    def _augment_obs(self, obs):
        """Add task one-hot to observation"""
        task_onehot = np.zeros(self.task_count, dtype=np.float32)
        task_onehot[self.current_task] = 1.0
        return np.concatenate([obs, task_onehot])
    
    def get_task_onehot(self):
        task_onehot = np.zeros(self.task_count, dtype=np.float32)
        task_onehot[self.current_task] = 1.0
        return task_onehot

# =======================================
# 2. SIMPLIFIED DPMM
# =======================================
class DirichletProcessMM:
    """Simplified DPMM for clustering"""
    
    def __init__(self, concentration, latent_dim, device='cpu'):
        self.concentration = concentration
        self.latent_dim = latent_dim
        self.device = device
        self.clusters = {}  # cluster_id -> {'mean': tensor, 'count': int, 'cov': tensor}
        self.next_cluster_id = 0
        print(f"🎯 DPMM initialized: α={concentration}, latent_dim={latent_dim}")
    
    def infer_cluster(self, latent):
        """Assign cluster using Chinese Restaurant Process"""
        if isinstance(latent, torch.Tensor):
            latent = latent.detach().cpu().numpy()
        
        if len(self.clusters) == 0:
            # First point creates first cluster
            self.clusters[0] = {
                'mean': latent.copy(), 
                'count': 1,
                'cov': np.eye(self.latent_dim) * 0.1
            }
            self.next_cluster_id = 1
            return 0
        
        # Compute probabilities for existing clusters
        probs = []
        cluster_ids = list(self.clusters.keys())
        
        for cluster_id in cluster_ids:
            cluster = self.clusters[cluster_id]
            # CRP probability
            crp_prob = cluster['count'] / (sum(c['count'] for c in self.clusters.values()) + self.concentration)
            
            # Gaussian likelihood
            diff = latent - cluster['mean']
            try:
                # Simple Gaussian probability
                dist_sq = np.sum(diff ** 2)
                likelihood = np.exp(-0.5 * dist_sq)
            except:
                likelihood = 0.1
            
            probs.append(crp_prob * likelihood)
        
        # New cluster probability
        new_cluster_prob = self.concentration / (sum(c['count'] for c in self.clusters.values()) + self.concentration)
        probs.append(new_cluster_prob * 0.5)  # Prior likelihood
        
        # Normalize probabilities
        probs = np.array(probs)
        probs = probs / (np.sum(probs) + 1e-8)
        
        # Sample cluster
        chosen_idx = np.random.choice(len(probs), p=probs)
        
        if chosen_idx == len(cluster_ids):
            # Create new cluster
            cluster_id = self.next_cluster_id
            self.clusters[cluster_id] = {
                'mean': latent.copy(), 
                'count': 1,
                'cov': np.eye(self.latent_dim) * 0.1
            }
            self.next_cluster_id += 1
            return cluster_id
        else:
            # Add to existing cluster
            cluster_id = cluster_ids[chosen_idx]
            cluster = self.clusters[cluster_id]
            old_count = cluster['count']
            new_count = old_count + 1
            cluster['mean'] = (old_count * cluster['mean'] + latent) / new_count
            cluster['count'] = new_count
            return cluster_id
    
    def infer_batch(self, latents):
        """Process batch of latents"""
        assignments = []
        for latent in latents:
            if isinstance(latent, torch.Tensor):
                latent = latent.detach().cpu().numpy()
            assignment = self.infer_cluster(latent)
            assignments.append(assignment)
        return torch.tensor(assignments)
    
    def update_memo_vb(self, Z: torch.Tensor) -> torch.Tensor:
        """Simplified VB update"""
        if Z.shape[0] == 0:
            return torch.tensor(0.0)
        
        # Simple ELBO computation
        elbo = torch.tensor(0.0)
        for z in Z:
            cluster_id = self.infer_cluster(z)
            elbo += 1.0  # Simplified ELBO
        
        return elbo / len(Z)
    
    def get_num_clusters(self):
        return len(self.clusters)

# =======================================
# 3. ATTENTION MECHANISM (ATLAS 핵심)
# =======================================
class AttentionGatingNetwork(nn.Module):
    """Attention-based gating network for expert selection"""
    
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 128):
        super().__init__()
        self.num_experts = num_experts
        
        # Attention network
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
        
        # Task segregation network
        self.task_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
    def forward(self, obs: torch.Tensor, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: observations [batch_size, obs_dim]
            latent: latent representations [batch_size, latent_dim]
        
        Returns:
            attention_weights: [batch_size, num_experts]
            task_embedding: [batch_size, hidden_dim//2]
        """
        # Combine observation and latent
        combined = torch.cat([obs, latent], dim=-1)
        
        # Compute attention weights
        attention_logits = self.attention_net(combined)
        attention_weights = F.softmax(attention_logits, dim=-1)
        
        # Compute task embedding for segregation
        task_embedding = self.task_embed(combined)
        
        return attention_weights, task_embedding

# =======================================
# 4. REPLAY BUFFER
# =======================================
class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, latent_dim: int):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.latents = np.zeros((capacity, latent_dim), dtype=np.float32)
        self.task_ids = np.zeros(capacity, dtype=np.int32)
    
    def add(self, obs, action, reward, next_obs, done, latent, task_id):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.latents[self.ptr] = latent.cpu().numpy() if torch.is_tensor(latent) else latent
        self.task_ids[self.ptr] = task_id
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        indices = np.random.choice(self.size, batch_size, replace=True)
        return {
            'observations': torch.FloatTensor(self.observations[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'next_observations': torch.FloatTensor(self.next_observations[indices]),
            'dones': torch.FloatTensor(self.dones[indices]),
            'latents': torch.FloatTensor(self.latents[indices]),
            'task_ids': torch.LongTensor(self.task_ids[indices])
        }

# =======================================
# 5. EXPERT POLICY
# =======================================
class ExpertPolicy(nn.Module):
    """Individual expert policy with SAC components"""
    
    def __init__(self, task_obs_dim: int, action_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.task_obs_dim = task_obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # Actor network (task observation + latent)
        actor_input_dim = task_obs_dim + latent_dim
        self.actor = nn.Sequential(
            nn.Linear(actor_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # mean and log_std
        )
        
        # Critic networks (task observation + action + latent)
        critic_input_dim = task_obs_dim + action_dim + latent_dim
        self.critic1 = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.critic2 = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Target networks
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        
        # Temperature parameter for SAC
        self.log_alpha = nn.Parameter(torch.zeros(1))
    
    def get_action(self, task_obs: torch.Tensor, latent: torch.Tensor, eval_mode: bool = False):
        """Get action from policy"""
        obs_latent = torch.cat([task_obs, latent], dim=-1)
        actor_output = self.actor(obs_latent)
        mean, log_std = actor_output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        if eval_mode:
            action = torch.tanh(mean)
            log_prob = None
        else:
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            
            # Compute log probability
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob, mean, log_std
    
    def get_q_values(self, task_obs: torch.Tensor, action: torch.Tensor, latent: torch.Tensor, target: bool = False):
        """Get Q-values from critics"""
        obs_action_latent = torch.cat([task_obs, action, latent], dim=-1)
        
        if target:
            q1 = self.target_critic1(obs_action_latent)
            q2 = self.target_critic2(obs_action_latent)
        else:
            q1 = self.critic1(obs_action_latent)
            q2 = self.critic2(obs_action_latent)
        
        return q1, q2

# =======================================
# 6. AUXILIARY MODULES
# =======================================
class EmbedReconstructor(nn.Module):
    """Reconstruct full input from latent embedding"""
    
    def __init__(self, latent_dim: int, input_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

class DynamicsPredictor(nn.Module):
    """Predict next latent from current latent and action"""
    
    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        za = torch.cat([z, a], dim=-1)
        return self.predictor(za)

# =======================================
# 7. CLUSTER-EXPERT MANAGER
# =======================================
class ClusterExpertManager:
    """Manage cluster-expert mapping and knowledge distillation"""
    
    def __init__(self, experts: List[ExpertPolicy], dpmm: DirichletProcessMM,
                 merge_threshold: float = 0.5, min_points: int = 30):
        self.experts = experts
        self.dpmm = dpmm
        self.merge_threshold = merge_threshold
        self.min_points = min_points
        
        # Cluster to expert mapping
        self.cluster_to_expert = {}
        self.expert_to_cluster = {}
        
        # Pending distillations
        self.pending_distillations = []
        
        # Initialize mapping
        for i in range(len(experts)):
            self.cluster_to_expert[i] = i
            self.expert_to_cluster[i] = i
    
    def get_expert_for_cluster(self, cluster_id: int) -> int:
        """Get expert index for given cluster"""
        if cluster_id in self.cluster_to_expert:
            return self.cluster_to_expert[cluster_id]
        
        # If cluster doesn't have expert, assign to closest existing expert
        if self.cluster_to_expert:
            return min(self.cluster_to_expert.values())
        return 0
    
    def compute_distillation_loss(self, batch, device):
        """Compute distillation loss for pending distillations"""
        if not self.pending_distillations:
            return torch.tensor(0.0, device=device)
        
        total_distill_loss = torch.tensor(0.0, device=device)
        
        # Process each pending distillation
        for distillation in self.pending_distillations[:]:
            source_idx = distillation['source_expert']
            target_idx = distillation['target_expert']
            
            if (source_idx >= len(self.experts) or target_idx >= len(self.experts) or
                source_idx == target_idx):
                continue
            
            source_expert = self.experts[source_idx]
            target_expert = self.experts[target_idx]
            
            # Move batch to device
            obs = batch['observations'].to(device)
            latents = batch['latents'].to(device)
            
            # Extract task-specific observations (remove task onehot)
            task_obs = obs[:, :-3]  # Remove task onehot (assuming 3 tasks)
            
            # Get outputs from both experts
            with torch.no_grad():
                source_action, source_log_prob, source_mean, source_log_std = source_expert.get_action(task_obs, latents)
            
            target_action, target_log_prob, target_mean, target_log_std = target_expert.get_action(task_obs, latents)
            
            # KL divergence loss between action distributions
            source_std = torch.exp(source_log_std)
            target_std = torch.exp(target_log_std)
            
            # KL(source || target)
            kl_loss = torch.distributions.kl_divergence(
                torch.distributions.Normal(source_mean, source_std),
                torch.distributions.Normal(target_mean, target_std)
            ).sum(dim=-1).mean()
            
            total_distill_loss += kl_loss
            
            # Decrease remaining steps
            distillation['steps_remaining'] -= 1
            
            # Remove completed distillations
            if distillation['steps_remaining'] <= 0:
                self.pending_distillations.remove(distillation)
                print(f"📚 Completed distillation: Expert {source_idx} → {target_idx}")
        
        return total_distill_loss

# =======================================
# 8. ATLAS AGENT
# =======================================
class ATLASAgent:
    """ATLAS: Attention-based Task Learning And Segregation Agent"""
    
    def __init__(self, config: Dict, obs_dim: int, action_dim: int):
        self.config = config
        self.device = torch.device(config['device'])
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = config['latent_dim']
        
        # Task encoder (전체 observation 사용)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.latent_dim)
        ).to(self.device)
        
        # Expert policies (task onehot 제외한 observation 사용)
        task_obs_dim = obs_dim - config['task_count']
        self.experts = nn.ModuleList([
            ExpertPolicy(task_obs_dim, action_dim, self.latent_dim, config['pmoe_hidden_dim'])
            for _ in range(config['init_experts'])
        ]).to(self.device)
        
        # ATLAS Attention Network
        self.attention_net = AttentionGatingNetwork(
            input_dim=obs_dim + self.latent_dim,
            num_experts=len(self.experts),
            hidden_dim=128
        ).to(self.device)
        
        # Expert buffers
        self.expert_buffers = [
            ReplayBuffer(config['buf_per_expert'], obs_dim, action_dim, self.latent_dim)
            for _ in range(config['init_experts'])
        ]
        
        # Auxiliary modules
        self.reconstructor = EmbedReconstructor(self.latent_dim, obs_dim).to(self.device)
        self.dynamics_predictor = DynamicsPredictor(self.latent_dim, action_dim).to(self.device)
        
        # DPMM
        self.dpmm = DirichletProcessMM(
            concentration=config['dp_concentration'],
            latent_dim=self.latent_dim,
            device='cpu'
        )
        
        # Cluster-Expert Manager
        self.manager = ClusterExpertManager(
            experts=self.experts,
            dpmm=self.dpmm,
            merge_threshold=config['cluster_merge_threshold'],
            min_points=config['cluster_min_points']
        )
        
        # Optimizers
        self.expert_optimizers = []
        for expert in self.experts:
            actor_params = list(expert.actor.parameters())
            critic_params = list(expert.critic1.parameters()) + list(expert.critic2.parameters())
            alpha_params = [expert.log_alpha]
            
            self.expert_optimizers.append({
                'actor': optim.Adam(actor_params, lr=config['learning_rate']),
                'critic': optim.Adam(critic_params, lr=config['learning_rate']),
                'alpha': optim.Adam(alpha_params, lr=config['learning_rate'])
            })
        
        # Auxiliary optimizers
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=config['learning_rate'])
        self.reconstructor_optimizer = optim.Adam(self.reconstructor.parameters(), lr=config['learning_rate'])
        self.dynamics_optimizer = optim.Adam(self.dynamics_predictor.parameters(), lr=config['learning_rate'])
        self.attention_optimizer = optim.Adam(self.attention_net.parameters(), lr=config['learning_rate'])
        
        # Target entropy for SAC
        self.target_entropy = -action_dim
        
        print(f"🧠 ATLAS Agent initialized:")
        print(f" Obs: {obs_dim}, Action: {action_dim}, Latent: {self.latent_dim}")
        print(f" Experts: {len(self.experts)}, Tasks: {config['task_count']}")
    
    def select_action(self, obs: np.ndarray, eval_mode: bool = False):
        """Select action using ATLAS attention mechanism"""
        # Extract task-specific observation
        task_obs = obs[:-self.config['task_count']]
        task_onehot = obs[-self.config['task_count']:]
        
        # Encode latent
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent = self.encoder(obs_tensor)
        
        # Get attention weights for expert selection
        with torch.no_grad():
            attention_weights, task_embedding = self.attention_net(obs_tensor, latent)
        
        # Select expert based on attention weights
        if eval_mode:
            expert_id = torch.argmax(attention_weights, dim=-1).item()
        else:
            expert_probs = F.softmax(attention_weights / 0.1, dim=-1)  # Temperature scaling
            expert_id = torch.multinomial(expert_probs, 1).item()
        
        # Get cluster assignment for tracking
        cluster_id = self.dpmm.infer_cluster(latent.squeeze(0).cpu())
        
        # Get action from selected expert
        expert = self.experts[expert_id]
        task_obs_tensor = torch.FloatTensor(task_obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _, _, _ = expert.get_action(task_obs_tensor, latent, eval_mode)
        
        return action.squeeze(0).cpu().numpy(), latent.squeeze(0), cluster_id, expert_id
    
    def add_transition(self, obs, action, reward, next_obs, done, latent, cluster_id, expert_id, task_id):
        """Add transition to appropriate expert buffer"""
        if expert_id < len(self.expert_buffers):
            self.expert_buffers[expert_id].add(obs, action, reward, next_obs, done, latent, task_id)
    
    def train_step(self):
        """Main training step with ATLAS attention mechanism"""
        total_losses = {}
        
        # Train each expert
        for i, expert in enumerate(self.experts):
            if len(self.expert_buffers[i]) >= self.config['batch_size']:
                expert_losses = self._train_expert(i)
                for key, value in expert_losses.items():
                    total_losses[f'expert_{i}_{key}'] = value
        
        # Train auxiliary modules
        aux_losses = self._train_auxiliary_modules()
        total_losses.update(aux_losses)
        
        # Train attention network
        attention_losses = self._train_attention()
        total_losses.update(attention_losses)
        
        return total_losses
    
    def _train_expert(self, expert_idx: int):
        """Train individual expert using SAC"""
        expert = self.experts[expert_idx]
        buffer = self.expert_buffers[expert_idx]
        optimizer = self.expert_optimizers[expert_idx]
        
        # Sample batch
        batch = buffer.sample(self.config['batch_size'])
        obs = torch.FloatTensor(batch['observations']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_obs = torch.FloatTensor(batch['next_observations']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        latents = torch.FloatTensor(batch['latents']).to(self.device)
        
        # Extract task-specific observations
        task_obs = obs[:, :-self.config['task_count']]
        next_task_obs = next_obs[:, :-self.config['task_count']]
        
        # Encode next latents
        with torch.no_grad():
            next_latents = self.encoder(next_obs)
        
        # ===== Critic Update =====
        with torch.no_grad():
            # Target actions
            next_actions, next_log_probs, _, _ = expert.get_action(next_task_obs, next_latents)
            
            # Target Q-values
            target_q1, target_q2 = expert.get_q_values(next_task_obs, next_actions, next_latents, target=True)
            target_q = torch.min(target_q1, target_q2)
            
            # SAC target
            alpha = torch.exp(expert.log_alpha)
            target_q = rewards + (1 - dones) * self.config['gamma'] * (target_q - alpha * next_log_probs)
        
        # Current Q-values
        current_q1, current_q2 = expert.get_q_values(task_obs, actions, latents)
        
        # Critic losses
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = critic1_loss + critic2_loss
        
        # Update critics
        optimizer['critic'].zero_grad()
        critic_loss.backward()
        optimizer['critic'].step()
        
        # ===== Actor Update =====
        # Current actions
        new_actions, log_probs, _, _ = expert.get_action(task_obs, latents)
        
        # Q-values for new actions
        q1_new, q2_new = expert.get_q_values(task_obs, new_actions, latents)
        q_new = torch.min(q1_new, q2_new)
        
        # Actor loss
        alpha = torch.exp(expert.log_alpha)
        actor_loss = (alpha * log_probs - q_new).mean()
        
        # Update actor
        optimizer['actor'].zero_grad()
        actor_loss.backward()
        optimizer['actor'].step()
        
        # ===== Alpha Update =====
        alpha_loss = -(expert.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        optimizer['alpha'].zero_grad()
        alpha_loss.backward()
        optimizer['alpha'].step()
        
        # ===== Soft Update Target Networks =====
        self._soft_update_target(expert.critic1, expert.target_critic1)
        self._soft_update_target(expert.critic2, expert.target_critic2)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha.item()
        }
    
    def _train_auxiliary_modules(self):
        """Train encoder, reconstructor, and dynamics predictor"""
        losses = {}
        
        # Collect data from all expert buffers
        all_obs = []
        all_actions = []
        all_next_obs = []
        all_latents = []
        
        for buffer in self.expert_buffers:
            if len(buffer) >= self.config['batch_size'] // len(self.expert_buffers):
                batch_size = min(self.config['batch_size'] // len(self.expert_buffers), len(buffer))
                batch = buffer.sample(batch_size)
                
                all_obs.append(torch.FloatTensor(batch['observations']))
                all_actions.append(torch.FloatTensor(batch['actions']))
                all_next_obs.append(torch.FloatTensor(batch['next_observations']))
                all_latents.append(torch.FloatTensor(batch['latents']))
        
        if not all_obs:
            return losses
        
        # Concatenate all data
        obs = torch.cat(all_obs, dim=0).to(self.device)
        actions = torch.cat(all_actions, dim=0).to(self.device)
        next_obs = torch.cat(all_next_obs, dim=0).to(self.device)
        latents = torch.cat(all_latents, dim=0).to(self.device)
        
        # ===== Encoder + Reconstruction Loss =====
        encoded_latents = self.encoder(obs)
        reconstructed_obs = self.reconstructor(encoded_latents)
        
        reconstruction_loss = F.mse_loss(reconstructed_obs, obs)
        
        # ===== Dynamics Prediction Loss =====
        next_encoded_latents = self.encoder(next_obs)
        predicted_next_latents = self.dynamics_predictor(encoded_latents, actions)
        
        dynamics_loss = F.mse_loss(predicted_next_latents, next_encoded_latents.detach())
        
        # ===== VB Loss from DPMM =====
        if len(encoded_latents) > 0:
            vb_loss = -self.dpmm.update_memo_vb(encoded_latents.detach().cpu())
            vb_loss = vb_loss.to(self.device) if torch.is_tensor(vb_loss) else torch.tensor(vb_loss, device=self.device)
        else:
            vb_loss = torch.tensor(0.0, device=self.device)
        
        # ===== Total Auxiliary Loss =====
        total_aux_loss = (self.config['lambda_rec'] * reconstruction_loss +
                         self.config['lambda_dyn'] * dynamics_loss +
                         self.config['lambda_vb'] * vb_loss)
        
        # Update encoder and auxiliary modules
        self.encoder_optimizer.zero_grad()
        self.reconstructor_optimizer.zero_grad()
        self.dynamics_optimizer.zero_grad()
        
        total_aux_loss.backward()
        
        self.encoder_optimizer.step()
        self.reconstructor_optimizer.step()
        self.dynamics_optimizer.step()
        
        losses.update({
            'reconstruction_loss': reconstruction_loss.item(),
            'dynamics_loss': dynamics_loss.item(),
            'vb_loss': vb_loss.item() if torch.is_tensor(vb_loss) else vb_loss,
            'total_aux_loss': total_aux_loss.item()
        })
        
        return losses
    
    def _train_attention(self):
        """Train ATLAS attention network"""
        losses = {}
        
        # Collect data from all expert buffers
        all_obs = []
        all_actions = []
        all_rewards = []
        all_expert_ids = []
        
        for expert_id, buffer in enumerate(self.expert_buffers):
            if len(buffer) >= self.config['batch_size'] // len(self.expert_buffers):
                batch_size = min(self.config['batch_size'] // len(self.expert_buffers), len(buffer))
                batch = buffer.sample(batch_size)
                
                all_obs.append(torch.FloatTensor(batch['observations']))
                all_actions.append(torch.FloatTensor(batch['actions']))
                all_rewards.append(torch.FloatTensor(batch['rewards']))
                all_expert_ids.extend([expert_id] * batch_size)
        
        if not all_obs:
            return losses
        
        # Concatenate all data
        obs = torch.cat(all_obs, dim=0).to(self.device)
        actions = torch.cat(all_actions, dim=0).to(self.device)
        rewards = torch.cat(all_rewards, dim=0).to(self.device)
        expert_ids = torch.LongTensor(all_expert_ids).to(self.device)
        
        # Encode latents
        with torch.no_grad():
            latents = self.encoder(obs)
        
        # Get attention weights
        attention_weights, task_embeddings = self.attention_net(obs, latents)
        
        # ===== Attention Supervision Loss =====
        # Cross-entropy loss to match the expert that was actually used
        attention_loss = F.cross_entropy(attention_weights, expert_ids)
        
        # ===== Diversity Regularization =====
        # Encourage diverse attention patterns
        avg_attention = torch.mean(attention_weights, dim=0)
        uniform_dist = torch.ones_like(avg_attention) / len(self.experts)
        diversity_loss = F.kl_div(F.log_softmax(avg_attention, dim=0), uniform_dist, reduction='sum')
        
        # ===== Total Attention Loss =====
        total_attention_loss = attention_loss + 0.01 * diversity_loss
        
        # Update attention network
        self.attention_optimizer.zero_grad()
        total_attention_loss.backward()
        self.attention_optimizer.step()
        
        losses.update({
            'attention_loss': attention_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'total_attention_loss': total_attention_loss.item()
        })
        
        return losses
    
    def _soft_update_target(self, source, target):
        """Soft update target network"""
        tau = self.config['tau']
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
    
    def update_dpmm(self):
        """Update DPMM with recent latent representations"""
        # Collect latents from all buffers
        all_latents = []
        
        for buffer in self.expert_buffers:
            if len(buffer) > 0:
                # Sample some recent latents
                sample_size = min(100, len(buffer))
                recent_batch = buffer.sample(sample_size)
                latents = torch.FloatTensor(recent_batch['latents'])
                all_latents.append(latents)
        
        if all_latents:
            latents_batch = torch.cat(all_latents, dim=0)
            
            # Update DPMM
            elbo = self.dpmm.update_memo_vb(latents_batch)
            
            # Prune small clusters
            removed = self.dpmm.prune_clusters(self.config['cluster_min_points'])
            
            print(f"🎯 DPMM Update: ELBO={elbo:.4f}, Clusters={len(self.dpmm.cluster_means)}, Removed={removed}")
            
            return elbo
        
        return 0.0
    
    def expand_experts(self):
        """Add new expert when new cluster is discovered"""
        current_num_experts = len(self.experts)
        current_num_clusters = len(self.dpmm.cluster_means)
        
        if current_num_clusters > current_num_experts:
            # Add new expert
            task_obs_dim = self.obs_dim - self.config['task_count']
            new_expert = ExpertPolicy(
                task_obs_dim, 
                self.action_dim, 
                self.latent_dim, 
                self.config['pmoe_hidden_dim']
            ).to(self.device)
            
            self.experts.append(new_expert)
            
            # Add new buffer
            new_buffer = ReplayBuffer(
                self.config['buf_per_expert'], 
                self.obs_dim, 
                self.action_dim, 
                self.latent_dim
            )
            self.expert_buffers.append(new_buffer)
            
            # Add new optimizer
            actor_params = list(new_expert.actor.parameters())
            critic_params = list(new_expert.critic1.parameters()) + list(new_expert.critic2.parameters())
            alpha_params = [new_expert.log_alpha]
            
            self.expert_optimizers.append({
                'actor': optim.Adam(actor_params, lr=self.config['learning_rate']),
                'critic': optim.Adam(critic_params, lr=self.config['learning_rate']),
                'alpha': optim.Adam(alpha_params, lr=self.config['learning_rate'])
            })
            
            # Update attention network to handle new expert
            old_attention = self.attention_net
            self.attention_net = AttentionGatingNetwork(
                input_dim=self.obs_dim + self.latent_dim,
                num_experts=len(self.experts),
                hidden_dim=128
            ).to(self.device)
            
            # Copy old weights where possible
            with torch.no_grad():
                # Copy shared parameters
                self.attention_net.shared_layers.load_state_dict(old_attention.shared_layers.state_dict())
                
                # Extend output layer
                old_weight = old_attention.output_layer.weight.data
                old_bias = old_attention.output_layer.bias.data
                
                # Initialize new expert weight randomly
                new_weight = torch.randn(1, old_weight.size(1), device=self.device) * 0.01
                new_bias = torch.zeros(1, device=self.device)
                
                self.attention_net.output_layer.weight.data = torch.cat([old_weight, new_weight], dim=0)
                self.attention_net.output_layer.bias.data = torch.cat([old_bias, new_bias], dim=0)
            
            # Update optimizer
            self.attention_optimizer = optim.Adam(
                self.attention_net.parameters(), 
                lr=self.config['learning_rate']
            )
            
            # Update cluster-expert mapping
            new_expert_id = len(self.experts) - 1
            new_cluster_id = current_num_clusters - 1
            self.manager.cluster_to_expert[new_cluster_id] = new_expert_id
            self.manager.expert_to_cluster[new_expert_id] = new_cluster_id
            
            print(f"🚀 Added Expert {new_expert_id} for Cluster {new_cluster_id}")
            print(f"   Total Experts: {len(self.experts)}, Total Clusters: {current_num_clusters}")
            
            return True
        
        return False
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'config': self.config,
            'encoder_state_dict': self.encoder.state_dict(),
            'experts_state_dict': [expert.state_dict() for expert in self.experts],
            'attention_state_dict': self.attention_net.state_dict(),
            'reconstructor_state_dict': self.reconstructor.state_dict(),
            'dynamics_state_dict': self.dynamics_predictor.state_dict(),
            'dpmm_state': {
                'cluster_means': [m.cpu() for m in self.dpmm.cluster_means],
                'cluster_covs': [c.cpu() for c in self.dpmm.cluster_covs],
                'cluster_counts': self.dpmm.cluster_counts.copy(),
                'cluster_weights': self.dpmm.cluster_weights.copy(),
                'alpha': self.dpmm.alpha,
                'memo_elbo': self.dpmm.memo_elbo.cpu() if torch.is_tensor(self.dpmm.memo_elbo) else self.dpmm.memo_elbo
            },
            'cluster_expert_mapping': {
                'cluster_to_expert': self.manager.cluster_to_expert.copy(),
                'expert_to_cluster': self.manager.expert_to_cluster.copy()
            }
        }
        torch.save(checkpoint, filepath)
        print(f"💾 Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load basic modules
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.reconstructor.load_state_dict(checkpoint['reconstructor_state_dict'])
        self.dynamics_predictor.load_state_dict(checkpoint['dynamics_state_dict'])
        
        # Load experts
        for i, expert_state in enumerate(checkpoint['experts_state_dict']):
            if i < len(self.experts):
                self.experts[i].load_state_dict(expert_state)
        
        # Load attention network
        self.attention_net.load_state_dict(checkpoint['attention_state_dict'])
        
        # Load DPMM state
        dpmm_state = checkpoint['dpmm_state']
        self.dpmm.cluster_means = [m.to(self.dpmm.device) for m in dpmm_state['cluster_means']]
        self.dpmm.cluster_covs = [c.to(self.dpmm.device) for c in dpmm_state['cluster_covs']]
        self.dpmm.cluster_counts = dpmm_state['cluster_counts']
        self.dpmm.cluster_weights = dpmm_state['cluster_weights']
        self.dpmm.alpha = dpmm_state['alpha']
        self.dpmm.memo_elbo = torch.tensor(dpmm_state['memo_elbo'], device=self.dpmm.device)
        
        # Load mappings
        mapping = checkpoint['cluster_expert_mapping']
        self.manager.cluster_to_expert = mapping['cluster_to_expert']
        self.manager.expert_to_cluster = mapping['expert_to_cluster']
        
        print(f"📁 Checkpoint loaded: {filepath}")
        print(f"   Experts: {len(self.experts)}, Clusters: {len(self.dpmm.cluster_means)}")

# =======================================
# 9. MULTI-TASK ENVIRONMENT WRAPPER
# =======================================
class MultiTaskWrapper:
    """Wrapper for multi-task environments with task switching"""
    
    def __init__(self, env_name: str, task_count: int):
        self.env_name = env_name
        self.task_count = task_count
        self.current_task = 0
        
        # Create base environment
        self.env = gym.make(env_name)
        self.base_obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        # Extended observation space (base + task onehot)
        self.obs_dim = self.base_obs_dim + task_count
        
        print(f"🌍 MultiTask Environment: {env_name}")
        print(f"   Base obs: {self.base_obs_dim}, Extended obs: {self.obs_dim}")
        print(f"   Action: {self.action_dim}, Tasks: {task_count}")
    
    def reset(self, task_id: int = None):
        """Reset environment with specific task"""
        if task_id is not None:
            self.current_task = task_id
        
        base_obs = self.env.reset()
        if isinstance(base_obs, tuple):
            base_obs = base_obs[0]  # Handle gym API changes
        
        # Add task onehot encoding
        task_onehot = np.zeros(self.task_count)
        task_onehot[self.current_task] = 1.0
        
        extended_obs = np.concatenate([base_obs, task_onehot])
        return extended_obs
    
    def step(self, action):
        """Step environment and add task information"""
        next_base_obs, reward, done, info = self.env.step(action)
        
        # Handle gym API changes
        if len(self.env.step(action)) == 5:  # New gym API
            next_base_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
        
        # Add task onehot encoding
        task_onehot = np.zeros(self.task_count)
        task_onehot[self.current_task] = 1.0
        
        extended_next_obs = np.concatenate([next_base_obs, task_onehot])
        
        # Modify reward based on task
        modified_reward = self._modify_reward(reward, self.current_task)
        
        return extended_next_obs, modified_reward, done, info
    
    def _modify_reward(self, base_reward: float, task_id: int) -> float:
        """Modify reward based on current task"""
        # Task-specific reward modifications
        if task_id == 0:  # Task 0: Original reward
            return base_reward
        elif task_id == 1:  # Task 1: Encourage forward movement
            return base_reward + 0.1  # Small bonus
        elif task_id == 2:  # Task 2: Penalize fast movement
            return base_reward * 0.8  # Reduce reward
        else:
            return base_reward
    
    def switch_task(self, task_id: int):
        """Switch to different task"""
        if 0 <= task_id < self.task_count:
            self.current_task = task_id
            print(f"🔄 Switched to Task {task_id}")
        else:
            print(f"⚠️ Invalid task ID: {task_id}")
    
    def get_task_id(self) -> int:
        """Get current task ID"""
        return self.current_task

# =======================================
# 10. TRAINING LOOP
# =======================================
def train_atlas(config: Dict):
    """Main training loop for ATLAS"""
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Create environment
    env = MultiTaskWrapper(config['env_name'], config['task_count'])
    
    # Create agent
    agent = ATLASAgent(config, env.obs_dim, env.action_dim)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    task_performance = {i: [] for i in range(config['task_count'])}
    
    # Training state
    total_steps = 0
    episode_count = 0
    
    print("🚀 Starting ATLAS Training...")
    print(f"📊 Config: {config}")
    
    while total_steps < config['total_steps']:
        # Task curriculum: cycle through tasks
        current_task = episode_count % config['task_count']
        
        # Reset environment
        obs = env.reset(current_task)
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\n📝 Episode {episode_count + 1}, Task {current_task}, Steps {total_steps}")
        
        while not done and total_steps < config['total_steps']:
            # Select action
            action, latent, cluster_id, expert_id = agent.select_action(obs, eval_mode=False)
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            
            # Store transition
            agent.add_transition(
                obs, action, reward, next_obs, done, 
                latent.cpu().numpy(), cluster_id, expert_id, current_task
            )
            
            # Update state
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            # Training
            if total_steps % config['train_freq'] == 0:
                train_losses = agent.train_step()
                
                if total_steps % (config['train_freq'] * 10) == 0 and train_losses:
                    print(f"🎯 Step {total_steps}: {train_losses}")
            
            # DPMM update
            if total_steps % config['dpmm_freq'] == 0:
                elbo = agent.update_dpmm()
                
                # Check for expert expansion
                expanded = agent.expand_experts()
                if expanded:
                    print(f"🆕 Expert expansion at step {total_steps}")
            
            # Evaluation
            if total_steps % config['eval_freq'] == 0:
                eval_results = evaluate_atlas(agent, env, config)
                print(f"📈 Evaluation at step {total_steps}: {eval_results}")
                
                # Save checkpoint
                checkpoint_path = f"atlas_checkpoint_step_{total_steps}.pth"
                agent.save_checkpoint(checkpoint_path)
        
        # Episode finished
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        task_performance[current_task].append(episode_reward)
        episode_count += 1
        
        print(f"✅ Episode {episode_count} finished: Reward={episode_reward:.2f}, Length={episode_length}")
        
        # Print statistics
        if episode_count % 10 == 0:
            recent_rewards = episode_rewards[-10:]
            avg_reward = np.mean(recent_rewards)
            print(f"📊 Last 10 episodes: Avg Reward={avg_reward:.2f}")
            
            # DPMM statistics
            dpmm_stats = agent.dpmm.get_cluster_statistics()
            print(f"🎯 DPMM Stats: {dpmm_stats}")
    
    print("🎉 Training completed!")
    
    # Final evaluation
    final_results = evaluate_atlas(agent, env, config, num_episodes=10)
    print(f"🏆 Final Results: {final_results}")
    
    # Save final model
    agent.save_checkpoint("atlas_final_model.pth")
    
    return agent, {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'task_performance': task_performance,
        'final_results': final_results
    }

def evaluate_atlas(agent: ATLASAgent, env: MultiTaskWrapper, config: Dict, num_episodes: int = 5):
    """Evaluate ATLAS agent across all tasks"""
    results = {}
    
    for task_id in range(config['task_count']):
        task_rewards = []
        task_lengths = []
        
        for episode in range(num_episodes):
            obs = env.reset(task_id)
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, latent, cluster_id, expert_id = agent.select_action(obs, eval_mode=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if episode_length > 1000:  # Prevent infinite episodes
                    break
            
            task_rewards.append(episode_reward)
            task_lengths.append(episode_length)
        
        results[f'task_{task_id}'] = {
            'mean_reward': np.mean(task_rewards),
            'std_reward': np.std(task_rewards),
            'mean_length': np.mean(task_lengths)
        }
    
    # Overall performance
    all_rewards = [results[f'task_{i}']['mean_reward'] for i in range(config['task_count'])]
    results['overall'] = {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards)
    }
    
    return results

# =======================================
# 11. MAIN EXECUTION
# =======================================
if __name__ == "__main__":
    from lifelong_rl.config import get_default_config
    
    
    # Get configuration
    config = get_default_config()
    
    # Modify config for testing
    config.update({
        'total_steps': 50000,
        'eval_freq': 2500,
        'dpmm_freq': 1000,
        'env_name': 'Ant-v4',  # Use v4 for compatibility
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })
    
    print("🧠 ATLAS - Attention-based Task Learning And Segregation")
    print("=" * 60)
    
    try:
        # Train agent
        agent, training_history = train_atlas(config)
        
        print("\n✨ Training Summary:")
        print(f"📈 Total Episodes: {len(training_history['episode_rewards'])}")
        print(f"🎯 Average Reward: {np.mean(training_history['episode_rewards'][-50:]):.2f}")
        print(f"🏆 Final Performance: {training_history['final_results']['overall']['mean_reward']:.2f}")
        print(f"🧠 Total Experts: {len(agent.experts)}")
        print(f"🎯 Total Clusters: {len(agent.dpmm.cluster_means)}")
        
        # Task-specific performance
        print("\n📊 Task-specific Performance:")
        for task_id in range(config['task_count']):
            if f'task_{task_id}' in training_history['final_results']:
                task_result = training_history['final_results'][f'task_{task_id}']
                print(f"   Task {task_id}: {task_result['mean_reward']:.2f} ± {task_result['std_reward']:.2f}")
        
        # Save training history
        import json
        import pickle
        
        # Save as JSON (for metrics)
        history_json = {
            'episode_rewards': training_history['episode_rewards'],
            'episode_lengths': training_history['episode_lengths'],
            'final_results': training_history['final_results'],
            'config': config
        }
        
        with open('atlas_training_history.json', 'w') as f:
            json.dump(history_json, f, indent=2)
        
        # Save full history as pickle (for task_performance dict)
        with open('atlas_training_history.pkl', 'wb') as f:
            pickle.dump(training_history, f)
        
        print("💾 Training history saved!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n🔚 ATLAS training session ended")

# =======================================
# 12. ADDITIONAL UTILITIES
# =======================================
def visualize_attention_weights(agent: ATLASAgent, env: MultiTaskWrapper, config: Dict):
    """Visualize attention weights across different tasks"""
    import matplotlib.pyplot as plt
    
    attention_data = {f'task_{i}': [] for i in range(config['task_count'])}
    
    # Collect attention weights for each task
    for task_id in range(config['task_count']):
        obs = env.reset(task_id)
        
        for _ in range(100):  # Collect 100 samples per task
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                latent = agent.encoder(obs_tensor)
                attention_weights, _ = agent.attention_net(obs_tensor, latent)
                attention_weights = F.softmax(attention_weights, dim=-1)
            
            attention_data[f'task_{task_id}'].append(attention_weights.cpu().numpy().flatten())
            
            # Take random action and step
            action = env.env.action_space.sample()
            obs, _, done, _ = env.step(action)
            
            if done:
                obs = env.reset(task_id)
    
    # Plot attention weights
    fig, axes = plt.subplots(1, config['task_count'], figsize=(5*config['task_count'], 4))
    if config['task_count'] == 1:
        axes = [axes]
    
    for task_id in range(config['task_count']):
        attention_matrix = np.array(attention_data[f'task_{task_id}'])
        mean_attention = np.mean(attention_matrix, axis=0)
        std_attention = np.std(attention_matrix, axis=0)
        
        expert_ids = list(range(len(agent.experts)))
        axes[task_id].bar(expert_ids, mean_attention, yerr=std_attention, alpha=0.7)
        axes[task_id].set_title(f'Task {task_id} Attention')
        axes[task_id].set_xlabel('Expert ID')
        axes[task_id].set_ylabel('Attention Weight')
        axes[task_id].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('atlas_attention_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Attention visualization saved as 'atlas_attention_visualization.png'")

def analyze_cluster_expert_mapping(agent: ATLASAgent):
    """Analyze the mapping between clusters and experts"""
    print("\n🔍 Cluster-Expert Mapping Analysis:")
    print("=" * 50)
    
    # DPMM statistics
    dpmm_stats = agent.dpmm.get_cluster_statistics()
    print(f"📊 DPMM Statistics:")
    print(f"   Total Clusters: {dpmm_stats['num_clusters']}")
    print(f"   Total Points: {dpmm_stats['total_points']}")
    print(f"   Concentration (α): {dpmm_stats['alpha']}")
    print(f"   Latest ELBO: {dpmm_stats['latest_elbo']:.4f}")
    
    # Cluster details
    print(f"\n🎯 Cluster Details:")
    for i, (count, weight) in enumerate(zip(dpmm_stats['cluster_counts'], dpmm_stats['cluster_weights'])):
        expert_id = agent.manager.get_expert_for_cluster(i)
        print(f"   Cluster {i}: {count} points ({weight:.3f} weight) → Expert {expert_id}")
    
    # Expert utilization
    print(f"\n🧠 Expert Utilization:")
    expert_usage = {}
    for cluster_id, expert_id in agent.manager.cluster_to_expert.items():
        if expert_id not in expert_usage:
            expert_usage[expert_id] = []
        expert_usage[expert_id].append(cluster_id)
    
    for expert_id in range(len(agent.experts)):
        clusters = expert_usage.get(expert_id, [])
        buffer_size = len(agent.expert_buffers[expert_id])
        print(f"   Expert {expert_id}: Clusters {clusters}, Buffer size: {buffer_size}")
    
    return dpmm_stats, expert_usage

def test_atlas_generalization(agent: ATLASAgent, env: MultiTaskWrapper, config: Dict):
    """Test ATLAS generalization to unseen task variations"""
    print("\n🧪 Testing ATLAS Generalization...")
    
    # Test with different reward modifications
    original_modify_reward = env._modify_reward
    
    def test_reward_modification(base_reward, task_id):
        """Alternative reward function for testing"""
        if task_id == 0:  # Make task 0 more challenging
            return base_reward * 0.5
        elif task_id == 1:  # Different bonus structure
            return base_reward + 0.2
        elif task_id == 2:  # Completely different objective
            return -abs(base_reward)  # Minimize reward instead of maximize
        return base_reward
    
    # Test generalization
    generalization_results = {}
    
    # Original tasks
    env._modify_reward = original_modify_reward
    original_results = evaluate_atlas(agent, env, config, num_episodes=5)
    generalization_results['original'] = original_results
    
    # Modified tasks
    env._modify_reward = test_reward_modification
    modified_results = evaluate_atlas(agent, env, config, num_episodes=5)
    generalization_results['modified'] = modified_results
    
    # Restore original reward function
    env._modify_reward = original_modify_reward
    
    # Print results
    print("📊 Generalization Results:")
    print("Original Tasks:")
    for task_id in range(config['task_count']):
        orig_reward = original_results[f'task_{task_id}']['mean_reward']
        print(f"   Task {task_id}: {orig_reward:.2f}")
    
    print("Modified Tasks:")
    for task_id in range(config['task_count']):
        mod_reward = modified_results[f'task_{task_id}']['mean_reward']
        print(f"   Task {task_id}: {mod_reward:.2f}")
    
    return generalization_results

def benchmark_atlas_performance():
    """Benchmark ATLAS against different configurations"""
    print("\n⚡ Benchmarking ATLAS Performance...")
    
    base_config = get_default_config()
    base_config['total_steps'] = 10000  # Shorter for benchmarking
    
    configurations = [
        {'name': 'Standard', 'changes': {}},
        {'name': 'High Concentration', 'changes': {'dp_concentration': 5.0}},
        {'name': 'Low Concentration', 'changes': {'dp_concentration': 0.1}},
        {'name': 'More Experts', 'changes': {'init_experts': 5}},
        {'name': 'Larger Latent', 'changes': {'latent_dim': 64}},
        {'name': 'No Attention', 'changes': {'lambda_vb': 0.0}}  # Disable DPMM influence
    ]
    
    benchmark_results = {}
    
    for config_info in configurations:
        print(f"\n🔧 Testing: {config_info['name']}")
        
        # Create modified config
        test_config = base_config.copy()
        test_config.update(config_info['changes'])
        
        try:
            # Quick training
            env = MultiTaskWrapper(test_config['env_name'], test_config['task_count'])
            agent = ATLASAgent(test_config, env.obs_dim, env.action_dim)
            
            # Run shortened training
            _, history = train_atlas(test_config)
            
            # Evaluate
            final_performance = history['final_results']['overall']['mean_reward']
            benchmark_results[config_info['name']] = {
                'performance': final_performance,
                'num_experts': len(agent.experts),
                'num_clusters': len(agent.dpmm.cluster_means)
            }
            
            print(f"   Performance: {final_performance:.2f}")
            print(f"   Experts: {len(agent.experts)}, Clusters: {len(agent.dpmm.cluster_means)}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            benchmark_results[config_info['name']] = {'performance': 0.0, 'error': str(e)}
    
    # Print benchmark summary
    print("\n📈 Benchmark Summary:")
    print("=" * 50)
    sorted_results = sorted(benchmark_results.items(), 
                          key=lambda x: x[1].get('performance', 0), reverse=True)
    
    for name, result in sorted_results:
        if 'error' not in result:
            print(f"{name:15}: {result['performance']:6.2f} "
                  f"(E:{result['num_experts']}, C:{result['num_clusters']})")
        else:
            print(f"{name:15}: FAILED")
    
    return benchmark_results

# =======================================
# 13. DEMO AND TESTING
# =======================================
def run_atlas_demo():
    """Run a quick demo of ATLAS"""
    print("🎮 ATLAS Demo Mode")
    print("=" * 30)
    
    # Quick config for demo
    config = get_default_config()
    config.update({
        'total_steps': 5000,
        'eval_freq': 1000,
        'dpmm_freq': 500,
        'init_experts': 2,
        'task_count': 2,  # Reduce for faster demo
        'env_name': 'Ant-v4'
    })
    
    # Run training
    agent, history = train_atlas(config)
    
    # Quick analysis
    analyze_cluster_expert_mapping(agent)
    
    print("\n✅ Demo completed!")
    return agent, history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ATLAS: Attention-based Task Learning And Segregation')
    parser.add_argument('--mode', choices=['train', 'demo', 'benchmark'], default='train',
                       help='Execution mode')
    parser.add_argument('--config', type=str, help='Config file path (optional)')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to load (optional)')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        agent, history = run_atlas_demo()
        
    elif args.mode == 'benchmark':
        benchmark_results = benchmark_atlas_performance()
        
    else:  # train mode
        config = get_default_config()
        
        # Load custom config if provided
        if args.config:
            import json
            with open(args.config, 'r') as f:
                custom_config = json.load(f)
            config.update(custom_config)
        
        # Train agent
        agent, history = train_atlas(config)
        
        # Load checkpoint if provided
        if args.checkpoint:
            agent.load_checkpoint(args.checkpoint)
        
        # Create visualizations if requested
        if args.visualize:
            env = MultiTaskWrapper(config['env_name'], config['task_count'])
            visualize_attention_weights(agent, env, config)
            analyze_cluster_expert_mapping(agent)
            test_atlas_generalization(agent, env, config)

