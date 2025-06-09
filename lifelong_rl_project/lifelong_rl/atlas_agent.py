import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .dpmm import DirichletProcessMM
from .networks import PolicyNetwork, AttentionMechanism, ValueNetwork
from .replay_buffer import ExpertReplayBuffer

class ATLASAgent:
    """ATLAS: Attention-based Task Learning And Segregation for Multi-Task Offline RL"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Environment parameters
        self.state_dim = config.get('state_dim', 17)  # Default for Ant-v5
        self.action_dim = config.get('action_dim', 8)
        self.latent_dim = config['latent_dim']
        
        # DPMM for task clustering
        self.dpmm = DirichletProcessMM(
            concentration=config['dp_concentration'],
            latent_dim=self.latent_dim,
            device=config['device']
        )
        
        # Expert networks (Parametric Mixture of Experts)
        self.experts = nn.ModuleList()
        self.expert_optimizers = []
        
        # Attention mechanism for expert selection
        self.attention = AttentionMechanism(
            state_dim=self.state_dim,
            latent_dim=self.latent_dim,
            hidden_dim=config['pmoe_hidden_dim']
        ).to(self.device)
        
        # Value networks for each expert
        self.value_networks = nn.ModuleList()
        self.value_optimizers = []
        
        # Replay buffers per expert
        self.expert_buffers = {}
        
        # Initialize experts
        for _ in range(config['init_experts']):
            self._create_new_expert()
            
        # Attention optimizer
        self.attention_optimizer = torch.optim.Adam(
            self.attention.parameters(), 
            lr=config['learning_rate']
        )
        
        # Training statistics
        self.training_stats = defaultdict(list)
        self.current_step = 0
        
        print(f"🚀 ATLAS Agent initialized with {len(self.experts)} experts")

    def _create_new_expert(self) -> int:
        """Create a new expert policy and value network"""
        expert_id = len(self.experts)
        
        # Policy network
        policy = PolicyNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config['pmoe_hidden_dim'],
            latent_dim=self.latent_dim
        ).to(self.device)
        
        # Value network
        value = ValueNetwork(
            state_dim=self.state_dim,
            hidden_dim=self.config['pmoe_hidden_dim']
        ).to(self.device)
        
        self.experts.append(policy)
        self.value_networks.append(value)
        
        # Optimizers
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=self.config['learning_rate'])
        value_optimizer = torch.optim.Adam(value.parameters(), lr=self.config['learning_rate'])
        
        self.expert_optimizers.append(policy_optimizer)
        self.value_optimizers.append(value_optimizer)
        
        # Replay buffer
        self.expert_buffers[expert_id] = ExpertReplayBuffer(
            capacity=self.config['buf_per_expert'],
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
        print(f"✨ Created new expert {expert_id}")
        return expert_id

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict]:
        """Select action using attention-weighted expert ensemble"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get latent representation and attention weights
            latent_z, attention_weights = self.attention(state_tensor)
            
            # Infer cluster assignment (for tracking)
            cluster_id = self.dpmm.infer_cluster(latent_z.squeeze(0))
            
            # Get actions from all experts
            expert_actions = []
            expert_log_probs = []
            
            for expert in self.experts:
                if deterministic:
                    action, log_prob = expert.get_action_deterministic(state_tensor)
                else:
                    action, log_prob = expert.get_action(state_tensor)
                expert_actions.append(action)
                expert_log_probs.append(log_prob)
            
            # Weighted combination of expert actions
            if expert_actions:
                expert_actions = torch.stack(expert_actions, dim=1)  # [1, num_experts, action_dim]
                attention_weights = attention_weights.unsqueeze(-1)  # [1, num_experts, 1]
                
                final_action = torch.sum(attention_weights * expert_actions, dim=1)
                final_action = final_action.squeeze(0).cpu().numpy()
            else:
                # Fallback if no experts
                final_action = np.random.randn(self.action_dim) * 0.1
        
        info = {
            'cluster_id': cluster_id,
            'attention_weights': attention_weights.squeeze(0).cpu().numpy(),
            'latent_z': latent_z.squeeze(0).cpu().numpy(),
            'num_experts': len(self.experts)
        }
        
        return final_action, info

    def update(self, batch_data: Dict) -> Dict:
        """Update ATLAS agent with a batch of experience"""
        self.current_step += 1
        losses = {}
        
        # Extract batch data
        states = batch_data['states'].to(self.device)
        actions = batch_data['actions'].to(self.device)
        rewards = batch_data['rewards'].to(self.device)
        next_states = batch_data['next_states'].to(self.device)
        dones = batch_data['dones'].to(self.device)
        
        batch_size = states.shape[0]
        
        # 1. Update attention mechanism and get latent representations
        latent_z, attention_weights = self.attention(states)
        
        # 2. Update DPMM with latent representations (periodic)
        if self.current_step % self.config['dpmm_freq'] == 0:
            elbo = self.dpmm.update_memo_vb(latent_z.detach())
            losses['dpmm_elbo'] = elbo.item()
        
        # 3. Assign experiences to experts based on clustering
        cluster_assignments = self.dpmm.infer_batch(latent_z.detach())
        
        # Create new experts if needed
        max_cluster = cluster_assignments.max().item() if len(cluster_assignments) > 0 else 0
        while len(self.experts) <= max_cluster:
            self._create_new_expert()
        
        # 4. Store experiences in expert-specific buffers
        for i in range(batch_size):
            expert_id = cluster_assignments[i].item()
            if expert_id < len(self.expert_buffers):
                self.expert_buffers[expert_id].add(
                    state=states[i].cpu().numpy(),
                    action=actions[i].cpu().numpy(),
                    reward=rewards[i].cpu().numpy(),
                    next_state=next_states[i].cpu().numpy(),
                    done=dones[i].cpu().numpy()
                )
        
        # 5. Update expert policies
        expert_losses = self._update_experts(states, actions, rewards, next_states, dones, 
                                           attention_weights, cluster_assignments)
        losses.update(expert_losses)
        
        # 6. Update attention mechanism
        attention_loss = self._update_attention(states, actions, latent_z, attention_weights)
        losses['attention_loss'] = attention_loss.item()
        
        # 7. Knowledge distillation (periodic)
        if self.current_step % self.config['distillation_steps'] == 0:
            distill_loss = self._knowledge_distillation()
            losses['distillation_loss'] = distill_loss.item()
        
        # Update statistics
        self._update_training_stats(losses)
        
        return losses

    def _update_experts(self, states, actions, rewards, next_states, dones, 
                       attention_weights, cluster_assignments) -> Dict:
        """Update expert policies using SAC-style updates"""
        losses = {}
        
        for expert_id in range(len(self.experts)):
            # Get mask for this expert
            expert_mask = (cluster_assignments == expert_id)
            if not expert_mask.any():
                continue
                
            # Expert-specific data
            expert_states = states[expert_mask]
            expert_actions = actions[expert_mask]
            expert_rewards = rewards[expert_mask]
            expert_next_states = next_states[expert_mask]
            expert_dones = dones[expert_mask]
            expert_weights = attention_weights[expert_mask, expert_id:expert_id+1]
            
            if len(expert_states) == 0:
                continue
            
            # Policy loss (weighted by attention)
            pred_actions, log_probs = self.experts[expert_id].get_action(expert_states)
            policy_loss = -torch.mean(expert_weights * log_probs)
            
            # Value loss
            pred_values = self.value_networks[expert_id](expert_states)
            target_values = expert_rewards + self.config['gamma'] * \
                           (1 - expert_dones) * self.value_networks[expert_id](expert_next_states).detach()
            value_loss = F.mse_loss(pred_values.squeeze(), target_values.squeeze())
            
            # Update expert
            self.expert_optimizers[expert_id].zero_grad()
            policy_loss.backward(retain_graph=True)
            self.expert_optimizers[expert_id].step()
            
            # Update value network
            self.value_optimizers[expert_id].zero_grad()
            value_loss.backward()
            self.value_optimizers[expert_id].step()
            
            losses[f'expert_{expert_id}_policy_loss'] = policy_loss.item()
            losses[f'expert_{expert_id}_value_loss'] = value_loss.item()
        
        return losses

    def _update_attention(self, states, actions, latent_z, attention_weights) -> torch.Tensor:
        """Update attention mechanism"""
        # Reconstruction loss: attention should help reconstruct state features
        reconstructed_states = torch.zeros_like(states)
        
        for expert_id in range(len(self.experts)):
            if expert_id < attention_weights.shape[1]:
                expert_contribution = self.experts[expert_id].get_state_features(states)
                expert_weight = attention_weights[:, expert_id:expert_id+1]
                reconstructed_states += expert_weight * expert_contribution
        
        reconstruction_loss = F.mse_loss(reconstructed_states, states)
        
        # Regularization: encourage diverse attention patterns
        attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=1)
        entropy_loss = -torch.mean(attention_entropy)  # Negative to encourage diversity
        
        total_loss = reconstruction_loss + 0.01 * entropy_loss
        
        self.attention_optimizer.zero_grad()
        total_loss.backward()
        self.attention_optimizer.step()
        
        return total_loss

    def _knowledge_distillation(self) -> torch.Tensor:
        """Knowledge distillation between experts"""
        if len(self.experts) < 2:
            return torch.tensor(0.0, device=self.device)
        
        distillation_loss = torch.tensor(0.0, device=self.device)
        num_pairs = 0
        
        # Sample states from expert buffers
        for expert_id, buffer in self.expert_buffers.items():
            if len(buffer) < 10:
                continue
                
            sample_states, _, _, _, _ = buffer.sample(min(32, len(buffer)))
            sample_states = torch.FloatTensor(sample_states).to(self.device)
            
            # Get action distributions from all experts
            teacher_actions = []
            for teacher_id, teacher in enumerate(self.experts):
                if teacher_id != expert_id:
                    with torch.no_grad():
                        teacher_action, _ = teacher.get_action(sample_states)
                        teacher_actions.append(teacher_action)
            
            if teacher_actions:
                # Student (current expert) learns from ensemble of teachers
                student_action, _ = self.experts[expert_id].get_action(sample_states)
                teacher_ensemble = torch.stack(teacher_actions, dim=0).mean(dim=0)
                
                distill_loss = F.mse_loss(student_action, teacher_ensemble.detach())
                distillation_loss += distill_loss
                num_pairs += 1
        
        if num_pairs > 0:
            distillation_loss /= num_pairs
            
            # Backpropagate distillation loss
            for expert_id in range(len(self.experts)):
                self.expert_optimizers[expert_id].zero_grad()
            
            distillation_loss.backward()
            
            for expert_id in range(len(self.experts)):
                self.expert_optimizers[expert_id].step()
        
        return distillation_loss

    def _update_training_stats(self, losses: Dict):
        """Update training statistics"""
        for key, value in losses.items():
            self.training_stats[key].append(value)
        
        # Log periodic summaries
        if self.current_step % 100 == 0:
            print(f"Step {self.current_step}: {len(self.experts)} experts, "
                  f"{len(self.dpmm.cluster_means)} clusters")

    def add_offline_data(self, dataset: Dict):
        """Add offline dataset to appropriate expert buffers"""
        states = dataset['observations']
        actions = dataset['actions']
        rewards = dataset['rewards']
        next_states = dataset['next_observations']
        dones = dataset['terminals']
        
        print(f"🗃️ Processing {len(states)} offline transitions...")
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        for i in range(0, len(states), batch_size):
            end_idx = min(i + batch_size, len(states))
            batch_states = torch.FloatTensor(states[i:end_idx]).to(self.device)
            
            # Get latent representations
            with torch.no_grad():
                latent_z, _ = self.attention(batch_states)
                cluster_assignments = self.dpmm.infer_batch(latent_z)
            
            # Create experts if needed
            max_cluster = cluster_assignments.max().item() if len(cluster_assignments) > 0 else 0
            while len(self.experts) <= max_cluster:
                self._create_new_expert()
            
            # Add to expert buffers
            for j in range(len(batch_states)):
                expert_id = cluster_assignments[j].item()
                global_idx = i + j
                
                if expert_id < len(self.expert_buffers):
                    self.expert_buffers[expert_id].add(
                        state=states[global_idx],
                        action=actions[global_idx],
                        reward=rewards[global_idx],
                        next_state=next_states[global_idx],
                        done=dones[global_idx]
                    )
        
        print(f"✅ Offline data distributed across {len(self.experts)} experts")

    def train_offline(self, num_epochs: int = 100):
        """Train agent on offline data"""
        print(f"🎓 Starting offline training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            epoch_losses = defaultdict(list)
            
            # Sample from all expert buffers
            for expert_id, buffer in self.expert_buffers.items():
                if len(buffer) < self.config['batch_size']:
                    continue
                
                # Sample batch
                states, actions, rewards, next_states, dones = buffer.sample(
                    self.config['batch_size']
                )
                
                batch_data = {
                    'states': torch.FloatTensor(states),
                    'actions': torch.FloatTensor(actions),
                    'rewards': torch.FloatTensor(rewards),
                    'next_states': torch.FloatTensor(next_states),
                    'dones': torch.FloatTensor(dones)
                }
                
                # Update agent
                losses = self.update(batch_data)
                
                for key, value in losses.items():
                    epoch_losses[key].append(value)
            
            # Print epoch summary
            if epoch % 10 == 0:
                avg_losses = {k: np.mean(v) for k, v in epoch_losses.items() if v}
                print(f"Epoch {epoch}: {avg_losses}")
        
        print("✅ Offline training completed")

    def evaluate(self, env, num_episodes: int = 10) -> Dict:
        """Evaluate agent performance"""
        episode_rewards = []
        episode_lengths = []
        cluster_usage = defaultdict(int)
        
        for episode in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            episode_reward = 0
            episode_length = 0
            
            while True:
                action, info = self.select_action(state, deterministic=True)
                cluster_usage[info['cluster_id']] += 1
                
                next_state, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if done or truncated:
                    break
                    
                state = next_state
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'num_experts': len(self.experts),
            'num_clusters': len(self.dpmm.cluster_means),
            'cluster_usage': dict(cluster_usage)
        }
        
        return results

    def save(self, filepath: str):
        """Save agent state"""
        checkpoint = {
            'config': self.config,
            'experts': [expert.state_dict() for expert in self.experts],
            'value_networks': [vnet.state_dict() for vnet in self.value_networks],
            'attention': self.attention.state_dict(),
            'dpmm_state': {
                'cluster_means': self.dpmm.cluster_means,
                'cluster_covs': self.dpmm.cluster_covs,
                'cluster_counts': self.dpmm.cluster_counts,
                'cluster_weights': self.dpmm.cluster_weights,
                'alpha': self.dpmm.alpha
            },
            'current_step': self.current_step,
            'training_stats': dict(self.training_stats)
        }
        
        torch.save(checkpoint, filepath)
        print(f"💾 Agent saved to {filepath}")

    def load(self, filepath: str):
        """Load agent state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load expert networks
        self.experts = nn.ModuleList()
        self.value_networks = nn.ModuleList()
        self.expert_optimizers = []
        self.value_optimizers = []
        
        for expert_state in checkpoint['experts']:
            expert = PolicyNetwork(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=self.config['pmoe_hidden_dim'],
                latent_dim=self.latent_dim
            ).to(self.device)
            expert.load_state_dict(expert_state)
            self.experts.append(expert)
            
            # Create optimizer
            optimizer = torch.optim.Adam(expert.parameters(), lr=self.config['learning_rate'])
            self.expert_optimizers.append(optimizer)
        
        for vnet_state in checkpoint['value_networks']:
            vnet = ValueNetwork(
                state_dim=self.state_dim,
                hidden_dim=self.config['pmoe_hidden_dim']
            ).to(self.device)
            vnet.load_state_dict(vnet_state)
            self.value_networks.append(vnet)
            
            # Create optimizer
            optimizer = torch.optim.Adam(vnet.parameters(), lr=self.config['learning_rate'])
            self.value_optimizers.append(optimizer)
        
        # Load attention mechanism
        self.attention.load_state_dict(checkpoint['attention'])
        
        # Load DPMM state
        dpmm_state = checkpoint['dpmm_state']
        self.dpmm.cluster_means = dpmm_state['cluster_means']
        self.dpmm.cluster_covs = dpmm_state['cluster_covs']
        self.dpmm.cluster_counts = dpmm_state['cluster_counts']
        self.dpmm.cluster_weights = dpmm_state['cluster_weights']
        self.dpmm.alpha = dpmm_state['alpha']
        
        # Load training state
        self.current_step = checkpoint['current_step']
        self.training_stats = defaultdict(list, checkpoint['training_stats'])
        
        print(f"📁 Agent loaded from {filepath}")

    def get_expert_info(self) -> Dict:
        """Get information about current experts"""
        info = {
            'num_experts': len(self.experts),
            'expert_buffer_sizes': {i: len(buf) for i, buf in self.expert_buffers.items()},
            'cluster_info': self.dpmm.get_cluster_statistics(),
            'total_training_steps': self.current_step
        }
        return info

    def prune_experts(self, min_buffer_size: int = 100):
        """Remove experts with insufficient data"""
        experts_to_remove = []
        
        for expert_id in range(len(self.experts)):
            if expert_id in self.expert_buffers:
                buffer_size = len(self.expert_buffers[expert_id])
                if buffer_size < min_buffer_size:
                    experts_to_remove.append(expert_id)
        
        # Remove experts in reverse order
        for expert_id in reversed(experts_to_remove):
            if expert_id < len(self.experts):
                del self.experts[expert_id]
                del self.value_networks[expert_id]
                del self.expert_optimizers[expert_id]
                del self.value_optimizers[expert_id]
                if expert_id in self.expert_buffers:
                    del self.expert_buffers[expert_id]
        
        # Prune DPMM clusters as well
        removed_clusters = self.dpmm.prune_clusters(min_points=min_buffer_size // 10)
        
        if experts_to_remove:
            print(f"🧹 Pruned {len(experts_to_remove)} experts and {removed_clusters} clusters")
        
        return len(experts_to_remove)
