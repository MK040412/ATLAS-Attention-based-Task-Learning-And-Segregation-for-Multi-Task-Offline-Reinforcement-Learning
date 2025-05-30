import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np

class PolicyNetwork(nn.Module):
    """Policy network for each expert"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Policy head
        self.mean_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.log_std_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Action bounds
        self.action_scale = 1.0
        self.action_bias = 0.0

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log_std"""
        features = self.feature_net(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, -20, 2)  # Clamp for stability
        return mean, log_std

    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        
        action = y_t * self.action_scale + self.action_bias
        
        # Compute log prob with correction for tanh
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

    def get_action_deterministic(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get deterministic action (mean)"""
        mean, log_std = self.forward(state)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        
        # For deterministic policy, we still need log_prob for compatibility
        # Use a small std for the log_prob calculation
        std = torch.ones_like(mean) * 0.01
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(mean).sum(1, keepdim=True)
        
        return action, log_prob

    def get_state_features(self, state: torch.Tensor) -> torch.Tensor:
        """Get state features for attention mechanism"""
        return self.feature_net(state)


class ValueNetwork(nn.Module):
    """Value network for critic"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class AttentionMechanism(nn.Module):
    """Attention mechanism for expert selection and latent representation"""
    
    def __init__(self, state_dim: int, latent_dim: int, hidden_dim: int = 128, max_experts: int = 10):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.max_experts = max_experts
        
        # State encoder to latent space
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Attention query network
        self.query_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention key network (for experts)
        self.key_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Expert embeddings (learnable)
        self.expert_embeddings = nn.Parameter(
            torch.randn(max_experts, latent_dim) * 0.1
        )
        
        # Temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: state -> latent representation + attention weights
        
        Args:
            state: [batch_size, state_dim]
            
        Returns:
            latent_z: [batch_size, latent_dim] - latent representation
            attention_weights: [batch_size, num_experts] - attention weights
        """
        batch_size = state.shape[0]
        
        # Encode state to latent space
        latent_z = self.state_encoder(state)  # [batch_size, latent_dim]
        
        # Compute queries
        queries = self.query_net(latent_z)  # [batch_size, hidden_dim]
        
        # Compute keys from expert embeddings
        current_num_experts = self._get_current_num_experts()
        expert_embs = self.expert_embeddings[:current_num_experts]  # [num_experts, latent_dim]
        keys = self.key_net(expert_embs)  # [num_experts, hidden_dim]
        
        # Compute attention scores
        scores = torch.matmul(queries, keys.t()) / self.temperature  # [batch_size, num_experts]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, num_experts]
        
        return latent_z, attention_weights

    def _get_current_num_experts(self) -> int:
        """Get current number of experts (this will be set dynamically)"""
        # This is a placeholder - in practice, this will be updated
        # by the ATLAS agent based on the current number of experts
        return min(self.max_experts, 3)  # Default to 3 for initialization

    def update_num_experts(self, num_experts: int):
        """Update the current number of experts for attention computation"""
        self.current_num_experts = min(num_experts, self.max_experts)

    def add_expert_embedding(self):
        """Add a new expert embedding (when a new expert is created)"""
        # In practice, expert embeddings are fixed size but we only use
        # the first N embeddings where N is the current number of experts
        pass

    def get_expert_similarities(self, latent_z: torch.Tensor) -> torch.Tensor:
        """Compute similarities between latent representation and expert embeddings"""
        current_num_experts = self._get_current_num_experts()
        expert_embs = self.expert_embeddings[:current_num_experts]
        
        # Cosine similarity
        latent_norm = F.normalize(latent_z, p=2, dim=-1)
        expert_norm = F.normalize(expert_embs, p=2, dim=-1)
        
        similarities = torch.matmul(latent_norm, expert_norm.t())
        return similarities


class TaskEncoder(nn.Module):
    """Task encoder for inferring task identity from trajectory"""
    
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # Encoder for state-action pairs
        self.sa_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Encode trajectory to task representation
        
        Args:
            states: [batch_size, seq_len, state_dim]
            actions: [batch_size, seq_len, action_dim]
            
        Returns:
            task_embedding: [batch_size, latent_dim]
        """
        batch_size, seq_len = states.shape[:2]
        
        # Concatenate states and actions
        sa_pairs = torch.cat([states, actions], dim=-1)  # [batch_size, seq_len, state_dim + action_dim]
        
        # Encode each state-action pair
        sa_encoded = self.sa_encoder(sa_pairs.view(-1, self.state_dim + self.action_dim))
        sa_encoded = sa_encoded.view(batch_size, seq_len, self.latent_dim)
        
        # Pass through LSTM
        lstm_out, (hidden, _) = self.lstm(sa_encoded)
        
        # Use final hidden state as task representation
        task_embedding = self.output_proj(hidden.squeeze(0))  # [batch_size, latent_dim]
        
        return task_embedding


class DynamicsModel(nn.Module):
    """Dynamics model for each expert"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + 1)  # +1 for reward prediction
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next state and reward
        
        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
            
        Returns:
            next_state: [batch_size, state_dim]
            reward: [batch_size, 1]
        """
        sa_input = torch.cat([state, action], dim=-1)
        output = self.network(sa_input)
        
        next_state_delta = output[:, :self.state_dim]
        reward = output[:, -1:]
        
        # Predict state change rather than absolute next state
        next_state = state + next_state_delta
        
        return next_state, reward


class DiscriminatorNetwork(nn.Module):
    """Discriminator for distinguishing between different tasks/experts"""
    
    def __init__(self, state_dim: int, action_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Classify state-action pairs
        
        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
            
        Returns:
            logits: [batch_size, num_classes]
        """
        sa_input = torch.cat([state, action], dim=-1)
        logits = self.network(sa_input)
        return logits

    def get_loss(self, state: torch.Tensor, action: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute discriminator loss"""
        logits = self.forward(state, action)
        loss = F.cross_entropy(logits, labels)
        return loss


def initialize_weights(module: nn.Module):
    """Initialize network weights"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)


# Helper function to create networks with proper initialization
def create_policy_network(state_dim: int, action_dim: int, hidden_dim: int = 128, latent_dim: int = 32) -> PolicyNetwork:
    """Create and initialize a policy network"""
    network = PolicyNetwork(state_dim, action_dim, hidden_dim, latent_dim)
    network.apply(initialize_weights)
    return network

def create_value_network(state_dim: int, hidden_dim: int = 128) -> ValueNetwork:
    """Create and initialize a value network"""
    network = ValueNetwork(state_dim, hidden_dim)
    network.apply(initialize_weights)
    return network

def create_attention_mechanism(state_dim: int, latent_dim: int, hidden_dim: int = 128, max_experts: int = 10) -> AttentionMechanism:
    """Create and initialize attention mechanism"""
    network = AttentionMechanism(state_dim, latent_dim, hidden_dim, max_experts)
    network.apply(initialize_weights)
    return network
