import torch

def get_default_config():
    """Get default configuration"""
    return {
        'env_name': 'Ant-v5',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'task_count': 3,
        'latent_dim': 32,
        'init_experts': 3,
        'pmoe_hidden_dim': 128, # Hidden dim for expert policy networks
        'dp_concentration': 1.0,
        'cluster_merge_threshold': 1e-4, # KL divergence threshold for merging clusters
        'cluster_min_points': 30, # Minimum points for a cluster to be considered stable
        'distillation_steps': 50, # Number of steps for knowledge distillation
        'learning_rate': 3e-4,
        'batch_size': 64,
        'gamma': 0.99,
        'tau': 0.005, # For soft target updates
        'buffer_size': 100000, # Total buffer size (not used directly if buf_per_expert is used)
        'buf_per_expert': 50000, # Replay buffer capacity per expert
        'train_freq': 1, # How often to train the agent (in steps)
        'dpmm_freq': 500, # How often to update DPMM statistics
        'eval_freq': 1500, # How often to run comprehensive evaluation
        'total_steps': 10000, # Total training steps
        'lambda_vb': 0.1, # Weight for Variational Bayes ELBO loss
        'lambda_rec': 0.1, # Weight for reconstruction loss
        'lambda_dyn': 0.1, # Weight for dynamics prediction loss
        'lambda_distill': 0.1, # Weight for distillation loss
        'seed': 42
    }

