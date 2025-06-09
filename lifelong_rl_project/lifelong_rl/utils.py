import torch
import numpy as np
import random
import os
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🎲 Random seed set to {seed}")

def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(experiment_name)
    return logger

def save_config(config: Dict, filepath: str):
    """Save configuration to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"💾 Config saved to {filepath}")

def load_config(filepath: str) -> Dict:
    """Load configuration from JSON file"""
    with open(filepath, 'r') as f:
        config = json.load(f)
    print(f"📁 Config loaded from {filepath}")
    return config

def save_results(results: Dict, filepath: str):
    """Save results to pickle file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    print(f"💾 Results saved to {filepath}")

def load_results(filepath: str) -> Dict:
    """Load results from pickle file"""
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    print(f"📁 Results loaded from {filepath}")
    return results

def moving_average(data: List[float], window: int) -> List[float]:
    """Compute moving average"""
    if len(data) < window:
        return data
    
    averaged = []
    for i in range(len(data)):
        start_idx = max(0, i - window + 1)
        averaged.append(np.mean(data[start_idx:i+1]))
    
    return averaged

def plot_training_curves(training_stats: Dict[str, List[float]], save_path: str = None):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Main metrics to plot
    metrics = ['total_reward', 'policy_loss', 'value_loss', 'attention_loss']
    
    for i, metric in enumerate(metrics):
        if metric in training_stats and training_stats[metric]:
            data = training_stats[metric]
            smoothed = moving_average(data, window=10)
            
            axes[i].plot(data, alpha=0.3, label='Raw')
            axes[i].plot(smoothed, label='Smoothed')
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_xlabel('Step')
            axes[i].legend()
            axes[i].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training curves saved to {save_path}")
    
    plt.show()

def plot_expert_evolution(expert_info: Dict[str, List], save_path: str = None):
    """Plot evolution of experts over time"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Number of experts over time
    if 'num_experts' in expert_info:
        ax1.plot(expert_info['num_experts'])
        ax1.set_title('Number of Experts Over Time')
        ax1.set_xlabel('Update Step')
        ax1.set_ylabel('Number of Experts')
        ax1.grid(True)
    
    # Number of clusters over time
    if 'num_clusters' in expert_info:
        ax2.plot(expert_info['num_clusters'])
        ax2.set_title('Number of Clusters Over Time')
        ax2.set_xlabel('Update Step')
        ax2.set_ylabel('Number of Clusters')
        ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Expert evolution plot saved to {save_path}")
    
    plt.show()

def plot_attention_heatmap(attention_weights: np.ndarray, save_path: str = None):
    """Plot attention weights heatmap"""
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(attention_weights.T, 
                annot=True, 
                fmt='.3f', 
                cmap='Blues',
                xticklabels=[f'Step {i}' for i in range(attention_weights.shape[0])],
                yticklabels=[f'Expert {i}' for i in range(attention_weights.shape[1])])
    
    plt.title('Attention Weights Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Experts')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Attention heatmap saved to {save_path}")
    
    plt.show()

def compute_cluster_metrics(cluster_assignments: List[int], true_labels: Optional[List[int]] = None) -> Dict[str, float]:
    """Compute clustering evaluation metrics"""
    metrics = {}
    
    # Basic statistics
    unique_clusters = set(cluster_assignments)
    metrics['num_clusters'] = len(unique_clusters)
    metrics['cluster_sizes'] = {cluster: cluster_assignments.count(cluster) for cluster in unique_clusters}
    
    # If true labels are available, compute external validation metrics
    if true_labels is not None:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
        
        metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, cluster_assignments)
        metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, cluster_assignments)
        
        # Note: silhouette_score requires the original data points, not just labels
        # This would need to be computed separately with the actual feature vectors
    
    return metrics

def compute_expert_utilization(attention_weights: np.ndarray) -> Dict[str, Any]:
    """Compute expert utilization statistics"""
    # Average attention per expert
    avg_attention = np.mean(attention_weights, axis=0)
    
    # Entropy of attention distribution
    def entropy(probs):
        probs = probs + 1e-8  # Avoid log(0)
        return -np.sum(probs * np.log(probs))
    
    attention_entropy = [entropy(weights) for weights in attention_weights]
    
    # Effective number of experts (based on entropy)
    max_entropy = np.log(attention_weights.shape[1])
    normalized_entropy = np.array(attention_entropy) / max_entropy
    effective_experts = np.exp(attention_entropy)
    
    return {
        'avg_attention_per_expert': avg_attention.tolist(),
        'attention_entropy': attention_entropy,
        'normalized_entropy': normalized_entropy.tolist(),
        'effective_num_experts': effective_experts.tolist(),
        'max_attention_expert': np.argmax(avg_attention),
        'min_attention_expert': np.argmin(avg_attention)
    }

def evaluate_expert_diversity(expert_policies: List, states: np.ndarray) -> Dict[str, float]:
    """Evaluate diversity between expert policies"""
    if len(expert_policies) < 2:
        return {'diversity_score': 0.0}
    
    # Get actions from all experts for the same states
    expert_actions = []
    for expert in expert_policies:
        with torch.no_grad():
            actions, _ = expert.get_action_deterministic(torch.FloatTensor(states))
            expert_actions.append(actions.numpy())
    
    expert_actions = np.array(expert_actions)  # [num_experts, num_states, action_dim]
    
    # Compute pairwise action differences
    diversities = []
    for i in range(len(expert_policies)):
        for j in range(i + 1, len(expert_policies)):
            diff = np.mean(np.linalg.norm(expert_actions[i] - expert_actions[j], axis=1))
            diversities.append(diff)
    
    return {
        'diversity_score': np.mean(diversities),
        'diversity_std': np.std(diversities),
        'pairwise_diversities': diversities
    }

def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """Create experiment directory with timestamp"""
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    
    print(f"📁 Experiment directory created: {exp_dir}")
    return exp_dir

def log_hyperparameters(logger: logging.Logger, config: Dict):
    """Log hyperparameters"""
    logger.info("🔧 Hyperparameters:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

def compute_gradient_norm(model: torch.nn.Module) -> float:
    """Compute gradient norm for monitoring training stability"""
    total_norm = 0.0
    param_count = 0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    if param_count > 0:
        total_norm = total_norm ** (1. / 2)
    
    return total_norm

def soft_update(target_net: torch.nn.Module, source_net: torch.nn.Module, tau: float):
    """Soft update target network parameters"""
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

def hard_update(target_net: torch.nn.Module, source_net: torch.nn.Module):
    """Hard update target network parameters"""
    target_net.load_state_dict(source_net.state_dict())

def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device_info() -> Dict[str, Any]:
    """Get device information"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        info['cuda_device_name'] = torch.cuda.get_device_name()
        info['cuda_memory_allocated'] = torch.cuda.memory_allocated()
        info['cuda_memory_reserved'] = torch.cuda.memory_reserved()
    
    return info

def create_video_from_frames(frames: List[np.ndarray], output_path: str, fps: int = 30):
    """Create video from list of frames"""
    try:
        import cv2
        
        if not frames:
            print("⚠️ No frames provided for video creation")
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            video_writer.write(frame_bgr)
        
        video_writer.release()
        print(f"🎬 Video saved to {output_path}")
        
    except ImportError:
        print("⚠️ OpenCV not available, cannot create video")

def normalize_data(data: np.ndarray, epsilon: float = 1e-8) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Normalize data to zero mean and unit variance"""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) + epsilon
    
    normalized_data = (data - mean) / std
    
    stats = {
        'mean': mean,
        'std': std
    }
    
    return normalized_data, stats

def denormalize_data(normalized_data: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Denormalize data using stored statistics"""
    return normalized_data * stats['std'] + stats['mean']

def compute_return(rewards: List[float], gamma: float) -> float:
    """Compute discounted return"""
    return_val = 0.0
    for i, reward in enumerate(rewards):
        return_val += (gamma ** i) * reward
    return return_val

def compute_gae(rewards: List[float], values: List[float], dones: List[bool], 
                gamma: float, lam: float) -> List[float]:
    """Compute Generalized Advantage Estimation"""
    advantages = []
    gae = 0.0
    
    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[i + 1]
        
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    
    return advantages

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.wait = 0
        self.best_weights = None
        self.stopped_epoch = 0
    
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """Check if training should stop"""
        if self.best_score is None:
            self.best_score = score
            self.best_weights = model.state_dict().copy() if self.restore_best_weights else None
            return False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.wait = 0
            self.best_weights = model.state_dict().copy() if self.restore_best_weights else None
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False

def print_system_info():
    """Print system information"""
    device_info = get_device_info()
    
    print("🖥️ System Information:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {device_info['cuda_available']}")
    
    if device_info['cuda_available']:
        print(f"  CUDA devices: {device_info['cuda_device_count']}")
        print(f"  Current device: {device_info['cuda_current_device']}")
        print(f"  Device name: {device_info['cuda_device_name']}")
        print(f"  Memory allocated: {device_info['cuda_memory_allocated'] / 1024**2:.1f} MB")
        print(f"  Memory reserved: {device_info['cuda_memory_reserved'] / 1024**2:.1f} MB")