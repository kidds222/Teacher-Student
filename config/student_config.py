#!/usr/bin/env python3
"""
Student network configuration - research edition
β-VAE Student network configuration parameters
"""

from dataclasses import dataclass
from typing import List

@dataclass
class StudentConfig:
    """Base Student configuration"""
    
    # === β-VAE core parameters ===
    z_dim: int = 64                 # content latent dim (task-agnostic)
    u_dim: int = 16                 # domain latent dim (task-specific)
    beta: float = 1.0               # β-VAE beta parameter
    
    # === Contrastive learning parameters ===
    contrastive_weight: float = 0.1 # contrastive loss weight
    temperature: float = 0.1        # contrastive temperature
    negative_samples: int = 64      # number of negative samples
    
    # === Knowledge distillation parameters ===
    kd_weight: float = 1.0          # KD weight
    kd_temperature: float = 3.0     # KD temperature
    
    # === Network architecture ===
    hidden_dims: List[int] = None   # encoder hidden dims
    decoder_hidden_dims: List[int] = None  # decoder hidden dims
    
    # === Training parameters ===
    learning_rate: float = 0.001    # Student learning rate
    weight_decay: float = 1e-5      # weight decay
    num_tasks: int = 3              # number of tasks
    
    def __post_init__(self):
        """Set default network structure"""
        if self.hidden_dims is None:
            self.hidden_dims = [64, 128, 256, 512]
        if self.decoder_hidden_dims is None:
            self.decoder_hidden_dims = [512, 256, 128, 64]
    
    def to_dict(self) -> dict:
        """Convert to dict"""
        return {
            'z_dim': self.z_dim,
            'u_dim': self.u_dim,
            'beta': self.beta,
            'contrastive_weight': self.contrastive_weight,
            'temperature': self.temperature,
            'negative_samples': self.negative_samples,
            'kd_weight': self.kd_weight,
            'kd_temperature': self.kd_temperature,
            'hidden_dims': self.hidden_dims,
            'decoder_hidden_dims': self.decoder_hidden_dims,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_tasks': self.num_tasks,
            'config_type': 'StudentConfig'
        }


@dataclass
class StudentConfigBalanced(StudentConfig):
    """Balanced Student configuration - balance performance and speed"""
    z_dim: int = 64                 # standard latent dim
    u_dim: int = 16
    beta: float = 1.0               # balanced disentanglement
    contrastive_weight: float = 0.1 # moderate contrastive learning
    learning_rate: float = 0.001    # stable learning rate


@dataclass
class StudentConfigHigh(StudentConfig):
    """High-performance Student configuration - for best paper results"""
    z_dim: int = 128                # larger latent dim
    u_dim: int = 32
    beta: float = 2.0               # stronger disentanglement
    contrastive_weight: float = 0.2 # stronger contrastive learning
    learning_rate: float = 0.0002   # more stable learning rate
    temperature: float = 0.05       # lower temperature
    hidden_dims: List[int] = None
    decoder_hidden_dims: List[int] = None
    
    def __post_init__(self):
        self.hidden_dims = [128, 256, 512, 1024]  # larger network
        self.decoder_hidden_dims = [1024, 512, 256, 128]
        super().__post_init__() 