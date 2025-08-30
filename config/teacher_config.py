#!/usr/bin/env python3
"""
Teacher network configuration file
WGAN-GP Teacher configuration parameters
"""

from dataclasses import dataclass

@dataclass
class TeacherConfig:
    """Base Teacher configuration - improved per theory-driven fixes"""
    
    # === Network architecture (enhanced) ===
    latent_dim: int = 256           # latent dimension - increased (128→256)
    gen_hidden_dim: int = 1024      # generator hidden dim - increased (512→1024)
    disc_hidden_dim: int = 1024     # discriminator hidden dim - increased (512→1024)
    
    # === Additional architectural options ===
    use_attention: bool = True      # enable self-attention
    use_residual: bool = True       # enable residual connections
    use_spectral_norm: bool = True  # enable spectral norm (stable training)
    
    # === Training parameters - image quality focused ===
    learning_rate: float = 0.0003   # generator lr - balance quality and stability
    disc_learning_rate: float = 0.0001  # discriminator lr - keep adversarial balance
    beta1: float = 0.5              # Adam beta1 - momentum helps convergence
    beta2: float = 0.999            # Adam beta2
    
    # === WGAN-GP parameters - quality optimization ===
    gradient_penalty_weight: float = 10.0   # gradient penalty weight - improve stability
    n_critic: int = 5              # discriminator training ratio - balance GAN
    
    # === Monitoring/adaptive params - quickly enable Teacher-Student ===
    target_d_loss: float = 6.0      # target D loss threshold - relaxed (3.5→6.0)
    target_g_loss: float = 2.5      # target G loss threshold - relaxed (1.5→2.5)
    enable_kd_threshold: bool = True # enable KD threshold check
    
    # === Expert management ===
    expert_lr_decay: float = 0.5    # expert lr decay factor
    expert_momentum: float = 0.9    # expert optimizer momentum
    
    def to_dict(self) -> dict:
        """Convert to dict"""
        return {
            'latent_dim': self.latent_dim,
            'gen_hidden_dim': self.gen_hidden_dim,
            'disc_hidden_dim': self.disc_hidden_dim,
            'use_attention': self.use_attention,
            'use_residual': self.use_residual,
            'use_spectral_norm': self.use_spectral_norm,
            'learning_rate': self.learning_rate,
            'disc_learning_rate': self.disc_learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'gradient_penalty_weight': self.gradient_penalty_weight,
            'n_critic': self.n_critic,
            'target_d_loss': self.target_d_loss,
            'target_g_loss': self.target_g_loss,
            'enable_kd_threshold': self.enable_kd_threshold,
            'expert_lr_decay': self.expert_lr_decay,
            'expert_momentum': self.expert_momentum,
            'config_type': 'TeacherConfig'
        }


@dataclass  
class TeacherConfigHigh(TeacherConfig):

    latent_dim: int = 128           # larger latent space
    gen_hidden_dim: int = 512       # larger network
    disc_hidden_dim: int = 512
    learning_rate: float = 0.0001   # more stable learning rate
    gradient_penalty_weight: float = 20.0  # stronger gradient penalty 