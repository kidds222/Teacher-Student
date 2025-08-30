

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from typing import Tuple, Dict, Optional, List


class TaskAwareEncoder(nn.Module):
    """Task-aware encoder"""
    
    def __init__(self, z_dim: int = 64, u_dim: int = 16, num_tasks: int = 3):
        super().__init__()
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.num_tasks = num_tasks
        
        # Shared feature extractor
        self.shared_conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 32x32 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16x16 -> 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 8x8 -> 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 1, 0),  # 4x4 -> 1x1
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        
        # Content encoding branch (task-agnostic)
        self.content_mu = nn.Linear(512, z_dim)
        self.content_logvar = nn.Linear(512, z_dim)
        
        # Domain encoding branch (task-specific)
        self.domain_mu = nn.Linear(512, u_dim)
        self.domain_logvar = nn.Linear(512, u_dim)
        
        # Task embedding
        self.task_embedding = nn.Embedding(num_tasks, 64)
        self.task_proj = nn.Linear(64, 512)
        
    def forward(self, x: torch.Tensor, task_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        # Feature extraction
        features = self.shared_conv(x).view(batch_size, -1)  # [B, 512]
        
        # Fuse task embedding
        # task_id should be shape [B] with identical values per batch
        # nn.Embedding expects index tensor
        task_emb = self.task_embedding(task_id)  # [B, 64]
        task_proj = self.task_proj(task_emb)      # [B, 512]
        features = features + task_proj
        
        # Content encoding (task-agnostic)
        z_mu = self.content_mu(features)
        z_logvar = self.content_logvar(features)
        
        # Domain encoding (task-specific)
        u_mu = self.domain_mu(features)
        u_logvar = self.domain_logvar(features)
        
        return z_mu, z_logvar, u_mu, u_logvar, features


class DomainGuidedDecoder(nn.Module):
    """Domain-guided decoder"""
    
    def __init__(self, z_dim: int = 64, u_dim: int = 16):
        super().__init__()
        self.z_dim = z_dim
        self.u_dim = u_dim
        
        # Domain fusion layer
        self.domain_fusion = nn.Sequential(
            nn.Linear(z_dim + u_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 1, 0),  # 1x1 -> 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4x4 -> 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8x8 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),     # 16x16 -> 32x32
            nn.Tanh()
        )
        
    def forward(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # Domain-guided fusion
        combined = torch.cat([z, u], dim=1)  # [B, z_dim + u_dim]
        fused = self.domain_fusion(combined)  # [B, 512]
        
        # Reshape and decode
        fused = fused.view(-1, 512, 1, 1)    # [B, 512, 1, 1]
        output = self.decoder(fused)          # [B, 3, 32, 32]
        
        return output


class TaskConditionalPrior(nn.Module):
    """Task-conditional prior p(z|task)"""
    
    def __init__(self, z_dim: int = 64, num_tasks: int = 3):
        super().__init__()
        self.z_dim = z_dim
        self.num_tasks = num_tasks
        
        # Prior parameters per task
        self.task_mu = nn.Parameter(torch.zeros(num_tasks, z_dim))
        self.task_logvar = nn.Parameter(torch.zeros(num_tasks, z_dim))
        
        # Initialization - different initial values per task
        nn.init.normal_(self.task_mu, 0, 0.1)
        nn.init.constant_(self.task_logvar, -1)  # smaller initial variance
        
        # Different initial means to help task separation
        for task_idx in range(num_tasks):
            self.task_mu.data[task_idx] = torch.randn(z_dim) * 0.1
        
    def forward(self, task_id: torch.Tensor) -> dist.Normal:
        """Get task-conditional prior distribution"""
        # Ensure correct shape for task_id
        if task_id.dim() == 0:
            task_id = task_id.unsqueeze(0)  # scalar -> [1]
        
        mu = self.task_mu[task_id]      # [B, z_dim]
        logvar = self.task_logvar[task_id]  # [B, z_dim]
        std = torch.exp(0.5 * logvar)
        
        return dist.Normal(mu, std)
    
    def sample(self, task_id: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Sample from task prior"""
        prior = self.forward(task_id)
        
        if num_samples == 1:
            # Single sample: (batch_size, z_dim)
            return prior.rsample()
        else:
            # Multiple samples: (num_samples, batch_size, z_dim)
            samples = prior.rsample((num_samples,))
            
            # If batch_size=1, simplify to (num_samples, z_dim)
            if samples.size(1) == 1:
                samples = samples.squeeze(1)
            
            return samples


class ContrastiveLearner(nn.Module):
    """Contrastive learning module"""
    
    def __init__(self, z_dim: int = 64, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
        )
        
    def forward(self, z: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss"""
        # Ensure batch sizes match for z and task_ids
        batch_size = z.size(0)
        if task_ids.size(0) != batch_size:
            # If mismatch, repeat last id or truncate
            if task_ids.size(0) < batch_size:
                last_task_id = task_ids[-1]
                task_ids = torch.cat([task_ids, last_task_id.repeat(batch_size - task_ids.size(0))])
            else:
                task_ids = task_ids[:batch_size]
        
        # Project to contrastive space
        projected = self.projection(z)  # [B, 64]
        projected = F.normalize(projected, dim=1)
        
        # Similarity matrix
        similarity = torch.matmul(projected, projected.T) / self.temperature  # [B, B]
        
        # Task labels matrix
        task_matrix = task_ids.unsqueeze(1) == task_ids.unsqueeze(0)  # [B, B]
        
        # Intra-task aggregation
        intra_mask = task_matrix & ~torch.eye(batch_size, dtype=torch.bool, device=z.device)
        if intra_mask.sum() > 0:
            intra_pos = similarity[intra_mask].mean()
        else:
            intra_pos = torch.tensor(0.0, device=z.device)
        
        # Inter-task separation
        inter_mask = ~task_matrix
        if inter_mask.sum() > 0:
            inter_neg = similarity[inter_mask].mean()
        else:
            inter_neg = torch.tensor(0.0, device=z.device)
        
        # Loss: maximize intra similarity, minimize inter similarity
        contrastive_loss = -intra_pos + inter_neg
        
        return contrastive_loss


class AdvancedStudentVAE(nn.Module):
    
    
    def __init__(self, 
                 z_dim: int = 64,
                 u_dim: int = 16, 
                 num_tasks: int = 3,
                 beta: float = 1.0,
                 contrastive_weight: float = 0.1,
                 temperature: float = 0.1):
        super().__init__()
        
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.num_tasks = num_tasks
        self.beta = beta
        self.contrastive_weight = contrastive_weight
        
        # Core components
        self.encoder = TaskAwareEncoder(z_dim, u_dim, num_tasks)
        self.decoder = DomainGuidedDecoder(z_dim, u_dim)
        self.task_prior = TaskConditionalPrior(z_dim, num_tasks)
        self.contrastive_learner = ContrastiveLearner(z_dim, temperature)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x: torch.Tensor, task_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input image"""
        z_mu, z_logvar, u_mu, u_logvar = self.encoder(x, task_id)
        return z_mu, z_logvar, u_mu, u_logvar
    
    def decode(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Decode latent variables"""
        return self.decoder(z, u)
    
    def forward(self, x: torch.Tensor, task_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Encode
        z_mu, z_logvar, u_mu, u_logvar = self.encode(x, task_id)
        
        # Reparameterize
        z = self.reparameterize(z_mu, z_logvar)
        u = self.reparameterize(u_mu, u_logvar)
        
        # Decode
        x_recon = self.decode(z, u)
        
        return {
            'x_recon': x_recon,
            'reconstruction': x_recon,  
            'z_mu': z_mu, 'z_logvar': z_logvar,
            'u_mu': u_mu, 'u_logvar': u_logvar,
            'z': z, 'u': u
        }
    
    def compute_loss(self, x: torch.Tensor, task_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute VAE loss"""
        outputs = self.forward(x, task_id)
        
        # 1. Reconstruction loss
        recon_loss = F.mse_loss(outputs['x_recon'], x, reduction='mean')
        
        # 2. KL divergence (task conditional prior)
        # Content encoding KL
        task_prior = self.task_prior(task_id)
        q_z = dist.Normal(outputs['z_mu'], torch.exp(0.5 * outputs['z_logvar']))
        
        kl_z = dist.kl_divergence(q_z, task_prior).mean()
        
        # Domain encoding KL (standard normal)
        standard_normal = dist.Normal(
            torch.zeros_like(outputs['u_mu']), 
            torch.ones_like(outputs['u_mu'])
        )
        q_u = dist.Normal(outputs['u_mu'], torch.exp(0.5 * outputs['u_logvar']))
        kl_u = dist.kl_divergence(q_u, standard_normal).mean()
        
        # 3. Contrastive loss
        contrastive_loss = self.contrastive_learner(outputs['z'], task_id)
        
        # Total loss
        total_loss = recon_loss + self.beta * (kl_z + kl_u) + self.contrastive_weight * contrastive_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss_z': kl_z,
            'kl_loss_u': kl_u,
            'contrastive_loss': contrastive_loss
        }
    
    def generate(self, task_id: torch.Tensor, num_samples: int = 1, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate new samples"""
        if z is None:
            # Ensure task_id is scalar or single-element tensor
            if task_id.numel() > 1:
                # If multiple task_ids provided, use the first
                task_id = task_id[0:1]
            
            # Sample from task prior
            z = self.task_prior.sample(task_id, num_samples)
            
            # Ensure shape is (num_samples, z_dim)
            if z.dim() == 3:
                # (num_samples, 1, z_dim) -> squeeze middle dim
                z = z.squeeze(1)
            elif z.dim() == 2 and z.size(0) == 1 and num_samples > 1:
                # (1, z_dim) but need multiple samples -> repeat
                z = z.repeat(num_samples, 1)
        
        # Generate domain code (standard normal)
        batch_size = z.size(0)
        u = torch.randn(batch_size, self.u_dim, device=z.device)
        
        return self.decode(z, u)
    
    def get_config(self) -> Dict:
        """Get model config"""
        return {
            'z_dim': self.z_dim,
            'u_dim': self.u_dim,
            'num_tasks': self.num_tasks,
            'beta': self.beta,
            'contrastive_weight': self.contrastive_weight,
            'type': 'AdvancedStudentVAE'
        }


def test_advanced_student():
    """Test Advanced Student VAE"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Testing device: {device}")
    
    # Create model
    student = AdvancedStudentVAE(
        z_dim=64, u_dim=16, num_tasks=3,
        beta=1.0, contrastive_weight=0.1
    ).to(device)
    
    # Test data
    batch_size = 8
    x = torch.randn(batch_size, 3, 32, 32).to(device)
    task_id = torch.randint(0, 3, (batch_size,)).to(device)
    
    print(" Testing forward pass...")
    outputs = student(x, task_id)
    print(f" Reconstruction shape: {outputs['x_recon'].shape}")
    print(f" Content z shape: {outputs['z'].shape}")
    print(f" Domain u shape: {outputs['u'].shape}")
    
    print(" Testing loss computation...")
    losses = student.compute_loss(x, task_id)
    print(f" Total loss: {losses['total_loss'].item():.4f}")
    print(f" Reconstruction loss: {losses['recon_loss'].item():.4f}")
    print(f" KL loss (z): {losses['kl_loss_z'].item():.4f}")
    print(f" KL loss (u): {losses['kl_loss_u'].item():.4f}")
    print(f" Contrastive loss: {losses['contrastive_loss'].item():.4f}")
    
    print(" Testing generation...")
    generated = student.generate(task_id[:2], num_samples=1)
    print(f" Generated image shape: {generated.shape}")
    
    print(" Advanced Student VAE test completed!")
    return student


if __name__ == "__main__":
    test_advanced_student() 