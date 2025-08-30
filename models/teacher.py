
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Dict


class TeacherGenerator(nn.Module):
    """
    Teacher Generator
    Input: 100-dim noise vector
    Output: 32x32x3 RGB image
    """
    
    def __init__(self, z_dim=256, channels=3, feature_count=4):  # use z_dim=256 as in training
        super(TeacherGenerator, self).__init__()
        
        self.z_dim = z_dim
        self.channels = channels
        self.feature_count = feature_count
        
        # Fully connected: z -> 4x4 feature map (larger initial features)
        self.fc = nn.Linear(z_dim, feature_count * 512 * 4 * 4)
        
        # Use GroupNorm instead of BatchNorm
        self.norm1 = nn.GroupNorm(32, feature_count * 512)
        self.norm2 = nn.GroupNorm(32, feature_count * 256)
        self.norm3 = nn.GroupNorm(16, feature_count * 128)
        
        # Transposed conv: 4x4 -> 8x8
        self.deconv1 = nn.ConvTranspose2d(
            feature_count * 512, feature_count * 256, 
            kernel_size=4, stride=2, padding=1
        )
        
        # Transposed conv: 8x8 -> 16x16
        self.deconv2 = nn.ConvTranspose2d(
            feature_count * 256, feature_count * 128,
            kernel_size=4, stride=2, padding=1
        )
        
        # Transposed conv: 16x16 -> 32x32
        self.deconv3 = nn.ConvTranspose2d(
            feature_count * 128, channels,
            kernel_size=4, stride=2, padding=1
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """WGAN-GP generator-specific initialization"""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
                # Normal init with moderate std
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        """Forward pass"""
        # z: (batch_size, z_dim)
        
        # FC + reshape
        x = self.fc(z)  # (batch_size, feature_count * 512 * 4 * 4)
        x = x.view(-1, self.feature_count * 512, 4, 4)
        
        # Upsampling blocks
        x = F.relu(self.norm1(x))  # norm then activation
        x = F.relu(self.norm2(self.deconv1(x)))  # (batch_size, 256, 8, 8)
        x = F.relu(self.norm3(self.deconv2(x)))  # (batch_size, 128, 16, 16)
        
        # Output layer - no norm
        x = torch.tanh(self.deconv3(x))  # (batch_size, 3, 32, 32)
        
        return x


class TeacherDiscriminator(nn.Module):
    """
    Teacher Discriminator
    Input: 32x32x3 RGB image
    Output: scalar logit (for WGAN loss)
    """
    
    def __init__(self, channels=3, feature_multiplier=2):  # increase feature_multiplier
        super(TeacherDiscriminator, self).__init__()
        
        self.channels = channels
        self.feature_multiplier = feature_multiplier
        
        # Enhanced conv: 32x32 -> 16x16
        self.conv1 = nn.Conv2d(channels, 64 * feature_multiplier, kernel_size=4, stride=2, padding=1)
        
        # Conv: 16x16 -> 8x8
        self.conv2 = nn.Conv2d(64 * feature_multiplier, 128 * feature_multiplier, kernel_size=4, stride=2, padding=1)
        self.layer_norm2 = nn.GroupNorm(8, 128 * feature_multiplier)
        
        # Conv: 8x8 -> 4x4
        self.conv3 = nn.Conv2d(128 * feature_multiplier, 256 * feature_multiplier, kernel_size=4, stride=2, padding=1)
        self.layer_norm3 = nn.GroupNorm(16, 256 * feature_multiplier)
        
        # Conv: 4x4 -> 2x2
        self.conv4 = nn.Conv2d(256 * feature_multiplier, 512 * feature_multiplier, kernel_size=4, stride=2, padding=1)
        self.layer_norm4 = nn.GroupNorm(32, 512 * feature_multiplier)
        
        # Extra conv for finer features
        self.conv5 = nn.Conv2d(512 * feature_multiplier, 1024 * feature_multiplier, kernel_size=2, stride=1, padding=0)
        self.layer_norm5 = nn.GroupNorm(64, 1024 * feature_multiplier)
        
        # Fully-connected to logit
        self.fc = nn.Linear(1024 * feature_multiplier * 1 * 1, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """WGAN-GP specific initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # Normal init with smaller std
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        # x: (batch_size, channels, 32, 32)
        
        # Downsampling block - no norm in the first layer
        x = F.leaky_relu(self.conv1(x), 0.2)
        
        # Other layers with GroupNorm
        x = F.leaky_relu(self.layer_norm2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.layer_norm3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.layer_norm4(self.conv4(x)), 0.2)
        
        # Extra fine feature extraction layer
        x = F.leaky_relu(self.layer_norm5(self.conv5(x)), 0.2)
        
        # Flatten + FC
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def get_features(self, x):
        """Get intermediate features for knowledge distillation"""
        # Layer 1: 32x32 -> 16x16
        x = F.leaky_relu(self.conv1(x), 0.2)
        
        # Layer 2: 16x16 -> 8x8
        x = F.leaky_relu(self.layer_norm2(self.conv2(x)), 0.2)
        
        # Layer 3: 8x8 -> 4x4
        x = F.leaky_relu(self.layer_norm3(self.conv3(x)), 0.2)
        
        # Layer 4: 4x4 -> 2x2
        x = F.leaky_relu(self.layer_norm4(self.conv4(x)), 0.2)
        
        # Layer 5: 2x2 -> 1x1 (advanced features)
        x = F.leaky_relu(self.layer_norm5(self.conv5(x)), 0.2)
        
        return x


def gradient_penalty(discriminator, real_images, fake_images, device='cuda'):
    """
    WGAN-GP gradient penalty
    """
    batch_size = real_images.size(0)
    
    # Random weights for interpolation
    epsilon = torch.rand(batch_size, 1, 1, 1).to(device)
    epsilon = epsilon.expand_as(real_images)
    
    # Interpolated images
    interpolated = epsilon * real_images + (1 - epsilon) * fake_images
    interpolated = interpolated.to(device)
    interpolated.requires_grad_(True)
    
    # Discriminator output on interpolated images
    disc_interpolated = discriminator(interpolated)
    
    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=disc_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(disc_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    
    return penalty


class TeacherNetwork(nn.Module):
   
    
    def __init__(self, z_dim=256, channels=3, learning_rate=0.0002, beta1=0.5, device='cuda'):  # fix: use correct latent dim
        super(TeacherNetwork, self).__init__()
        
        self.z_dim = z_dim
        self.channels = channels
        self.device = device
        
        # Initialize networks
        self.generator = TeacherGenerator(z_dim=z_dim, channels=channels).to(device)
        self.discriminator = TeacherDiscriminator(channels=channels).to(device)
        
        # Initialize optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(), 
            lr=learning_rate, betas=(beta1, 0.999)
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(), 
            lr=learning_rate, betas=(beta1, 0.999)
        )
        
        # Gradient penalty weight
        self.gradient_penalty_weight = 10.0
    
    def train_step(self, real_images: torch.Tensor) -> Dict[str, float]:
        """
        Perform one training step
        Return training metrics
        """
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)
        
        # ============ Train Discriminator ============
        self.optimizer_d.zero_grad()
        
        # Real images
        real_validity = self.discriminator(real_images)
        
        # Generate fake images
        z = torch.randn(batch_size, self.z_dim).to(self.device)
        fake_images = self.generator(z).detach()
        fake_validity = self.discriminator(fake_images)
        
        # WGAN losses
        d_loss_real = -torch.mean(real_validity)
        d_loss_fake = torch.mean(fake_validity)
        
        # Gradient penalty
        gp = gradient_penalty(self.discriminator, real_images, fake_images, self.device)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake + self.gradient_penalty_weight * gp
        d_loss.backward()
        self.optimizer_d.step()
        
        # ============ Train Generator ============
        self.optimizer_g.zero_grad()
        
        # Generate new fake images
        z = torch.randn(batch_size, self.z_dim).to(self.device)
        fake_images = self.generator(z)
        fake_validity = self.discriminator(fake_images)
        
        # Generator loss (make discriminator believe fakes are real)
        g_loss = -torch.mean(fake_validity)
        g_loss.backward()
        self.optimizer_g.step()
        
        return {
            'discriminator_loss': d_loss.item(),
            'generator_loss': g_loss.item(),
            'gradient_penalty': gp.item(),
            'd_real': torch.mean(real_validity).item(),
            'd_fake': torch.mean(fake_validity).item()
        }
    
    def generate_samples(self, num_samples: int = 64) -> torch.Tensor:
        """Generate samples"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.z_dim).to(self.device)
            samples = self.generator(z)
        self.generator.train()
        return samples
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict']) 