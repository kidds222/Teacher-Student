

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional

try:
    from .teacher import TeacherGenerator, TeacherDiscriminator, gradient_penalty
    from .student import AdvancedStudentVAE
except ImportError:
    # Absolute imports when running this file standalone
    from teacher import TeacherGenerator, TeacherDiscriminator, gradient_penalty
    from student import AdvancedStudentVAE

import torchvision.models as models
import torch.nn.functional as F


class DynamicTeacherStudent(nn.Module):
   
    
    def __init__(self, 
                 z_dim: int = 256,  # set to 256 to match TeacherGenerator
                 student_z_dim: int = 64,
                 student_u_dim: int = 16,
                 channels: int = 3,
                 num_tasks: int = 2,
                 fid_threshold: float = 100.0,  # match config file
                 learning_rate: float = 0.0002,
                 disc_learning_rate: float = None,  # discriminator separate lr
                 student_lr: float = 0.001,
                 beta1: float = 0.5,
                 beta: float = 1.0,
                 contrastive_weight: float = 0.1,
                 kd_weight: float = 1.0,
                 kd_feature_weight: float = 0.5,
                 kd_adaptive: bool = True,
                 use_mixed_precision: bool = True,  # mixed precision training
                 n_critic: int = 5,  # discriminator training ratio
                 gradient_penalty_weight: float = 10.0,  # gradient penalty weight
                 target_d_loss: float = 6.0,  # from TeacherConfig
                 target_g_loss: float = 2.5,  # from TeacherConfig
                 enable_kd_threshold: bool = True,  # KD threshold control
                 device: str = 'cuda'):
        super().__init__()
        
        self.z_dim = z_dim
        self.student_z_dim = student_z_dim
        self.student_u_dim = student_u_dim
        self.channels = channels
        self.num_tasks = num_tasks
        self.fid_threshold = fid_threshold
        self.learning_rate = learning_rate
        self.disc_learning_rate = disc_learning_rate if disc_learning_rate is not None else learning_rate * 0.5
        self.student_lr = student_lr
        self.beta1 = beta1
        self.beta = beta
        self.contrastive_weight = contrastive_weight
        self.kd_weight = kd_weight
        self.kd_feature_weight = kd_feature_weight
        self.kd_adaptive = kd_adaptive
        self.use_mixed_precision = use_mixed_precision  # mixed precision
        self.n_critic = n_critic  # ratio
        self.gradient_penalty_weight = gradient_penalty_weight  # gp weight
        self.target_d_loss = target_d_loss # target
        self.target_g_loss = target_g_loss # target
        self.enable_kd_threshold = enable_kd_threshold # enable KD threshold
        self.device = device
        
        # WGAN-GP training counter
        self.critic_iterations = 0
        
        # Simplified KD control - enabled by default
        self.kd_enabled = True  # enable KD by default
        self.teacher_stable_epochs = 1  # require only 1 epoch stable
        self.stability_check_interval = 50  # check stability every 50 iterations
        self.min_stable_epochs = 1  # only 1 stable epoch needed to turn on KD
        
        # Simplified G-loss quality counters
        self.good_g_loss_count = 0  # count of good G losses
        self.total_g_loss_count = 0  # total checks
        self.min_good_ratio = 0.5  # require 50% good ratio
        
        # Teacher stability history
        self.recent_d_losses = []
        self.recent_g_losses = []
        self.loss_history_size = 15  # keep last 15 batches (reduced from 20)
        
        # Mixed precision setup
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Teacher components - enhanced architecture
        self.discriminator = TeacherDiscriminator(channels, feature_multiplier=2).to(device)
        self.teacher_experts = nn.ModuleDict()
        self.teacher_optimizers = {}
        
        # Perceptual loss network - improve image quality
        self.vgg = models.vgg16(pretrained=True).features[:16].to(device)  # use early layers only
        for param in self.vgg.parameters():
            param.requires_grad = False  # freeze VGG
        self.perceptual_weight = 0.1  # perceptual weight
        
        # Student network
        self.student = AdvancedStudentVAE(
            z_dim=student_z_dim,
            u_dim=student_u_dim,
            num_tasks=num_tasks,
            beta=beta,
            contrastive_weight=contrastive_weight,
            temperature=0.1  # default from StudentConfig
        ).to(device)
        
        # Enhanced KD mapping layer - fix dimensionality issue
        self.kd_mapping = nn.Sequential(
            nn.Linear(student_z_dim, z_dim // 2),
            nn.BatchNorm1d(z_dim // 2),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(z_dim // 2, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ReLU(True)
        ).to(device)
        
        # Feature alignment layer
        self.feature_aligner = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 512)
        ).to(device)
        
        # Adaptive KD weights
        self.kd_weight_scheduler = nn.Parameter(torch.tensor(kd_weight))
        self.kd_feature_weight_scheduler = nn.Parameter(torch.tensor(kd_feature_weight))
        
        # Optimizers - balanced learning rates
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=self.disc_learning_rate, betas=(beta1, 0.999)
        )
        self.student_optimizer = optim.Adam(
            self.student.parameters(), 
            lr=student_lr, betas=(beta1, 0.999), weight_decay=1e-5  # from StudentConfig
        )
        
        # Disable schedulers for now - keep fixed lr
        # self.disc_scheduler = optim.lr_scheduler.ExponentialLR(self.disc_optimizer, gamma=0.99)
        # self.student_scheduler = optim.lr_scheduler.ExponentialLR(self.student_optimizer, gamma=0.999)
        self.disc_scheduler = None
        self.student_scheduler = None
        
        # KD optimizer
        self.kd_optimizer = optim.Adam([
            {'params': self.kd_mapping.parameters()},
            {'params': self.feature_aligner.parameters()},
            {'params': [self.kd_weight_scheduler, self.kd_feature_weight_scheduler]}
        ], lr=student_lr * 0.1)
        
        # Expert management
        self.current_expert_id = 0
        self.expert_performance = {}
        self.teacher_schedulers = {}  # pre-initialize
        
        # Initialize the first expert
        self.add_new_expert()
    
    def add_new_expert(self) -> int:
        """Add a new Teacher expert"""
        expert_id = len(self.teacher_experts)
        
        # Create new generator expert - enhanced architecture
        new_expert = TeacherGenerator(
            z_dim=self.z_dim, 
            channels=self.channels,
            feature_count=4  # 4x capacity
        ).to(self.device)
        
        # Create corresponding optimizer - use Adam
        new_optimizer = optim.Adam(
            new_expert.parameters(), 
            lr=self.learning_rate, 
            betas=(self.beta1, 0.999)
        )
        
        # Disable scheduler for the new expert
        # new_scheduler = optim.lr_scheduler.ExponentialLR(new_optimizer, gamma=0.999)
        new_scheduler = None
        
        # Add to dict (string keys)
        expert_key = f"expert_{expert_id}"
        self.teacher_experts[expert_key] = new_expert
        self.teacher_optimizers[expert_key] = new_optimizer
        
        # Add scheduler
        self.teacher_schedulers[expert_key] = new_scheduler
        
        self.expert_performance[expert_id] = float('inf')
        
        # Update current expert id
        self.current_expert_id = expert_id
        
        print(f" Added new expert #{expert_id}")
        return expert_id
    
    def select_best_expert(self, task_id: Optional[int] = None) -> int:
        """Select the best-performing expert"""
        if not self.expert_performance:
            return 0
        
        if task_id is not None and hasattr(self, 'task_expert_performance'):
            # Task-aware selection: prefer expert performing best on this task
            if task_id in self.task_expert_performance:
                task_performance = self.task_expert_performance[task_id]
                if task_performance:
                    best_expert_id = min(task_performance.keys(), key=lambda k: task_performance[k])
                    return best_expert_id
        
        # Global best expert selection
        best_expert_id = min(self.expert_performance.keys(), key=lambda k: self.expert_performance[k])
        return best_expert_id
    
    def update_expert_performance(self, expert_id: int, fid_score: float, task_id: Optional[int] = None):
        """Update expert performance"""
        self.expert_performance[expert_id] = fid_score
        
        # Task-level performance tracking
        if task_id is not None:
            if not hasattr(self, 'task_expert_performance'):
                self.task_expert_performance = {}
            if task_id not in self.task_expert_performance:
                self.task_expert_performance[task_id] = {}
            self.task_expert_performance[task_id][expert_id] = fid_score
    
    def should_add_expert(self, current_fid: float) -> bool:
        """Determine if a new expert should be added"""
        return current_fid > self.fid_threshold
    
    def generate_with_expert(self, z: torch.Tensor, expert_id: Optional[int] = None) -> torch.Tensor:
        """Generate images using a specified expert"""
        if expert_id is None:
            expert_id = self.current_expert_id
        
        expert_key = f"expert_{expert_id}"
        if expert_key not in self.teacher_experts:
            raise ValueError(f"Expert #{expert_id} does not exist")
        
        return self.teacher_experts[expert_key](z)
    
    def generate_with_student(self, task_id: torch.Tensor, num_samples: int = 1, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate images using the Student VAE"""
        return self.student.generate(task_id, num_samples, z)
    
    def compute_teacher_loss(self, 
                           real_images: torch.Tensor, 
                           expert_id: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Compute loss for the Teacher module (WGAN-GP)"""
        if expert_id is None:
            expert_id = self.current_expert_id
        
        batch_size = real_images.size(0)
        
        # Generate noise
        z = torch.randn(batch_size, self.z_dim).to(self.device)
        
        # Generate fake images
        fake_images = self.generate_with_expert(z, expert_id)
        
        # Compute discriminator loss (WGAN-GP)
        real_logits = self.discriminator(real_images)
        fake_logits = self.discriminator(fake_images.detach())
        
        # WGAN loss
        d_loss_real = -torch.mean(real_logits)
        d_loss_fake = torch.mean(fake_logits)
        d_loss_wgan = d_loss_real + d_loss_fake
        
        # Gradient penalty
        gp = gradient_penalty(self.discriminator, real_images, fake_images, self.device)
        d_loss = d_loss_wgan + self.gradient_penalty_weight * gp
        
        # Compute generator loss
        fake_logits_for_gen = self.discriminator(fake_images)
        g_loss = -torch.mean(fake_logits_for_gen)
        
        return {
            'discriminator_loss': d_loss,
            'generator_loss': g_loss,
            'gradient_penalty': gp
        }
    
    def compute_student_loss(self, 
                          real_images: torch.Tensor,
                          task_id: torch.Tensor,
                          teacher_expert_id: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Compute Student loss, intelligently controlling KD enable/disable"""
        # 1. Compute basic VAE loss
        vae_losses = self.student.compute_loss(real_images, task_id)
        
        # 2. Intelligent KD control
        kd_loss = torch.tensor(0.0).to(self.device)
        kd_feature_loss = torch.tensor(0.0).to(self.device)
        
        # Only perform KD when Teacher is stable and KD is enabled
        if self.kd_enabled and teacher_expert_id is not None and teacher_expert_id >= 0:
            try:
                # Get z and u from Student encoding
                student_outputs = self.student(real_images, task_id)
                student_z = student_outputs['z']
                student_u = student_outputs['u']
                
                # Map Student's z to Teacher's z space using KD mapping layer
                with torch.no_grad():
                    # Convert Student's z through the mapping layer
                    teacher_z = self.kd_mapping(student_z)
                    
                    # Generate corresponding images using Teacher
                    teacher_images = self.generate_with_expert(teacher_z, teacher_expert_id)
                
                # Multi-scale generation level KD: let Student learn Teacher's generation ability
                student_generated = self.student.decode(student_z, student_u)
                
                # 1. Pixel-level loss
                pixel_loss = nn.MSELoss()(student_generated, teacher_images.detach())
                
                # 2. Perceptual loss (using VGG features)
                if student_generated.size(1) == 1:  # Grayscale to RGB
                    student_rgb = student_generated.repeat(1, 3, 1, 1)
                    teacher_rgb = teacher_images.repeat(1, 3, 1, 1)
                else:
                    student_rgb = student_generated
                    teacher_rgb = teacher_images.detach()
                
                # Normalize to [0,1]
                student_norm = (student_rgb + 1.0) / 2.0
                teacher_norm = (teacher_rgb + 1.0) / 2.0
                
                try:
                    student_vgg_feat = self.vgg(student_norm)
                    teacher_vgg_feat = self.vgg(teacher_norm)
                    perceptual_loss = nn.MSELoss()(student_vgg_feat, teacher_vgg_feat.detach())
                except:
                    perceptual_loss = torch.tensor(0.0, device=self.device)
                
                # Combine KD loss
                kd_loss = pixel_loss + 0.1 * perceptual_loss
                
                # Feature-level KD: align latent space
                student_features = self.student.encoder.shared_conv(real_images)
                student_features = student_features.view(student_features.size(0), -1)
                
                with torch.no_grad():
                    teacher_features = self.discriminator.get_features(teacher_images.detach())
                    teacher_features = teacher_features.view(teacher_features.size(0), -1)
                    
                    # Use feature alignment layer
                    aligned_student_features = self.feature_aligner(student_features)
                    
                    # Feature alignment loss
                    if aligned_student_features.size() == teacher_features.size():
                        kd_feature_loss = nn.MSELoss()(aligned_student_features, teacher_features)
                
                # Adaptive weight adjustment - increase KD strength
                if self.kd_adaptive:
                    current_kd_weight = torch.sigmoid(self.kd_weight_scheduler) * 1.5  # increase KD weight
                    current_kd_feature_weight = torch.sigmoid(self.kd_feature_weight_scheduler) * 1.0  # increase feature distillation
                else:
                    current_kd_weight = self.kd_weight * 1.5  # increase base weight
                    current_kd_feature_weight = self.kd_feature_weight * 1.0
                    
            except Exception as e:
                print(f" KD calculation failed: {e}")
                kd_loss = torch.tensor(0.0).to(self.device)
                kd_feature_loss = torch.tensor(0.0).to(self.device)
                current_kd_weight = 0.0
                current_kd_feature_weight = 0.0
        else:
            current_kd_weight = 0.0
            current_kd_feature_weight = 0.0
        
        # Total loss calculation
        if self.kd_enabled:
            total_loss = vae_losses['total_loss'] + current_kd_weight * kd_loss + current_kd_feature_weight * kd_feature_loss
        else:
            total_loss = vae_losses['total_loss']
        
        return {
            'student_total_loss': total_loss,
            'vae_loss': vae_losses['total_loss'],
            'recon_loss': vae_losses['recon_loss'],
            'kl_loss_z': vae_losses['kl_loss_z'],
            'kl_loss_u': vae_losses['kl_loss_u'],
            'contrastive_loss': vae_losses['contrastive_loss'],
            'knowledge_distillation_loss': kd_loss,
            'kd_feature_loss': kd_feature_loss
        }
    
    def train_teacher_step(self, 
                          real_images: torch.Tensor, 
                          expert_id: Optional[int] = None) -> Dict[str, float]:
        """Perform one training step for the Teacher module - correct WGAN-GP training ratio"""
        if expert_id is None:
            expert_id = self.current_expert_id
        
        expert_key = f"expert_{expert_id}"
        if expert_key not in self.teacher_experts:
            raise ValueError(f"Expert #{expert_id} does not exist")
        
        # Ensure optimizer and scheduler exist
        if expert_key not in self.teacher_optimizers:
            raise ValueError(f"Optimizer for expert #{expert_id} does not exist")
        if expert_key not in self.teacher_schedulers:
            print(f" Warning: Scheduler for expert #{expert_id} does not exist, skipping scheduler update")
        
        batch_size = real_images.size(0)
        
        # 1. Train discriminator (train every time)
        z = torch.randn(batch_size, self.z_dim).to(self.device)
        fake_images = self.generate_with_expert(z, expert_id)
        real_logits = self.discriminator(real_images)
        fake_logits = self.discriminator(fake_images.detach())
        
        d_loss_real = -torch.mean(real_logits)
        d_loss_fake = torch.mean(fake_logits)
        d_loss_wgan = d_loss_real + d_loss_fake
        
        gp = gradient_penalty(self.discriminator, real_images, fake_images.detach(), self.device)
        d_loss = d_loss_wgan + self.gradient_penalty_weight * gp
        
        self.disc_optimizer.zero_grad()
        d_loss.backward()
        
        # Gentle gradient clipping
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.disc_optimizer.step()
        
        # Adaptive learning rate adjustment: based on loss balance
        if self.critic_iterations % 100 == 0:
            # Disable scheduler update temporarily
            # if self.disc_scheduler:
            #     self.disc_scheduler.step()
            
            # If discriminator is too strong, further reduce its learning rate
            if abs(d_loss.item()) > 20:
                for param_group in self.disc_optimizer.param_groups:
                    param_group['lr'] *= 0.95
        
        # Update counter
        self.critic_iterations += 1
        
        # 2. Update generator based on n_critic ratio
        g_loss = torch.tensor(0.0)
        
        if self.critic_iterations % self.n_critic == 0:
            # Enhanced generator training - add perceptual loss to improve image quality
            z_gen = torch.randn(batch_size, self.z_dim).to(self.device)
            fake_images_gen = self.generate_with_expert(z_gen, expert_id)
            fake_logits_gen = self.discriminator(fake_images_gen)
            
            # Basic WGAN loss
            g_loss_wgan = -torch.mean(fake_logits_gen)
            
            # Perceptual loss - improve image quality
            perceptual_loss = self.compute_perceptual_loss(fake_images_gen, real_images)
            
            # Total generator loss
            g_loss = g_loss_wgan + self.perceptual_weight * perceptual_loss
            
            self.teacher_optimizers[expert_key].zero_grad()
            g_loss.backward()
            
            # Gentle gradient clipping
            torch.nn.utils.clip_grad_norm_(self.teacher_experts[expert_key].parameters(), max_norm=1.0)
            self.teacher_optimizers[expert_key].step()
            
            # Disable generator learning rate scheduler temporarily
            # if expert_key in self.teacher_schedulers and self.teacher_schedulers[expert_key]:
            #     try:
            #         self.teacher_schedulers[expert_key].step()
            #     except Exception as e:
            #         print(f"WARNING: Scheduler update failed for expert #{expert_id}: {e}")
        
        # Update loss history and check stability
        d_loss_val = d_loss.item() if d_loss is not None else 0.0
        g_loss_val = g_loss.item() if isinstance(g_loss, torch.Tensor) else 0.0
        
        self.update_loss_history(d_loss_val, g_loss_val)
        
        # Check stability every 50 iterations (reduced frequency to avoid too frequent checks)
        if self.critic_iterations % 50 == 0:
            stability_changed = self.check_teacher_stability()
        
        # Safe return value handling - ensure training continues, while monitoring quality
        quality_metrics = {
            'discriminator_loss': d_loss_val,
            'generator_loss': g_loss_val,
            'gradient_penalty': gp.item() if gp is not None else 0.0,
            'critic_iterations': self.critic_iterations,
            'generator_updated': self.critic_iterations % self.n_critic == 0,
            'kd_enabled': self.kd_enabled,  # new: return KD state
            'teacher_stable_epochs': self.teacher_stable_epochs,  # new: return stable epochs
            'training_continued': True,  # new: ensure training continues
            'kd_weight': self.kd_weight,  # new: return current KD weight
            'quality_warning': g_loss_val > self.target_g_loss * 2.0  # new: quality warning flag
        }
        
        return quality_metrics
    
    def train_student_step(self, 
                          real_images: torch.Tensor,
                          task_id: torch.Tensor,
                          teacher_expert_id: Optional[int] = None) -> Dict[str, float]:
        """Perform one training step for the Student module"""
        # Compute Student loss
        losses = self.compute_student_loss(real_images, task_id, teacher_expert_id)
        
        # Update Student
        self.student_optimizer.zero_grad()
        losses['student_total_loss'].backward()
        
        # Gradient clipping (new)
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        
        self.student_optimizer.step()
        
        # Disable Student learning rate scheduler temporarily
        # if hasattr(self, 'student_scheduler') and self.student_scheduler and self.critic_iterations % 200 == 0:
        #     self.student_scheduler.step()
        
        # Update KD components (fix: avoid duplicate backward pass)
        # Only update KD components if KD is enabled, but do not interrupt training
        if teacher_expert_id is not None and teacher_expert_id >= 0 and self.kd_adaptive and self.kd_enabled:
            # Recalculate KD loss (not included in student_total_loss)
            student_outputs = self.student(real_images, task_id)
            student_z = student_outputs['z']
            student_u = student_outputs['u']
            
            with torch.no_grad():
                teacher_z = self.kd_mapping(student_z)
                teacher_images = self.generate_with_expert(teacher_z, teacher_expert_id)
            
            student_generated = self.student.decode(student_z, student_u)
            kd_loss = nn.MSELoss()(student_generated, teacher_images.detach())
            
            # Update KD components separately
            self.kd_optimizer.zero_grad()
            kd_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.kd_mapping.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.feature_aligner.parameters(), max_norm=1.0)
            self.kd_optimizer.step()
        
        # Return scalar metrics - safe handling, ensure training does not interrupt
        result = {}
        for k, v in losses.items():
            if v is None:
                result[k] = 0.0
            elif isinstance(v, torch.Tensor):
                result[k] = v.item()
            else:
                result[k] = float(v) if v is not None else 0.0
        
        # Add training status indicator
        result['training_continued'] = True
        result['kd_enabled'] = self.kd_enabled
        
        return result
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'num_experts': len(self.teacher_experts),
            'current_expert_id': self.current_expert_id,
            'expert_performance': self.expert_performance.copy(),
            'fid_threshold': self.fid_threshold,
            'best_expert_id': self.select_best_expert() if self.expert_performance else -1,
            'student_config': self.student.get_config(),
            'system_type': 'DynamicTeacherStudent'
        }
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'teacher_experts': {k: v.state_dict() for k, v in self.teacher_experts.items()},
            'teacher_optimizers': {k: v.state_dict() for k, v in self.teacher_optimizers.items()},
            'discriminator': self.discriminator.state_dict(),
            'disc_optimizer': self.disc_optimizer.state_dict(),
            'student': self.student.state_dict(),
            'student_optimizer': self.student_optimizer.state_dict(),
            'kd_mapping': self.kd_mapping.state_dict(),
            'feature_aligner': self.feature_aligner.state_dict(),
            'kd_weight_scheduler': self.kd_weight_scheduler.data,  # fix: use .data instead of .state_dict()
            'kd_feature_weight_scheduler': self.kd_feature_weight_scheduler.data,  # fix
            'current_expert_id': self.current_expert_id,
            'expert_performance': self.expert_performance,
            'model_config': {
                'z_dim': self.z_dim,
                'student_z_dim': self.student_z_dim,
                'student_u_dim': self.student_u_dim,
                'channels': self.channels,
                'num_tasks': self.num_tasks,
                'fid_threshold': self.fid_threshold,
                'learning_rate': self.learning_rate,
                'student_lr': self.student_lr,
                'beta1': self.beta1,
                'beta': self.beta,
                'contrastive_weight': self.contrastive_weight,
                'kd_weight': self.kd_weight,
                'kd_feature_weight': self.kd_feature_weight,
                'kd_adaptive': self.kd_adaptive
            }
        }
        torch.save(checkpoint, filepath)
        print(f" Model saved to: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Rebuild experts
        self.teacher_experts = nn.ModuleDict()
        self.teacher_optimizers = {}
        
        for expert_key, expert_state in checkpoint['teacher_experts'].items():
            expert = TeacherGenerator(self.z_dim, self.channels, feature_count=4).to(self.device)
            expert.load_state_dict(expert_state)
            optimizer = optim.Adam(expert.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999))
            
            self.teacher_experts[expert_key] = expert
            self.teacher_optimizers[expert_key] = optimizer
        
        # Load other components
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
        self.student.load_state_dict(checkpoint['student'])
        self.student_optimizer.load_state_dict(checkpoint['student_optimizer'])
        self.kd_mapping.load_state_dict(checkpoint['kd_mapping'])
        self.feature_aligner.load_state_dict(checkpoint['feature_aligner'])
        self.kd_weight_scheduler.data = checkpoint['kd_weight_scheduler']  # fix: directly assign .data
        self.kd_feature_weight_scheduler.data = checkpoint['kd_feature_weight_scheduler']  # fix
        
        # Load state
        self.current_expert_id = checkpoint['current_expert_id']
        self.expert_performance = checkpoint['expert_performance']
        
        print(f" Model loaded from {filepath}")
        print(f" Current number of experts: {len(self.teacher_experts)}")
        print(f" Current expert ID: {self.current_expert_id}")
    
    def compute_perceptual_loss(self, fake_images, real_images):
        """Compute perceptual loss - improve image quality"""
        if fake_images.size(1) == 1:  # Grayscale to RGB
            fake_images = fake_images.repeat(1, 3, 1, 1)
        if real_images.size(1) == 1:
            real_images = real_images.repeat(1, 3, 1, 1)
            
        # Ensure images are in [0,1] range
        fake_images = (fake_images + 1.0) / 2.0
        real_images = (real_images + 1.0) / 2.0
        
        # VGG feature extraction
        fake_features = self.vgg(fake_images)
        real_features = self.vgg(real_images)
        
        # Perceptual loss
        perceptual_loss = F.mse_loss(fake_features, real_features)
        return perceptual_loss

    def check_teacher_stability(self, current_epoch: int = None) -> bool:
        """Check if Teacher is stable, decide whether to enable KD"""
        if not self.enable_kd_threshold or len(self.recent_d_losses) < 5:  # reduce history requirement
            return False
            
        # Calculate recent loss statistics
        recent_d_avg = sum(abs(loss) for loss in self.recent_d_losses[-5:]) / 5  # only look at recent 5
        recent_g_avg = sum(abs(loss) for loss in self.recent_g_losses[-5:]) / 5
        
        # Simplified stability condition - adjust based on actual training data
        d_stable = recent_d_avg < 8.0  # loosen D loss requirement (actual average 3.82)
        g_stable = recent_g_avg < 15.0  # loosen G loss requirement (actual average 2.38, but with large fluctuations)
        
        # Simple stability judgment
        is_stable = d_stable and g_stable
        
        if is_stable:
            self.teacher_stable_epochs += 1
            if self.teacher_stable_epochs >= self.min_stable_epochs and not self.kd_enabled:
                self.kd_enabled = True
                print(f" Teacher stable {self.teacher_stable_epochs} epochs, enabling KD!")
                print(f"   D loss: {recent_d_avg:.3f} < 8.0 (stable)")
                print(f"   G loss: {recent_g_avg:.3f} < 15.0 (stable)")
                
                # Adjust KD weight based on loss level
                if recent_d_avg < 5.0 and recent_g_avg < 5.0:  # both losses are low
                    self.kd_weight = min(1.0, self.kd_weight * 1.2)
                    print(f"     Both losses are low, increasing KD weight to: {self.kd_weight:.3f}")
                elif recent_d_avg < 6.0 and recent_g_avg < 10.0:  # moderate losses
                    self.kd_weight = min(1.0, self.kd_weight * 1.1)
                    print(f"     Moderate losses, slightly increasing KD weight to: {self.kd_weight:.3f}")
                else:  # high losses
                    self.kd_weight = max(0.1, self.kd_weight * 0.9)
                    print(f"     High losses, decreasing KD weight to: {self.kd_weight:.3f}")
                
                return True
        else:
            self.teacher_stable_epochs = 0
            if self.kd_enabled:
                self.kd_enabled = False
                print(f" Teacher unstable, pausing KD")
                print(f"   D loss: {recent_d_avg:.3f} >= 8.0 or G loss: {recent_g_avg:.3f} >= 15.0")
        
        return self.kd_enabled
    
    def update_loss_history(self, d_loss: float, g_loss: float):
        """Update loss history"""
        self.recent_d_losses.append(d_loss)
        self.recent_g_losses.append(g_loss)
        
        # Keep history size
        if len(self.recent_d_losses) > self.loss_history_size:
            self.recent_d_losses.pop(0)
        if len(self.recent_g_losses) > self.loss_history_size:
            self.recent_g_losses.pop(0)
    
    def get_training_status(self) -> dict:
        """Get current training status"""
        if len(self.recent_d_losses) >= 5:
            recent_d_avg = sum(abs(loss) for loss in self.recent_d_losses[-5:]) / 5
            recent_g_avg = sum(abs(loss) for loss in self.recent_g_losses[-5:]) / 5
        else:
            recent_d_avg = float('inf')
            recent_g_avg = float('inf')
            
        return {
            'kd_enabled': self.kd_enabled,
            'teacher_stable_epochs': self.teacher_stable_epochs,
            'recent_d_avg': recent_d_avg,
            'recent_g_avg': recent_g_avg,
            'd_target': self.target_d_loss,
            'g_target': self.target_g_loss,
            'progress_to_stability': min(self.teacher_stable_epochs / self.min_stable_epochs, 1.0) if self.teacher_stable_epochs < self.min_stable_epochs else 1.0
        }

    def test_teacher_memory(self, expert_id: int = 0, num_samples: int = 16):
        """Test if Teacher expert remembers previous tasks"""
        print(f" Testing Teacher expert #{expert_id}'s memory...")
        
        expert_key = f"expert_{expert_id}"
        if expert_key not in self.teacher_experts:
            print(f" Expert #{expert_id} does not exist")
            return None
            
        with torch.no_grad():
            # Generate images
            z = torch.randn(num_samples, self.z_dim).to(self.device)
            fake_images = self.teacher_experts[expert_key](z)
            
            print(f"    Generated {num_samples} images")
            print(f"    Image value range: [{fake_images.min():.3f}, {fake_images.max():.3f}]")
            print(f"    Image shape: {fake_images.shape}")
            
            # Simple visual check metrics
            mean_intensity = fake_images.mean().item()
            std_intensity = fake_images.std().item()
            
            print(f"    Mean intensity: {mean_intensity:.3f}")
            print(f"    Std intensity: {std_intensity:.3f}")
            
            # Determine generation quality
            if std_intensity < 0.1:
                print(f"    Possible mode collapse (low std)")
            elif abs(mean_intensity) > 0.8:
                print(f"    Possible saturation (high mean)")
            else:
                print(f"    Generation looks normal")
                
            return fake_images

    def test_student_forgetting(self, old_task_data, old_task_id: int = 0):
        """Test if Student forgot old tasks"""
        print(f"🎓 Testing Student's memory for task {old_task_id}...")
        
        self.student.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(old_task_data):
                if batch_idx >= 5:  # only test a few batches
                    break
                    
                images = images.to(self.device)
                task_ids = torch.full((images.size(0),), old_task_id, 
                                    dtype=torch.long, device=self.device)
                
                # Student reconstruction
                outputs = self.student(images, task_ids)
                reconstruction = outputs['reconstruction']
                
                # Compute reconstruction loss
                recon_loss = torch.nn.MSELoss()(reconstruction, images)
                total_loss += recon_loss.item()
                num_batches += 1
                
                if batch_idx == 0:
                    print(f"  First batch:")
                    print(f"      Original value range: [{images.min():.3f}, {images.max():.3f}]")
                    print(f"      Reconstruction value range: [{reconstruction.min():.3f}, {reconstruction.max():.3f}]")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(f"    Average reconstruction loss: {avg_loss:.4f}")
        
        # Determine forgetting level
        if avg_loss > 0.5:
            print(f"    Severe forgetting (loss > 0.5)")
            forgetting_level = "Severe"
        elif avg_loss > 0.2:
            print(f"    Mild forgetting (loss > 0.2)")
            forgetting_level = "Mild"
        else:
            print(f"    Good memory (loss <= 0.2)")
            forgetting_level = "Good"
            
        return {"avg_loss": avg_loss, "forgetting_level": forgetting_level}

    def test_knowledge_distillation(self, test_images, task_id: int = 0):
        """Test if KD effectively transfers knowledge"""
        print(f" Testing KD effectiveness...")
        print(f"   KD enabled state: {'Enabled' if self.kd_enabled else 'Disabled'}")
        
        if not self.kd_enabled:
            print(f"     KD is disabled, cannot test effectiveness")
            return {"kd_enabled": False}
        
        with torch.no_grad():
            # Student encoding
            student_outputs = self.student(test_images, 
                                         torch.full((test_images.size(0),), task_id, 
                                                   dtype=torch.long, device=self.device))
            student_z = student_outputs['z']
            
            # Map to Teacher space
            teacher_z = self.kd_mapping(student_z)
            
            # Teacher generation
            teacher_images = self.generate_with_expert(teacher_z, self.current_expert_id)
            
            # Student reconstruction
            student_recon = student_outputs['reconstruction']
            
            # Compute knowledge transfer quality
            teacher_student_sim = torch.nn.functional.cosine_similarity(
                teacher_images.view(teacher_images.size(0), -1),
                student_recon.view(student_recon.size(0), -1),
                dim=1
            ).mean().item()
            
            print(f"    Teacher-Student similarity: {teacher_student_sim:.3f}")
            print(f"    Teacher image quality: mean={teacher_images.mean():.3f}, std={teacher_images.std():.3f}")
            print(f"    Student reconstruction quality: mean={student_recon.mean():.3f}, std={student_recon.std():.3f}")
            
            # Determine KD effectiveness
            if teacher_student_sim > 0.7:
                print(f"     Good knowledge transfer (similarity > 0.7)")
                kd_quality = "Good"
            elif teacher_student_sim > 0.5:
                print(f"     Moderate knowledge transfer (similarity > 0.5)")
                kd_quality = "Moderate"
            else:
                print(f"     Poor knowledge transfer (similarity <= 0.5)")
                kd_quality = "Poor"
                
            return {
                "kd_enabled": True,
                "similarity": teacher_student_sim,
                "kd_quality": kd_quality,
                "teacher_stable_epochs": self.teacher_stable_epochs
            }


def test_dynamic_system():
    """Test the dynamic teacher-student system"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Using device: {device}")
    
    # Create system
    dts = DynamicTeacherStudent(
        z_dim=256,  # fix: use correct latent dimension
        student_z_dim=64,
        student_u_dim=16,
        channels=3,
        num_tasks=3,
        fid_threshold=100.0,  # fix: use correct FID threshold
        beta=1.0,
        contrastive_weight=0.1,
        kd_weight=1.0,
        kd_feature_weight=0.5,
        kd_adaptive=True,
        device=device
    )
    
    # Test data
    batch_size = 8
    real_images = torch.randn(batch_size, 3, 32, 32).to(device)
    task_id = torch.randint(0, 3, (batch_size,)).to(device)
    z = torch.randn(batch_size, 256).to(device)  # fix: use correct latent dimension
    
    print(" Testing basic functionality...")
    
    # Test generation
    teacher_images = dts.generate_with_expert(z)
    student_images = dts.generate_with_student(task_id)
    
    print(f" Teacher generated: {teacher_images.shape}")
    print(f" Student generated: {student_images.shape}")
    
    # Test training step
    teacher_metrics = dts.train_teacher_step(real_images)
    student_metrics = dts.train_student_step(real_images, task_id, teacher_expert_id=0)
    
    print(f" Teacher training metrics: {list(teacher_metrics.keys())}")
    print(f" Student training metrics: {list(student_metrics.keys())}")
    
    # Print loss values
    print(f"   Teacher D loss: {teacher_metrics['discriminator_loss']:.4f}")
    print(f"   Teacher G loss: {teacher_metrics['generator_loss']:.4f}")
    print(f"   Student total loss: {student_metrics['student_total_loss']:.4f}")
    print(f"   Student reconstruction loss: {student_metrics['recon_loss']:.4f}")
    print(f"   Student contrastive loss: {student_metrics['contrastive_loss']:.4f}")
    
    # Test expert addition
    dts.add_new_expert()
    info = dts.get_model_info()
    print(f" Model info: Experts={info['num_experts']}, Student type={info['student_config']['type']}")
    
    print(" Dynamic teacher-student system test completed!")
    return dts


if __name__ == "__main__":
    test_dynamic_system() 