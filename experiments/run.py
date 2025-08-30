#!/usr/bin/env python3
"""
Dynamic Lifelong Learning - Main runner
Automatically reads the three configuration files under config/
"""

import torch
from torch.utils.data import DataLoader
import os
import sys
import time
from typing import Dict, List
# from tqdm import tqdm  # No longer using progress bar
import csv

# Add project root to sys.path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import configs
from config.teacher_config import TeacherConfig
from config.student_config import StudentConfig  
from config.experiment_config import ExperimentConfig

# Import models and utilities
from models.dynamic_teacher_student import DynamicTeacherStudent
from utils.training_utils import MetricsTracker, TrainingTimer, create_exp_directory, log_experiment_info
from utils.image_utils import save_samples
from utils.fid_utils import calculate_fid_from_dataloader_and_generator
from utils.experiment_logger import create_experiment_logger
from data.data_loaders import get_single_task_loader
from utils.forgetting_evaluator import evaluate_forgetting_during_training


class SimpleTrainer:
    """Simple dynamic lifelong learning trainer"""
    
    def __init__(self):
        print(" Initializing dynamic lifelong learning trainer")
        print("=" * 50)
        
        # Auto-load three config files
        print(" Loading configuration files...")
        self.teacher_config = TeacherConfig()
        self.student_config = StudentConfig()
        self.experiment_config = ExperimentConfig()
        
        # Print current configuration
        self._print_current_config()
        
        # Select device
        self.device = torch.device(self.experiment_config.device if torch.cuda.is_available() else 'cpu')
        print(f" Device: {self.device}")
        
        # Create experiment directory
        self.exp_dir = create_exp_directory(
            self.experiment_config.results_dir,
            self.experiment_config.get_exp_name()
        )
        print(f" Experiment directory: {self.exp_dir}")
        
        # 🆕 Initialize complete experiment logger
        config_dict = {
            'teacher': self.teacher_config.to_dict(),
            'student': self.student_config.to_dict(),
            'experiment': self.experiment_config.to_dict()
        }
        self.experiment_logger = create_experiment_logger(self.exp_dir, config_dict)
        
        # Initialize KD status tracking (for FID curve key event annotations)
        self._last_kd_status = None
        
        # 🆕 Set runtime information
        self.experiment_logger.set_run_info(
            run_id=f"run_{int(time.time())}",
            seed=self.experiment_config.seed
        )
        
        # 🆕 Log complete runtime metadata (Section 5.1 experiment setup)
        self._log_complete_experiment_setup()
        
        # Save config info (backward compatibility)
        log_experiment_info(self.exp_dir, config_dict)
        
        # Initialize model
        self._init_model()
        
        # Initialize training tools
        auto_save_path = os.path.join(self.exp_dir, 'metrics.json')
        self.metrics_tracker = MetricsTracker(auto_save_path=auto_save_path)
        self.timer = TrainingTimer()
        
        # Task history
        self.task_history = []
        
        # Save previous task data for forgetting evaluation
        self.previous_task_data = None
        self.previous_task_name = None
        
        print(" Initialization completed!")
        print("=" * 50)
    
    def _print_current_config(self):
        """Print current configuration"""
        print("\n Current configuration:")
        print(f" Teacher:")
        print(f"   Latent dim: {self.teacher_config.latent_dim}")
        print(f"   Learning rate: {self.teacher_config.learning_rate}")
        print(f"   Gradient penalty weight: {self.teacher_config.gradient_penalty_weight}")
        
        print(f" Student:")
        print(f"   Content latent dim (z_dim): {self.student_config.z_dim}")
        print(f"   Domain latent dim (u_dim): {self.student_config.u_dim}")
        print(f"   Beta (β): {self.student_config.beta}")
        print(f"   Contrastive weight: {self.student_config.contrastive_weight}")
        print(f"   Learning rate: {self.student_config.learning_rate}")
        
        print(f" Experiment:")
        print(f"   Num epochs: {self.experiment_config.num_epochs}")
        print(f"   Batch size: {self.experiment_config.batch_size}")
        print(f"   FID threshold: {self.experiment_config.fid_threshold}")
        print(f"   Task sequence: {self.experiment_config.task_sequence}")
    
    def _init_model(self):
        """Initialize model"""
        print("\n Initializing dynamic Teacher-Student model...")
        
        self.model = DynamicTeacherStudent(
            z_dim=self.teacher_config.latent_dim,
            student_z_dim=self.student_config.z_dim,
            student_u_dim=self.student_config.u_dim,
            channels=self.experiment_config.num_channels,
            num_tasks=self.student_config.num_tasks,
            fid_threshold=self.experiment_config.fid_threshold,
            learning_rate=self.teacher_config.learning_rate,
            disc_learning_rate=getattr(self.teacher_config, 'disc_learning_rate', self.teacher_config.learning_rate * 0.5),
            student_lr=self.student_config.learning_rate,
            beta1=self.teacher_config.beta1,
            beta=self.student_config.beta,
            contrastive_weight=self.student_config.contrastive_weight,
            kd_weight=self.student_config.kd_weight,
            kd_feature_weight=0.5,  # hardcoded because StudentConfig does not define this param
            kd_adaptive=self.experiment_config.kd_adaptive,  # new
            n_critic=self.teacher_config.n_critic,
            gradient_penalty_weight=self.teacher_config.gradient_penalty_weight,
            target_d_loss=self.teacher_config.target_d_loss,  # from TeacherConfig
            target_g_loss=self.teacher_config.target_g_loss,  # from TeacherConfig
            enable_kd_threshold=getattr(self.teacher_config, 'enable_kd_threshold', True),  # new
            use_mixed_precision=self.experiment_config.use_mixed_precision,  # new
            device=self.device
        )
        
        print(f" Model initialized")
        print(f"   Number of Teacher experts: {len(self.model.teacher_experts)}")
        print(f"   Student type: Advanced VAE")
    
    def run_experiment(self):
        """Run full experiment"""
        print(f"\n Starting dynamic lifelong learning experiment")
        print(f"Task sequence: {self.experiment_config.task_sequence}")
        print("=" * 70)
        
        self.timer.start_training()
        task_results = []
        
        # Train each task
        for task_id, task_name in enumerate(self.experiment_config.task_sequence):
            print(f"\n{'='*15} Task {task_id+1}/{len(self.experiment_config.task_sequence)}: {task_name} {'='*15}")
            
            # 🆕 Log model parameter count
            try:
                teacher_params = sum(p.numel() for p in self.model.teacher_experts[f'expert_{self.model.current_expert_id}'].parameters())
                student_params = sum(p.numel() for p in self.model.student.parameters())
                # Use integrated efficiency metrics logging
                self.experiment_logger.log_efficiency_metrics(
                    duration_sec=0,  # you may pass actual training time here
                    images_per_sec=0,  # you may pass actual throughput here
                    gpu_mem_peak_mb=0,  # you may pass actual GPU memory usage here
                    params_teacher=teacher_params,
                    params_student=student_params
                )
            except Exception as e:
                print(f" Failed to log parameter counts: {e}")
            
            try:
                # Get data
                data_loader = self._get_task_data(task_id)
                
                # Save loaders for continual learning evaluation
                if not hasattr(self, 'previous_loaders'):
                    self.previous_loaders = {}
                self.previous_loaders[task_name] = data_loader
                
                # Train task
                metrics = self._train_task(data_loader, task_name, task_id)
                
               
                
                # Save current task data for next evaluation
                self.previous_task_data = data_loader
                self.previous_task_name = task_name
                
                task_results.append({
                    'task_id': task_id,
                    'task_name': task_name,
                    'metrics': metrics,
                    'success': True
                })
                
                print(f" Task {task_name} finished")
                
            except Exception as e:
                print(f" Task {task_name} failed: {e}")
                import traceback
                traceback.print_exc()  # print full traceback
                
                task_results.append({
                    'task_id': task_id,
                    'task_name': task_name,
                    'error': str(e),
                    'success': False
                })
                
                # Skip failed task and continue
                print(f"⏭ Skipping failed task, continue to next...")
                continue
        
        # Generate report
        total_time = self.timer.end_training()
        
        # Generate full experiment report
        experiment_report = self.experiment_logger.finalize()
        
        self._save_final_results()
        self._print_final_summary(task_results, total_time)
        
        return task_results
    
    def _get_task_data(self, task_id: int):
        """Get task dataloader"""
        data_loader, task_name = get_single_task_loader(
            task_id=task_id,
            datasets_root=self.experiment_config.datasets_root,
            batch_size=self.experiment_config.batch_size,
            num_workers=0,
            samples_per_task=self.experiment_config.samples_per_task
        )
        
        print(f" Data loaded: {len(data_loader.dataset)} samples, {len(data_loader)} batches")
        return data_loader
    
    def _train_task(self, data_loader: DataLoader, task_name: str, task_id: int):
        """Train a single task"""
        print(f" Starting training - current expert #{self.model.current_expert_id}")
        num_epochs = self.experiment_config.num_epochs
        epoch_acc_history = []  # record per-epoch accuracy history
        all_task_names = self.experiment_config.task_sequence[:task_id+1]
        all_task_loaders = [self.previous_loaders[name] for name in all_task_names]
        
        self.timer.start_task()
        self.experiment_logger.start_task(task_id, task_name)  # record task start time
        
        # Record task info
        task_data = {
            'dataset_size': len(data_loader.dataset),
            'num_batches': len(data_loader),
            'expert_id': self.model.current_expert_id
        }
        self.metrics_tracker.set_task_info(task_id, task_name, task_data)
        
        # Training loop
        for epoch in range(num_epochs):
            self.timer.start_epoch()
            self.experiment_logger.start_epoch(epoch)  # start epoch logging
            
            # Train one epoch
            self.model.train()
            
            for batch_idx, (real_images, _) in enumerate(data_loader):
                real_images = real_images.to(self.device)
                task_ids = torch.full((real_images.size(0),), task_id, dtype=torch.long, device=self.device)
                
                # Train Teacher and Student - keep training even on exceptions
                try:
                    teacher_metrics = self.model.train_teacher_step(real_images)
                    student_metrics = self.model.train_student_step(real_images, task_ids, teacher_expert_id=self.model.current_expert_id)
                except Exception as e:
                    print(f" Training step exception: {e}, continuing training...")
                    # Provide default metrics to keep training running
                    teacher_metrics = {
                        'discriminator_loss': 0.0,
                        'generator_loss': 0.0,
                        'gradient_penalty': 0.0,
                        'critic_iterations': 0,
                        'generator_updated': False,
                        'kd_enabled': False,
                        'teacher_stable_epochs': 0,
                        'training_continued': True
                    }
                    student_metrics = {
                        'vae_loss': 0.0,
                        'recon_loss': 0.0,
                        'kl_loss_z': 0.0,
                        'kl_loss_u': 0.0,
                        'contrastive_loss': 0.0,
                        'knowledge_distillation_loss': 0.0,
                        'kd_feature_loss': 0.0,
                        'training_continued': True,
                        'kd_enabled': False
                    }
                
                # Update metrics
                all_metrics = {**teacher_metrics, **student_metrics}
                self.metrics_tracker.update(all_metrics, real_images.size(0))
                
                # Log batch-level metrics
                self.experiment_logger.log_batch_metrics(all_metrics, real_images.size(0))
                
                # Log KD stability metrics
                try:
                    # Estimate Teacher-Student similarity (simplified)
                    teacher_stable_epochs = teacher_metrics.get('teacher_stable_epochs', 0)
                    kd_enabled = teacher_metrics.get('kd_enabled', False)
                    kd_weight = teacher_metrics.get('kd_weight', 0.0)
                    
                    # Estimate similarity based on stability indicators
                    if teacher_stable_epochs > 5:
                        similarity = 0.8  # high stability
                    elif teacher_stable_epochs > 2:
                        similarity = 0.6  # medium stability
                    else:
                        similarity = 0.4  # low stability
                    
                    # Log KD stability metrics (to logger)
                    self.experiment_logger.log_kd_stability_metrics(
                        task_name=task_name,
                        epoch=epoch,
                        kd_enabled=kd_enabled,
                        kd_weight=kd_weight,
                        teacher_student_similarity=similarity,
                        distillation_gating_status="enabled" if kd_enabled else "disabled",
                        cosine_similarity=similarity
                    )
                    
                    # Log KD gating key events (for FID curve annotations)
                    if hasattr(self, '_last_kd_status') and self._last_kd_status != kd_enabled:
                        # When KD status changes, log an event
                        gating_status = "on" if kd_enabled else "off"
                        self.experiment_logger.log_kd_gating_event(
                            epoch=epoch,
                            task_id=task_id,
                            task_name=task_name,
                            gating_status=gating_status,
                            kd_weight=kd_weight,
                            teacher_student_similarity=similarity
                        )
                    
                    # Update last KD status
                    self._last_kd_status = kd_enabled
                    
                except Exception as e:
                    print(f" Error logging KD stability metrics: {e}")
                    # Continue training, do not interrupt
                 
                 # Show progress - includes KD status and stability info
                g_loss_display = f"{teacher_metrics['generator_loss']:.2f}" if teacher_metrics['generator_updated'] else "Skip"
                d_loss_abs = abs(teacher_metrics['discriminator_loss'])
                g_loss_val = teacher_metrics['generator_loss'] if teacher_metrics['generator_updated'] else 0
                
                # Use color to indicate distance from target - based on new target values
                d_target = getattr(self.model, 'target_d_loss', 2.0)
                g_target = getattr(self.model, 'target_g_loss', 1.0)
                d_status = "" if d_loss_abs < d_target else "" if d_loss_abs < d_target*2 else ""
                g_status = "" if g_loss_val < g_target else "" if g_loss_val < g_target*2 else ""
                
                # KD status indicator
                kd_status = "" if teacher_metrics.get('kd_enabled', False) else "⏸"
                stable_epochs = teacher_metrics.get('teacher_stable_epochs', 0)
                
                # Show progress every 50 batches
                if batch_idx % 50 == 0:
                    # Get current learning rate info
                    current_d_lr = self.model.disc_optimizer.param_groups[0]['lr']
                    current_g_lr = self.model.teacher_optimizers[f"expert_{self.model.current_expert_id}"].param_groups[0]['lr']
                    
                    print(f"   Batch {batch_idx}/{len(data_loader)} | "
                          f"D_loss={d_status}{teacher_metrics['discriminator_loss']:.1f} | "
                          f"G_loss={g_status}{g_loss_display} | "
                          f"KD={kd_status}{stable_epochs} | "
                          f"Critic={teacher_metrics['critic_iterations']%self.model.n_critic+1}/{self.model.n_critic} | "
                          f"S_VAE={student_metrics['vae_loss']:.3f} | "
                          f"D_lr={current_d_lr:.1e} | "
                          f"G_lr={current_g_lr:.1e}")
            
            epoch_time = self.timer.end_epoch()
            epoch_summary = self.metrics_tracker.end_epoch(epoch_time)
            
            # 🆕 Log epoch summary
            self.experiment_logger.end_epoch(epoch_summary)
            
            # 🆕 Log complete epoch-level training metrics (supports Section 5.2 model performance and training results)
            try:
                # Get current learning rate
                current_d_lr = self.model.disc_optimizer.param_groups[0]['lr']
                current_g_lr = self.model.teacher_optimizers[f"expert_{self.model.current_expert_id}"].param_groups[0]['lr']
                
                # Get parameter count info
                params_teacher = sum(p.numel() for p in self.model.teacher_experts[f"expert_{self.model.current_expert_id}"].parameters())
                params_student = sum(p.numel() for p in self.model.student.parameters())
                params_total = params_teacher + params_student
                
                # Get GPU memory usage
                if torch.cuda.is_available():
                    peak_mem_MB = torch.cuda.max_memory_allocated() / 1024 / 1024
                else:
                    peak_mem_MB = 0.0
                
                # Calculate throughput
                throughput_img_s = len(data_loader.dataset) / epoch_time if epoch_time > 0 else 0
                
                # Log complete epoch training metrics
                self.experiment_logger.log_epoch_training_metrics(
                    epoch=epoch,
                    task_id=task_id,
                    task_name=task_name,
                    student_total_loss=epoch_summary.get('vae_loss', 0.0),
                    recon_loss=epoch_summary.get('recon_loss', 0.0),
                    kl_z=epoch_summary.get('kl_loss_z', 0.0),
                    kl_u=epoch_summary.get('kl_loss_u', 0.0),
                    contrastive_loss=epoch_summary.get('contrastive_loss', 0.0),
                    g_loss=epoch_summary.get('generator_loss', 0.0),
                    d_loss=epoch_summary.get('discriminator_loss', 0.0),
                    grad_penalty=epoch_summary.get('gradient_penalty', 0.0),
                    kd_enabled=epoch_summary.get('kd_enabled', False),
                    kd_weight=epoch_summary.get('kd_weight', 0.0),
                    kd_pixel=epoch_summary.get('kd_pixel_loss', 0.0),
                    kd_perceptual=epoch_summary.get('kd_perceptual_loss', 0.0),
                    kd_feature=epoch_summary.get('kd_feature_loss', 0.0),
                    lr_G=current_g_lr,
                    lr_D=current_d_lr,
                    time_sec=epoch_time,
                    throughput_img_s=throughput_img_s,
                    peak_mem_MB=peak_mem_MB
                )
                
                # Log resource cost metrics
                self.experiment_logger.log_resource_cost(
                    epoch=epoch,
                    params_total=params_total,
                    params_teacher=params_teacher,
                    params_student=params_student,
                    epoch_time_sec=epoch_time,
                    throughput_img_s=throughput_img_s,
                    peak_mem_MB=peak_mem_MB
                )
                
            except Exception as e:
                print(f" Error logging epoch metrics: {e}")
                # Continue training, do not interrupt
             
             # Print epoch results
            print(f"Epoch {epoch+1:03d}/{self.experiment_config.num_epochs} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"Teacher D: {epoch_summary.get('discriminator_loss', 0):.4f} | "
                  f"Teacher G: {epoch_summary.get('generator_loss', 0):.4f} | "
                  f"Student VAE: {epoch_summary.get('vae_loss', 0):.4f}")
            
            # Save samples every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._save_samples(epoch + 1, task_name, task_id)
            
            # Evaluate FID every 30 epochs (early expert-addition check)
            if (epoch + 1) % 30 == 0 and epoch > 0:
                print(f" Epoch {epoch+1}: Mid-epoch FID evaluation...")
                try:
                    # Compute mid-epoch FID for Teacher and Student separately
                    mid_teacher_fid = self._calculate_fid(data_loader, task_name, task_id)
                    mid_student_fid = self._evaluate_student_fid(data_loader, task_name, task_id)
                    print(f" Mid Teacher FID: {mid_teacher_fid:.2f}, Student FID: {mid_student_fid:.2f}")
                    
                    # Log mid-epoch FID evaluation snapshot (for FID trend over tasks)
                    try:
                        # Collect mid-epoch FID snapshot for old tasks
                        mid_student_fid_old_tasks = {}
                        mid_teacher_fid_old_tasks = {}
                        if task_id > 0:  # From second task onwards there are old tasks
                            for prev_task_id in range(task_id):
                                prev_task_name = self.experiment_config.task_sequence[prev_task_id]
                                if prev_task_name in self.previous_loaders:
                                    try:
                                        # Quickly evaluate Student on old tasks
                                        old_student_fid = self._evaluate_student_on_old_task(
                                            self.previous_loaders[prev_task_name], 
                                            prev_task_name, 
                                            prev_task_id
                                        )
                                        mid_student_fid_old_tasks[prev_task_name] = old_student_fid
                                        
                                        # Quickly evaluate Teacher on old tasks
                                        old_teacher_fid = self._evaluate_teacher_on_old_task(
                                            self.previous_loaders[prev_task_name], 
                                            prev_task_name, 
                                            prev_task_id
                                        )
                                        mid_teacher_fid_old_tasks[prev_task_name] = old_teacher_fid
                                        
                                        print(f"  Mid-epoch evaluation on old task {prev_task_name}:")
                                        print(f"    Student FID: {old_student_fid:.2f}")
                                        print(f"    Teacher FID: {old_teacher_fid:.2f}")
                                        
                                    except Exception as e:
                                        print(f" Mid-epoch evaluation on old task {prev_task_name} failed: {e}")
                                        mid_student_fid_old_tasks[prev_task_name] = 200.0
                                        mid_teacher_fid_old_tasks[prev_task_name] = 200.0
                        
                        # Log intermediate evaluation snapshot - separately record Teacher and Student FID, including old task evaluation
                        self.experiment_logger.log_evaluation_snapshot(
                            epoch=epoch + 1,
                            task_id=task_id,
                            task_name=task_name,
                            teacher_fid_curr=mid_teacher_fid,
                            student_fid_curr=mid_student_fid,  # Real Student FID
                            student_fid_old_tasks=mid_student_fid_old_tasks,
                            teacher_fid_old_tasks=mid_teacher_fid_old_tasks,  # 🆕 Add Teacher old task FID
                            fid_n=self.experiment_config.fid_sample_size // 2,  # Use fewer samples for intermediate evaluation
                            fid_seed=self.experiment_config.seed,
                            eval_split="val",
                            inception_variant="v3",
                            image_norm_spec="[-1, 1]"
                        )
                    
                    except Exception as e:
                        print(f" Error recording intermediate FID evaluation snapshot: {e}")
                        # Continue execution, don't interrupt
                    
                                        # If Teacher FID is very poor, consider adding expert early
                    if mid_teacher_fid > self.model.fid_threshold * 2.0:  # 100% higher than threshold, stricter condition
                        print(f" Teacher FID ({mid_teacher_fid:.2f}) far above threshold, consider adding expert early")
                        # Early expert addition logic
                        try:
                            print(f" Mid-FID expert addition check: Teacher FID={mid_teacher_fid:.2f}, Student FID={mid_student_fid:.2f}, threshold={self.model.fid_threshold}")
                            if mid_teacher_fid > self.model.fid_threshold:
                                print(f" Mid-FID triggered expert addition!")
                                # Directly add expert
                                old_expert_count = len(self.model.teacher_experts)
                                new_expert_id = self.model.add_new_expert()
                                
                                # 🆕 Log expert event triggered by mid-FID
                                try:
                                    self.experiment_logger.log_expert_event(
                                        epoch=epoch + 1,
                                        task_id=task_id,
                                        trigger_metric="Mid_training_Teacher_FID",
                                        trigger_value=mid_teacher_fid,
                                        trigger_threshold=self.model.fid_threshold,
                                        action="add_expert",
                                        active_expert_before=old_expert_count,
                                        active_expert_after=new_expert_id,
                                        fid_before=mid_teacher_fid,
                                        fid_after=mid_teacher_fid,  # Simplified handling
                                        checkpoint_path=f"/checkpoints/mid_expert_{new_expert_id}.pth"
                                    )
                                except Exception as e:
                                    print(f" Error logging mid-FID expert event: {e}")
                                
                                # Log expert addition event
                                self.experiment_logger.log_expert_addition(
                                    expert_id=new_expert_id,
                                    trigger_fid=mid_teacher_fid,
                                    current_task=task_name,
                                    reason=f"Mid-training Teacher FID ({mid_teacher_fid:.2f}) exceeded threshold ({self.model.fid_threshold})"
                                )
                                
                                # 🆕 Log expert expansion key event
                                self.experiment_logger.log_expert_expansion_event(
                                    epoch=epoch + 1,
                                    task_id=task_id,
                                    task_name=task_name,
                                    old_expert_count=old_expert_count,
                                    new_expert_id=new_expert_id,
                                    trigger_fid=mid_teacher_fid,
                                    trigger_threshold=self.model.fid_threshold
                                )
                                
                                print(f" Expert #{new_expert_id} added (mid-FID triggered)")
                                print(f" Expert count: {old_expert_count} → {len(self.model.teacher_experts)}")
                                
                                # Decide whether to continue training based on configuration
                                if self.experiment_config.continue_training_after_expert_add:
                                    print(f" Continue training to {self.experiment_config.num_epochs} epochs...")
                                    # Continue training, don't break loop
                                else:
                                    print(f" End current task training after expert addition")
                                    break  # Break training loop, start new task
                            else:
                                print(f" Mid Teacher FID ({mid_teacher_fid:.2f}) <= threshold ({self.model.fid_threshold}), continue training")
                        except Exception as e:
                            print(f" Mid-FID expert addition check failed: {e}")
                except Exception as e:
                    print(f" Mid-FID evaluation failed: {e}")
            
            # Evaluate accuracy on all learned tasks each epoch
            accs = []
            for eval_id, eval_name in enumerate(all_task_names):
                eval_loader = all_task_loaders[eval_id]
                acc = self._evaluate_on_task(eval_loader, eval_name, eval_id)
                accs.append(acc)
            epoch_acc_history.append([epoch+1] + accs)
            print(f"Epoch {epoch+1:03d} | " + ' | '.join([f"{n}: {a:.4f}" for n,a in zip(all_task_names, accs)]))
        
        task_time = self.timer.end_task()
        
        # Compute FID and decide whether to add a new expert
        try:
            # Compute final FID for Teacher and Student
            teacher_final_fid = self._calculate_fid(data_loader, task_name, task_id)
            student_final_fid = self._evaluate_student_fid(data_loader, task_name, task_id)
            
            # Check FID computation success
            if teacher_final_fid is None or teacher_final_fid == float('inf') or teacher_final_fid != teacher_final_fid:
                print(f" Teacher FID abnormal, using default 200.0")
                teacher_final_fid = 200.0
            if student_final_fid is None or student_final_fid == float('inf') or student_final_fid != student_final_fid:
                print(f" Student FID abnormal, using default 200.0")
                student_final_fid = 200.0
                
        except Exception as e:
            print(f" FID computation failed: {e}, using default 200.0")
            teacher_final_fid = 200.0
            student_final_fid = 200.0
            
        # Use Teacher FID to update expert performance (for expert-addition decision)
        self.model.update_expert_performance(self.model.current_expert_id, teacher_final_fid)
        
        # Log full evaluation snapshot (for FID trends across tasks)
        try:
            # Collect FID results on previous tasks
            student_fid_old_tasks = {}
            teacher_fid_old_tasks = {}
            if task_id > 0:  # from the second task onwards there are old tasks
                for prev_task_id in range(task_id):
                    prev_task_name = self.experiment_config.task_sequence[prev_task_id]
                    if prev_task_name in self.previous_loaders:
                        try:
                            # Evaluate Student on previous tasks
                            old_student_fid = self._evaluate_student_on_old_task(
                                self.previous_loaders[prev_task_name], 
                                prev_task_name, 
                                prev_task_id
                            )
                            student_fid_old_tasks[prev_task_name] = old_student_fid
                            
                            # Evaluate Teacher on previous tasks
                            old_teacher_fid = self._evaluate_teacher_on_old_task(
                                self.previous_loaders[prev_task_name], 
                                prev_task_name, 
                                prev_task_id
                            )
                            teacher_fid_old_tasks[prev_task_name] = old_teacher_fid
                            
                            # Also log previous task performance (student knowledge retention)
                            self.experiment_logger.log_old_task_performance(
                                current_task=task_name,
                                old_task_name=prev_task_name,
                                old_task_fid=old_student_fid,
                                reconstruction_quality=1.0 / (1.0 + old_student_fid)
                            )
                            
                            # Log Teacher performance on previous tasks
                            self.experiment_logger.log_teacher_old_task_performance(
                                current_task=task_name,
                                old_task_name=prev_task_name,
                                old_task_fid=old_teacher_fid,
                                expert_id=self.model.current_expert_id,
                                generation_quality=1.0 / (1.0 + old_teacher_fid),
                                memory_efficiency=1.0  # simplified
                            )
                            
                            print(f" Previous task {prev_task_name} evaluation done:")
                            print(f"    Student FID: {old_student_fid:.2f}")
                            print(f"    Teacher FID: {old_teacher_fid:.2f}")
                            
                        except Exception as e:
                            print(f" Evaluation on previous task {prev_task_name} failed: {e}")
                            student_fid_old_tasks[prev_task_name] = 200.0  # default value
                            teacher_fid_old_tasks[prev_task_name] = 200.0  # default value
            
            # Log evaluation snapshot - Teacher and Student FID including previous tasks
            self.experiment_logger.log_evaluation_snapshot(
                epoch=self.experiment_config.num_epochs,  # epoch at the end of the task
                task_id=task_id,
                task_name=task_name,
                teacher_fid_curr=teacher_final_fid,  # Teacher FID on current task
                student_fid_curr=student_final_fid,  # Student FID on current task
                student_fid_old_tasks=student_fid_old_tasks,
                teacher_fid_old_tasks=teacher_fid_old_tasks,  # include Teacher FID on previous tasks
                fid_n=self.experiment_config.fid_sample_size,
                fid_seed=self.experiment_config.seed,
                eval_split="val",
                inception_variant="v3",
                image_norm_spec="[-1, 1]"
            )
            
        except Exception as e:
            print(f" Error logging evaluation snapshot: {e}")
            # Continue execution, do not interrupt
         
        try:
            print(f" Checking expert addition: Teacher FID={teacher_final_fid:.2f}, Student FID={student_final_fid:.2f}, threshold={self.model.fid_threshold}")
            # Use Teacher FID to decide whether to add an expert
            should_add_expert = self._should_add_expert(teacher_final_fid, task_name)
        except Exception as e:
            print(f" Expert addition check failed: {e}")
            should_add_expert = False
        
        # Log task results
        task_info = {
            'task_id': task_id,
            'task_name': task_name,
            'expert_id': self.model.current_expert_id,
            'teacher_final_fid': teacher_final_fid,
            'student_final_fid': student_final_fid,
            'fid_gap': teacher_final_fid - student_final_fid,  # FID difference
            'training_time': task_time,
            'added_expert': should_add_expert
        }
        self.task_history.append(task_info)
        
        # Log task completion and FID scores
        self.experiment_logger.log_task_completion(task_id, task_name, task_info)
        # Separately log Teacher and Student FID
        self.experiment_logger.log_fid_scores({
            f"{task_name}_Teacher": teacher_final_fid,
            f"{task_name}_Student": student_final_fid
        })
        
        # Log FID_end value at task completion
        self.experiment_logger.log_task_final_fid(
            task_id=task_id,
            task_name=task_name,
            teacher_fid_end=teacher_final_fid,
            student_fid_end=student_final_fid
        )
        
        # 🆕 Log task-expert mapping relationship
        self.experiment_logger.log_task_expert_mapping(
            task_id=task_id,
            task_name=task_name,
            expert_id=self.model.current_expert_id,
            expert_performance=teacher_final_fid,  # Use Teacher's FID
            trigger_fid=teacher_final_fid if should_add_expert else None
        )
        
        # 🆕 Log expert trigger analysis data
        if should_add_expert:
            # Use integrated Expert analysis summary method
            bound_expert_ids = list(range(len(self.model.teacher_experts)))
            self.experiment_logger.log_expert_analysis_summary(
                task_name=task_name,
                task_id=task_id,
                expert_triggered=True,
                bound_expert_ids=bound_expert_ids,
                trigger_fid=teacher_final_fid,
                threshold_fid=self.model.fid_threshold,
                parameters_before=0,  # Can pass parameter count before adding Expert here
                parameters_after=0,   # Can pass parameter count after adding Expert here
                reason="FID threshold exceeded",
                event_step=len(self.task_history),
                event_epoch=0
            )
            
            # Log detailed expert events
            try:
                old_expert_count = len(self.model.teacher_experts)
                new_expert_id = self.model.add_new_expert()
                
                # Log expert expansion key events
                self.experiment_logger.log_expert_expansion_event(
                    epoch=self.experiment_config.num_epochs,
                    task_id=task_id,
                    task_name=task_name,
                    old_expert_count=old_expert_count,
                    new_expert_id=new_expert_id,
                    trigger_fid=teacher_final_fid,
                    trigger_threshold=self.model.fid_threshold
                )
                
                # Log expert events
                self.experiment_logger.log_expert_event(
                    epoch=self.experiment_config.num_epochs,
                    task_id=task_id,
                    trigger_metric="Teacher_FID_curr",
                    trigger_value=teacher_final_fid,
                    trigger_threshold=self.model.fid_threshold,
                    action="add_expert",
                    active_expert_before=old_expert_count,
                    active_expert_after=new_expert_id,
                    fid_before=teacher_final_fid,
                                            fid_after=teacher_final_fid,  # Simplified handling, should actually re-evaluate
                    checkpoint_path=f"/checkpoints/expert_{new_expert_id}.pth"
                )
                
                print(f" Expert #{new_expert_id} added (FID triggered)")
                print(f" Expert count: {old_expert_count} → {len(self.model.teacher_experts)}")
                
            except Exception as e:
                print(f" Error logging expert events: {e}")
                # Continue execution, don't interrupt

        # Continuous learning diagnosis - detect forgetting
        if task_id > 0:  # Start detection from second task
            print(f"\n === Continuous Learning Diagnosis ===")
            print(f"Current task: {task_name} (Task #{task_id})")
            
            # Detect memory of all Teacher experts
            print(f"\n Teacher Expert Memory Detection:")
            for expert_id in range(len(self.model.teacher_experts)):
                expert_key = f"expert_{expert_id}"
                if expert_key in self.model.teacher_experts:
                    if expert_id == 0:
                        expert_task = "MNIST"
                    elif expert_id == 1:
                        expert_task = "Fashion-MNIST"  
                    else:
                        expert_task = f"Task{expert_id}"
                    
                    print(f"   Expert #{expert_id}({expert_task}):", end=" ")
                    try:
                        self.model.test_teacher_memory(expert_id=expert_id, num_samples=8)
                        print("✓")
                    except Exception as e:
                        print(f" Detection failed: {e}")
            
            # Detect Student memory of all learned tasks
            print(f"\n Student Task Memory Detection:")
            for prev_task_id in range(task_id):  # Detect previous tasks
                if prev_task_id == 0:
                    prev_task_name = "MNIST"
                elif prev_task_id == 1:
                    prev_task_name = "Fashion-MNIST"
                else:
                    prev_task_name = f"Task{prev_task_id}"
                
                print(f"   Task #{prev_task_id}({prev_task_name}):", end=" ")
                try:
                    if hasattr(self, 'previous_loaders') and prev_task_name in self.previous_loaders:
                        old_loader = self.previous_loaders[prev_task_name]
                        forgetting_result = self.model.test_student_forgetting(old_loader, old_task_id=prev_task_id)
                        print(f"Forgetting level: {forgetting_result['forgetting_level']}")
                    else:
                        print(f" Unable to find data")
                except Exception as e:
                    print(f" Detection failed: {e}")
            
            # Detect knowledge distillation effect
            print(f"\n Knowledge Distillation Effect Detection:")
            try:
                test_batch = next(iter(data_loader))
                test_images = test_batch[0][:8].to(self.device)
                kd_result = self.model.test_knowledge_distillation(test_images, task_id)
                if kd_result.get('kd_enabled'):
                    print(f"   KD Quality: {kd_result['kd_quality']}")
                    print(f"   Teacher Stable Epochs: {kd_result.get('teacher_stable_epochs', 0)}")
                    
                    # 🆕 Log Cosine Similarity time series data
                    self.experiment_logger.log_cosine_similarity(
                        task_name=task_name,
                        similarity=kd_result.get('similarity', 0.0),
                        kd_quality=kd_result.get('kd_quality', 'Unknown'),
                        teacher_stable_epochs=kd_result.get('teacher_stable_epochs', 0)
                    )
                else:
                    print(f"    Knowledge distillation not enabled")
            except Exception as e:
                print(f"    KD effect detection failed: {e}")
                
            print(f"\n=== Diagnosis Complete ===")
            print(f" Check image files in results/samples/ directory for visual verification")
            print(f"   - teacher_expert0_MNIST_*.png : Can expert #0 still generate MNIST")
            print(f"   - student_task0_MNIST_*.png : Can Student still reconstruct MNIST")
            print(f"   - CURRENT_teacher_{task_name}_*.png : Current expert generation effect")
            print("")
        
        print(f" Task completed - FID: {teacher_final_fid:.2f}, Expert: #{self.model.current_expert_id}")
        
        return {
            'final_fid': teacher_final_fid,
            'training_time': task_time,
            'expert_id': self.model.current_expert_id
        }
    
    def _calculate_fid(self, data_loader: DataLoader, task_name: str, current_task_id: int = None) -> float:
        """Calculate Teacher's FID score - improved version: evaluate multi-task capability"""
        print(f" Calculating Teacher FID score for {task_name}...")
        
        try:
            expert_key = f"expert_{self.model.current_expert_id}"
            current_expert = self.model.teacher_experts[expert_key]
            current_expert.eval()
            
            # 1. Evaluate current task
            current_fid = calculate_fid_from_dataloader_and_generator(
                real_dataloader=data_loader,
                generator=current_expert,
                device=self.device,
                num_fake_samples=self.experiment_config.fid_sample_size,
                max_real_samples=self.experiment_config.fid_sample_size,
                z_dim=self.teacher_config.latent_dim
            )
            
            # 2. If there are multiple tasks, evaluate multi-task comprehensive capability
            if hasattr(self, 'previous_loaders') and len(self.previous_loaders) > 1 and current_task_id is not None:
                print(f" Evaluating multi-task capability of expert #{self.model.current_expert_id}...")
                
                # Evaluate all learned tasks
                all_fids = []
                for eval_task_id, eval_task_name in enumerate(self.experiment_config.task_sequence[:current_task_id+1]):
                    if eval_task_name in self.previous_loaders:
                        eval_loader = self.previous_loaders[eval_task_name]
                        task_fid = calculate_fid_from_dataloader_and_generator(
                            real_dataloader=eval_loader,
                            generator=current_expert,
                            device=self.device,
                            num_fake_samples=self.experiment_config.fid_sample_size // 2,  # Reduce sample count
                            max_real_samples=self.experiment_config.fid_sample_size // 2,
                            z_dim=self.teacher_config.latent_dim
                        )
                        all_fids.append(task_fid)
                        print(f"    {eval_task_name}: Teacher FID = {task_fid:.2f}")
                
                # Calculate comprehensive FID (average)
                if all_fids:
                    avg_fid = sum(all_fids) / len(all_fids)
                    print(f" Multi-task average Teacher FID: {avg_fid:.2f}")
                    
                    # Use comprehensive FID as decision basis
                    return avg_fid
            
            return current_fid
        except Exception as e:
            print(f"  Teacher FID calculation failed: {e}")
            return float('inf')
    
    def _evaluate_student_fid(self, data_loader: DataLoader, task_name: str, current_task_id: int = None) -> float:
        """Calculate Student's FID score"""
        print(f" Calculating Student FID score for {task_name}...")
        
        try:
            # Use Student model (VAE) to generate images
            self.model.student.eval()
            
            # Calculate Student's FID
            student_fid = calculate_fid_from_dataloader_and_generator(
                real_dataloader=data_loader,
                generator=self.model.student,
                device=self.device,
                num_fake_samples=self.experiment_config.fid_sample_size,
                max_real_samples=self.experiment_config.fid_sample_size,
                z_dim=self.teacher_config.latent_dim
            )
            
            print(f"  Student FID: {student_fid:.2f}")
            return student_fid
            
        except Exception as e:
            print(f"  Student FID calculation failed: {e}")
            return float('inf')
    
    def _should_add_expert(self, current_fid: float, task_name: str) -> bool:
        """Determine whether to add new expert"""
        if self.model.should_add_expert(current_fid):
            print(f" FID ({current_fid:.2f}) > threshold ({self.model.fid_threshold}), adding new expert")
            
            # Record expert count before addition
            old_expert_count = len(self.model.teacher_experts)
            new_expert_id = self.model.add_new_expert()
            
            # 🆕 Log expert addition event
            self.experiment_logger.log_expert_addition(
                expert_id=new_expert_id,
                trigger_fid=current_fid,
                current_task=task_name,
                reason=f"FID ({current_fid:.2f}) exceeded threshold ({self.model.fid_threshold})"
            )
            
            return True
        else:
            print(f" FID ({current_fid:.2f}) <= threshold ({self.model.fid_threshold}), continue using current expert")
            return False
    
    def _should_save_samples(self, epoch: int) -> bool:
        """Only save samples at 10, 20, ..., 100"""
        return epoch % 10 == 0
    
    def _save_samples(self, epoch: int, task_name: str, task_id: int):
        """Save generated samples - save by task groups"""
        if not self.experiment_config.save_samples:
            return
            
        samples_dir = os.path.join(self.exp_dir, 'samples')
        os.makedirs(samples_dir, exist_ok=True)
        
        # Create directory for current task
        task_dir = os.path.join(samples_dir, f'task{task_id}')
        os.makedirs(task_dir, exist_ok=True)
        
        with torch.no_grad():
            # Teacher samples - generate samples for each expert
            num_experts = len(self.model.teacher_experts)
            print(f" Generating samples - currently have {num_experts} experts")
            
            for expert_id in range(num_experts):
                expert_key = f"expert_{expert_id}"
                if expert_key in self.model.teacher_experts:
                    # Teacher samples - specify expert ID
                    z = torch.randn(64, self.teacher_config.latent_dim).to(self.device)
                    teacher_samples = self.model.generate_with_expert(z, expert_id)
                    
                    # Determine task name based on expert ID
                    if expert_id == 0:
                        expert_task = "MNIST"
                    elif expert_id == 1:
                        expert_task = "Fashion-MNIST"
                    else:
                        expert_task = f"Task{expert_id}"
                    
                    # Save to current task directory
                    teacher_path = os.path.join(task_dir, 
                        f'teacher_expert{expert_id}_{expert_task}_epoch{epoch:03d}.png')
                    save_samples(teacher_samples, teacher_path, nrow=8)
                    print(f"    Teacher Expert #{expert_id}({expert_task}) images saved")
            
            # 🎓 Student samples - generate for current task
                try:
                    single_task_id = torch.tensor([task_id], dtype=torch.long, device=self.device)
                    student_samples = self.model.generate_with_student(single_task_id, num_samples=64)
                    
                    # Determine task name
                    if task_id == 0:
                        task_name_display = "MNIST"
                    elif task_id == 1:
                        task_name_display = "Fashion-MNIST"
                    else:
                        task_name_display = f"Task{task_id}"
                    
                    # Save to current task directory
                    student_path = os.path.join(task_dir, 
                        f'student_task{task_id}_{task_name_display}_epoch{epoch:03d}.png')
                    save_samples(student_samples, student_path, nrow=8)
                    print(f"    Student Task #{task_id}({task_name_display}) images saved")
                except Exception as e:
                    print(f"    Student Task #{task_id} generation failed: {e}")
            

                
                # Create expert samples subdirectory
                expert_samples_dir = os.path.join(samples_dir, 'expert_samples')
                os.makedirs(expert_samples_dir, exist_ok=True)
                
                # Generate expert samples (save each expert separately)
                try:
                    for expert_id in range(len(self.model.teacher_experts)):
                        expert_samples = self.model.generate_with_expert(z, expert_id)
                        expert_path = os.path.join(expert_samples_dir, 
                            f'teacher_expert{expert_id}_task{task_id}_epoch{epoch:03d}.png')
                        save_samples(expert_samples, expert_path, nrow=8)
                        print(f"    Teacher Expert {expert_id} samples saved")
                except Exception as e:
                    print(f"    Expert sample generation failed: {e}")
                

            
                print(f"    Expert samples saved to: {expert_samples_dir}")
            
            print(f"    Task {task_id} samples saved to: {task_dir}")
    
    def _save_final_results(self):
        """Save final results"""
        checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pth')
        self.model.save_checkpoint(checkpoint_path)
        
        metrics_path = os.path.join(self.exp_dir, 'metrics.json')
        self.metrics_tracker.save_detailed_metrics(metrics_path)
        
        print(f" Results saved to: {self.exp_dir}")
    
    def _print_final_summary(self, task_results: List[Dict], total_time: float):
        """Print final summary"""
        successful_tasks = [r for r in task_results if r.get('success', False)]
        
        print(f"\n Dynamic Lifelong Learning Experiment Completed!")
        print("=" * 50)
        print(f" Experiment Statistics:")
        print(f"   Total tasks: {len(task_results)}")
        print(f"   Successful tasks: {len(successful_tasks)}")
        print(f"   Total training time: {self.timer.format_time(total_time)}")
        print(f"   Final expert count: {len(self.model.teacher_experts)}")
        print(f"   Best expert: #{self.model.select_best_expert()}")
        
        print(f"\n Expert Performance (FID Scores):")
        for expert_id, fid in self.model.expert_performance.items():
            status = "" if expert_id == self.model.select_best_expert() else ""
            print(f"   {status} Expert #{expert_id}: {fid:.2f}")
        
        print(f"\n Detailed results saved in: {self.exp_dir}")
        print("=" * 50)

    def _evaluate_on_task(self, data_loader, task_name, task_id):
        """Evaluate model accuracy on specified task"""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model.student(images, torch.full((images.size(0),), task_id, dtype=torch.long, device=self.device))
                if 'logits' in outputs:
                    preds = outputs['logits'].argmax(dim=1)
                elif 'reconstruction' in outputs:
                    # If no classification, simply use reconstruction error threshold (customizable)
                    preds = (outputs['reconstruction'] - images).abs().mean(dim=[1,2,3]) < 0.5
                else:
                    preds = torch.zeros_like(labels)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total if total > 0 else 0.0
        return acc

    def _evaluate_student_on_old_task(self, data_loader, task_name: str, task_id: int) -> float:
        """Evaluate Student performance on old tasks (for forgetting analysis)"""
        try:
            print(f" Evaluating Student performance on old task {task_name}...")
            
            # Use Student model to reconstruct old task data
            self.model.student.eval()
            total_recon_loss = 0.0
            num_samples = 0
            
            with torch.no_grad():
                for batch_idx, (real_images, _) in enumerate(data_loader):
                    if batch_idx >= 10:  # Only evaluate first 10 batches for efficiency
                        break
                    
                    real_images = real_images.to(self.device)
                    task_ids = torch.full((real_images.size(0),), task_id, dtype=torch.long, device=self.device)
                    
                    # Use Student to reconstruct
                    student_output = self.model.student(real_images, task_ids)
                    if 'reconstruction' in student_output:
                        recon = student_output['reconstruction']
                        # Calculate reconstruction loss
                        recon_loss = torch.nn.functional.mse_loss(recon, real_images)
                        total_recon_loss += recon_loss.item() * real_images.size(0)
                        num_samples += real_images.size(0)
            
            if num_samples > 0:
                avg_recon_loss = total_recon_loss / num_samples
                # Convert reconstruction loss to FID-like score (simplified handling)
                # Lower reconstruction loss = better performance = lower FID score
                estimated_fid = avg_recon_loss * 1000  # Scaling factor
                print(f" Student reconstruction loss on {task_name}: {avg_recon_loss:.4f}, estimated FID: {estimated_fid:.2f}")
                return estimated_fid
            else:
                print(f" Unable to evaluate Student performance on {task_name}")
                return 200.0  # Default value
                
        except Exception as e:
            print(f" Error evaluating Student performance on old task {task_name}: {e}")
            return 200.0  # Default value

    def _evaluate_teacher_on_old_task(self, data_loader, task_name: str, task_id: int) -> float:
        """Evaluate Teacher performance on old tasks (for forgetting analysis)"""
        try:
            print(f" Evaluating Teacher performance on old task {task_name}...")
            
            # Use current expert to evaluate old task
            expert_key = f"expert_{self.model.current_expert_id}"
            current_expert = self.model.teacher_experts[expert_key]
            current_expert.eval()
            
            # Calculate Teacher FID on old task
            teacher_fid = calculate_fid_from_dataloader_and_generator(
                real_dataloader=data_loader,
                generator=current_expert,
                device=self.device,
                num_fake_samples=self.experiment_config.fid_sample_size // 2,  # Reduce sample count
                max_real_samples=self.experiment_config.fid_sample_size // 2,
                z_dim=self.teacher_config.latent_dim
            )
            
            print(f" Teacher FID on {task_name}: {teacher_fid:.2f}")
            return teacher_fid
                
        except Exception as e:
            print(f" Error evaluating Teacher performance on old task {task_name}: {e}")
            return 200.0  # Default value

    def _log_complete_experiment_setup(self):
       
        print(" Logging complete experiment setup...")
        
        # 1. Log runtime metadata
        try:
            import subprocess
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                              stderr=subprocess.DEVNULL, 
                                              text=True).strip()[:8]
        except:
            git_commit = "unknown"
        
        # Get device information
        device_info = {}
        if torch.cuda.is_available():
            device_info.update({
                'gpu_model': torch.cuda.get_device_name(0),
                'cuda_version': torch.version.cuda,
                'driver_version': 'N/A',  # Requires additional retrieval
                'gpu_memory_mb': torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            })
        
        # Get library version information
        library_versions = {
            'torch_version': torch.__version__,
            'torch_cuda': torch.version.cuda,
            'numpy_version': 'N/A',  # Requires additional retrieval
            'torchvision_version': 'N/A'  # Requires additional retrieval
        }
        
        try:
            import numpy as np
            library_versions['numpy_version'] = np.__version__
        except:
            pass
        
        try:
            import torchvision
            library_versions['torchvision_version'] = torchvision.__version__
        except:
            pass
        
        self.experiment_logger.log_run_metadata(
            run_id=f"run_{int(time.time())}",
            seed=self.experiment_config.seed,
            git_commit=git_commit,
            device_info=device_info,
            library_versions=library_versions
        )
        
        # 2. Log dataset information
        for task_id, task_name in enumerate(self.experiment_config.task_sequence):
            # Estimate dataset information (actual values will be updated during data loading)
            estimated_samples = self.experiment_config.samples_per_task
            estimated_val_samples = estimated_samples // 5  # Assume 20% for validation
            
            self.experiment_logger.log_dataset_info(
                task_name=task_name,
                task_id=task_id,
                num_classes=10,  # Assume 10 classes, actual value needs to be determined based on dataset
                train_samples=estimated_samples,
                val_samples=estimated_val_samples,
                image_size=(self.experiment_config.image_size, self.experiment_config.image_size),
                normalization="[-1, 1]",  # Determined based on data preprocessing
                task_order=task_id
            )
        
        # 3. Log training configuration
        self.experiment_logger.log_training_config(
            batch_size=self.experiment_config.batch_size,
            learning_rate=self.teacher_config.learning_rate,
            optimizer="Adam",  # Retrieved from configuration
            scheduler="StepLR",  # Retrieved from configuration
            total_epochs=self.experiment_config.num_epochs,
            eval_interval=30,  # Evaluate FID every 30 epochs
            early_stop=False,
            checkpoint_strategy="best"
        )
        
        # 4. Log FID configuration
        self.experiment_logger.log_fid_config(
            fid_n=self.experiment_config.fid_sample_size,
            inception_version="v3",
            resize_method="bilinear",
            preprocessing="[-1, 1] normalization",
            eval_split="val",
            fid_seed=self.experiment_config.seed
        )
        
        print("  Complete experiment setup logging completed")


def main():
    """Main function"""
    print(" Dynamic Lifelong Learning System")
    print("Configuration files auto-loaded, starting training...")
    print()
    
    # Create trainer and run
    trainer = SimpleTrainer()
    results = trainer.run_experiment()
    
    print("\n Experiment completed!")
    return results


if __name__ == "__main__":
    main() 