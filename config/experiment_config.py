#!/usr/bin/env python3


import os
from dataclasses import dataclass
from typing import List

@dataclass
class ExperimentConfig:
    """Standard experiment configuration - for research"""
    
    # === Basic experiment settings ===
    experiment_name: str = "AdvancedDynamicTeaching"
    seed: int = 42                  # random seed
    device: str = "cuda"            # device
    gpu_id: int = 0                 # GPU device id
    
    # === Training parameters ===
    num_epochs: int = 150            # training epochs - increased for better results
    batch_size: int = 128           # batch size - increased for stability
    samples_per_task: int = 40000  # samples per task - increased for quality
    
    # === Dataset settings ===
    datasets_root: str = "./datasets"          # datasets root - relative path
    task_sequence: List[str] = None             # task sequence
    image_size: int = 32                        # image size
    num_channels: int = 3                       # num channels
    
    # === Expert management ===
    fid_threshold: float = 100.0    # FID threshold (to add new expert) - higher to reduce triggers
    max_experts: int = 10           # max number of experts
    
    # === Saving ===
    results_dir: str = "./results"  # results directory
    save_interval: int = 10         # save interval (epoch)
    sample_grid_size: int = 8       # grid size for generated samples
    save_samples: bool = True       # whether to save generated samples
    save_plots: bool = True         # whether to save plots
    
    # === Evaluation ===
    fid_sample_size: int = 2000     # number of samples for FID
    
    # === Optimization ===
    use_mixed_precision: bool = True    # mixed precision training
    gradient_clip_norm: float = 1.0     # gradient clipping norm
    kd_adaptive: bool = True            # adaptive knowledge distillation
    task_aware_expert_selection: bool = True  # task-aware expert selection
    
    # === Expert training control ===
    continue_training_after_expert_add: bool = True  # continue training after adding an expert
    
    def __post_init__(self):
        """Parameter validation and defaults"""
        if self.task_sequence is None:
            self.task_sequence = ["MNIST", "FashionMNIST"]  # supported datasets
        
        # Ensure directories exist
        os.makedirs(self.datasets_root, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def get_exp_name(self) -> str:
        """Generate experiment name"""
        return f"{self.experiment_name}_epochs{self.num_epochs}_bs{self.batch_size}"
    
    def to_dict(self) -> dict:
        """Convert to dict"""
        return {
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'device': self.device,
            'gpu_id': self.gpu_id,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'samples_per_task': self.samples_per_task,
            'datasets_root': self.datasets_root,
            'task_sequence': self.task_sequence,
            'image_size': self.image_size,
            'num_channels': self.num_channels,
            'fid_threshold': self.fid_threshold,
            'max_experts': self.max_experts,
            'results_dir': self.results_dir,
            'save_interval': self.save_interval,
            'sample_grid_size': self.sample_grid_size,
            'save_samples': self.save_samples,
            'save_plots': self.save_plots,
            'fid_sample_size': self.fid_sample_size,
            'continue_training_after_expert_add': self.continue_training_after_expert_add,
            'config_type': 'ExperimentConfig'
        }


@dataclass
class ExperimentConfigLong(ExperimentConfig):
    """Longer training configuration - for final results"""
    num_epochs: int = 100           # more epochs
    batch_size: int = 128           # large batch size
    samples_per_task: int = 20000   # more samples
    save_interval: int = 20         # less frequent saves
    fid_threshold: float = 150.0    # stricter FID threshold
    continue_training_after_expert_add: bool = True  # continue training after adding expert 


@dataclass
class ExperimentConfigMultiExpert(ExperimentConfig):
    """Multi-expert testing configuration - for full Teacher-Student system"""
    
    # === Expert management ===
    fid_threshold: float = 100.0    # lower threshold, easier to trigger new experts
    max_experts: int = 5           # cap number of experts
    
    # === Training parameters ===
    num_epochs: int = 100          # fewer epochs for quick tests
    batch_size: int = 64           # smaller batch to increase steps
    samples_per_task: int = 20000  # fewer samples
    
    # === Evaluation ===
    fid_sample_size: int = 1000    # fewer samples for faster eval
    save_interval: int = 5         # save more frequently
    
    # === Expert training control ===
    continue_training_after_expert_add: bool = True  # continue training after adding expert
    
    def __post_init__(self):
        """Parameter validation and defaults"""
        if self.task_sequence is None:
            self.task_sequence = ["MNIST", "FashionMNIST"]
        
        # Ensure directories exist
        os.makedirs(self.datasets_root, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def get_exp_name(self) -> str:
        """Generate experiment name"""
        return f"{self.experiment_name}_MultiExpert_epochs{self.num_epochs}_bs{self.batch_size}_fid{self.fid_threshold}" 