"""
Training utilities
"""

import os
import time
import json
import torch
from typing import Dict, Any
from collections import defaultdict


class MetricsTracker:
    
    
    def __init__(self, auto_save_path: str = None):
        self.metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(list)
        self.current_epoch = 0
        self.auto_save_path = auto_save_path
        
        # Enhanced feature: detailed logging
        self.detailed_log = []
        self.task_info = {}
        self.timing_info = {}
        
        import time
        self.start_time = time.time()
        
    def update(self, metrics: Dict[str, float], batch_size: int = 1):
        """Update metrics"""
        for key, value in metrics.items():
            self.epoch_metrics[key].append(value)
    
        # Detailed logging: save per-batch information
        import time
        log_entry = {
            'timestamp': time.time() - self.start_time,
            'epoch': self.current_epoch,
            'batch_metrics': metrics.copy(),
            'batch_size': batch_size
        }
        self.detailed_log.append(log_entry)
    
    def end_epoch(self, epoch_time: float = 0.0) -> Dict[str, float]:
        """End current epoch and return averaged metrics"""
        epoch_avg = {}
        for key, values in self.epoch_metrics.items():
            if values:
                avg_value = sum(values) / len(values)
                epoch_avg[key] = avg_value
                self.metrics[key].append(avg_value)
        
        # Record epoch timing information
        import time
        self.timing_info[f'epoch_{self.current_epoch}'] = {
            'duration': epoch_time,
            'timestamp': time.time() - self.start_time,
            'metrics': epoch_avg.copy()
        }
        
        # Clear current epoch metrics
        self.epoch_metrics.clear()
        self.current_epoch += 1
        
        # Save in real time
        if self.auto_save_path:
            self.save_detailed_metrics(self.auto_save_path)
        
        return epoch_avg
    
    def get_epoch_averages(self) -> Dict[str, float]:
        """Get current epoch average metrics"""
        epoch_avg = {}
        for key, values in self.epoch_metrics.items():
            if values:
                epoch_avg[key] = sum(values) / len(values)
        return epoch_avg
    
    def save_metrics(self, filepath: str):
        """Save basic metrics to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
    
    def set_task_info(self, task_id: int, task_name: str, task_data: dict = None):
        """Record task information"""
        import time
        self.task_info[f'task_{task_id}'] = {
            'task_id': task_id,
            'task_name': task_name,
            'start_time': time.time() - self.start_time,
            'start_epoch': self.current_epoch,
            'data': task_data or {}
        }
    
    def save_detailed_metrics(self, base_path: str):
        """Save detailed metrics to multiple files"""
        import os
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        
        # Save basic metrics
        with open(base_path, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
        
        # Save detailed logs (keep last 1000 entries to avoid huge files)
        detailed_path = base_path.replace('.json', '_detailed.json')
        recent_logs = self.detailed_log[-1000:] if len(self.detailed_log) > 1000 else self.detailed_log
        with open(detailed_path, 'w') as f:
            json.dump(recent_logs, f, indent=2)
        
        # Save timing information
        timing_path = base_path.replace('.json', '_timing.json')
        with open(timing_path, 'w') as f:
            json.dump(self.timing_info, f, indent=2)
        
        # Save task information
        task_path = base_path.replace('.json', '_tasks.json')
        with open(task_path, 'w') as f:
            json.dump(self.task_info, f, indent=2)
    
    def get_training_summary(self) -> dict:
        """Get training summary"""
        import time
        total_time = time.time() - self.start_time
        
        summary = {
            'total_epochs': self.current_epoch,
            'total_time': total_time,
            'avg_epoch_time': total_time / max(1, self.current_epoch),
            'total_batches': len(self.detailed_log),
            'tasks_completed': len(self.task_info),
            'final_metrics': {k: v[-1] if v else 0 for k, v in self.metrics.items()}
        }
        return summary


class TrainingTimer:
    """Training timer"""
    
    def __init__(self):
        self.start_time = None
        self.epoch_start = None
        self.task_start = None
        self.training_start = None
        
    def start_training(self):
        """Start training timer"""
        self.training_start = time.time()
        
    def end_training(self) -> float:
        """End training timer"""
        if self.training_start:
            return time.time() - self.training_start
        return 0.0
        
    def start_task(self):
        """Start task timer"""
        self.task_start = time.time()
        
    def end_task(self) -> float:
        """End task timer"""
        if self.task_start:
            return time.time() - self.task_start
        return 0.0
        
    def start_epoch(self):
        """Start epoch timer"""
        self.epoch_start = time.time()
        
    def end_epoch(self) -> float:
        """End epoch timer"""
        if self.epoch_start:
            return time.time() - self.epoch_start
        return 0.0
    
    def format_time(self, seconds: float) -> str:
        """Format time string"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m{seconds%60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h{minutes:.0f}m"
    
    def get_average_times(self) -> Dict[str, float]:
        """Get average times (simplified)"""
        return {
            'avg_epoch_time': 0.0,
            'avg_task_time': 0.0
        }


def create_exp_directory(results_dir: str, exp_name: str) -> str:
    """Create experiment directory"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(results_dir, f"{exp_name}_{timestamp}")
    
    # Create directory structure
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    
    print(f"  Created experiment directory: {exp_dir}")
    return exp_dir


def log_experiment_info(exp_dir: str, config_dict: Dict[str, Any]):
    """Record experiment information"""
    config_path = os.path.join(exp_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f" Config saved to: {config_path}")


def save_checkpoint(checkpoint_data: Dict[str, Any], filepath: str, is_best: bool = False):
    """Save checkpoint"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint_data, filepath)
    
    if is_best:
        best_path = filepath.replace('.pth', '_best.pth')
        torch.save(checkpoint_data, best_path)


def print_model_info(model: torch.nn.Module, model_name: str):
    """Print model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f" {model_name} model info:")
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")


if __name__ == "__main__":
    # Test utilities
    print(" Testing training utilities...")
    
    # Test MetricsTracker
    tracker = MetricsTracker()
    tracker.update({'loss': 1.5, 'accuracy': 0.8})
    tracker.update({'loss': 1.2, 'accuracy': 0.85})
    epoch_avg = tracker.end_epoch()
    print(f" MetricsTracker: {epoch_avg}")
    
    # Test TrainingTimer
    timer = TrainingTimer()
    timer.start_epoch()
    time.sleep(0.1)
    elapsed = timer.end_epoch()
    print(f" TrainingTimer: {timer.format_time(elapsed)}")
    
    print(" Training utilities test completed!") 