#!/usr/bin/env python3
"""
Independent forgetting evaluation script
Run manually after experiment completion to evaluate if model has forgotten previously learned tasks
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dynamic_teacher_student import DynamicTeacherStudent
from data.data_loaders import get_single_task_loader
from utils.forgetting_evaluator import evaluate_forgetting_during_training
from config.experiment_config import ExperimentConfig


def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """Load model from checkpoint"""
    print(f" Loading model checkpoint: {checkpoint_path}")
    
    # Create model instance (specific structure will be restored in load_checkpoint)
    model = DynamicTeacherStudent(
        z_dim=256,
        student_z_dim=64,
        student_u_dim=16,
        channels=3,
        num_tasks=3,
        fid_threshold=100.0,
        device=device
    )
    
    # Use model's built-in loading method (completely matches save_checkpoint)
    if os.path.exists(checkpoint_path):
        model.load_checkpoint(checkpoint_path)
        print(f" Model loaded successfully")
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"Checkpoint file doesn't exist: {checkpoint_path}")


def get_experiment_results_dir():
    """Get latest experiment results directory"""
    results_dir = Path("./results")
    if not results_dir.exists():
        raise FileNotFoundError("results directory doesn't exist")
    
    # Find latest experiment directory
    experiment_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not experiment_dirs:
        raise FileNotFoundError("No experiment results directory found")
    
    # Sort by modification time, take the latest
    latest_dir = max(experiment_dirs, key=lambda x: x.stat().st_mtime)
    print(f" Using latest experiment results directory: {latest_dir}")
    return latest_dir


def main():
    parser = argparse.ArgumentParser(description="Forgetting evaluation script")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    parser.add_argument("--results_dir", type=str, help="Experiment results directory")
    parser.add_argument("--task_id", type=int, default=1, help="Task ID to evaluate")
    parser.add_argument("--previous_task_id", type=int, default=0, help="Previous task ID")
    parser.add_argument("--device", type=str, default="cuda", help="Computing device")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--samples_per_task", type=int, default=40000, help="Samples per task")
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Auto-find latest checkpoint
        results_dir = get_experiment_results_dir()
        checkpoint_path = results_dir / "checkpoints" / "final_model.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file doesn't exist: {checkpoint_path}")
    
    # Determine results save directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = get_experiment_results_dir()
    
    # Load model
    model = load_model_from_checkpoint(str(checkpoint_path), args.device)
    
    print(f" Starting forgetting evaluation...")
    print(f"   Current task ID: {args.task_id}")
    print(f"   Previous task ID: {args.previous_task_id}")
    print(f"   Results save directory: {results_dir}")
    
    # Load previous task data
    print(f" Loading previous task data...")
    previous_task_loader, previous_task_name = get_single_task_loader(
        task_id=args.previous_task_id,
        datasets_root="./datasets",
        batch_size=args.batch_size,
        num_workers=0,
        samples_per_task=args.samples_per_task
    )
    
    # Determine task name
    if args.previous_task_id == 0:
        previous_task_name = "MNIST"
    elif args.previous_task_id == 1:
        previous_task_name = "Fashion-MNIST"
    else:
        previous_task_name = f"Task{args.previous_task_id}"
    
    print(f"   Previous task: {previous_task_name}")
    
    # Create save directory
    save_dir = Path("analyze_results") / f"forgetting_evaluation_task{args.task_id}_vs_task{args.previous_task_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Run forgetting evaluation
    try:
        forgetting_results = evaluate_forgetting_during_training(
            model=model,
            previous_task_data=previous_task_loader,
            previous_task_name=previous_task_name,
            current_task_id=args.task_id,
            save_dir=str(save_dir)
        )
        
        print(f" Forgetting evaluation completed!")
        print(f" Results saved in: {save_dir}")
        print(f"   - forgetting_evaluation.json: detailed evaluation results")
        print(f"   - forgetting_report.txt: text report")
        print(f"   - forgetting_visualization.png: visualization charts")
        
    except Exception as e:
        print(f" Forgetting evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 