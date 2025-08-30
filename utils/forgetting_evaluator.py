#!/usr/bin/env python3
"""
Forgetting evaluation utilities
Evaluate whether the model forgot previously learned tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os
import json
from datetime import datetime

from utils.fid_utils import calculate_fid_from_dataloader_and_generator
from utils.image_utils import save_samples


class ForgettingEvaluator:
    """Forgetting evaluator"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.evaluation_results = {}
    
    def evaluate_teacher_forgetting(self, 
                                  model, 
                                  previous_task_data: DataLoader,
                                  previous_task_name: str,
                                  task_id: int) -> Dict:
        """
        Evaluate whether the Teacher has forgotten the previous task
        
        Args:
            model: Teacher-Student model
            previous_task_data: dataloader of the previous task
            previous_task_name: name of the previous task
            task_id: current task id
        
        Returns:
            Evaluation result dictionary
        """
        print(f"Evaluating Teacher forgetting for task {task_id-1} ({previous_task_name})...")
        
        results = {
            'task_name': previous_task_name,
            'current_task_id': task_id,
            'evaluation_time': datetime.now().isoformat()
        }
        
        try:
            # Generate samples for each expert on previous task
            expert_results = {}
            for expert_id in range(len(model.teacher_experts)):
                print(f"    Evaluating expert {expert_id}...")
                
                try:
                    # Generate samples
                    z = torch.randn(1000, model.z_dim).to(self.device)
                    generated_samples = model.generate_with_expert(z, expert_id)
                    
                    # Compute FID score
                    try:
                        fid_score = calculate_fid_from_dataloader_and_generator(
                            real_dataloader=previous_task_data,
                            generator=lambda z: model.generate_with_expert(z, expert_id),
                            device=self.device,
                            num_fake_samples=1000,
                            max_real_samples=1000,
                            z_dim=model.z_dim
                        )
                    except Exception as e:
                        print(f"       FID calculation failed: {e}")
                        fid_score = float('inf')
                    
                    expert_results[f'expert_{expert_id}'] = {
                        'fid_score': fid_score,
                        'generated_samples': generated_samples
                    }
                    
                    print(f"       Expert {expert_id} FID: {fid_score:.2f}")
                except Exception as e:
                    print(f"       Expert {expert_id} evaluation failed: {e}")
                    expert_results[f'expert_{expert_id}'] = {
                        'fid_score': float('inf'),
                        'error': str(e)
                    }
            
            results['teacher_experts'] = expert_results
            
        except Exception as e:
            print(f"    Teacher evaluation failed: {e}")
            results['teacher_error'] = str(e)
        
        return results
    
    def evaluate_student_forgetting(self, 
                                  model, 
                                  previous_task_data: DataLoader,
                                  previous_task_name: str,
                                  task_id: int) -> Dict:
        """
        Evaluate whether the Student has forgotten the previous task
        
        Args:
            model: Teacher-Student model
            previous_task_data: dataloader of the previous task
            previous_task_name: name of the previous task
            task_id: current task id
        
        Returns:
            Evaluation result dictionary
        """
        print(f" Evaluating Student forgetting for task {task_id-1} ({previous_task_name})...")
        
        results = {
            'task_name': previous_task_name,
            'current_task_id': task_id,
            'evaluation_time': datetime.now().isoformat()
        }
        
        try:
            # 1. Reconstruction quality evaluation
            recon_losses = []
            kl_losses = []
            
            model.student.eval()
            with torch.no_grad():
                for batch_idx, (images, _) in enumerate(previous_task_data):
                    if batch_idx >= 10:  # evaluate first 10 batches only
                        break
                    
                    images = images.to(self.device)
                    batch_size = images.size(0)
                    
                    # Create previous task_id
                    prev_task_id = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                    
                    # Compute reconstruction losses
                    try:
                        student_outputs = model.student(images, prev_task_id)
                        recon_loss = student_outputs.get('recon_loss', torch.tensor(0.0))
                        kl_loss_z = student_outputs.get('kl_loss_z', torch.tensor(0.0))
                        kl_loss_u = student_outputs.get('kl_loss_u', torch.tensor(0.0))
                    except Exception as e:
                        print(f"       Failed to parse Student outputs: {e}")
                        recon_loss = torch.tensor(0.0)
                        kl_loss_z = torch.tensor(0.0)
                        kl_loss_u = torch.tensor(0.0)
                    
                    recon_losses.append(recon_loss.item())
                    kl_losses.append(kl_loss_z.item() + kl_loss_u.item())
            
            avg_recon_loss = np.mean(recon_losses)
            avg_kl_loss = np.mean(kl_losses)
            
            print(f"    Average reconstruction loss: {avg_recon_loss:.4f}")
            print(f"    Average KL loss: {avg_kl_loss:.4f}")
            
            # 2. Generation quality evaluation
            try:
                prev_task_id = torch.zeros(1000, dtype=torch.long, device=self.device)
                generated_samples = model.generate_with_student(prev_task_id, num_samples=1000)
                
                # Compute FID score
                fid_score = calculate_fid_from_dataloader_and_generator(
                    real_dataloader=previous_task_data,
                    generator=lambda z: model.generate_with_student(prev_task_id, num_samples=z.size(0)),
                    device=self.device,
                    num_fake_samples=1000,
                    max_real_samples=1000,
                    z_dim=model.student.z_dim
                )
            except Exception as e:
                print(f"    Student generation evaluation failed: {e}")
                generated_samples = None
                fid_score = float('inf')
            
            print(f"    Student FID: {fid_score:.2f}")
            
            results.update({
                'avg_recon_loss': avg_recon_loss,
                'avg_kl_loss': avg_kl_loss,
                'fid_score': fid_score,
                'generated_samples': generated_samples
            })
            
        except Exception as e:
            print(f"    Student evaluation failed: {e}")
            results['student_error'] = str(e)
        
        return results
    
    def create_forgetting_report(self, 
                               evaluation_results: Dict,
                               save_dir: str) -> str:
        """
        Create forgetting evaluation report
        
        Args:
            evaluation_results: evaluation results
            save_dir: directory to save
        
        Returns:
            Report file path
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Create visualizations
        self._create_forgetting_visualization(evaluation_results, save_dir)
        
        # Create text report
        report_path = os.path.join(save_dir, 'forgetting_report.txt')
        self._create_text_report(evaluation_results, report_path)
        
        # Save JSON results
        results_path = os.path.join(save_dir, 'forgetting_evaluation.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
        return report_path
    
    def _create_forgetting_visualization(self, results: Dict, save_dir: str):
        """Create forgetting visualization charts"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Forgetting Evaluation Results', fontsize=16)
            
            # Teacher FID scores
            if 'teacher_evaluation' in results and 'teacher_experts' in results['teacher_evaluation']:
                teacher_fids = []
                expert_names = []
                for expert_name, expert_data in results['teacher_evaluation']['teacher_experts'].items():
                    if 'fid_score' in expert_data and expert_data['fid_score'] != float('inf'):
                        teacher_fids.append(expert_data['fid_score'])
                        expert_names.append(expert_name)
                
                if teacher_fids:
                    axes[0, 0].bar(expert_names, teacher_fids)
                    axes[0, 0].set_title('Teacher Expert FID Scores')
                    axes[0, 0].set_ylabel('FID Score')
                    axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Student evaluation
            if 'student_evaluation' in results:
                student_data = results['student_evaluation']
                if 'avg_recon_loss' in student_data:
                    axes[0, 1].bar(['Reconstruction Loss', 'KL Loss'], 
                                  [student_data['avg_recon_loss'], student_data['avg_kl_loss']])
                    axes[0, 1].set_title('Student Losses')
                    axes[0, 1].set_ylabel('Loss')
                
                if 'fid_score' in student_data and student_data['fid_score'] != float('inf'):
                    axes[1, 0].bar(['Student FID'], [student_data['fid_score']])
                    axes[1, 0].set_title('Student FID Score')
                    axes[1, 0].set_ylabel('FID Score')
            
            # Forgetting assessment
            axes[1, 1].text(0.1, 0.8, 'Forgetting Assessment:', fontsize=12, fontweight='bold')
            
            # Simple assessment
            teacher_forgetting = False
            student_forgetting = False
            
            if 'teacher_evaluation' in results and 'teacher_experts' in results['teacher_evaluation']:
                teacher_fids = []
                for expert_data in results['teacher_evaluation']['teacher_experts'].values():
                    if 'fid_score' in expert_data and expert_data['fid_score'] != float('inf'):
                        teacher_fids.append(expert_data['fid_score'])
                
                if teacher_fids:
                    avg_teacher_fid = np.mean(teacher_fids)
                    if avg_teacher_fid > 50:  # adjustable threshold
                        teacher_forgetting = True
            
            if 'student_evaluation' in results:
                student_data = results['student_evaluation']
                if 'avg_recon_loss' in student_data and student_data['avg_recon_loss'] > 0.5:  # adjustable threshold
                    student_forgetting = True
            
            y_pos = 0.6
            if teacher_forgetting:
                axes[1, 1].text(0.1, y_pos, ' Teacher may exhibit forgetting', color='red', fontsize=10)
            else:
                axes[1, 1].text(0.1, y_pos, ' Teacher shows good retention', color='green', fontsize=10)
            
            y_pos -= 0.2
            if student_forgetting:
                axes[1, 1].text(0.1, y_pos, ' Student may exhibit forgetting', color='red', fontsize=10)
            else:
                axes[1, 1].text(0.1, y_pos, ' Student shows good retention', color='green', fontsize=10)
            
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Save figure
            viz_path = os.path.join(save_dir, 'forgetting_visualization.png')
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    Visualization saved: {viz_path}")
            
        except Exception as e:
            print(f"    Failed to create visualization: {e}")
    
    def _create_text_report(self, results: Dict, report_path: str):
        """Create text report"""
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("Forgetting Evaluation Report\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Evaluation time: {results.get('evaluation_time', 'N/A')}\n")
                f.write(f"Current task ID: {results.get('current_task_id', 'N/A')}\n")
                f.write(f"Evaluated task: {results.get('task_name', 'N/A')}\n\n")
                
                # Teacher results
                if 'teacher_evaluation' in results:
                    teacher_data = results['teacher_evaluation']
                    f.write("Teacher Evaluation:\n")
                    f.write("-" * 30 + "\n")
                    if 'teacher_experts' in teacher_data:
                        for expert_name, expert_data in teacher_data['teacher_experts'].items():
                            if 'fid_score' in expert_data:
                                f.write(f"{expert_name}: FID = {expert_data['fid_score']:.2f}\n")
                    f.write("\n")
                
                # Student results
                if 'student_evaluation' in results:
                    student_data = results['student_evaluation']
                    f.write("Student Evaluation:\n")
                    f.write("-" * 30 + "\n")
                    if 'avg_recon_loss' in student_data:
                        f.write(f"Average reconstruction loss: {student_data['avg_recon_loss']:.4f}\n")
                        f.write(f"Average KL loss: {student_data['avg_kl_loss']:.4f}\n")
                    if 'fid_score' in student_data:
                        f.write(f"FID score: {student_data['fid_score']:.2f}\n")
                    f.write("\n")
                
                # Assessment
                f.write("Assessment:\n")
                f.write("-" * 30 + "\n")
                
                # Simple logic
                teacher_fids = []
                if 'teacher_evaluation' in results and 'teacher_experts' in results['teacher_evaluation']:
                    for expert_data in results['teacher_evaluation']['teacher_experts'].values():
                        if 'fid_score' in expert_data and expert_data['fid_score'] != float('inf'):
                            teacher_fids.append(expert_data['fid_score'])
                
                if teacher_fids:
                    avg_teacher_fid = np.mean(teacher_fids)
                    if avg_teacher_fid > 50:  # adjustable threshold
                        f.write(" Teacher may exhibit forgetting (FID > 50)\n")
                    else:
                        f.write(" Teacher shows good retention\n")
                
                if 'student_evaluation' in results:
                    student_data = results['student_evaluation']
                    if 'avg_recon_loss' in student_data and student_data['avg_recon_loss'] > 0.5:  # adjustable threshold
                        f.write(" Student may exhibit forgetting (reconstruction loss > 0.5)\n")
                    else:
                        f.write(" Student shows good retention\n")
                
                f.write("\nRecommendations:\n")
                f.write("- If FID is too high, consider adjusting training strategy\n")
                f.write("- If reconstruction loss is too high, consider stronger regularization\n")
                f.write("- Perform forgetting evaluation regularly\n")
            
            print(f"    Text report saved: {report_path}")
            
        except Exception as e:
            print(f"    Failed to create text report: {e}")


def evaluate_forgetting_during_training(model, 
                                      previous_task_data: DataLoader,
                                      previous_task_name: str,
                                      current_task_id: int,
                                      save_dir: str) -> Dict:
    """
    Evaluate forgetting during training
    
    Args:
        model: Teacher-Student model
        previous_task_data: previous task data
        previous_task_name: name of previous task
        current_task_id: current task ID
        save_dir: directory to save
    
    Returns:
        Evaluation results
    """
    evaluator = ForgettingEvaluator()
    
    print(f"\n Starting forgetting evaluation...")
    print(f"   Evaluated task: {previous_task_name}")
    print(f"   Current task: {current_task_id}")
    
    # Evaluate Teacher
    teacher_results = evaluator.evaluate_teacher_forgetting(
        model, previous_task_data, previous_task_name, current_task_id
    )
    
    # Evaluate Student
    student_results = evaluator.evaluate_student_forgetting(
        model, previous_task_data, previous_task_name, current_task_id
    )
    
    # Merge results
    combined_results = {
        'teacher_evaluation': teacher_results,
        'student_evaluation': student_results,
        'evaluation_time': datetime.now().isoformat()
    }
    
    # Create report
    evaluator.create_forgetting_report(combined_results, save_dir)
    
    return combined_results 