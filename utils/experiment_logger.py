


import os
import json
import csv
import time
import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
import psutil
import torch


class ExperimentLogger:
   
    
    def __init__(self, experiment_dir: str, experiment_name: str = None):
        """
        Initialize experiment logger
        
        Args:
            experiment_dir: directory to save experiment results
            experiment_name: experiment name
        """
        self.experiment_dir = experiment_dir
        self.experiment_name = experiment_name or f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create experiment directory structure
        self.setup_directories()
        
        # Initialize data structures for logging
        self.metrics = defaultdict(list)  # store all metrics
        self.epoch_data = []  # per-epoch summary
        self.task_data = []  # per-task data
        self.expert_data = []  # expert-related data
        
        # Evaluation timeline records
        self.evaluation_data = []  # evaluation checkpoints
        self.fid_curves = defaultdict(list)  # FID curve data
        self.forgetting_metrics = {}  # forgetting metrics
        self.expert_growth_timeline = []  # expert growth timeline
        self.kd_gating_status = []  # KD gating status
        self.sample_paths = {}  # sample path records
        self.efficiency_metrics = []  # efficiency/resource metrics
        
        # Task-expert mapping records
        self.task_expert_mapping = {}  # mapping from task to expert
        self.expert_task_performance = {}  # expert performance per task
        self.cosine_similarity_timeline = []  # Cosine similarity timeline
        self.teacher_student_comparison = {}  # Teacher vs Student comparison
        
        # Timing records
        self.experiment_start_time = time.time()
        self.epoch_start_time = None
        self.task_start_time = None
        
        # Config records
        self.config_data = {}
        
        # Resource monitoring init
        self.gpu_memory_tracker = []
        self.peak_gpu_memory = 0
        
        print(f" Experiment logger initialized: {self.experiment_name}")
        print(f" Save directory: {self.experiment_dir}")
    
    def setup_directories(self):
        """Set up experiment directory structure"""
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Create subdirectories
        self.logs_dir = os.path.join(self.experiment_dir, "logs")
        self.plots_dir = os.path.join(self.experiment_dir, "plots")
        self.data_dir = os.path.join(self.experiment_dir, "data")
        self.reports_dir = os.path.join(self.experiment_dir, "reports")
        
        for dir_path in [self.logs_dir, self.plots_dir, self.data_dir, self.reports_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def log_config(self, config_dict: Dict[str, Any]):
        """Record experiment configuration"""
        self.config_data.update(config_dict)
        
        # Save config to file
        config_file = os.path.join(self.data_dir, "experiment_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f" Config recorded: {len(self.config_data)} parameters")
    
    def start_epoch(self, epoch: int):
        """Start a new epoch log"""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self.current_epoch_metrics = defaultdict(list)
        
        print(f" Start logging Epoch {epoch}")
    
    def start_task(self, task_id: int, task_name: str):
        """Start a new task log"""
        self.task_start_time = time.time()
        print(f" Start logging task: {task_name} (ID: {task_id})")
    
    def log_batch_metrics(self, metrics: Dict[str, float], batch_size: int = None):
        """Log batch-level metrics"""
        timestamp = time.time() - self.experiment_start_time
        
        # Add timestamp and epoch info
        log_entry = {
            'epoch': getattr(self, 'current_epoch', 0),
            'timestamp': timestamp,
            'batch_size': batch_size,
            **metrics
        }
        
        # Store in current epoch metrics
        for key, value in metrics.items():
            self.current_epoch_metrics[key].append(value)
            self.metrics[key].append({
                'epoch': self.current_epoch,
                'timestamp': timestamp,
                'value': value
            })
        
        # Save to detailed log in real-time
        self._append_to_detailed_log(log_entry)
    
    def end_epoch(self, additional_metrics: Dict[str, float] = None):
        """End current epoch and compute summary statistics"""
        if self.epoch_start_time is None:
            return
        
        epoch_duration = time.time() - self.epoch_start_time
        
        # Build epoch summary
        epoch_summary = {
            'epoch': self.current_epoch,
            'duration': epoch_duration,
            'timestamp': time.time() - self.experiment_start_time
        }
        
        # Aggregate statistics for each metric
        for metric_name, values in self.current_epoch_metrics.items():
            if values:
                epoch_summary[f'{metric_name}_mean'] = np.mean(values)
                epoch_summary[f'{metric_name}_std'] = np.std(values)
                epoch_summary[f'{metric_name}_min'] = np.min(values)
                epoch_summary[f'{metric_name}_max'] = np.max(values)
                epoch_summary[f'{metric_name}_final'] = values[-1]
        
        # Merge additional metrics
        if additional_metrics:
            epoch_summary.update(additional_metrics)
        
        self.epoch_data.append(epoch_summary)
        
        # Save epoch summary
        self._save_epoch_summary()
        
        print(f" Epoch {self.current_epoch} logging completed (duration: {epoch_duration:.1f}s)")
        
        return epoch_summary
    
    def log_task_completion(self, task_id: int, task_name: str, 
                          task_metrics: Dict[str, float]):
        """Log task completion info"""
        # Handle None task_start_time safely
        if hasattr(self, 'task_start_time') and self.task_start_time is not None:
            task_duration = time.time() - self.task_start_time
        else:
            task_duration = 0.0  # if no start time, set to 0
        
        task_summary = {
            'task_id': task_id,
            'task_name': task_name,
            'duration': task_duration,
            'completion_time': datetime.datetime.now().isoformat(),
            **task_metrics
        }
        
        self.task_data.append(task_summary)
        
        # Save task data
        self._save_task_data()
        
        print(f" Task {task_name} completion logged")
    
    def log_expert_addition(self, expert_id: int, trigger_fid: float, 
                          current_task: str, reason: str = "FID threshold exceeded"):
        """Log expert addition event"""
        expert_event = {
            'expert_id': expert_id,
            'trigger_fid': trigger_fid,
            'current_task': current_task,
            'reason': reason,
            'timestamp': time.time() - self.experiment_start_time,
            'datetime': datetime.datetime.now().isoformat()
        }
        
        self.expert_data.append(expert_event)
        
        # Save expert data
        self._save_expert_data()
        
        print(f" Expert #{expert_id} addition event logged")
    
    def _save_expert_expansion_trigger_logs(self):
        """Save expert expansion trigger logs (supports Table 5.3-A)"""
        if 'expert_expansion_trigger_logs' in self.metrics and self.metrics['expert_expansion_trigger_logs']:
            # Save as JSON format
            logs_file = os.path.join(self.data_dir, 'expert_expansion_trigger_logs.json')
            with open(logs_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics['expert_expansion_trigger_logs'], f, indent=2, default=str)
            
            
            logs_csv_file = os.path.join(self.data_dir, 'expert_expansion_trigger_logs.csv')
            df_logs = pd.DataFrame(self.metrics['expert_expansion_trigger_logs'])
            
            # Reorder columns 
            column_order = [
                'task_id', 'task_name', 'epoch', 'iteration', 'trigger_fid', 
                'trigger_threshold', 'FID_before', 'FID_after', 'kd_gating_status', 
                'stability_score', 'trigger_reason', 'timestamp', 'datetime'
            ]
            
            # Keep only existing columns
            existing_columns = [col for col in column_order if col in df_logs.columns]
            df_logs = df_logs[existing_columns]
            
            df_logs.to_csv(logs_csv_file, index=False, encoding='utf-8')
            
            print(f" Expert expansion trigger logs saved: {len(self.metrics['expert_expansion_trigger_logs'])} records")
            print(f"   JSON: {logs_file}")
            print(f"   CSV: {logs_csv_file}")
            
            # Generate statistics for Table 
            self._generate_expert_expansion_summary()
    
    def _generate_expert_expansion_summary(self):
        """Generate statistical summary of expert expansion trigger logs"""
        if 'expert_expansion_trigger_logs' not in self.metrics or not self.metrics['expert_expansion_trigger_logs']:
            return
        
        logs = self.metrics['expert_expansion_trigger_logs']
        
        # Statistics by task
        task_stats = {}
        for log in logs:
            task_key = f"{log['task_id']}_{log['task_name']}"
            if task_key not in task_stats:
                task_stats[task_key] = {
                    'task_id': log['task_id'],
                    'task_name': log['task_name'],
                    'trigger_count': 0,
                    'avg_trigger_fid': 0,
                    'avg_fid_before': 0,
                    'avg_fid_after': 0,
                    'kd_on_count': 0,
                    'kd_off_count': 0,
                    'total_stability_score': 0,
                    'stability_count': 0
                }
            
            stats = task_stats[task_key]
            stats['trigger_count'] += 1
            stats['avg_trigger_fid'] += log['trigger_fid']
            if log['FID_before']:
                stats['avg_fid_before'] += log['FID_before']
            if log['FID_after']:
                stats['avg_fid_after'] += log['FID_after']
            
            if log['kd_gating_status'] == 'on':
                stats['kd_on_count'] += 1
            elif log['kd_gating_status'] == 'off':
                stats['kd_off_count'] += 1
            
            if log['stability_score']:
                stats['total_stability_score'] += log['stability_score']
                stats['stability_count'] += 1
        
        # Calculate averages
        for stats in task_stats.values():
            if stats['trigger_count'] > 0:
                stats['avg_trigger_fid'] /= stats['trigger_count']
            if stats['avg_fid_before'] > 0:
                stats['avg_fid_before'] /= stats['trigger_count']
            if stats['avg_fid_after'] > 0:
                stats['avg_fid_after'] /= stats['trigger_count']
            if stats['stability_count'] > 0:
                stats['avg_stability_score'] = stats['total_stability_score'] / stats['stability_count']
            else:
                stats['avg_stability_score'] = None
        
        # Save statistical summary
        summary_file = os.path.join(self.data_dir, 'expert_expansion_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(task_stats, f, indent=2, default=str)
        
      
        summary_csv_file = os.path.join(self.data_dir, 'expert_expansion_summary_table_5_3a.csv')
        summary_data = []
        
        for task_key, stats in task_stats.items():
            summary_data.append({
                'Task_ID': stats['task_id'],
                'Task_Name': stats['task_name'],
                'Trigger_Count': stats['trigger_count'],
                'Avg_Trigger_FID': f"{stats['avg_trigger_fid']:.4f}",
                'Avg_FID_Before': f"{stats['avg_fid_before']:.4f}" if stats['avg_fid_before'] > 0 else 'N/A',
                'Avg_FID_After': f"{stats['avg_fid_after']:.4f}" if stats['avg_fid_after'] > 0 else 'N/A',
                'KD_On_Count': stats['kd_on_count'],
                'KD_Off_Count': stats['kd_off_count'],
                'Avg_Stability_Score': f"{stats['avg_stability_score']:.4f}" if stats['avg_stability_score'] else 'N/A'
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(summary_csv_file, index=False, encoding='utf-8')
        
        print(f" Expert expansion summary generated: {summary_file}")
        print(f" format data: {summary_csv_file}")
        
        # Output statistical summary
        print("\n Expert expansion trigger statistics summary:")
        for task_key, stats in task_stats.items():
            print(f"   Task {stats['task_id']}({stats['task_name']}): {stats['trigger_count']} triggers")
            print(f"     Average trigger FID: {stats['avg_trigger_fid']:.4f}")
            print(f"     KD status: ON({stats['kd_on_count']}) OFF({stats['kd_off_count']})")
            if stats['avg_stability_score']:
                print(f"     Average stability score: {stats['avg_stability_score']:.4f}")
            print()
        
       
        self._plot_fid_slope_comparison()
        
        # Generate ablation comparison plots 
        self._plot_ablation_comparison()
    
    def _save_fid_slope_analysis_data(self):
        """Save FID slope analysis data"""
        if 'fid_slope_analysis' in self.metrics and self.metrics['fid_slope_analysis']:
            # Save as JSON format
            slope_file = os.path.join(self.data_dir, 'fid_slope_analysis.json')
            with open(slope_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics['fid_slope_analysis'], f, indent=2, default=str)
            
            # Save as CSV format
            slope_csv_file = os.path.join(self.data_dir, 'fid_slope_analysis.csv')
            df_slope = pd.DataFrame(self.metrics['fid_slope_analysis'])
            df_slope.to_csv(slope_csv_file, index=False, encoding='utf-8')
            
            print(f"  FID slope analysis data saved: {len(self.metrics['fid_slope_analysis'])} records")
            print(f"   JSON: {slope_file}")
            print(f"   CSV: {slope_csv_file}")
    
    def _save_ablation_comparison_data(self):
        """Save ablation comparison data"""
        if 'ablation_comparison' in self.metrics and self.metrics['ablation_comparison']:
            # Save as JSON format
            ablation_file = os.path.join(self.data_dir, 'ablation_comparison.json')
            with open(ablation_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics['ablation_comparison'], f, indent=2, default=str)
            
            # Save as CSV format
            ablation_csv_file = os.path.join(self.data_dir, 'ablation_comparison.csv')
            df_ablation = pd.DataFrame(self.metrics['ablation_comparison'])
            df_ablation.to_csv(ablation_csv_file, index=False, encoding='utf-8')
            
            print(f"  Ablation comparison data saved: {len(self.metrics['ablation_comparison'])} records")
            print(f"   JSON: {ablation_file}")
            print(f"   CSV: {ablation_csv_file}")
    
    
    
    def log_fid_per_epoch(self, task_name: str, epoch: int, teacher_fid: float, student_fid: float, **kwargs):
        """
        Log Teacher and Student FID values for each epoch
        
        Args:
            task_name: task name
            epoch: current epoch
            teacher_fid: Teacher FID value
            student_fid: Student FID value
            **kwargs: other metrics
        """
        timestamp = time.time() - self.experiment_start_time
        
        # Record to FID curve data
        if f"{task_name}_epoch" not in self.fid_curves:
            self.fid_curves[f"{task_name}_epoch"] = []
        
        epoch_fid_entry = {
            'epoch': epoch,
            'task_name': task_name,
            'teacher_fid': teacher_fid,
            'student_fid': student_fid,
            'fid_gap': teacher_fid - student_fid,
            'timestamp': timestamp,
            'datetime': datetime.datetime.now().isoformat(),
            **kwargs
        }
        
        self.fid_curves[f"{task_name}_epoch"].append(epoch_fid_entry)
        
        # Also record to metrics
        teacher_key = f'fid_epoch_{task_name}_Teacher'
        student_key = f'fid_epoch_{task_name}_Student'
        
        if teacher_key not in self.metrics:
            self.metrics[teacher_key] = []
        if student_key not in self.metrics:
            self.metrics[student_key] = []
        
        self.metrics[teacher_key].append({
            'epoch': epoch,
            'task_name': task_name,
            'fid_score': teacher_fid,
            'timestamp': timestamp,
            'datetime': datetime.datetime.now().isoformat(),
            **kwargs
        })
        
        self.metrics[student_key].append({
            'epoch': epoch,
            'task_name': task_name,
            'fid_score': student_fid,
            'timestamp': timestamp,
            'datetime': datetime.datetime.now().isoformat(),
            **kwargs
        })
        
        print(f" Epoch {epoch} FID logged: {task_name} - Teacher: {teacher_fid:.2f}, Student: {student_fid:.2f}")
    
    def log_old_task_performance(self, current_task: str, old_task_name: str, 
                                old_task_fid: float, reconstruction_quality: float):
        """Log Student performance on old tasks"""
        if 'old_task_performance' not in self.metrics:
            self.metrics['old_task_performance'] = []
        
        performance_data = {
            'current_task': current_task,
            'old_task_name': old_task_name,
            'old_task_fid': old_task_fid,
            'reconstruction_quality': reconstruction_quality,
            'timestamp': time.time(),
            'epoch': getattr(self, 'current_epoch', 0)
        }
        
        self.metrics['old_task_performance'].append(performance_data)
        
        # Silent logging, no console output
    
    def log_teacher_retention_metrics(self, task_name: str, task_id: int, epoch: int,
                                    teacher_fid_current: float, teacher_fid_old_tasks: Dict[str, float],
                                    teacher_stability_score: float, expert_utilization: Dict[int, float]):
        """Log Teacher model retention capability metrics"""
        if 'teacher_retention_metrics' not in self.metrics:
            self.metrics['teacher_retention_metrics'] = []
        
        retention_data = {
            'task_name': task_name,
            'task_id': task_id,
            'epoch': epoch,
            'teacher_fid_current': teacher_fid_current,
            'teacher_fid_old_tasks': teacher_fid_old_tasks,
            'teacher_stability_score': teacher_stability_score,
            'expert_utilization': expert_utilization,
            'timestamp': time.time()
        }
        
        self.metrics['teacher_retention_metrics'].append(retention_data)
        
        # Silent logging, no console output
    
    def log_teacher_old_task_performance(self, current_task: str, old_task_name: str,
                                        old_task_fid: float, expert_id: int, 
                                        generation_quality: float, memory_efficiency: float):
        """Log Teacher performance on old tasks"""
        if 'teacher_old_task_performance' not in self.metrics:
            self.metrics['teacher_old_task_performance'] = []
        
        performance_data = {
            'current_task': current_task,
            'old_task_name': old_task_name,
            'old_task_fid': old_task_fid,
            'expert_id': expert_id,
            'generation_quality': generation_quality,
            'memory_efficiency': memory_efficiency,
            'timestamp': time.time(),
            'epoch': getattr(self, 'current_epoch', 0)
        }
        
        self.metrics['teacher_old_task_performance'].append(performance_data)
        
        # Silent logging, no console output
    
    def log_teacher_expert_analysis(self, task_name: str, task_id: int, epoch: int,
                                   expert_performance: Dict[int, float], expert_stability: Dict[int, float],
                                   expert_memory_usage: Dict[int, float], expert_utilization_rate: Dict[int, float]):
        """Log Teacher expert analysis data"""
        if 'teacher_expert_analysis' not in self.metrics:
            self.metrics['teacher_expert_analysis'] = []
        
        analysis_data = {
            'task_name': task_name,
            'task_id': task_id,
            'epoch': epoch,
            'expert_performance': expert_performance,
            'expert_stability': expert_stability,
            'expert_memory_usage': expert_memory_usage,
            'expert_utilization_rate': expert_utilization_rate,
            'timestamp': time.time()
        }
        
        self.metrics['teacher_expert_analysis'].append(analysis_data)
        
        # Silent logging, no console output
    
    def log_kd_stability_metrics(self, task_name: str, epoch: int, kd_enabled: bool, kd_weight: float,
                                teacher_student_similarity: float, distillation_gating_status: str,
                                cosine_similarity: float = None, **kwargs):
        """
        Log knowledge distillation stability metrics
        
        Args:
            task_name: task name
            epoch: current epoch
            kd_enabled: whether knowledge distillation is enabled
            kd_weight: knowledge distillation weight
            teacher_student_similarity: Teacher-Student similarity
            distillation_gating_status: distillation gating status
            cosine_similarity: cosine similarity
            **kwargs: other metrics
        """
        timestamp = time.time() - self.experiment_start_time
        
        # Record to KD stability metrics
        if 'kd_stability_metrics' not in self.metrics:
            self.metrics['kd_stability_metrics'] = []
        self.metrics['kd_stability_metrics'].append({
            'task_name': task_name,
            'epoch': epoch,
            'kd_enabled': kd_enabled,
            'kd_weight': kd_weight,
            'teacher_student_similarity': teacher_student_similarity,
            'distillation_gating_status': distillation_gating_status,
            'cosine_similarity': cosine_similarity,
            'timestamp': time.time() - self.experiment_start_time,
            'datetime': datetime.datetime.now().isoformat(),
            **kwargs
        })
        
        # Silent logging, no console output
    
    def log_task_completion_with_old_task_evaluation(self, task_id: int, task_name: str, 
                                                   task_metrics: Dict[str, float],
                                                   old_task_evaluations: Dict[str, float] = None):
        """
        Log task completion info, including evaluation results on old tasks
        
        Args:
            task_id: task ID
            task_name: task name
            task_metrics: task metrics
            old_task_evaluations: evaluation results on old tasks {old_task_name: fid_score}
        """
        # Call the original task completion logging method
        self.log_task_completion(task_id, task_name, task_metrics)
        
        # Log evaluation results on old tasks
        if old_task_evaluations:
            for old_task_name, old_task_fid in old_task_evaluations.items():
                self.log_old_task_performance(
                    current_task=task_name,
                    old_task_name=old_task_name,
                    old_task_fid=old_task_fid,
                    reconstruction_quality=1.0 / (1.0 + old_task_fid)  # simple reconstruction quality metric
                )
        
        print(f" Task {task_name} completion logged (including old task evaluation)")
    
    def log_epoch_fid_evaluation(self, epoch: int, task_name: str, teacher_fid: float, student_fid: float,
                                kd_enabled: bool, kd_weight: float, teacher_student_similarity: float,
                                **kwargs):
        """
        Log complete evaluation metrics for each epoch (integrates data collection from three sections)
        
        Args:
            epoch: current epoch
            task_name: task name
            teacher_fid: Teacher FID value
            student_fid: Student FID value
            kd_enabled: whether knowledge distillation is enabled
            kd_weight: knowledge distillation weight
            teacher_student_similarity: Teacher-Student similarity
            **kwargs: other metrics
        """
        # Log FID trends across tasks
        self.log_fid_per_epoch(task_name, epoch, teacher_fid, student_fid, **kwargs)
        
        # Log knowledge distillation stability
        distillation_gating_status = "enabled" if kd_enabled else "disabled"
        self.log_kd_stability_metrics(
            task_name=task_name,
            epoch=epoch,
            kd_enabled=kd_enabled,
            kd_weight=kd_weight,
            teacher_student_similarity=teacher_student_similarity,
            distillation_gating_status=distillation_gating_status,
            cosine_similarity=teacher_student_similarity,  # use similarity as cosine similarity
            **kwargs
        )
        
        print(f" Epoch {epoch} complete evaluation logged: {task_name}")
    
   
    
    def log_expert_trigger_summary(self, task_name: str, task_id: int, expert_triggered: bool,
                                  trigger_fid: float = None, threshold_fid: float = None,
                                  expert_id: int = None, **kwargs):
        """
        Log Expert trigger summary
        
        Args:
            task_name: task name
            task_id: task ID
            expert_triggered: whether Expert addition was triggered
            trigger_fid: FID value at trigger
            threshold_fid: threshold FID value
            expert_id: newly added Expert ID
            **kwargs: other metrics
        """
        timestamp = time.time() - self.experiment_start_time
        
        trigger_summary = {
            'task_name': task_name,
            'task_id': task_id,
            'expert_triggered': expert_triggered,
            'trigger_fid': trigger_fid,
            'threshold_fid': threshold_fid,
            'expert_id': expert_id,
            'fid_gap': trigger_fid - threshold_fid if trigger_fid and threshold_fid else None,
            'timestamp': timestamp,
            'datetime': datetime.datetime.now().isoformat(),
            **kwargs
        }
        
        # Record to Expert trigger summary metrics
        if 'expert_trigger_summary' not in self.metrics:
            self.metrics['expert_trigger_summary'] = []
        
        self.metrics['expert_trigger_summary'].append(trigger_summary)
        
        status_str = "triggered" if expert_triggered else "not triggered"
        print(f" Expert trigger summary: {task_name} - {status_str} - Expert ID: {expert_id}")
    
    def log_expert_task_correlation(self, task_name: str, task_id: int, 
                                   bound_expert_ids: List[int], expert_reuse_info: Dict[str, Any] = None,
                                   **kwargs):
        """
        Log Expert-task correlation analysis
        
        Args:
            task_name: task name
            task_id: task ID
            bound_expert_ids: list of Expert IDs bound to this task
            expert_reuse_info: expert reuse information
            **kwargs: other metrics
        """
        timestamp = time.time() - self.experiment_start_time
        
        correlation_entry = {
            'task_name': task_name,
            'task_id': task_id,
            'bound_expert_ids': bound_expert_ids,
            'num_bound_experts': len(bound_expert_ids),
            'expert_reuse_info': expert_reuse_info or {},
            'timestamp': timestamp,
            'datetime': datetime.datetime.now().isoformat(),
            **kwargs
        }
        
        # Record to Expert-task correlation metrics
        if 'expert_task_correlation' not in self.metrics:
            self.metrics['expert_task_correlation'] = []
        
        self.metrics['expert_task_correlation'].append(correlation_entry)
        
        # Update task-expert mapping
        if task_name not in self.task_expert_mapping:
            self.task_expert_mapping[task_name] = []
        
        for expert_id in bound_expert_ids:
            mapping_entry = {
                'task_id': task_id,
                'task_name': task_name,
                'expert_id': expert_id,
                'timestamp': timestamp,
                'datetime': datetime.datetime.now().isoformat()
            }
            self.task_expert_mapping[task_name].append(mapping_entry)
        
        print(f" Expert-task correlation: {task_name} - bound experts: {bound_expert_ids}")
    
    def log_expert_resource_consumption(self, expert_id: int, task_name: str, 
                                       parameters_before: int, parameters_after: int,
                                       storage_size_mb: float = None, inference_time_ms: float = None,
                                       **kwargs):
        """
        Log Expert resource consumption analysis
        
        Args:
            expert_id: Expert ID
            task_name: task name
            parameters_before: parameter count before adding Expert
            parameters_after: parameter count after adding Expert
            storage_size_mb: storage size (MB)
            inference_time_ms: inference time (ms)
            **kwargs: other metrics
        """
        timestamp = time.time() - self.experiment_start_time
        
        # Calculate resource changes
        parameter_increase = parameters_after - parameters_before
        parameter_increase_percent = (parameter_increase / parameters_before * 100) if parameters_before > 0 else 0
        
        resource_entry = {
            'expert_id': expert_id,
            'task_name': task_name,
            'parameters_before': parameters_before,
            'parameters_after': parameters_after,
            'parameter_increase': parameter_increase,
            'parameter_increase_percent': parameter_increase_percent,
            'storage_size_mb': storage_size_mb,
            'inference_time_ms': inference_time_ms,
            'timestamp': timestamp,
            'datetime': datetime.datetime.now().isoformat(),
            **kwargs
        }
        
        # Record to Expert resource consumption metrics
        if 'expert_resource_consumption' not in self.metrics:
            self.metrics['expert_resource_consumption'] = []
        
        self.metrics['expert_resource_consumption'].append(resource_entry)
        
        # Also record to efficiency metrics
        efficiency_entry = {
            'expert_id': expert_id,
            'task_name': task_name,
            'parameters_teacher': parameters_after,
            'parameters_student': 0,  # Student parameter count remains unchanged
            'total_params': parameters_after,
            'parameter_increase': parameter_increase,
            'storage_size_mb': storage_size_mb,
            'inference_time_ms': inference_time_ms,
            'timestamp': timestamp,
            'datetime': datetime.datetime.now().isoformat()
        }
        self.efficiency_metrics.append(efficiency_entry)
        
        print(f" Expert resource consumption: Expert #{expert_id} - parameter increase: {parameter_increase:,} ({parameter_increase_percent:.1f}%)")
    
    def log_expert_analysis_summary(self, task_name: str, task_id: int, 
                                   expert_triggered: bool, bound_expert_ids: List[int],
                                   trigger_fid: float = None, threshold_fid: float = None,
                                   parameters_before: int = None, parameters_after: int = None,
                                   **kwargs):
        """
        Log Expert analysis summary 
        
        Args:
            task_name: task name
            task_id: task ID
            expert_triggered: whether Expert addition was triggered
            bound_expert_ids: list of Expert IDs bound to this task
            trigger_fid: FID value at trigger
            threshold_fid: threshold FID value
            parameters_before: parameter count before adding Expert
            parameters_after: parameter count after adding Expert
            **kwargs: other metrics
        """
        # Log Expert trigger summary
        self.log_expert_trigger_summary(
            task_name=task_name,
            task_id=task_id,
            expert_triggered=expert_triggered,
            trigger_fid=trigger_fid,
            threshold_fid=threshold_fid,
            expert_id=bound_expert_ids[-1] if bound_expert_ids and expert_triggered else None,
            **kwargs
        )
        
        # Log Expert-task correlation
        self.log_expert_task_correlation(
            task_name=task_name,
            task_id=task_id,
            bound_expert_ids=bound_expert_ids,
            **kwargs
        )
        
        # Log Expert resource consumption
        if expert_triggered and parameters_before and parameters_after:
            self.log_expert_resource_consumption(
                expert_id=bound_expert_ids[-1],
                task_name=task_name,
                parameters_before=parameters_before,
                parameters_after=parameters_after,
                **kwargs
            )
        
        print(f" Expert analysis summary completed: {task_name} - triggered: {expert_triggered} - bound experts: {bound_expert_ids}")
    
    def log_task_expert_mapping(self, task_id: int, task_name: str, expert_id: int, 
                               expert_performance: float, trigger_fid: float = None):
        """
        Log task-to-expert mapping relationship
        
        Args:
            task_id: task ID
            task_name: task name
            expert_id: expert ID
            expert_performance: expert performance on this task
            trigger_fid: FID value when this expert was triggered
        """
        timestamp = time.time() - self.experiment_start_time
        
        # Log task-expert mapping
        mapping_entry = {
            'task_id': task_id,
            'task_name': task_name,
            'expert_id': expert_id,
            'expert_performance': expert_performance,
            'trigger_fid': trigger_fid,
            'timestamp': timestamp,
            'datetime': datetime.datetime.now().isoformat()
        }
        
        # Store to task-expert mapping
        if task_name not in self.task_expert_mapping:
            self.task_expert_mapping[task_name] = []
        self.task_expert_mapping[task_name].append(mapping_entry)
        
        # Store to expert-task performance
        if expert_id not in self.expert_task_performance:
            self.expert_task_performance[expert_id] = {}
        self.expert_task_performance[expert_id][task_name] = {
            'performance': expert_performance,
            'trigger_fid': trigger_fid,
            'timestamp': timestamp,
            'datetime': datetime.datetime.now().isoformat()
        }
        
        print(f" Task-expert mapping logged: {task_name} -> Expert #{expert_id}, performance: {expert_performance:.2f}")
    
    def log_cosine_similarity(self, task_name: str, similarity: float, kd_quality: str = None, **kwargs):
        """
        Log Cosine Similarity timeline data
        
        Args:
            task_name: task name
            similarity: similarity value
            kd_quality: knowledge distillation quality rating
            **kwargs: other related metrics
        """
        timestamp = time.time() - self.experiment_start_time
        
        similarity_entry = {
            'task_name': task_name,
            'similarity': similarity,
            'kd_quality': kd_quality,
            'epoch': getattr(self, 'current_epoch', 0),
            'timestamp': timestamp,
            'datetime': datetime.datetime.now().isoformat(),
            **kwargs
        }
        
        self.cosine_similarity_timeline.append(similarity_entry)
        
        # Also record to metrics
        metric_key = f'cosine_similarity_{task_name}'
        if metric_key not in self.metrics:
            self.metrics[metric_key] = []
        self.metrics[metric_key].append(similarity_entry)
        
        print(f" Cosine Similarity logged: {task_name} - {similarity:.3f} ({kd_quality})")
    
    
    
    def log_evaluation_point(self, phase: str, task_id: int, task_name: str, 
                           fid_score: float, num_experts: int, **kwargs):
        """
        Log evaluation checkpoint data
        
        Args:
            phase: evaluation checkpoint (mid_eval / task_end / final)
            task_id: task ID
            task_name: task name
            fid_score: FID score
            num_experts: current total number of experts
            **kwargs: other metrics
        """
        evaluation_entry = {
            'phase': phase,
            'task_id': task_id,
            'task_name': task_name,
            'fid_score': fid_score,
            'num_experts_total': num_experts,
            'timestamp': time.time() - self.experiment_start_time,
            'datetime': datetime.datetime.now().isoformat(),
            **kwargs
        }
        
        self.evaluation_data.append(evaluation_entry)
        
        # Log FID curve data
        self.fid_curves[f"{task_name}_{phase}"].append({
            'epoch': getattr(self, 'current_epoch', 0),
            'fid_score': fid_score,
            'num_experts': num_experts
        })
        
        print(f" Evaluation checkpoint logged: {phase} - {task_name} - FID: {fid_score:.2f} - Experts: {num_experts}")
    
    def log_expert_growth_event(self, event_step: int, event_epoch: int, 
                               action: str, active_expert_id: int, 
                               growth_threshold: float, fid_current: float,
                               **kwargs):
        """
        Log expert expansion event timeline
        
        Args:
            event_step: event step
            event_epoch: event epoch
            action: action type (add_expert, remove_expert, etc.)
            active_expert_id: active expert ID
            growth_threshold: expansion threshold
            fid_current: current FID score
            **kwargs: other event information
        """
        growth_event = {
            'event_step': event_step,
            'event_epoch': event_epoch,
            'action': action,
            'active_expert_id': active_expert_id,
            'growth_threshold': growth_threshold,
            'fid_current': fid_current,
            'timestamp': time.time() - self.experiment_start_time,
            'datetime': datetime.datetime.now().isoformat(),
            **kwargs
        }
        
        self.expert_growth_timeline.append(growth_event)
        
        print(f" Expert expansion event: {action} - Expert {active_expert_id} - FID: {fid_current:.2f}")
    
    
    
    def log_sample_paths(self, task_name: str, samples_teacher_path: str, 
                        samples_student_path: str, sample_seed: int):
        """
        Log sample paths and seeds
        
        Args:
            task_name: task name
            samples_teacher_path: Teacher sample path
            samples_student_path: Student sample path
            sample_seed: sampling seed
        """
        self.sample_paths[task_name] = {
            'samples_teacher_path': samples_teacher_path,
            'samples_student_path': samples_student_path,
            'sample_seed': sample_seed,
            'timestamp': time.time() - self.experiment_start_time
        }
        
        print(f" Sample paths logged: {task_name} - seed: {sample_seed}")
    
    def log_efficiency_metrics(self, duration_sec: float, images_per_sec: float,
                              gpu_mem_peak_mb: float, params_teacher: int,
                              params_student: int, **kwargs):
        """
        Log efficiency and resource metrics
        
        Args:
            duration_sec: duration (seconds)
            images_per_sec: images processed per second
            gpu_mem_peak_mb: GPU memory peak (MB)
            params_teacher: Teacher parameter count
            params_student: Student parameter count
            **kwargs: other efficiency metrics
        """
        efficiency_entry = {
            'duration_sec': duration_sec,
            'images_per_sec': images_per_sec,
            'gpu_mem_peak_mb': gpu_mem_peak_mb,
            'params_teacher': params_teacher,
            'params_student': params_student,
            'timestamp': time.time() - self.experiment_start_time,
            'datetime': datetime.datetime.now().isoformat(),
            **kwargs
        }
        
        self.efficiency_metrics.append(efficiency_entry)
        
        print(f" Efficiency metrics: {duration_sec:.1f}s - {images_per_sec:.1f} img/s - GPU: {gpu_mem_peak_mb:.1f}MB")
    
    def log_fid_split_info(self, task_name: str, fid_split: str, 
                          fid_train: float = None, fid_val: float = None, 
                          fid_test: float = None):
        """
        Log FID evaluation set information
        
        Args:
            task_name: task name
            fid_split: evaluation set type (train/val/test)
            fid_train: training set FID
            fid_val: validation set FID
            fid_test: test set FID
        """
        fid_split_entry = {
            'task_name': task_name,
            'fid_split': fid_split,
            'fid_train': fid_train,
            'fid_val': fid_val,
            'fid_test': fid_test,
            'timestamp': time.time() - self.experiment_start_time,
            'datetime': datetime.datetime.now().isoformat()
        }
        
        # Store to metrics
        self.metrics[f'fid_split_{task_name}'].append(fid_split_entry)
        
        print(f" FID evaluation set logged: {task_name} - {fid_split}")
    
    def calculate_forgetting_metrics(self):
        """Calculate forgetting metrics"""
        print(" Calculating forgetting metrics...")
        
        # Group evaluation data by task
        task_evaluations = defaultdict(list)
        for eval_data in self.evaluation_data:
            task_evaluations[eval_data['task_name']].append(eval_data)
        
        for task_name, evaluations in task_evaluations.items():
            # Find task_end and final evaluations
            task_end_eval = None
            final_eval = None
            
            for eval_data in evaluations:
                if eval_data['phase'] == 'task_end':
                    task_end_eval = eval_data
                elif eval_data['phase'] == 'final':
                    final_eval = eval_data
            
            if task_end_eval and final_eval:
                forgetting = final_eval['fid_score'] - task_end_eval['fid_score']
                
                self.forgetting_metrics[task_name] = {
                    'fid_task_end': task_end_eval['fid_score'],
                    'fid_final': final_eval['fid_score'],
                    'forgetting': forgetting,
                    'task_id': task_end_eval['task_id']
                }
                
                print(f" Task {task_name} forgetting: {forgetting:.2f}")
        
        # Save forgetting metrics
        self._save_forgetting_metrics()
    
    def monitor_gpu_memory(self):
        """Monitor GPU memory usage"""
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                self.gpu_memory_tracker.append({
                    'timestamp': time.time() - self.experiment_start_time,
                    'gpu_memory_mb': gpu_memory
                })
                
                if gpu_memory > self.peak_gpu_memory:
                    self.peak_gpu_memory = gpu_memory
                
                return {
                    'gpu_memory_mb': gpu_memory,
                    'peak_gpu_memory_mb': self.peak_gpu_memory,
                    'cuda_available': True
                }
        except:
            pass
        return {
            'gpu_memory_mb': 0.0,
            'peak_gpu_memory_mb': self.peak_gpu_memory,
            'cuda_available': False
        }
    
    def get_system_metrics(self):
        """Get system resource metrics"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            gpu_memory = self.monitor_gpu_memory()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / 1024 / 1024 / 1024,
                'gpu_memory_mb': gpu_memory,
                'peak_gpu_memory_mb': self.peak_gpu_memory
            }
        except:
            return {}
    
    def generate_plots(self):
        """Generate all visualization charts"""
        print(" Generating visualization charts...")
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Training loss curves
        self._plot_training_losses()
        
        # 2. FID score changes
        self._plot_fid_scores()
        
        # 3. Expert addition timeline
        self._plot_expert_timeline()
        
        # 4. Task performance comparison
        self._plot_task_performance()
        
        # 5. Detailed metrics heatmap
        self._plot_metrics_heatmap()
        
        # New: Evaluation function related charts
        # 6. FID curve analysis
        self._plot_fid_curves()
        
        # 7. Expert growth timeline
        self._plot_expert_growth_timeline()
        
        # 8. KD gating status changes
        self._plot_kd_gating_status()
        
        # 9. Forgetting analysis
        self._plot_forgetting_analysis()
        
        # 10. Efficiency resource monitoring
        self._plot_efficiency_metrics()
        
        # 11. FID trend by task
        self._plot_fid_trend_by_task()
        
        # 12. Student model retention of old tasks capability
        self._plot_old_task_performance()
        
        # 13. Knowledge distillation stability analysis
        self._plot_kd_stability_analysis()
        
        # New: Teacher model retention capability analysis
        self._plot_teacher_retention_analysis()
        
        # 14. Expert trigger record analysis
        self._plot_expert_trigger_analysis()
        
        # 15. Expert count and task correlation analysis
        self._plot_expert_task_correlation()
        
        # 16. Expert resource consumption analysis
        self._plot_expert_resource_consumption()
        
        # 17. Efficiency and resource monitoring
        self._plot_efficiency_metrics()
        
        # 18. FID trend by task
        self._plot_fid_trend_by_task()
        
        # Generate expert growth timeline charts
        #  Generate FID curves with key event annotations
        self.generate_fid_curves_with_events()
        
        #  Generate task completion analysis charts
        self._plot_task_completion_analysis()
        
        #  Generate retention analysis charts
        self.generate_retention_analysis_plots()
        
        #  Generate retention heatmap
        self.generate_retention_heatmap()
        
        #  Generate Recon/KL curves
        self.generate_recon_kl_curves()
        
        # Generate fixed seed sample grid
        self.generate_fixed_seed_samples()
        
        #  Generate KD timeline with FID overlay
        self.generate_kd_timeline_with_fid()
        
        #  Generate representation alignment curves
        self.generate_representation_alignment_curves()
        
        #  Generate KD ablation comparison analysis
        self.generate_kd_ablation_comparison()
        
        # Generate knowledge distillation gating status charts
        # self._generate_kd_gating_charts()  # Already integrated into other methods
        
        print(f" Charts saved to: {self.plots_dir}")
    
    def _plot_training_losses(self):
        """Plot training loss curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Losses Over Time', fontsize=16)
        
        # Discriminator loss
        if 'discriminator_loss' in self.metrics:
            epochs = [item['epoch'] for item in self.metrics['discriminator_loss']]
            values = [item['value'] for item in self.metrics['discriminator_loss']]
            axes[0, 0].plot(epochs, values, label='Discriminator Loss', alpha=0.7)
            axes[0, 0].set_title('Discriminator Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
        
        # Generator loss
        if 'generator_loss' in self.metrics:
            epochs = [item['epoch'] for item in self.metrics['generator_loss']]
            values = [item['value'] for item in self.metrics['generator_loss']]
            axes[0, 1].plot(epochs, values, label='Generator Loss', alpha=0.7, color='orange')
            axes[0, 1].set_title('Generator Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)
        
        # Student VAE loss
        if 'vae_loss' in self.metrics:
            epochs = [item['epoch'] for item in self.metrics['vae_loss']]
            values = [item['value'] for item in self.metrics['vae_loss']]
            axes[1, 0].plot(epochs, values, label='VAE Loss', alpha=0.7, color='green')
            axes[1, 0].set_title('Student VAE Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        
        # Contrastive learning loss
        if 'contrastive_loss' in self.metrics:
            epochs = [item['epoch'] for item in self.metrics['contrastive_loss']]
            values = [item['value'] for item in self.metrics['contrastive_loss']]
            axes[1, 1].plot(epochs, values, label='Contrastive Loss', alpha=0.7, color='red')
            axes[1, 1].set_title('Contrastive Learning Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'training_losses.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_fid_scores(self):
        """Plot FID score changes"""
        # Check if there's FID-related data
        fid_metrics = {k: v for k, v in self.metrics.items() if k.startswith('fid_')}
        evaluation_snapshots = self.metrics.get('evaluation_snapshots', [])
        
        if not fid_metrics and not evaluation_snapshots:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # If there's evaluation snapshot data, use it
        if evaluation_snapshots:
            epochs = [entry['epoch'] for entry in evaluation_snapshots]
            teacher_fids = [entry['Teacher_FID_curr'] for entry in evaluation_snapshots]
            student_fids = [entry['Student_FID_curr'] for entry in evaluation_snapshots]
            
            ax.plot(epochs, teacher_fids, 'bo-', label='Teacher FID', linewidth=2, markersize=6)
            ax.plot(epochs, student_fids, 'ro-', label='Student FID', linewidth=2, markersize=6)
        
        # If there are other FID metrics, also plot them
        for metric_name, data in fid_metrics.items():
            if isinstance(data, list) and data:
                task_name = metric_name.replace('fid_', '')
                timestamps = [item.get('timestamp', 0) for item in data]
                
                # Handle FID data with different structures
                fid_scores = []
                for item in data:
                    if 'fid_score' in item:
                        fid_scores.append(item['fid_score'])
                    elif 'value' in item:
                        fid_scores.append(item['value'])
                    else:
                        # Skip data without FID scores
                        continue
                
                if fid_scores:  # Only plot when there are valid FID scores
                    ax.plot(timestamps[:len(fid_scores)], fid_scores, label=f'{task_name} FID', marker='o', alpha=0.7)
        
        ax.set_title('FID Scores Over Time')
        ax.set_xlabel('Epoch' if evaluation_snapshots else 'Training Time (seconds)')
        ax.set_ylabel('FID Score')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'fid_scores.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_expert_timeline(self):
        """Plot expert addition timeline"""
        if not self.expert_data:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        timestamps = [item['timestamp'] for item in self.expert_data]
        expert_ids = [item['expert_id'] for item in self.expert_data]
        fid_triggers = [item['trigger_fid'] for item in self.expert_data]
        
        scatter = ax.scatter(timestamps, expert_ids, c=fid_triggers, 
                           cmap='viridis', s=100, alpha=0.7)
        
        for i, (ts, eid, fid) in enumerate(zip(timestamps, expert_ids, fid_triggers)):
            ax.annotate(f'FID:{fid:.1f}', (ts, eid), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_title('Expert Addition Timeline')
        ax.set_xlabel('Training Time (seconds)')
        ax.set_ylabel('Expert ID')
        plt.colorbar(scatter, label='Trigger FID Score')
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'expert_timeline.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_task_performance(self):
        """Plot task performance comparison"""
        if not self.task_data:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        task_names = [item['task_name'] for item in self.task_data]
        final_fids = [item.get('final_fid', 0) for item in self.task_data]
        durations = [item['duration'] for item in self.task_data]
        
        # FID score comparison
        axes[0].bar(task_names, final_fids, alpha=0.7)
        axes[0].set_title('Final FID Scores by Task')
        axes[0].set_ylabel('FID Score')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Training time comparison
        axes[1].bar(task_names, durations, alpha=0.7, color='orange')
        axes[1].set_title('Training Duration by Task')
        axes[1].set_ylabel('Duration (seconds)')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'task_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_heatmap(self):
        """Plot metrics heatmap"""
        if not self.epoch_data:
            return
        
        # Create DataFrame from epoch summary data
        df = pd.DataFrame(self.epoch_data)
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if not col in ['epoch', 'timestamp', 'duration']]
        
        if len(numeric_cols) == 0:
            return
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        heatmap_data = df[numeric_cols].T
        sns.heatmap(heatmap_data, annot=False, cmap='viridis', ax=ax)
        
        ax.set_title('Training Metrics Heatmap')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metrics')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'metrics_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    
    
    def _plot_fid_curves(self):
        """Plot FID curves analysis"""
        if not self.fid_curves:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('FID Curves Analysis by Task and Phase', fontsize=16)
        
        # Plot grouped by task
        task_phases = defaultdict(list)
        for key, curves in self.fid_curves.items():
            if '_' in key:
                parts = key.split('_', 1)
                if len(parts) == 2:
                    task_name, phase = parts
                    task_phases[task_name].append((phase, curves))
        
        plot_idx = 0
        for task_name, phase_data in task_phases.items():
            if plot_idx >= 4:  # Maximum 4 subplots
                break
                
            row, col = plot_idx // 2, plot_idx % 2
            ax = axes[row, col]
            
            for phase, curves in phase_data:
                if curves:
                    epochs = [item.get('epoch', 0) for item in curves]
                    fid_scores = [item.get('fid_score', 0) for item in curves]
                    num_experts = [item.get('num_experts', 1) for item in curves]
                    
                    # Filter valid data
                    valid_data = [(e, f, n) for e, f, n in zip(epochs, fid_scores, num_experts) 
                                if f > 0 and n > 0]
                    
                    if valid_data:
                        valid_epochs, valid_fids, valid_experts = zip(*valid_data)
                        
                        # Main FID curve
                        line1 = ax.plot(valid_epochs, valid_fids, label=f'{phase} FID', marker='o', alpha=0.7)
                        
                        # Expert count changes (secondary axis)
                        ax2 = ax.twinx()
                        line2 = ax2.plot(valid_epochs, valid_experts, label=f'{phase} Experts', 
                                       color='red', linestyle='--', alpha=0.7)
                        
                        ax.set_title(f'{task_name} - {phase}')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('FID Score', color=line1[0].get_color())
                        ax2.set_ylabel('Number of Experts', color=line2[0].get_color())
                        ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'fid_curves_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_expert_growth_timeline(self):
        """Plot expert growth timeline"""
        if not self.expert_growth_timeline:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Expert Growth Timeline Analysis', fontsize=16)
        
        # Sort events by time
        sorted_events = sorted(self.expert_growth_timeline, key=lambda x: x.get('timestamp', 0))
        
        if sorted_events:
            timestamps = [event.get('timestamp', 0) for event in sorted_events]
            actions = [event.get('action', 'unknown') for event in sorted_events]
            expert_ids = [event.get('active_expert_id', 0) for event in sorted_events]
            fid_scores = [event.get('fid_current', 0) for event in sorted_events]
            thresholds = [event.get('growth_threshold', 0) for event in sorted_events]
            
            # Filter valid data
            valid_data = [(t, e, f, th) for t, e, f, th in zip(timestamps, expert_ids, fid_scores, thresholds)
                         if t > 0 and e > 0 and f > 0 and th > 0]
            
            if valid_data:
                valid_timestamps, valid_expert_ids, valid_fid_scores, valid_thresholds = zip(*valid_data)
                
                # Upper plot: Expert ID changes
                ax1.plot(valid_timestamps, valid_expert_ids, 'bo-', label='Active Expert ID', linewidth=2)
                ax1.set_ylabel('Expert ID')
                ax1.set_title('Expert ID Changes Over Time')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Lower plot: FID scores and thresholds
                ax2.plot(valid_timestamps, valid_fid_scores, 'ro-', label='Current FID', linewidth=2)
                ax2.plot(valid_timestamps, valid_thresholds, 'g--', label='Growth Threshold', linewidth=2)
                ax2.set_xlabel('Time (seconds)')
                ax2.set_ylabel('FID Score')
                ax2.set_title('FID Scores vs Growth Threshold')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                # Annotate key events
                for i, event in enumerate(sorted_events):
                    if event.get('action') == 'add_expert' and event.get('timestamp', 0) > 0:
                        ax1.annotate(f"Add\nExpert{event.get('active_expert_id', 0)}", 
                                    xy=(event.get('timestamp', 0), event.get('active_expert_id', 0)),
                                    xytext=(10, 10), textcoords='offset points',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'expert_growth_timeline.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_kd_gating_status(self):
        """Plot KD gating status changes"""
        if not self.kd_gating_status:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Knowledge Distillation Gating Status', fontsize=16)
        
        if self.kd_gating_status:
            timestamps = [status.get('timestamp', 0) for status in self.kd_gating_status]
            kd_enabled = [status.get('kd_enabled', False) for status in self.kd_gating_status]
            kd_weights = [status.get('kd_weight', 0.0) for status in self.kd_gating_status]
            kd_intervals = [status.get('kd_check_interval', 0) for status in self.kd_gating_status]
            
            # Filter valid data
            valid_data = [(t, e, w, i) for t, e, w, i in zip(timestamps, kd_enabled, kd_weights, kd_intervals)
                         if t > 0 and i > 0]
            
            if valid_data:
                valid_timestamps, valid_kd_enabled, valid_kd_weights, valid_kd_intervals = zip(*valid_data)
                
                # Upper plot: KD enable status
                ax1.plot(valid_timestamps, valid_kd_enabled, 'bo-', label='KD Enabled', linewidth=2, markersize=8)
                ax1.set_ylabel('KD Status')
                ax1.set_title('Knowledge Distillation Enable/Disable Status')
                ax1.set_yticks([0, 1])
                ax1.set_yticklabels(['Disabled', 'Enabled'])
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Lower plot: KD weight and check interval
                ax2_twin = ax2.twinx()
                line1 = ax2.plot(valid_timestamps, valid_kd_weights, 'ro-', label='KD Weight', linewidth=2)
                line2 = ax2_twin.plot(valid_timestamps, valid_kd_intervals, 'g^-', label='Check Interval', linewidth=2)
                
                ax2.set_xlabel('Time (seconds)')
                ax2.set_ylabel('KD Weight', color=line1[0].get_color())
                ax2_twin.set_ylabel('Check Interval', color=line2[0].get_color())
                ax2.set_title('KD Weight and Check Interval Changes')
                ax2.grid(True, alpha=0.3)
                
                # Combine legends
                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax2_twin.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'kd_gating_status.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_forgetting_analysis(self):
        """Plot forgetting analysis"""
        if not self.forgetting_metrics:
            return
        
        # Check completeness of forgetting metric data
        valid_tasks = []
        for task_name, task_data in self.forgetting_metrics.items():
            if 'fid_task_end' in task_data and 'fid_final' in task_data:
                valid_tasks.append(task_name)
        
        if not valid_tasks:
            print(" No complete forgetting data, skipping forgetting analysis charts")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Catastrophic Forgetting Analysis', fontsize=16)
        
        task_names = valid_tasks
        fid_task_end = [self.forgetting_metrics[task]['fid_task_end'] for task in task_names]
        fid_final = [self.forgetting_metrics[task]['fid_final'] for task in task_names]
        forgetting_values = [self.forgetting_metrics[task]['forgetting'] for task in task_names]
        
        # Left plot: FID comparison
        x = np.arange(len(task_names))
        width = 0.35
        
        ax1.bar(x - width/2, fid_task_end, width, label='FID (Task End)', alpha=0.8)
        ax1.bar(x + width/2, fid_final, width, label='FID (Final)', alpha=0.8)
        ax1.set_xlabel('Tasks')
        ax1.set_ylabel('FID Score')
        ax1.set_title('FID Comparison: Task End vs Final')
        ax1.set_xticks(x)
        ax1.set_xticklabels(task_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Forgetting scores
        colors = ['red' if f > 0 else 'green' for f in forgetting_values]
        bars = ax2.bar(task_names, forgetting_values, color=colors, alpha=0.7)
        ax2.set_xlabel('Tasks')
        ax2.set_ylabel('Forgetting Score')
        ax2.set_title('Catastrophic Forgetting by Task')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, forgetting_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.01),
                    f'{value:.2f}', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'forgetting_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_efficiency_metrics(self):
        """Plot efficiency and resource monitoring charts"""
        if not self.efficiency_metrics:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Efficiency and Resource Monitoring', fontsize=16)
        
        if self.efficiency_metrics:
            timestamps = [metric.get('timestamp', 0) for metric in self.efficiency_metrics]
            durations = [metric.get('duration_sec', 0) for metric in self.efficiency_metrics]
            images_per_sec = [metric.get('images_per_sec', 0) for metric in self.efficiency_metrics]
            gpu_memory = [metric.get('gpu_mem_peak_mb', 0) for metric in self.efficiency_metrics]
            params_teacher = [metric.get('params_teacher', 0) for metric in self.efficiency_metrics]
            params_student = [metric.get('params_student', 0) for metric in self.efficiency_metrics]
            
            # Filter valid data
            valid_data = [(t, d, i, g, pt, ps) for t, d, i, g, pt, ps in 
                         zip(timestamps, durations, images_per_sec, gpu_memory, params_teacher, params_student)
                         if t > 0 and d > 0 and i > 0 and g > 0 and pt > 0 and ps > 0]
            
            if valid_data:
                valid_timestamps, valid_durations, valid_images_per_sec, valid_gpu_memory, valid_params_teacher, valid_params_student = zip(*valid_data)
                
                # Top left: Processing speed
                axes[0, 0].plot(valid_timestamps, valid_images_per_sec, 'bo-', linewidth=2)
                axes[0, 0].set_title('Processing Speed')
                axes[0, 0].set_ylabel('Images per Second')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Top right: GPU memory usage
                axes[0, 1].plot(valid_timestamps, valid_gpu_memory, 'ro-', linewidth=2)
                axes[0, 1].set_title('GPU Memory Usage')
                axes[0, 1].set_ylabel('GPU Memory (MB)')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Bottom left: Parameter comparison
                axes[1, 0].plot(valid_timestamps, valid_params_teacher, 'go-', label='Teacher', linewidth=2)
                axes[1, 0].plot(valid_timestamps, valid_params_student, 'mo-', label='Student', linewidth=2)
                axes[1, 0].set_title('Model Parameters')
                axes[1, 0].set_ylabel('Number of Parameters')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # Bottom right: Processing time
                axes[1, 1].plot(valid_timestamps, valid_durations, 'co-', linewidth=2)
                axes[1, 1].set_title('Processing Duration')
                axes[1, 1].set_xlabel('Time (seconds)')
                axes[1, 1].set_ylabel('Duration (seconds)')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'efficiency_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    
    
    def _plot_fid_trend_by_task(self):
        """Plot FID trend by task"""
        # Collect all epoch-level FID data
        fid_trend_data = []
        
        # First try to collect from evaluation snapshot data
        evaluation_snapshots = self.metrics.get('evaluation_snapshots', [])
        if evaluation_snapshots:
            for entry in evaluation_snapshots:
                fid_trend_data.append({
                    'task_name': entry.get('task_name', ''),
                    'epoch': entry.get('epoch', 0),
                    'model_type': 'Teacher',
                    'fid_score': entry.get('Teacher_FID_curr', 0)
                })
                fid_trend_data.append({
                    'task_name': entry.get('task_name', ''),
                    'epoch': entry.get('epoch', 0),
                    'model_type': 'Student',
                    'fid_score': entry.get('Student_FID_curr', 0)
                })
        
        # Then try to collect from other FID metrics
        for metric_key, metric_list in self.metrics.items():
            if metric_key.startswith('fid_epoch_') and metric_list:
                for entry in metric_list:
                    if 'epoch' in entry and 'fid_score' in entry:
                        fid_trend_data.append({
                            'task_name': entry.get('task_name', ''),
                            'epoch': entry.get('epoch', 0),
                            'model_type': 'Teacher' if 'Teacher' in metric_key else 'Student',
                            'fid_score': entry.get('fid_score', 0)
                        })
        
        if not fid_trend_data:
            return
        
        # Create charts
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('FID Trend Analysis by Task', fontsize=16)
        
        # Group data by task
        task_data = defaultdict(lambda: {'Teacher': [], 'Student': []})
        for entry in fid_trend_data:
            task_name = entry['task_name']
            model_type = entry['model_type']
            task_data[task_name][model_type].append((entry['epoch'], entry['fid_score']))
        
        # Plot FID trend for each task
        colors = plt.cm.Set3(np.linspace(0, 1, len(task_data)))
        
        for i, (task_name, models_data) in enumerate(task_data.items()):
            color = colors[i]
            
            # Teacher FID trend
            if models_data['Teacher']:
                epochs, fids = zip(*sorted(models_data['Teacher']))
                axes[0].plot(epochs, fids, 'o-', label=f'{task_name} (Teacher)', 
                           color=color, linewidth=2, markersize=6)
            
            # Student FID trend
            if models_data['Student']:
                epochs, fids = zip(*sorted(models_data['Student']))
                axes[1].plot(epochs, fids, 's-', label=f'{task_name} (Student)', 
                           color=color, linewidth=2, markersize=6)
        
        # Set labels and titles
        axes[0].set_title('Teacher FID Trend by Task')
        axes[0].set_ylabel('FID Score')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title('Student FID Trend by Task')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('FID Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'fid_trend_by_task.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(" FID trend by task chart generated")
    
    def _plot_old_task_performance(self):
        """Plot student model's old task retention analysis"""
        if 'old_task_performance' not in self.metrics or not self.metrics['old_task_performance']:
            return
        
        # Create charts
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Student Model Old Task Retention Analysis', fontsize=16)
        
        old_task_data = self.metrics['old_task_performance']
        
        # Group by current task
        current_task_groups = defaultdict(list)
        for entry in old_task_data:
            current_task_groups[entry['current_task']].append(entry)
        
        # 1. Old task FID distribution
        all_old_fids = [entry['old_task_fid'] for entry in old_task_data]
        axes[0, 0].hist(all_old_fids, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Old Task FID Distribution')
        axes[0, 0].set_xlabel('FID Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Reconstruction quality analysis
        reconstruction_qualities = [entry.get('reconstruction_quality', 0) for entry in old_task_data]
        axes[0, 1].hist(reconstruction_qualities, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Reconstruction Quality Distribution')
        axes[0, 1].set_xlabel('Reconstruction Quality')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Current task vs old task performance
        current_tasks = list(current_task_groups.keys())
        avg_old_fids = []
        for current_task in current_tasks:
            task_entries = current_task_groups[current_task]
            avg_fid = np.mean([entry['old_task_fid'] for entry in task_entries])
            avg_old_fids.append(avg_fid)
        
        if current_tasks and avg_old_fids:
            axes[1, 0].bar(current_tasks, avg_old_fids, alpha=0.7, color='orange')
            axes[1, 0].set_title('Average FID on Old Tasks by Current Task')
            axes[1, 0].set_xlabel('Current Task')
            axes[1, 0].set_ylabel('Average Old Task FID')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Forgetting analysis
        if len(current_tasks) > 1:
            # Calculate forgetting score (using first task as baseline)
            baseline_task = current_tasks[0]
            baseline_entries = current_task_groups[baseline_task]
            baseline_fids = [entry['old_task_fid'] for entry in baseline_entries]
            
            forgetting_scores = []
            for current_task in current_tasks[1:]:
                task_entries = current_task_groups[current_task]
                task_fids = [entry['old_task_fid'] for entry in task_entries]
                
                # Calculate forgetting score (FID increase indicates forgetting)
                if baseline_fids and task_fids:
                    baseline_avg = np.mean(baseline_fids)
                    task_avg = np.mean(task_fids)
                    forgetting = task_avg - baseline_avg
                    forgetting_scores.append(forgetting)
            
            if forgetting_scores:
                axes[1, 1].bar(current_tasks[1:], forgetting_scores, alpha=0.7, 
                              color=['red' if f > 0 else 'green' for f in forgetting_scores])
                axes[1, 1].set_title('Forgetting Analysis (Relative to Baseline Task)')
                axes[1, 1].set_xlabel('Task')
                axes[1, 1].set_ylabel('Forgetting Score (FID Increase)')
                axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'old_task_performance_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(" Old task performance analysis chart generated")
    
    def _plot_kd_stability_analysis(self):
        """Plot knowledge distillation stability analysis"""
        if 'kd_stability_metrics' not in self.metrics or not self.metrics['kd_stability_metrics']:
            return
        
        # Create charts
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Knowledge Distillation Stability Analysis', fontsize=16)
        
        kd_data = self.metrics['kd_stability_metrics']
        
        # Group by task
        task_groups = defaultdict(list)
        for entry in kd_data:
            task_groups[entry['task_name']].append(entry)
        
        # 1. Teacher-Student similarity changes
        for task_name, entries in task_groups.items():
            epochs = [entry['epoch'] for entry in entries]
            similarities = [entry['teacher_student_similarity'] for entry in entries]
            axes[0, 0].plot(epochs, similarities, 'o-', label=task_name, linewidth=2, markersize=6)
        
        axes[0, 0].set_title('Teacher-Student Similarity Changes')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Similarity')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Knowledge distillation weight changes
        for task_name, entries in task_groups.items():
            epochs = [entry['epoch'] for entry in entries]
            weights = [entry['kd_weight'] for entry in entries]
            axes[0, 1].plot(epochs, weights, 's-', label=task_name, linewidth=2, markersize=6)
        
        axes[0, 1].set_title('Knowledge Distillation Weight Changes')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('KD Weight')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distillation gating status statistics
        kd_enabled_counts = defaultdict(int)
        total_counts = defaultdict(int)
        
        for entry in kd_data:
            task_name = entry['task_name']
            total_counts[task_name] += 1
            if entry['kd_enabled']:
                kd_enabled_counts[task_name] += 1
        
        if total_counts:
            task_names = list(total_counts.keys())
            enabled_ratios = [kd_enabled_counts[task] / total_counts[task] * 100 for task in task_names]
            
            bars = axes[1, 0].bar(task_names, enabled_ratios, alpha=0.7, color='lightblue')
            axes[1, 0].set_title('KD Enable Ratio by Task')
            axes[1, 0].set_xlabel('Task')
            axes[1, 0].set_ylabel('KD Enable Ratio (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add value labels
            for bar, ratio in zip(bars, enabled_ratios):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{ratio:.1f}%', ha='center', va='bottom')
        
        # 4. Cosine similarity distribution
        all_cosine_similarities = [entry.get('cosine_similarity', 0) for entry in kd_data 
                                 if entry.get('cosine_similarity') is not None]
        
        if all_cosine_similarities:
            axes[1, 1].hist(all_cosine_similarities, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[1, 1].set_title('Cosine Similarity Distribution')
            axes[1, 1].set_xlabel('Cosine Similarity')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add quality rating lines
            axes[1, 1].axvline(x=0.7, color='green', linestyle='--', alpha=0.7, label='High Quality Threshold')
            axes[1, 1].axvline(x=0.4, color='orange', linestyle='--', alpha=0.7, label='Medium Quality Threshold')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'kd_stability_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(" Knowledge distillation stability analysis chart generated")
    
    def generate_report(self):
        """Generate experiment report"""
        print(" Generating experiment report...")
        
        report = {
            'experiment_name': self.experiment_name,
            'generation_time': datetime.datetime.now().isoformat(),
            'total_duration': time.time() - self.experiment_start_time,
            'config': self.config_data,
            'summary': self._generate_summary(),
            'detailed_analysis': self._generate_detailed_analysis()
        }
        
        # Save JSON report
        report_file = os.path.join(self.reports_dir, 'experiment_summary.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # Save Markdown report
        markdown_file = os.path.join(self.reports_dir, 'detailed_analysis.md')
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(self._generate_markdown_report(report))
        
        # Save text format report
        analysis_report_file = os.path.join(self.reports_dir, 'analysis_report.txt')
        with open(analysis_report_file, 'w', encoding='utf-8') as f:
            f.write(f"Experiment Name: {self.experiment_name}\n")
            f.write(f"Generation Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Training Duration: {time.time() - self.experiment_start_time:.2f} seconds\n\n")
            f.write("Experiment Summary:\n")
            f.write(json.dumps(report['summary'], indent=2, ensure_ascii=False, default=str))
            f.write("\n\nDetailed Analysis:\n")
            f.write(json.dumps(report['detailed_analysis'], indent=2, ensure_ascii=False, default=str))
        
        # Save detailed analysis report
        detailed_report_file = os.path.join(self.reports_dir, 'detailed_analysis.txt')
        with open(detailed_report_file, 'w', encoding='utf-8') as f:
            f.write(f"Experiment Name: {self.experiment_name}\n")
            f.write(f"Generation Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            f.write(self._generate_markdown_report(report))
        
        print(f" Reports generated: {self.reports_dir}")
        print(f" Analysis reports: analysis_report.txt")
        print(f" Detailed report: detailed_analysis.txt")
        
        return report
    
    def _generate_summary(self):
        """Generate experiment summary"""
        summary = {
            'total_epochs': len(self.epoch_data),
            'total_tasks': len(self.task_data),
            'total_experts': len(self.expert_data) + 1,  # +1 for initial expert
            'training_duration': time.time() - self.experiment_start_time
        }
        
        if self.task_data:
            final_fids = [task.get('final_fid', float('inf')) for task in self.task_data]
            summary['average_final_fid'] = np.mean(final_fids)
            summary['best_final_fid'] = np.min(final_fids)
            summary['worst_final_fid'] = np.max(final_fids)
        
        return summary
    
    def _generate_detailed_analysis(self):
        """Generate detailed analysis"""
        analysis = {}
        
        # Loss analysis
        if 'discriminator_loss' in self.metrics:
            d_losses = [item['value'] for item in self.metrics['discriminator_loss']]
            analysis['discriminator_loss_stats'] = {
                'mean': np.mean(d_losses),
                'std': np.std(d_losses),
                'trend': 'decreasing' if d_losses[-1] < d_losses[0] else 'increasing'
            }
        
        if 'generator_loss' in self.metrics:
            g_losses = [item['value'] for item in self.metrics['generator_loss']]
            analysis['generator_loss_stats'] = {
                'mean': np.mean(g_losses),
                'std': np.std(g_losses),
                'trend': 'decreasing' if g_losses[-1] < g_losses[0] else 'increasing'
            }
        
        # Expert analysis
        if self.expert_data:
            trigger_fids = [item['trigger_fid'] for item in self.expert_data]
            analysis['expert_addition_analysis'] = {
                'average_trigger_fid': np.mean(trigger_fids),
                'expert_addition_rate': len(self.expert_data) / len(self.task_data) if self.task_data else 0
            }
        
        # Forgetting analysis
        if self.forgetting_metrics:
            # Safe access to forgetting values, handling potentially missing keys
            forgetting_values = []
            for data in self.forgetting_metrics.values():
                if isinstance(data, dict) and 'forgetting' in data:
                    forgetting_values.append(data['forgetting'])
                elif isinstance(data, (int, float)):
                    # If it is a direct numeric value, use it directly
                    forgetting_values.append(data)
            
            if forgetting_values:  # Only analyze when valid data exists
                analysis['forgetting_analysis'] = {
                    'average_forgetting': np.mean(forgetting_values),
                    'max_forgetting': np.max(forgetting_values),
                    'min_forgetting': np.min(forgetting_values),
                    'forgetting_tasks': len(self.forgetting_metrics),
                    'positive_forgetting_tasks': len([f for f in forgetting_values if f > 0])
                }
        
        # Expert growth analysis
        if self.expert_growth_timeline:
            growth_events = [e for e in self.expert_growth_timeline if e['action'] == 'add_expert']
            analysis['expert_growth_analysis'] = {
                'total_growth_events': len(growth_events),
                'average_growth_interval': np.mean([e['event_epoch'] for e in growth_events]) if growth_events else 0,
                'growth_triggers': list(set([e['reason'] for e in growth_events if 'reason' in e]))
            }
        
        # KD gating analysis
        if self.kd_gating_status:
            kd_enabled_count = sum(1 for status in self.kd_gating_status if status['kd_enabled'])
            analysis['kd_gating_analysis'] = {
                'total_status_changes': len(self.kd_gating_status),
                'kd_enabled_percentage': kd_enabled_count / len(self.kd_gating_status) * 100,
                'average_kd_weight': np.mean([status['kd_weight'] for status in self.kd_gating_status])
            }
        
        # Efficiency analysis
        if self.efficiency_metrics:
            # Safe access to efficiency metrics, handling potentially missing keys
            processing_speeds = []
            gpu_memories = []
            durations = []
            
            for m in self.efficiency_metrics:
                if isinstance(m, dict):
                    # Handle different possible key names
                    if 'images_per_sec' in m:
                        processing_speeds.append(m['images_per_sec'])
                    elif 'throughput_img_s' in m:
                        processing_speeds.append(m['throughput_img_s'])
                    
                    if 'gpu_mem_peak_mb' in m:
                        gpu_memories.append(m['gpu_mem_peak_mb'])
                    elif 'peak_mem_MB' in m:
                        gpu_memories.append(m['peak_mem_MB'])
                    
                    if 'duration_sec' in m:
                        durations.append(m['duration_sec'])
                    elif 'time_sec' in m:
                        durations.append(m['time_sec'])
            
            if processing_speeds or gpu_memories or durations:
                analysis['efficiency_analysis'] = {
                    'average_processing_speed': np.mean(processing_speeds) if processing_speeds else 0,
                    'peak_gpu_memory': max(gpu_memories) if gpu_memories else 0,
                    'average_duration': np.mean(durations) if durations else 0
                }
        
        return analysis
    
    def _generate_markdown_report(self, report_data):
        """Generate Markdown format report"""
        md_content = f"""# Experiment Report: {self.experiment_name}

**Generation Time**: {report_data['generation_time']}
**Total Training Time**: {report_data['total_duration']:.1f} seconds

##  Experiment Summary

- **Total Training Epochs**: {report_data['summary'].get('total_epochs', 0)}
- **Number of Tasks**: {report_data['summary'].get('total_tasks', 0)}
- **Number of Experts**: {report_data['summary'].get('total_experts', 1)}
- **AverageFID**: {report_data['summary'].get('average_final_fid', 'N/A')}

##  Key Metrics

### Loss Statistics
"""
        if 'detailed_analysis' in report_data:
            analysis = report_data['detailed_analysis']
            if 'discriminator_loss_stats' in analysis:
                stats = analysis['discriminator_loss_stats']
                md_content += f"""
**Discriminator Loss**:
- Mean: {stats['mean']:.4f}
- Std Dev: {stats['std']:.4f}
- Trend: {stats['trend']}
"""
            
            if 'generator_loss_stats' in analysis:
                stats = analysis['generator_loss_stats']
                md_content += f"""
**Generator Loss**:
- Mean: {stats['mean']:.4f}
- Std Dev: {stats['std']:.4f}
- Trend: {stats['trend']}
"""
            
            if 'expert_addition_analysis' in analysis:
                stats = analysis['expert_addition_analysis']
                md_content += f"""
**Expert Addition Analysis**:
- AverageTrigger FID: {stats['average_trigger_fid']:.2f}
- Expert Addition Rate: {stats['expert_addition_rate']:.2f}
"""
            
            # New: Evaluation function analysis
            if 'forgetting_analysis' in analysis:
                stats = analysis['forgetting_analysis']
                md_content += f"""
**Forgetting analysis**:
- Average Forgetting: {stats['average_forgetting']:.2f}
- Max Forgetting: {stats['max_forgetting']:.2f}
- Min Forgetting: {stats['min_forgetting']:.2f}
- Forgetting Tasks: {stats['forgetting_tasks']}
- Positive Forgetting Tasks: {stats['positive_forgetting_tasks']}
"""
            
            if 'expert_growth_analysis' in analysis:
                stats = analysis['expert_growth_analysis']
                md_content += f"""
**Expert growth analysis**:
- Total Growth Events: {stats['total_growth_events']}
- Average Growth Interval: {stats['average_growth_interval']:.1f} epochs
- Growth Triggers: {', '.join(stats['growth_triggers']) if stats['growth_triggers'] else 'N/A'}
"""
            
            if 'kd_gating_analysis' in analysis:
                stats = analysis['kd_gating_analysis']
                md_content += f"""
**KD gating analysis**:
- Status Changes: {stats['total_status_changes']}
- KD Enabled Percentage: {stats['kd_enabled_percentage']:.1f}%
- Average KD Weight: {stats['average_kd_weight']:.3f}
"""
            
            if 'efficiency_analysis' in analysis:
                stats = analysis['efficiency_analysis']
                md_content += f"""
**Efficiency analysis**:
- Average Processing Speed: {stats['average_processing_speed']:.1f} img/s
- Peak GPU Memory: {stats['peak_gpu_memory']:.1f} MB
- Average Processing Time: {stats['average_duration']:.1f} s
"""
        
        # New: Three section analysis results
        md_content += f"""

"""
        
        md_content += f"""

##  Experiment Configuration

```json
{json.dumps(report_data['config'], indent=2)}
```

##  File Structure

- `logs/`: Detailed training logs
- `plots/`: Visualization charts
- `data/`: Structured data files
- `reports/`: Analysis reports

---
*Report automatically generated by ExperimentLogger*
"""
        
        return md_content
    
    def _append_to_detailed_log(self, log_entry):
        """Append detailed log"""
        log_file = os.path.join(self.logs_dir, 'detailed_training.jsonl')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, default=str) + '\n')
    
    def _save_epoch_summary(self):
        """Save epoch summary"""
        # Save as JSON format
        summary_file = os.path.join(self.data_dir, 'epoch_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.epoch_data, f, indent=2, default=str)
        
        # Also save as CSV format
        if self.epoch_data:
            try:
                df_epoch = pd.DataFrame(self.epoch_data)
                csv_file = os.path.join(self.data_dir, 'epoch_summary.csv')
                df_epoch.to_csv(csv_file, index=False, encoding='utf-8')
                print(f" Epoch summary data saved: {len(self.epoch_data)} records")
            except Exception as e:
                print(f" Failed to save epoch_summary.csv: {e}")
    
    def _save_task_data(self):
        """Save task data"""
        task_file = os.path.join(self.data_dir, 'task_data.json')
        with open(task_file, 'w', encoding='utf-8') as f:
            json.dump(self.task_data, f, indent=2, default=str)
    
    def _save_expert_data(self):
        """Save expert data"""
        expert_file = os.path.join(self.data_dir, 'expert_data.json')
        with open(expert_file, 'w', encoding='utf-8') as f:
            json.dump(self.expert_data, f, indent=2, default=str)
    
    def _save_kd_gating_status(self):
        """Save KD gating status data"""
        if self.kd_gating_status:
            kd_file = os.path.join(self.data_dir, 'kd_gating_status.json')
            with open(kd_file, 'w', encoding='utf-8') as f:
                json.dump(self.kd_gating_status, f, indent=2, default=str)
            
            # Also save as CSV format
            df_kd = pd.DataFrame(self.kd_gating_status)
            df_kd.to_csv(os.path.join(self.data_dir, 'kd_gating_status.csv'), index=False)
            
            print(f" KD gating status data saved: {len(self.kd_gating_status)} records")
    
    def _save_sample_paths(self):
        """Save sample paths data"""
        if self.sample_paths:
            paths_file = os.path.join(self.data_dir, 'sample_paths.json')
            with open(paths_file, 'w', encoding='utf-8') as f:
                json.dump(self.sample_paths, f, indent=2, default=str)
    
    def _save_efficiency_metrics(self):
        """Save efficiency metrics data"""
        if self.efficiency_metrics:
            eff_file = os.path.join(self.data_dir, 'efficiency_metrics.json')
            with open(eff_file, 'w', encoding='utf-8') as f:
                json.dump(self.efficiency_metrics, f, indent=2, default=str)
            
            # Also save as CSV format
            df_eff = pd.DataFrame(self.efficiency_metrics)
            df_eff.to_csv(os.path.join(self.data_dir, 'efficiency_metrics.csv'), index=False)
            
            print(f" Efficiency metrics data saved: {len(self.efficiency_metrics)} records")
    
    def _save_evaluation_data(self):
        """Save evaluation data"""
        # Save as JSON format
        eval_file = os.path.join(self.data_dir, 'evaluation_data.json')
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_data, f, indent=2, default=str)
        
        # Save evaluation snapshot data as CSV format
        if 'evaluation_snapshots' in self.metrics and self.metrics['evaluation_snapshots']:
            eval_csv_file = os.path.join(self.data_dir, 'evaluation_snapshots.csv')
            df_eval = pd.DataFrame(self.metrics['evaluation_snapshots'])
            df_eval.to_csv(eval_csv_file, index=False, encoding='utf-8')
            print(f" Evaluation snapshot data saved: {len(self.metrics['evaluation_snapshots'])} records")
            
            # Also save as JSON format
            eval_snapshots_file = os.path.join(self.data_dir, 'evaluation_snapshots.json')
            with open(eval_snapshots_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics['evaluation_snapshots'], f, indent=2, default=str)

    def _save_fid_curves(self):
        """Save FID curve data"""
        # Save as JSON format
        for task_name, curves in self.fid_curves.items():
            file_name = os.path.join(self.data_dir, f'fid_curve_{task_name}.json')
            with open(file_name, 'w', encoding='utf-8') as f:
                json.dump(curves, f, indent=2, default=str)
        
        # Save FID curve data as CSV format
        if 'fid_curves' in self.metrics and self.metrics['fid_curves']:
            for task_name, curves in self.metrics['fid_curves'].items():
                if curves:  # Ensure there is data
                    csv_file = os.path.join(self.data_dir, f'fid_curve_{task_name}.csv')
                    df_curve = pd.DataFrame(curves)
                    df_curve.to_csv(csv_file, index=False, encoding='utf-8')
                    print(f" Task {task_name}  FID curve data saved: {len(curves)} data points")
            
            # Save FID curve summary for all tasks
            all_curves = []
            for task_name, curves in self.metrics['fid_curves'].items():
                for curve in curves:
                    curve['task_name'] = task_name
                    all_curves.append(curve)
            
            if all_curves:
                summary_file = os.path.join(self.data_dir, 'fid_curves_summary.csv')
                df_summary = pd.DataFrame(all_curves)
                df_summary.to_csv(summary_file, index=False, encoding='utf-8')
                print(f" FID curve summary data saved: {len(all_curves)} data points")

    def _save_forgetting_metrics(self):
        """Save forgetting metrics"""
        forget_file = os.path.join(self.data_dir, 'forgetting_metrics.json')
        with open(forget_file, 'w', encoding='utf-8') as f:
            json.dump(self.forgetting_metrics, f, indent=2, default=str)

    def _save_expert_growth_timeline(self):
        """Save expert growth timeline data"""
        if self.expert_growth_timeline:
            growth_file = os.path.join(self.data_dir, 'expert_growth_timeline.json')
            with open(growth_file, 'w', encoding='utf-8') as f:
                json.dump(self.expert_growth_timeline, f, indent=2, default=str)
            
            # Also save as CSV format
            df_growth = pd.DataFrame(self.expert_growth_timeline)
            df_growth.to_csv(os.path.join(self.data_dir, 'expert_growth_timeline.csv'), index=False)
            
            print(f" Expert growth timeline data saved: {len(self.expert_growth_timeline)} records")
    
    def _save_task_expert_mapping(self):
        """Save task-expert mapping data"""
        if self.task_expert_mapping:
            # Save as JSON format
            mapping_file = os.path.join(self.data_dir, 'task_expert_mapping.json')
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.task_expert_mapping, f, indent=2, default=str)
            
            # Save as CSV format
            mapping_records = []
            for task_name, mappings in self.task_expert_mapping.items():
                for mapping in mappings:
                    mapping_records.append(mapping)
            
            if mapping_records:
                df_mapping = pd.DataFrame(mapping_records)
                df_mapping.to_csv(os.path.join(self.data_dir, 'task_expert_mapping.csv'), index=False)
                print(f" Task-expert mapping data saved: {len(mapping_records)} records")
    
    def _save_cosine_similarity_timeline(self):
        """Save cosine similarity timeline data"""
        if self.cosine_similarity_timeline:
            # Save as JSON format
            similarity_file = os.path.join(self.data_dir, 'cosine_similarity_timeline.json')
            with open(similarity_file, 'w', encoding='utf-8') as f:
                json.dump(self.cosine_similarity_timeline, f, indent=2, default=str)
            
            # Save as CSV format
            df_similarity = pd.DataFrame(self.cosine_similarity_timeline)
            df_similarity.to_csv(os.path.join(self.data_dir, 'cosine_similarity_timeline.csv'), index=False)
            
            print(f" Cosine similarity timeline data saved: {len(self.cosine_similarity_timeline)} records")
    
    def _save_teacher_student_comparison(self):
        """Save Teacher vs Student comparison data"""
        if self.teacher_student_comparison:
            # Save as JSON format
            comparison_file = os.path.join(self.data_dir, 'teacher_student_comparison.json')
            with open(comparison_file, 'w', encoding='utf-8') as f:
                json.dump(self.teacher_student_comparison, f, indent=2, default=str)
            
            # Save as CSV format
            comparison_records = []
            for task_name, comparisons in self.teacher_student_comparison.items():
                for comparison in comparisons:
                    comparison['task_name'] = task_name
                    comparison_records.append(comparison)
            
            if comparison_records:
                df_comparison = pd.DataFrame(comparison_records)
                df_comparison.to_csv(os.path.join(self.data_dir, 'teacher_student_comparison.csv'), index=False)
                print(f" Teacher vs Student comparison data saved: {len(comparison_records)} records")
    
    def _save_expert_task_performance(self):
        """Save expert-task performance data"""
        if self.expert_task_performance:
            # Save as JSON format
            performance_file = os.path.join(self.data_dir, 'expert_task_performance.json')
            with open(performance_file, 'w', encoding='utf-8') as f:
                json.dump(self.expert_task_performance, f, indent=2, default=str)
            
            # Save as CSV format
            performance_records = []
            for expert_id, tasks in self.expert_task_performance.items():
                for task_name, performance in tasks.items():
                    record = {
                        'expert_id': expert_id,
                        'task_name': task_name,
                        **performance
                    }
                    performance_records.append(record)
            
            if performance_records:
                df_performance = pd.DataFrame(performance_records)
                df_performance.to_csv(os.path.join(self.data_dir, 'expert_task_performance.csv'), index=False)
                print(f" Expert-task performance data saved: {len(performance_records)} records")

    def finalize(self):
        """End experiment, generate final report"""
        print(f"\n Experiment logging completed")
        print("=" * 50)
        
        # Calculate forgetting metrics
        self.calculate_forgetting_metrics()
        
        #  Calculate retention rate metrics (retention rate analysis based on FID_end(j))
        self.calculate_retention_rates()
        
        # Generate all charts and reports
        self.generate_plots()
        report = self.generate_report()
        
        # Save final data
        self._save_all_data()
        
        # Save new evaluation function data
        self._save_evaluation_data()
        self._save_fid_curves()
        self._save_expert_growth_timeline()
        self._save_kd_gating_status()
        self._save_sample_paths()
        self._save_efficiency_metrics()
        self._save_task_expert_mapping()
        self._save_cosine_similarity_timeline()
        self._save_teacher_student_comparison()
        self._save_expert_task_performance()
        
        # Save key event data
        self._save_key_events()
        
        # Save task final FID data
        self._save_task_final_fids()
        
        # Save retention rate data
        self._save_retention_rates()
        
        # Save training metrics data (
        self._save_training_metrics()
        
        # Save representation similarity data
        self._save_representation_similarity()
        
        # 🆕 Save sample generation data
        self._save_sample_generation()
        
        # 🆕 Save KD ablation experiment data
        self._save_kd_ablation_experiments()
        
        print(f" Experiment results saved in: {self.experiment_dir}")
        print(f" Visualization charts: {self.plots_dir}")
        print(f" Analysis reports: {self.reports_dir}")
        print("=" * 50)
        
        return report
    
    def _save_all_data(self):
        """Save all data to CSV files"""
        # Save training metrics
        if self.epoch_data:
            df_epochs = pd.DataFrame(self.epoch_data)
            df_epochs.to_csv(os.path.join(self.data_dir, 'epoch_data.csv'), index=False)
        
        # Save task data
        if self.task_data:
            df_tasks = pd.DataFrame(self.task_data)
            df_tasks.to_csv(os.path.join(self.data_dir, 'task_data.csv'), index=False)
        
        # Save expert data
        if self.expert_data:
            df_experts = pd.DataFrame(self.expert_data)
            df_experts.to_csv(os.path.join(self.data_dir, 'expert_data.csv'), index=False)
        
        # New: Save evaluation function related data
        if self.evaluation_data:
            df_eval = pd.DataFrame(self.evaluation_data)
            df_eval.to_csv(os.path.join(self.data_dir, 'evaluation_data.csv'), index=False)
        
        if self.expert_growth_timeline:
            df_growth = pd.DataFrame(self.expert_growth_timeline)
            df_growth.to_csv(os.path.join(self.data_dir, 'expert_growth_timeline.csv'), index=False)
        
        if self.kd_gating_status:
            df_kd = pd.DataFrame(self.kd_gating_status)
            df_kd.to_csv(os.path.join(self.data_dir, 'kd_gating_status.csv'), index=False)
        
        if self.efficiency_metrics:
            df_eff = pd.DataFrame(self.efficiency_metrics)
            df_eff.to_csv(os.path.join(self.data_dir, 'efficiency_metrics.csv'), index=False)
        
        # Save forgetting metrics to CSV
        if self.forgetting_metrics:
            # Safe access to forgetting data, handling potentially missing keys
            forgetting_data = []
            for task, data in self.forgetting_metrics.items():
                row = {'task_name': task}
                
                # Safe access to all fields
                if isinstance(data, dict):
                    row['fid_task_end'] = data.get('fid_task_end', 0.0)
                    row['fid_final'] = data.get('fid_final', 0.0)
                    row['forgetting'] = data.get('forgetting', 0.0)
                    row['task_id'] = data.get('task_id', 0)
                else:
                    # If data is not a dictionary, use default values
                    row['fid_task_end'] = 0.0
                    row['fid_final'] = 0.0
                    row['forgetting'] = float(data) if isinstance(data, (int, float)) else 0.0
                    row['task_id'] = 0
                
                forgetting_data.append(row)
            
            if forgetting_data:
                df_forget = pd.DataFrame(forgetting_data)
                df_forget.to_csv(os.path.join(self.data_dir, 'forgetting_metrics.csv'), index=False)
        
        # Save Teacher-Student FID comparison data to CSV
        self._save_teacher_student_fid_comparison()
        
        
        self._save_three_section_data()
        
       
        self._save_expert_analysis_data()
        
        
        self._save_expert_expansion_trigger_logs()
        
       
        self._save_fid_slope_analysis_data()
        
        
        self._save_ablation_comparison_data()
        
       
        self._save_complete_csv_schema_data()
        
        print(" All data saved as CSV format")
    
    def _save_complete_csv_schema_data(self):
        """Save complete CSV schema data"""
        
        # 1. Training process (epoch level, metrics_train.csv)
        if 'epoch_training_metrics' in self.metrics and self.metrics['epoch_training_metrics']:
            df_train = pd.DataFrame(self.metrics['epoch_training_metrics'])
            df_train.to_csv(os.path.join(self.data_dir, 'metrics_train.csv'), index=False)
            print(f" Training process metrics saved: {len(self.metrics['epoch_training_metrics'])} records")
        
        # 2. Evaluation snapshots (metrics_eval.csv)
        if 'evaluation_snapshots' in self.metrics and self.metrics['evaluation_snapshots']:
            df_eval = pd.DataFrame(self.metrics['evaluation_snapshots'])
            df_eval.to_csv(os.path.join(self.data_dir, 'metrics_eval.csv'), index=False)
            print(f" Evaluation snapshot data saved: {len(self.metrics['evaluation_snapshots'])} records")
        
        # 3. expert events（expert_events.csv）
        if 'expert_events' in self.metrics and self.metrics['expert_events']:
            df_expert = pd.DataFrame(self.metrics['expert_events'])
            df_expert.to_csv(os.path.join(self.data_dir, 'expert_events.csv'), index=False)
            print(f" Expert event data saved: {len(self.metrics['expert_events'])} records")
        
        # 4. Resources/costs (costs.csv)
        if 'resource_costs' in self.metrics and self.metrics['resource_costs']:
            df_costs = pd.DataFrame(self.metrics['resource_costs'])
            df_costs.to_csv(os.path.join(self.data_dir, 'costs.csv'), index=False)
            print(f" Resource cost data saved: {len(self.metrics['resource_costs'])} records")
        
        # 5. Run metadata (run_meta.csv)
        if 'run_metadata' in self.config_data:
            run_meta = self.config_data['run_metadata']
            # Flatten nested dictionary structure
            flat_run_meta = {
                'run_id': run_meta.get('run_id', 'unknown'),
                'seed': run_meta.get('seed', 0),
                'start_time': run_meta.get('start_time', ''),
                'git_commit': run_meta.get('git_commit', ''),
                'gpu_model': run_meta.get('device_info', {}).get('gpu_model', 'N/A'),
                'cuda_version': run_meta.get('device_info', {}).get('cuda_version', 'N/A'),
                'driver_version': run_meta.get('device_info', {}).get('driver_version', 'N/A'),
                'torch_version': run_meta.get('library_versions', {}).get('torch_version', 'N/A'),
                'torch_cuda': run_meta.get('library_versions', {}).get('torch_cuda', 'N/A'),
                'numpy_version': run_meta.get('library_versions', {}).get('numpy_version', 'N/A')
            }
            df_run_meta = pd.DataFrame([flat_run_meta])
            df_run_meta.to_csv(os.path.join(self.data_dir, 'run_meta.csv'), index=False)
            print(" Run metadata saved")
        
        # 6. Dataset information (datasets.csv)
        if 'datasets' in self.config_data and self.config_data['datasets']:
            df_datasets = pd.DataFrame(self.config_data['datasets'])
            df_datasets.to_csv(os.path.join(self.data_dir, 'datasets.csv'), index=False)
            print(f" Dataset information saved: {len(self.config_data['datasets'])} records")
        
        # 7. Training configuration (training_config.csv)
        if 'training_config' in self.config_data:
            df_training_config = pd.DataFrame([self.config_data['training_config']])
            df_training_config.to_csv(os.path.join(self.data_dir, 'training_config.csv'), index=False)
            print(" Training configuration saved")
        
        # 8. FID configuration (fid_config.csv)
        if 'fid_config' in self.config_data:
            df_fid_config = pd.DataFrame([self.config_data['fid_config']])
            df_fid_config.to_csv(os.path.join(self.data_dir, 'fid_config.csv'), index=False)
            print(" FID configuration saved")
        
        # 9. Teacher model retention capability data (teacher_retention_metrics.csv)
        if 'teacher_retention_metrics' in self.metrics and self.metrics['teacher_retention_metrics']:
            df_teacher_retention = pd.DataFrame(self.metrics['teacher_retention_metrics'])
            df_teacher_retention.to_csv(os.path.join(self.data_dir, 'teacher_retention_metrics.csv'), index=False)
            print(f" Teacher model retention capability data saved: {len(self.metrics['teacher_retention_metrics'])} records")
        
        # 10. Teacher old task performance data（teacher_old_task_performance.csv）
        if 'teacher_old_task_performance' in self.metrics and self.metrics['teacher_old_task_performance']:
            df_teacher_old_perf = pd.DataFrame(self.metrics['teacher_old_task_performance'])
            df_teacher_old_perf.to_csv(os.path.join(self.data_dir, 'teacher_old_task_performance.csv'), index=False)
            print(f" Teacher old task performance data saved: {len(self.metrics['teacher_old_task_performance'])} records")
    
    def _save_three_section_data(self):
        """Save specialized data supporting three section analysis"""
        # 1. Save FID trend data by task
        self._save_fid_trend_data()
        
        # 2. Save old task performance data
        self._save_old_task_performance_data()
        
        # 3. Save knowledge distillation stability data
        self._save_kd_stability_data()
        
        print(" Three section specialized data saved")
    
    def _save_expert_analysis_data(self):
       
        # 1. Save expert trigger summary data
        if 'expert_trigger_summary' in self.metrics and self.metrics['expert_trigger_summary']:
            df_trigger = pd.DataFrame(self.metrics['expert_trigger_summary'])
            df_trigger.to_csv(os.path.join(self.data_dir, 'expert_trigger_summary.csv'), index=False)
        
        # 2. Save expert-task correlation data
        if 'expert_task_correlation' in self.metrics and self.metrics['expert_task_correlation']:
            df_correlation = pd.DataFrame(self.metrics['expert_task_correlation'])
            df_correlation.to_csv(os.path.join(self.data_dir, 'expert_task_correlation.csv'), index=False)
        
        # 3. Save expert resource consumption data
        if 'expert_resource_consumption' in self.metrics and self.metrics['expert_resource_consumption']:
            df_resource = pd.DataFrame(self.metrics['expert_resource_consumption'])
            df_resource.to_csv(os.path.join(self.data_dir, 'expert_resource_consumption.csv'), index=False)
        
        print(" Expert analysis data saved")
    
    def _save_fid_trend_data(self):
        """Save FID trend data by task"""
        fid_trend_data = []
        
        # Collect all epoch-level FID data
        for metric_key, metric_list in self.metrics.items():
            if metric_key.startswith('fid_epoch_') and metric_list:
                for entry in metric_list:
                    if 'epoch' in entry and 'fid_score' in entry:
                        fid_trend_data.append({
                            'task_name': entry.get('task_name', ''),
                            'epoch': entry.get('epoch', 0),
                            'model_type': 'Teacher' if 'Teacher' in metric_key else 'Student',
                            'fid_score': entry.get('fid_score', 0),
                            'timestamp': entry.get('timestamp', 0),
                            'datetime': entry.get('datetime', '')
                        })
        
        if fid_trend_data:
            df_fid_trend = pd.DataFrame(fid_trend_data)
            df_fid_trend.to_csv(os.path.join(self.data_dir, 'fid_trend_by_epoch.csv'), index=False)
            print(f" FID trend data saved: {len(fid_trend_data)} records")
    
    def _save_old_task_performance_data(self):
        """Save old task performance data"""
        if 'old_task_performance' in self.metrics and self.metrics['old_task_performance']:
            df_old_task = pd.DataFrame(self.metrics['old_task_performance'])
            df_old_task.to_csv(os.path.join(self.data_dir, 'old_task_performance.csv'), index=False)
            print(f" Old task performance data saved: {len(self.metrics['old_task_performance'])} records")
    
    def _save_kd_stability_data(self):
        """Save knowledge distillation stability data"""
        if 'kd_stability_metrics' in self.metrics and self.metrics['kd_stability_metrics']:
            df_kd_stability = pd.DataFrame(self.metrics['kd_stability_metrics'])
            df_kd_stability.to_csv(os.path.join(self.data_dir, 'kd_stability_metrics.csv'), index=False)
            print(f" KD stability data saved: {len(self.metrics['kd_stability_metrics'])} records")
    
    def _save_teacher_student_fid_comparison(self):
        """Save Teacher-Student FID comparison data to CSV"""
        # Collect all Teacher and Student FID data
        comparison_data = []
        
        for metric_key, metric_list in self.metrics.items():
            if 'fid_' in metric_key and ('_Teacher' in metric_key or '_Student' in metric_key):
                for entry in metric_list:
                    if 'model' in entry and 'fid_score' in entry:
                        comparison_data.append({
                            'task': entry['task'],
                            'model': entry['model'],
                            'fid_score': entry['fid_score'],
                            'epoch': entry.get('epoch', 0),
                            'timestamp': entry.get('timestamp', 0),
                            'datetime': entry.get('datetime', '')
                        })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            df_comparison.to_csv(os.path.join(self.data_dir, 'teacher_student_fid_comparison.csv'), index=False)
            print(f" Teacher-Student FID comparison data saved: {len(comparison_data)} records")

    def log_run_metadata(self, run_id: str, seed: int, git_commit: str = None, 
                         device_info: Dict[str, Any] = None, library_versions: Dict[str, str] = None):
        """
        Log run metadata
        
        Args:
            run_id: Run ID
            seed: Random seed
            git_commit: Git commit hash
            device_info: Device information
            library_versions: Key library versions
        """
        timestamp = time.time() - self.experiment_start_time
        
        # Get system information
        if device_info is None:
            device_info = {}
            if torch.cuda.is_available():
                device_info.update({
                    'gpu_model': torch.cuda.get_device_name(0),
                    'cuda_version': torch.version.cuda,
                    'driver_version': 'N/A',  # Need additional retrieval
                    'gpu_memory_mb': torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                })
        
        if library_versions is None:
            library_versions = {
                'torch_version': torch.__version__,
                'torch_cuda': torch.version.cuda,
                'numpy_version': np.__version__,
                'torchvision_version': 'N/A'  # Need additional retrieval
            }
        
        run_metadata = {
            'run_id': run_id,
            'seed': seed,
            'git_commit': git_commit,
            'start_time': datetime.datetime.now().isoformat(),
            'timestamp': timestamp,
            'device_info': device_info,
            'library_versions': library_versions
        }
        
        # Store in configuration data
        self.config_data['run_metadata'] = run_metadata
        
        print(f" Run metadata logged: {run_id} (seed: {seed})")
    
    def log_dataset_info(self, task_name: str, task_id: int, num_classes: int, 
                        train_samples: int, val_samples: int, image_size: tuple,
                        normalization: str, task_order: int):
        """
        Log dataset information
        
        Args:
            task_name: Task name
            task_id: TaskID
            num_classes: Number of classes
            train_samples: Number of training samples
            val_samples: Number of validation samples
            image_size: Image size
            normalization: Normalization method
            task_order: Task order
        """
        dataset_info = {
            'task_name': task_name,
            'task_id': task_id,
            'num_classes': num_classes,
            'train_samples': train_samples,
            'val_samples': val_samples,
            'image_size': image_size,
            'normalization': normalization,
            'task_order': task_order,
            'timestamp': time.time() - self.experiment_start_time,
            'datetime': datetime.datetime.now().isoformat()
        }
        
        # Store in configuration data
        if 'datasets' not in self.config_data:
            self.config_data['datasets'] = []
        self.config_data['datasets'].append(dataset_info)
        
        print(f" Dataset information logged: {task_name} - {num_classes}classes, {train_samples}training samples")
    
    def log_training_config(self, batch_size: int, learning_rate: float, optimizer: str,
                           scheduler: str, total_epochs: int, eval_interval: int,
                           early_stop: bool = False, checkpoint_strategy: str = "best"):
        """
        Log training configuration
        
        Args:
            batch_size: Batch size
            learning_rate: Learning rate
            optimizer: Optimizer
            scheduler: Scheduler
            total_epochs: Total epochs
            eval_interval: Evaluation interval
            early_stop: Whether to early stop
            checkpoint_strategy: Checkpoint strategy
        """
        training_config = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'total_epochs': total_epochs,
            'eval_interval': eval_interval,
            'early_stop': early_stop,
            'checkpoint_strategy': checkpoint_strategy,
            'timestamp': time.time() - self.experiment_start_time,
            'datetime': datetime.datetime.now().isoformat()
        }
        
        # Store in configuration data
        self.config_data['training_config'] = training_config
        
        print(f" Training config logged: batch_size={batch_size}, lr={learning_rate}, epochs={total_epochs}")
    
    def log_fid_config(self, fid_n: int, inception_version: str, resize_method: str,
                      preprocessing: str, eval_split: str, fid_seed: int = None):
        """
        Log FID configuration
        
        Args:
            fid_n: Sampling count N
            inception_version: Inceptionversion
            resize_method: Resize method
            preprocessing: Preprocessing details
            eval_split: Evaluation split
            fid_seed: Fixed random seed
        """
        if fid_seed is None:
            fid_seed = 42  # Default seed
        
        fid_config = {
            'fid_n': fid_n,
            'inception_version': inception_version,
            'resize_method': resize_method,
            'preprocessing': preprocessing,
            'eval_split': eval_split,
            'fid_seed': fid_seed,
            'timestamp': time.time() - self.experiment_start_time,
            'datetime': datetime.datetime.now().isoformat()
        }
        
        # Store in configuration data
        self.config_data['fid_config'] = fid_config
        
        print(f" FID config logged: N={fid_n}, version={inception_version}, split={eval_split}")
    
    def log_epoch_training_metrics(self, epoch: int, task_id: int, task_name: str,
                                 student_total_loss: float, recon_loss: float, kl_z: float,
                                 kl_u: float, contrastive_loss: float, g_loss: float,
                                 d_loss: float, grad_penalty: float, kd_enabled: bool,
                                 kd_weight: float, kd_pixel: float = None, kd_perceptual: float = None,
                                 kd_feature: float = None, lr_G: float = None, lr_D: float = None,
                                 time_sec: float = None, throughput_img_s: float = None,
                                 peak_mem_MB: float = None):
        """
        Log epoch-level training metrics
        
        Args:
            epoch: Current epoch
            task_id: TaskID
            task_name: Task name
            student_total_loss: Student total loss
            recon_loss: Reconstruction loss
            kl_z: KL divergence Z
            kl_u: KL divergence U
            contrastive_loss: Contrastive learning loss
            g_loss: Generator Loss
            d_loss: Discriminator Loss
            grad_penalty: Gradient penalty
            kd_enabled: Whether knowledge distillation is enabled
            kd_weight: Knowledge distillation weight
            kd_pixel: Pixel-level knowledge distillation loss
            kd_perceptual: Perceptual-level knowledge distillation loss
            kd_feature: Feature-level knowledge distillation loss
            lr_G: Generator learning rate
            lr_D: Discriminator learning rate
            time_sec: Time elapsed (seconds)
            throughput_img_s: Throughput (images/second)
            peak_mem_MB: Peak memory (MB)
        """
        # Log to training process metrics
        training_metrics = {
            'run_id': getattr(self, 'run_id', 'unknown'),
            'seed': getattr(self, 'seed', 0),
            'epoch': epoch,
            'task_id': task_id,
            'task_name': task_name,
            'student_total_loss': student_total_loss,
            'recon_loss': recon_loss,
            'kl_z': kl_z,
            'kl_u': kl_u,
            'contrastive_loss': contrastive_loss,
            'g_loss': g_loss,
            'd_loss': d_loss,
            'grad_penalty': grad_penalty,
            'kd_enabled': 1 if kd_enabled else 0,
            'kd_weight': kd_weight,
            'kd_pixel': kd_pixel,
            'kd_perceptual': kd_perceptual,
            'kd_feature': kd_feature,
            'lr_G': lr_G,
            'lr_D': lr_D,
            'time_sec': time_sec,
            'throughput_img_s': throughput_img_s,
            'peak_mem_MB': peak_mem_MB,
            'timestamp': time.time() - self.experiment_start_time,
            'datetime': datetime.datetime.now().isoformat()
        }
        
        # Store in training metrics
        if 'epoch_training_metrics' not in self.metrics:
            self.metrics['epoch_training_metrics'] = []
        self.metrics['epoch_training_metrics'].append(training_metrics)
        
        # Only output concise training progress information
        batch_num = getattr(self, 'current_batch', 0)
        total_batches = getattr(self, 'total_batches', 313)
        g_loss_str = "Skip" if g_loss is None else f"{g_loss:.3f}"
        kd_status = "1" if kd_enabled else "0"
        
        print(f"Batch {batch_num}/{total_batches} | D_loss={d_loss:.1f} | G_loss={g_loss_str} | KD={kd_status} | Critic={grad_penalty:.1f} | S_VAE={student_total_loss:.3f} | D_lr={lr_D:.1e} | G_lr={lr_G:.1e}")
    
    def log_evaluation_snapshot(self, epoch: int, task_id: int, task_name: str,
                              teacher_fid_curr: float, student_fid_curr: float,
                              student_fid_old_tasks: Dict[str, float] = None,
                              teacher_fid_old_tasks: Dict[str, float] = None,  
                              fid_n: int = None, fid_seed: int = None,
                              eval_split: str = None, inception_variant: str = None,
                              image_norm_spec: str = None):
        """
        Log evaluation snapshot
        Args:
            epoch: Current epoch
            task_id: TaskID
            task_name: Task name
            teacher_fid_curr: Current task Teacher FID
            student_fid_curr: Current task Student FID
            student_fid_old_tasks: Old task Student FID {task_name: fid_score}
            teacher_fid_old_tasks: Old task Teacher FID {task_name: fid_score} 
            fid_n: FID sampling count
            fid_seed: FIDRandom seed
            eval_split: Evaluation set
            inception_variant: Inceptionversion
            image_norm_spec: Image normalization specification
        """
        # Build old task FID columns
        old_task_fids = {}
        if student_fid_old_tasks:
            for i, (old_task_name, old_fid) in enumerate(student_fid_old_tasks.items(), 1):
                old_task_fids[f'Student_FID_old_t{i}'] = old_fid
        
        # Build Teacher old task FID columns
        if teacher_fid_old_tasks:
            for i, (old_task_name, old_fid) in enumerate(teacher_fid_old_tasks.items(), 1):
                old_task_fids[f'Teacher_FID_old_t{i}'] = old_fid
        
        evaluation_snapshot = {
            'run_id': getattr(self, 'run_id', 'unknown'),
            'seed': getattr(self, 'seed', 0),
            'epoch': epoch,
            'task_id': task_id,
            'task_name': task_name,
            'Teacher_FID_curr': teacher_fid_curr,
            'Student_FID_curr': student_fid_curr,
            'fid_N': fid_n,
            'fid_seed': fid_seed,
            'eval_split': eval_split,
            'inception_variant': inception_variant,
            'image_norm_spec': image_norm_spec,
            'timestamp': time.time() - self.experiment_start_time,
            'datetime': datetime.datetime.now().isoformat(),
            **old_task_fids
        }
        
        # Store in evaluation metrics
        if 'evaluation_snapshots' not in self.metrics:
            self.metrics['evaluation_snapshots'] = []
        self.metrics['evaluation_snapshots'].append(evaluation_snapshot)
        
        #  Also log to FID curve data
        if 'fid_curves' not in self.metrics:
            self.metrics['fid_curves'] = {}
        
        # Create FID curve data for each task
        if task_name not in self.metrics['fid_curves']:
            self.metrics['fid_curves'][task_name] = []
        
        # Log current task FID curve points
        fid_curve_point = {
            'epoch': epoch,
            'task_id': task_id,
            'teacher_fid': teacher_fid_curr,
            'student_fid': student_fid_curr,
            'fid_gap': teacher_fid_curr - student_fid_curr,
            'timestamp': time.time() - self.experiment_start_time,
            'datetime': datetime.datetime.now().isoformat()
        }
        self.metrics['fid_curves'][task_name].append(fid_curve_point)
        
        # Silent logging, no console output
    
    def log_expert_event(self, epoch: int, task_id: int, trigger_metric: str, 
                        trigger_value: float, trigger_threshold: float,
                        action: str, active_expert_before: int, active_expert_after: int,
                        fid_before: float = None, fid_after: float = None,
                        checkpoint_path: str = None):
        """
        Log expert events
        
        Args:
            epoch: Current epoch
            task_id: TaskID
            trigger_metric: Trigger metric
            trigger_value: Trigger value
            trigger_threshold: Trigger threshold
            action: Action type
            active_expert_before: Previous active expert ID
            active_expert_after: Next active expert ID
            fid_before: FID before trigger
            fid_after: FID after trigger
            checkpoint_path: Checkpoint path
        """
        expert_event = {
            'run_id': getattr(self, 'run_id', 'unknown'),
            'seed': getattr(self, 'seed', 0),
            'epoch': epoch,
            'task_id': task_id,
            'trigger_metric': trigger_metric,
            'trigger_value': trigger_value,
            'trigger_threshold': trigger_threshold,
            'action': action,
            'active_expert_before': active_expert_before,
            'active_expert_after': active_expert_after,
            'FID_before': fid_before,
            'FID_after': fid_after,
            'checkpoint_path': checkpoint_path,
            'timestamp': time.time() - self.experiment_start_time,
            'datetime': datetime.datetime.now().isoformat()
        }
        
        # Store in expert events metrics
        if 'expert_events' not in self.metrics:
            self.metrics['expert_events'] = []
        self.metrics['expert_events'].append(expert_event)
        
        print(f" Expert event logged: {action} - expert{active_expert_before} -> {active_expert_after}")
    
    def log_resource_cost(self, epoch: int, params_total: int, params_teacher: int,
                         params_student: int, epoch_time_sec: float,
                         throughput_img_s: float, peak_mem_MB: float):
        """
        Log resource/cost metrics
        
        Args:
            epoch: Current epoch
            params_total: Total parameters
            params_teacher: Teacher parameters
            params_student: Student parameters
            epoch_time_sec: epochTime elapsed (seconds)
            throughput_img_s: Throughput (images/second)
            peak_mem_MB: Peak memory (MB)
        """
        resource_cost = {
            'run_id': getattr(self, 'run_id', 'unknown'),
            'seed': getattr(self, 'seed', 0),
            'epoch': epoch,
            'params_total': params_total,
            'params_teacher': params_teacher,
            'params_student': params_student,
            'epoch_time_sec': epoch_time_sec,
            'throughput_img_s': throughput_img_s,
            'peak_mem_MB': peak_mem_MB,
            'timestamp': time.time() - self.experiment_start_time,
            'datetime': datetime.datetime.now().isoformat()
        }
        
        # Store in resource cost metrics
        if 'resource_costs' not in self.metrics:
            self.metrics['resource_costs'] = []
        self.metrics['resource_costs'].append(resource_cost)
        
        # Silent logging, no console output
    
    def set_run_info(self, run_id: str, seed: int):
        """Set run information"""
        self.run_id = run_id
        self.seed = seed
        print(f" Run info set: {run_id} (seed: {seed})")
    
    def set_batch_info(self, current_batch: int, total_batches: int):
        """Set current batch info for training progress display"""
        self.current_batch = current_batch
        self.total_batches = total_batches

    def _plot_expert_resource_consumption(self):
        """Plot expert resource consumption analysis"""
        if 'resource_costs' not in self.metrics or not self.metrics['resource_costs']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Expert Resource Consumption Analysis', fontsize=16)
        
        epochs = [item['epoch'] for item in self.metrics['resource_costs']]
        
        # Parameter count changes
        if 'params_total' in self.metrics['resource_costs'][0]:
            params_total = [item['params_total'] for item in self.metrics['resource_costs']]
            axes[0, 0].plot(epochs, params_total, label='Total Parameters', marker='o')
            axes[0, 0].set_title('Parameter Count Over Time')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Parameters')
            axes[0, 0].grid(True)
        
        # Memory usage
        if 'peak_mem_MB' in self.metrics['resource_costs'][0]:
            peak_mem = [item['peak_mem_MB'] for item in self.metrics['resource_costs']]
            axes[0, 1].plot(epochs, peak_mem, label='Peak Memory', marker='s', color='orange')
            axes[0, 1].set_title('Peak Memory Usage')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Memory (MB)')
            axes[0, 1].grid(True)
        
        # Training time
        if 'epoch_time_sec' in self.metrics['resource_costs'][0]:
            epoch_time = [item['epoch_time_sec'] for item in self.metrics['resource_costs']]
            axes[1, 0].plot(epochs, epoch_time, label='Epoch Time', marker='^', color='green')
            axes[1, 0].set_title('Epoch Training Time')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].grid(True)
        
        # Throughput
        if 'throughput_img_s' in self.metrics['resource_costs'][0]:
            throughput = [item['throughput_img_s'] for item in self.metrics['resource_costs']]
            axes[1, 1].plot(epochs, throughput, label='Throughput', marker='d', color='red')
            axes[1, 1].set_title('Training Throughput')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Images/Second')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'expert_resource_consumption.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_expert_trigger_analysis(self):
        """Plot expert trigger analysis charts"""
        if 'expert_events' not in self.metrics or not self.metrics['expert_events']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Expert Trigger Analysis', fontsize=16)
        
        # 1. Expert trigger timeline
        trigger_epochs = [item['epoch'] for item in self.metrics['expert_events'] if item['action'] == 'add_expert']
        trigger_tasks = [item['task_id'] for item in self.metrics['expert_events'] if item['action'] == 'add_expert']
        
        if trigger_epochs:
            axes[0, 0].scatter(trigger_epochs, trigger_tasks, s=100, alpha=0.7, color='red')
            axes[0, 0].set_title('Expert Addition Timeline')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Task ID')
            axes[0, 0].grid(True)
        
        # 2. FID comparison before and after trigger
        fid_before = [item['FID_before'] for item in self.metrics['expert_events'] if 'FID_before' in item and item['FID_before'] is not None]
        fid_after = [item['FID_after'] for item in self.metrics['expert_events'] if 'FID_after' in item and item['FID_after'] is not None]
        
        if fid_before and fid_after:
            x_pos = range(len(fid_before))
            axes[0, 1].bar([x - 0.2 for x in x_pos], fid_before, width=0.4, label='FID Before', alpha=0.7)
            axes[0, 1].bar([x + 0.2 for x in x_pos], fid_after, width=0.4, label='FID After', alpha=0.7)
            axes[0, 1].set_title('FID Before vs After Expert Addition')
            axes[0, 1].set_xlabel('Expert Addition Event')
            axes[0, 1].set_ylabel('FID Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # 3. Trigger threshold distribution
        trigger_values = [item['trigger_value'] for item in self.metrics['expert_events'] if 'trigger_value' in item]
        trigger_thresholds = [item['trigger_threshold'] for item in self.metrics['expert_events'] if 'trigger_threshold' in item]
        
        if trigger_values and trigger_thresholds:
            axes[1, 0].scatter(trigger_values, trigger_thresholds, alpha=0.7, color='green')
            axes[1, 0].plot([min(trigger_values), max(trigger_values)], [min(trigger_thresholds), max(trigger_thresholds)], 'r--', alpha=0.5)
            axes[1, 0].set_title('Trigger Value vs Threshold')
            axes[1, 0].set_xlabel('Trigger Value')
            axes[1, 0].set_ylabel('Threshold')
            axes[1, 0].grid(True)
        
        # 4. Expert count growth
        if trigger_epochs:
            # Ensure array lengths are consistent
            expert_counts = list(range(1, len(trigger_epochs) + 2))  # Starting from 1, add +1 each time
            if len(expert_counts) > len(trigger_epochs):
                expert_counts = expert_counts[:len(trigger_epochs)]
            elif len(expert_counts) < len(trigger_epochs):
                trigger_epochs = trigger_epochs[:len(expert_counts)]
            
            if len(trigger_epochs) == len(expert_counts) and len(trigger_epochs) > 0:
                axes[1, 1].step(trigger_epochs, expert_counts, where='post', marker='o', linewidth=2, markersize=8)
                axes[1, 1].set_title('Expert Count Growth')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Number of Experts')
                axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'expert_trigger_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_expert_task_correlation(self):
        """Plot expert-task correlation analysis (Section 5.3.2)"""
        # Check if relevant data exists
        task_expert_mapping = self.metrics.get('task_expert_mapping', [])
        expert_events = self.metrics.get('expert_events', [])
        teacher_expert_analysis = self.metrics.get('teacher_expert_analysis', [])
        
        if not task_expert_mapping and not expert_events and not teacher_expert_analysis:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Expert-Task Correlation Analysis', fontsize=16)
        
        # 1. Task-expert mapping heatmap (if data exists)
        if task_expert_mapping:
            task_ids = [item['task_id'] for item in task_expert_mapping]
            expert_ids = [item['expert_id'] for item in task_expert_mapping]
            
            if task_ids and expert_ids:
                # Create simple mapping matrix
                max_task = max(task_ids) if task_ids else 0
                max_expert = max(expert_ids) if expert_ids else 0
                
                if max_task > 0 and max_expert > 0:
                    mapping_matrix = np.zeros((max_task + 1, max_expert + 1))
                    for item in task_expert_mapping:
                        mapping_matrix[item['task_id'], item['expert_id']] = 1
                    
                    im = axes[0, 0].imshow(mapping_matrix, cmap='Blues', aspect='auto')
                    axes[0, 0].set_title('Task-Expert Mapping Matrix')
                    axes[0, 0].set_xlabel('Expert ID')
                    axes[0, 0].set_ylabel('Task ID')
                    plt.colorbar(im, ax=axes[0, 0])
        else:
            # If no mapping data, show expert events statistics
            if expert_events:
                event_types = [item['action'] for item in expert_events]
                event_counts = {}
                for event_type in event_types:
                    event_counts[event_type] = event_counts.get(event_type, 0) + 1
                
                if event_counts:
                    axes[0, 0].pie(event_counts.values(), labels=event_counts.keys(), autopct='%1.1f%%', startangle=90)
                    axes[0, 0].set_title('Expert Event Types')
        
        # 2. Expert performance distribution (using teacher_expert_analysis data)
        if teacher_expert_analysis:
            expert_performances = {}
            for item in teacher_expert_analysis:
                if 'expert_performance' in item and isinstance(item['expert_performance'], dict):
                    for expert_id, performance in item['expert_performance'].items():
                        if expert_id not in expert_performances:
                            expert_performances[expert_id] = []
                        expert_performances[expert_id].append(performance)
            
            if expert_performances:
                expert_ids = list(expert_performances.keys())
                avg_performances = [np.mean(perfs) for perfs in expert_performances.values()]
                axes[0, 1].bar(expert_ids, avg_performances, alpha=0.7, color='orange')
                axes[0, 1].set_title('Average Expert Performance')
                axes[0, 1].set_xlabel('Expert ID')
                axes[0, 1].set_ylabel('Average Performance')
                axes[0, 1].grid(True)
        
        # 3. Relationship between task complexity and number of experts
        if expert_events:
            task_ids = [item['task_id'] for item in expert_events]
            expert_counts = [item['active_expert_after'] for item in expert_events]
            
            if task_ids and expert_counts:
                axes[1, 0].scatter(task_ids, expert_counts, alpha=0.7, color='green')
                axes[1, 0].set_title('Task Complexity vs Expert Count')
                axes[1, 0].set_xlabel('Task ID (Complexity)')
                axes[1, 0].set_ylabel('Expert Count')
                axes[1, 0].grid(True)
        
        # 4. Expert utilization rate
        if teacher_expert_analysis:
            expert_utilizations = {}
            for item in teacher_expert_analysis:
                if 'expert_utilization_rate' in item and isinstance(item['expert_utilization_rate'], dict):
                    for expert_id, utilization in item['expert_utilization_rate'].items():
                        if expert_id not in expert_utilizations:
                            expert_utilizations[expert_id] = []
                        expert_utilizations[expert_id].append(utilization)
            
            if expert_utilizations:
                expert_ids = list(expert_utilizations.keys())
                avg_utilizations = [np.mean(utils) for utils in expert_utilizations.values()]
                axes[1, 1].pie(avg_utilizations, labels=[f'Expert {eid}' for eid in expert_ids], 
                              autopct='%1.1f%%', startangle=90)
                axes[1, 1].set_title('Expert Utilization Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'expert_task_correlation.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_teacher_retention_analysis(self):
        """Plot teacher model retention capability analysis"""
        if 'teacher_retention_metrics' not in self.metrics or not self.metrics['teacher_retention_metrics']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Teacher Model Retention Analysis', fontsize=16)
        
        retention_data = self.metrics['teacher_retention_metrics']
        
        # Group by task
        task_groups = defaultdict(list)
        for entry in retention_data:
            task_groups[entry['task_name']].append(entry)
        
        # 1. Teacher FID change
        for task_name, entries in task_groups.items():
            epochs = [entry['epoch'] for entry in entries]
            teacher_fids = [entry['teacher_fid_current'] for entry in entries]
            axes[0, 0].plot(epochs, teacher_fids, 'o-', label=task_name, linewidth=2, markersize=6)
        
        axes[0, 0].set_title('Teacher FID Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Teacher FID')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Stability score changes
        for task_name, entries in task_groups.items():
            epochs = [entry['epoch'] for entry in entries]
            stability_scores = [entry['teacher_stability_score'] for entry in entries]
            axes[0, 1].plot(epochs, stability_scores, 's-', label=task_name, linewidth=2, markersize=6)
        
        axes[0, 1].set_title('Teacher Stability Score Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Stability Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Old task FID change (take first old task)
        for task_name, entries in task_groups.items():
            epochs = [entry['epoch'] for entry in entries]
            old_task_fids = []
            for entry in entries:
                if entry['teacher_fid_old_tasks']:
                    # Take FID of first old task
                    first_old_task_fid = list(entry['teacher_fid_old_tasks'].values())[0]
                    old_task_fids.append(first_old_task_fid)
                else:
                    old_task_fids.append(0)  # If no old task data, set to 0
            
            if any(fid > 0 for fid in old_task_fids):  # Only plot when valid data exists
                axes[1, 0].plot(epochs, old_task_fids, '^-', label=f"{task_name} (Old Task)", linewidth=2, markersize=6)
        
        axes[1, 0].set_title('Teacher Old Task FID Over Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Old Task FID')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Expert utilization rate changes (take first expert utilization)
        for task_name, entries in task_groups.items():
            epochs = [entry['epoch'] for entry in entries]
            expert_utilizations = []
            for entry in entries:
                if entry['expert_utilization']:
                    # Take first expert utilization
                    first_expert_util = list(entry['expert_utilization'].values())[0]
                    expert_utilizations.append(first_expert_util)
                else:
                    expert_utilizations.append(0)  # If no expert data, set to 0
            
            if any(util > 0 for util in expert_utilizations):  # Only plot when valid data exists
                axes[1, 1].plot(epochs, expert_utilizations, 'd-', label=f"{task_name} (Expert 1)", linewidth=2, markersize=6)
        
        axes[1, 1].set_title('Expert Utilization Over Time')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Expert Utilization')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'teacher_retention_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(" Teacher model retention capability analysis chart generated")

    def log_fid_scores(self, fid_scores: Dict[str, float]):
        """Log FID scores"""
        if 'fid_scores' not in self.metrics:
            self.metrics['fid_scores'] = []
        
        timestamp = time.time() - self.experiment_start_time
        
        for task_name, fid_score in fid_scores.items():
            fid_entry = {
                'task_name': task_name,
                'fid_score': fid_score,
                'timestamp': timestamp,
                'datetime': datetime.datetime.now().isoformat()
            }
            self.metrics['fid_scores'].append(fid_entry)
        
        # Save FID scores
        self._save_fid_scores()
        
        # Output FID scores
        for task_name, fid_score in fid_scores.items():
            print(f" FID logged: {task_name} = {fid_score:.2f}")

    def _save_fid_scores(self):
        """Save FID score data"""
        try:
            fid_file = os.path.join(self.data_dir, "fid_scores.csv")
            
            if self.metrics['fid_scores']:
                df = pd.DataFrame(self.metrics['fid_scores'])
                df.to_csv(fid_file, index=False, encoding='utf-8')
                
                # Also save as JSON format
                json_file = os.path.join(self.data_dir, "fid_scores.json")
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(self.metrics['fid_scores'], f, indent=2, ensure_ascii=False, default=str)
                    
        except Exception as e:
            print(f" Failed to save FID score data: {e}")

    def log_key_event(self, epoch: int, task_id: int, task_name: str, event_type: str, 
                      event_details: dict, **kwargs):
        """
        Log key events
        
        Args:
            epoch: Current epoch
            task_id: TaskID
            task_name: Task name
            event_type: Event type ('expert_expansion', 'expert_switch', 'kd_gating_on', 'kd_gating_off')
            event_details: Event details
            **kwargs: Other parameters
        """
        timestamp = time.time() - self.experiment_start_time
        
        key_event = {
            'epoch': epoch,
            'task_id': task_id,
            'task_name': task_name,
            'event_type': event_type,
            'event_details': event_details,
            'timestamp': timestamp,
            'datetime': datetime.datetime.now().isoformat(),
            **kwargs
        }
        
        # Store in key event metrics
        if 'key_events' not in self.metrics:
            self.metrics['key_events'] = []
        self.metrics['key_events'].append(key_event)
        
        # Also log to FID curve data for annotation
        if 'fid_curves' in self.metrics and task_name in self.metrics['fid_curves']:
            # Add event markers to corresponding FID curve data points
            for curve_point in self.metrics['fid_curves'][task_name]:
                if curve_point['epoch'] == epoch:
                    curve_point['key_event'] = key_event
                    break
        
        print(f" Key event logged: {event_type} - {task_name} (Epoch {epoch})")
    
    def log_expert_expansion_event(self, epoch: int, task_id: int, task_name: str, 
                                  old_expert_count: int, new_expert_id: int, 
                                  trigger_fid: float, **kwargs):
        """
        Log expert expansion event
        
        Args:
            epoch: Current epoch
            task_id: TaskID
            task_name: Task name
            old_expert_count: Number of experts before expansion
            new_expert_id: New expert ID
            trigger_fid: Trigger FID value
            **kwargs: Other parameters, including:
                - fid_before: FID before trigger
                - fid_after: FID after trigger (obtained at first evaluation point after trigger)
                - trigger_threshold: Trigger thresholdτ
                - kd_gating_status: KD gating status（on/off）
                - stability_score: Stability score
                - trigger_reason: Trigger reason
                - iteration: Current iteration count
        """
        # Get key parameters
        fid_before = kwargs.get('fid_before', None)
        fid_after = kwargs.get('fid_after', None)
        trigger_threshold = kwargs.get('trigger_threshold', None)
        kd_gating_status = kwargs.get('kd_gating_status', 'unknown')
        stability_score = kwargs.get('stability_score', None)
        trigger_reason = kwargs.get('trigger_reason', 'FID threshold exceeded')
        iteration = kwargs.get('iteration', 0)
        
        # Build complete expert expansion event record
        expansion_event = {
            'run_id': getattr(self, 'run_id', 'unknown'),
            'seed': getattr(self, 'seed', 0),
            'task_id': task_id,
            'task_name': task_name,
            'epoch': epoch,
            'iteration': iteration,
            'old_expert_count': old_expert_count,
            'new_expert_id': new_expert_id,
            'trigger_fid': trigger_fid,
            'FID_before': fid_before,
            'FID_after': fid_after,
            'trigger_threshold': trigger_threshold,
            'kd_gating_status': kd_gating_status,
            'stability_score': stability_score,
            'trigger_reason': trigger_reason,
            'action': 'expert_expansion',
            'timestamp': time.time() - self.experiment_start_time,
            'datetime': datetime.datetime.now().isoformat()
        }
        
        # Store in expert expansion event metrics
        if 'expert_expansion_events' not in self.metrics:
            self.metrics['expert_expansion_events'] = []
        self.metrics['expert_expansion_events'].append(expansion_event)
        
        # Also log to key events
        event_details = {
            'old_expert_count': old_expert_count,
            'new_expert_id': new_expert_id,
            'trigger_fid': trigger_fid,
            'FID_before': fid_before,
            'FID_after': fid_after,
            'trigger_threshold': trigger_threshold,
            'kd_gating_status': kd_gating_status,
            'stability_score': stability_score,
            'trigger_reason': trigger_reason,
            'action': 'add_expert'
        }
        self.log_key_event(epoch, task_id, task_name, 'expert_expansion', event_details, **kwargs)
        
        print(f" Expert expansion event logged: Task{task_id}({task_name}) Epoch{epoch} - expert{old_expert_count}->{old_expert_count+1}")
        if trigger_threshold:
            print(f"   Trigger threshold: {trigger_threshold:.4f}, Current FID: {trigger_fid:.4f}")
        if kd_gating_status != 'unknown':
            print(f"   KD gating status: {kd_gating_status}")
        if stability_score is not None:
            print(f"   Stability score: {stability_score:.4f}")
    
    def log_expert_expansion_trigger_log(self, task_id: int, task_name: str, epoch: int, 
                                       iteration: int, trigger_fid: float, trigger_threshold: float,
                                       fid_before: float, fid_after: float = None,
                                       kd_gating_status: str = 'unknown', stability_score: float = None,
                                       trigger_reason: str = "FID threshold exceeded", **kwargs):
        """
        Log expert expansion trigger log
        
        Args:
            task_id: TaskID
            task_name: Task name
            epoch: Current epoch
            iteration: Current iteration count
            trigger_fid: Trigger FID value
            trigger_threshold: Trigger thresholdτ
            fid_before: FID before trigger
            fid_after: FID after trigger (obtained at first evaluation point after trigger)
            kd_gating_status: KD gating status（on/off）
            stability_score: Stability score
            trigger_reason: Trigger reason
            **kwargs: Other parameters
        """
        # Build complete trigger log record
        trigger_log_entry = {
            'run_id': getattr(self, 'run_id', 'unknown'),
            'seed': getattr(self, 'seed', 0),
            'task_id': task_id,
            'task_name': task_name,
            'epoch': epoch,
            'iteration': iteration,
            'trigger_fid': trigger_fid,
            'trigger_threshold': trigger_threshold,
            'FID_before': fid_before,
            'FID_after': fid_after,
            'kd_gating_status': kd_gating_status,
            'stability_score': stability_score,
            'trigger_reason': trigger_reason,
            'timestamp': time.time() - self.experiment_start_time,
            'datetime': datetime.datetime.now().isoformat(),
            **kwargs
        }
        
        # Store in expert expansion trigger log
        if 'expert_expansion_trigger_logs' not in self.metrics:
            self.metrics['expert_expansion_trigger_logs'] = []
        self.metrics['expert_expansion_trigger_logs'].append(trigger_log_entry)
        
        # Output trigger log information
        print(f" Expert expansion trigger log: Task{task_id}({task_name}) Epoch{epoch} Iter{iteration}")
        print(f"   Trigger FID: {trigger_fid:.4f}, threshold: {trigger_threshold:.4f}")
        print(f"   FID change: {fid_before:.4f} -> {fid_after if fid_after else 'pending evaluation'}")
        if stability_score is not None:
            print(f"   KD status: {kd_gating_status}, stability: {stability_score:.4f}")
        else:
            print(f"   KD status: {kd_gating_status}, stability: N/A")
        print(f"   Trigger reason: {trigger_reason}")
    
    def log_expert_switch_event(self, epoch: int, task_id: int, task_name: str,
                                old_expert_id: int, new_expert_id: int, 
                                switch_reason: str, **kwargs):
        """Log expert switch events"""
        event_details = {
            'old_expert_id': old_expert_id,
            'new_expert_id': new_expert_id,
            'switch_reason': switch_reason,
            'action': 'switch_expert'
        }
        self.log_key_event(epoch, task_id, task_name, 'expert_switch', event_details, **kwargs)
    
    def log_kd_gating_event(self, epoch: int, task_id: int, task_name: str, 
                           gating_status: str, kd_weight: float, 
                           teacher_student_similarity: float, **kwargs):
        """Log KD gating events"""
        event_details = {
            'gating_status': gating_status,  # 'on' or 'off'
            'kd_weight': kd_weight,
            'teacher_student_similarity': teacher_student_similarity,
            'action': f'kd_gating_{gating_status}'
        }
        self.log_key_event(epoch, task_id, task_name, f'kd_gating_{gating_status}', event_details, **kwargs)

    def _save_key_events(self):
        """Save key event data"""
        if 'key_events' in self.metrics and self.metrics['key_events']:
            # Save as JSON format
            events_file = os.path.join(self.data_dir, 'key_events.json')
            with open(events_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics['key_events'], f, indent=2, default=str)
            
            # Save as CSV format
            events_csv_file = os.path.join(self.data_dir, 'key_events.csv')
            df_events = pd.DataFrame(self.metrics['key_events'])
            df_events.to_csv(events_csv_file, index=False, encoding='utf-8')
            
            print(f" Key event data saved: {len(self.metrics['key_events'])} events")
            
            # Statistics by event type
            event_counts = {}
            for event in self.metrics['key_events']:
                event_type = event['event_type']
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            print(" Event type statistics:")
            for event_type, count in event_counts.items():
                print(f"    {event_type}: {count} items")

    def generate_fid_curves_with_events(self, save_path: str = None):
        """
        Generate FID curve chart with key event annotations
        
        Args:
            save_path: Save path, use default path if None
        """
        if 'fid_curves' not in self.metrics or not self.metrics['fid_curves']:
            print(" No FID curve data to plot")
            return
        
        if save_path is None:
            save_path = os.path.join(self.plots_dir, 'fid_curves_with_events.png')
        
        # Set Chinese font
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Create subplots
        num_tasks = len(self.metrics['fid_curves'])
        fig, axes = plt.subplots(num_tasks, 1, figsize=(12, 6 * num_tasks))
        if num_tasks == 1:
            axes = [axes]
        
        # Define event type colors and markers
        event_colors = {
            'expert_expansion': 'red',
            'expert_switch': 'orange', 
            'kd_gating_on': 'green',
            'kd_gating_off': 'purple'
        }
        event_markers = {
            'expert_expansion': '^',
            'expert_switch': 's',
            'kd_gating_on': 'o',
            'kd_gating_off': 'x'
        }
        
        for i, (task_name, curves) in enumerate(self.metrics['fid_curves'].items()):
            ax = axes[i]
            
            if not curves:
                continue
            
            # Extract data
            epochs = [curve['epoch'] for curve in curves]
            teacher_fids = [curve['teacher_fid'] for curve in curves]
            student_fids = [curve['student_fid'] for curve in curves]
            
            # Plot FID curves
            ax.plot(epochs, teacher_fids, 'b-', label='Teacher FID', linewidth=2, marker='o', markersize=4)
            ax.plot(epochs, student_fids, 'r-', label='Student FID', linewidth=2, marker='s', markersize=4)
            
            # Annotate key events
            for curve in curves:
                if 'key_event' in curve:
                    event = curve['key_event']
                    event_type = event['event_type']
                    epoch = curve['epoch']
                    
                    if event_type in event_colors:
                        color = event_colors[event_type]
                        marker = event_markers[event_type]
                        
                        # Annotate events on Student FID curve
                        student_fid = curve['student_fid']
                        ax.scatter(epoch, student_fid, c=color, marker=marker, s=100, 
                                 label=f'{event_type.replace("_", " ").title()}', zorder=5)
                        
                        # Add event labels
                        ax.annotate(f'{event_type.replace("_", " ").title()}', 
                                   xy=(epoch, student_fid), xytext=(10, 10),
                                   textcoords='offset points', fontsize=8, 
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('FID Score')
            ax.set_title(f'{task_name} - FID change curve (with key event annotations)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Set y-axis range to avoid curve being unclear when FID values are too large
            max_fid = max(max(teacher_fids), max(student_fids))
            if max_fid > 100:
                ax.set_ylim(0, min(max_fid * 1.1, 200))  # Limit y-axis range
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" FID curve chart with key event annotations saved: {save_path}")
        
        # Also save event statistics
        if 'key_events' in self.metrics and self.metrics['key_events']:
            events_summary = {}
            for event in self.metrics['key_events']:
                task_name = event['task_name']
                event_type = event['event_type']
                if task_name not in events_summary:
                    events_summary[task_name] = {}
                if event_type not in events_summary[task_name]:
                    events_summary[task_name][event_type] = []
                events_summary[task_name][event_type].append(event)
            
            # Save event statistics
            events_file = os.path.join(self.data_dir, 'key_events_summary.json')
            with open(events_file, 'w', encoding='utf-8') as f:
                json.dump(events_summary, f, indent=2, default=str)
            
            print(f" Key event statistics saved: {events_file}")

    def log_task_final_fid(self, task_id: int, task_name: str, 
                           teacher_fid_end: float, student_fid_end: float,
                              **kwargs):
        """
        Log FID_end values when task completes
        
        Args:
            task_id: TaskID
            task_name: Task name
            teacher_fid_end: Teacher FID value when task completes
            student_fid_end: Student FID value when task completes
            **kwargs: Other parameters
        """
        timestamp = time.time() - self.experiment_start_time
        
        task_final_fid = {
            'task_id': task_id,
            'task_name': task_name,
            'teacher_fid_end': teacher_fid_end,
            'student_fid_end': student_fid_end,
            'fid_gap_end': teacher_fid_end - student_fid_end,
            'completion_time': datetime.datetime.now().isoformat(),
            'timestamp': timestamp,
            **kwargs
        }
        
        # Store in task final FID metrics
        if 'task_final_fids' not in self.metrics:
            self.metrics['task_final_fids'] = []
        self.metrics['task_final_fids'].append(task_final_fid)
        
        print(f" Task {task_name} Completion FID logged: Teacher={teacher_fid_end:.2f}, Student={student_fid_end:.2f}")
    
    def calculate_forgetting_metrics(self):
        """
        Calculate forgetting metrics (based on FID_end(j) and subsequent evaluation FID values)
        """
        if 'task_final_fids' not in self.metrics or not self.metrics['task_final_fids']:
            print(" No task completion FID data, cannot calculate forgetting")
            return
        
        if 'evaluation_snapshots' not in self.metrics or not self.metrics['evaluation_snapshots']:
            print(" No evaluation snapshot data, cannot calculate forgetting")
            return
        
        # Organize FID_end data by task
        task_fid_ends = {}
        for task_fid in self.metrics['task_final_fids']:
            task_name = task_fid['task_name']
            task_fid_ends[task_name] = {
                'teacher_fid_end': task_fid['teacher_fid_end'],
                'student_fid_end': task_fid['student_fid_end']
            }
        
        # Calculate forgetting
        forgetting_metrics = {}
        for snapshot in self.metrics['evaluation_snapshots']:
            current_task = snapshot['task_name']
            current_task_id = snapshot['task_id']
            
            # Only process snapshots with old task evaluations
            if 'Student_FID_old_t1' in snapshot:
                for i in range(1, current_task_id + 1):
                    old_task_key = f'Student_FID_old_t{i}'
                    teacher_old_task_key = f'Teacher_FID_old_t{i}'
                    
                    if old_task_key in snapshot:
                        # Get old task name
                        old_task_name = self._get_old_task_name(i, current_task_id)
                        if old_task_name and old_task_name in task_fid_ends:
                            # Calculate Student forgetting
                            student_fid_end = task_fid_ends[old_task_name]['student_fid_end']
                            student_fid_current = snapshot[old_task_key]
                            student_forgetting = student_fid_current - student_fid_end
                            
                            # Calculate Teacher forgetting
                            teacher_fid_end = task_fid_ends[old_task_name]['teacher_fid_end']
                            teacher_fid_current = snapshot.get(teacher_old_task_key, teacher_fid_end)
                            teacher_forgetting = teacher_fid_current - teacher_fid_end
                            
                            # Record forgetting metrics
                            if old_task_name not in forgetting_metrics:
                                forgetting_metrics[old_task_name] = {}
                            
                            forgetting_metrics[old_task_name][f'task_{current_task_id}'] = {
                                'student_fid_end': student_fid_end,
                                'student_fid_current': student_fid_current,
                                'student_forgetting': student_forgetting,
                                'teacher_fid_end': teacher_fid_end,
                                'teacher_fid_current': teacher_fid_current,
                                'teacher_forgetting': teacher_forgetting,
                                'evaluation_epoch': snapshot['epoch'],
                                'evaluation_task': current_task
                            }
        
        # Save forgetting metrics
        self.metrics['forgetting_metrics'] = forgetting_metrics
        
        print(f" Forgetting metrics calculation completed: {len(forgetting_metrics)} old tasks")
        return forgetting_metrics
    
    def _get_old_task_name(self, old_task_index: int, current_task_id: int) -> str:
        """Get task name by old task index"""
        # First try to get task sequence from config
        if hasattr(self, 'experiment_config') and hasattr(self.experiment_config, 'task_sequence'):
            if old_task_index < len(self.experiment_config.task_sequence):
                return self.experiment_config.task_sequence[old_task_index]
        
        # If no config, use default task name
        default_tasks = ['MNIST', 'Fashion-MNIST', 'CIFAR-10', 'SVHN', 'STL10']
        if old_task_index < len(default_tasks):
            return default_tasks[old_task_index]
        
        # If none available, return generic name
        return f'Task_{old_task_index}'

    def _save_task_final_fids(self):
        """Save task final FID data"""
        if 'task_final_fids' in self.metrics and self.metrics['task_final_fids']:
            # Save as JSON format
            final_fids_file = os.path.join(self.data_dir, 'task_final_fids.json')
            with open(final_fids_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics['task_final_fids'], f, indent=2, default=str)
            
            # Save as CSV format
            final_fids_csv_file = os.path.join(self.data_dir, 'task_final_fids.csv')
            df_final_fids = pd.DataFrame(self.metrics['task_final_fids'])
            df_final_fids.to_csv(final_fids_csv_file, index=False, encoding='utf-8')
            
            print(f" Task final FID data saved: {len(self.metrics['task_final_fids'])} tasks")
            
            # Show FID_end statistics
            if self.metrics['task_final_fids']:
                teacher_fids = [task['teacher_fid_end'] for task in self.metrics['task_final_fids']]
                student_fids = [task['student_fid_end'] for task in self.metrics['task_final_fids']]
                
                print(" FID_end statistics:")
                print(f"   Teacher FID_end: Average={np.mean(teacher_fids):.2f}, Min={np.min(teacher_fids):.2f}, Max={np.max(teacher_fids):.2f}")
                print(f"   Student FID_end: Average={np.mean(student_fids):.2f}, Min={np.min(student_fids):.2f}, Max={np.max(student_fids):.2f}")

    def calculate_retention_rates(self):
        """
        Calculate old task FID retention rates at each evaluation point
        Retention rate = FID_end(j) / FID_current, values closer to 1 indicate better retention
        """
        if 'task_final_fids' not in self.metrics or not self.metrics['task_final_fids']:
            print(" No task completion FID data, cannot calculate retention rates")
            return
        
        if 'evaluation_snapshots' not in self.metrics or not self.metrics['evaluation_snapshots']:
            print(" No evaluation snapshot data, cannot calculate retention rates")
            return
        
        # Organize FID_end data by task
        task_fid_ends = {}
        for task_fid in self.metrics['task_final_fids']:
            task_name = task_fid['task_name']
            task_fid_ends[task_name] = {
                'teacher_fid_end': task_fid['teacher_fid_end'],
                'student_fid_end': task_fid['student_fid_end']
            }
        
        # Calculate retention rates
        retention_rates = []
        
        for snapshot in self.metrics['evaluation_snapshots']:
            current_task = snapshot['task_name']
            current_task_id = snapshot['task_id']
            epoch = snapshot['epoch']
            
            # Only process snapshots with old task evaluations
            if 'Student_FID_old_t1' in snapshot:
                for i in range(1, current_task_id + 1):
                    old_task_key = f'Student_FID_old_t{i}'
                    teacher_old_task_key = f'Teacher_FID_old_t{i}'
                    
                    if old_task_key in snapshot:
                        # Get old task name
                        old_task_name = self._get_old_task_name(i, current_task_id)
                        
                        # If old task name not in task_fid_ends, try direct index matching
                        if old_task_name not in task_fid_ends and i <= len(task_fid_ends):
                            # Use index to directly get task name
                            available_task_names = list(task_fid_ends.keys())
                            old_task_name = available_task_names[i-1] if i-1 < len(available_task_names) else None
                        
                        if old_task_name and old_task_name in task_fid_ends:
                            # Calculate Student retention rate
                            student_fid_end = task_fid_ends[old_task_name]['student_fid_end']
                            student_fid_current = snapshot[old_task_key]
                            
                            # Avoid division by zero error
                            if student_fid_end > 0:
                                student_retention_rate = student_fid_end / student_fid_current
                                student_retention_percentage = (student_fid_end / student_fid_current) * 100
                            else:
                                student_retention_rate = 0.0
                                student_retention_percentage = 0.0
                            
                            # Calculate Teacher retention rate
                            teacher_fid_end = task_fid_ends[old_task_name]['teacher_fid_end']
                            teacher_fid_current = snapshot.get(teacher_old_task_key, teacher_fid_end)
                            
                            if teacher_fid_end > 0:
                                teacher_retention_rate = teacher_fid_end / teacher_fid_current
                                teacher_retention_percentage = (teacher_fid_end / teacher_fid_current) * 100
                            else:
                                teacher_retention_rate = 0.0
                                teacher_retention_percentage = 0.0
                            
                            # Record retention rate data
                            retention_data = {
                                'evaluation_epoch': epoch,
                                'evaluation_task': current_task,
                                'evaluation_task_id': current_task_id,
                                'old_task_name': old_task_name,
                                'old_task_index': i,
                                
                                # Student retention rate
                                'student_fid_end': student_fid_end,
                                'student_fid_current': student_fid_current,
                                'student_retention_rate': student_retention_rate,
                                'student_retention_percentage': student_retention_percentage,
                                'student_forgetting_rate': 1.0 - student_retention_rate,
                                
                                # Teacher retention rate
                                'teacher_fid_end': teacher_fid_end,
                                'teacher_fid_current': teacher_fid_current,
                                'teacher_retention_rate': teacher_retention_rate,
                                'teacher_retention_percentage': teacher_retention_percentage,
                                'teacher_forgetting_rate': 1.0 - teacher_retention_rate,
                                
                                # Retention rate difference
                                'retention_rate_gap': teacher_retention_rate - student_retention_rate,
                                'retention_percentage_gap': teacher_retention_percentage - student_retention_percentage,
                                
                                # Timestamp
                                'timestamp': time.time() - self.experiment_start_time
                            }
                            
                            retention_rates.append(retention_data)
        
        # Save retention rate metrics
        self.metrics['retention_rates'] = retention_rates
        
        print(f" Retention rate metrics calculation completed: {len(retention_rates)} evaluation points")
        
        # Show retention rate statistics
        if retention_rates:
            student_rates = [r['student_retention_percentage'] for r in retention_rates]
            teacher_rates = [r['teacher_retention_percentage'] for r in retention_rates]
            
            print(" Retention rate statistics:")
            print(f"   Student average retention rate: {np.mean(student_rates):.2f}% (Range: {np.min(student_rates):.2f}% - {np.max(student_rates):.2f}%)")
            print(f"   Teacher average retention rate: {np.mean(teacher_rates):.2f}% (Range: {np.min(teacher_rates):.2f}% - {np.max(teacher_rates):.2f}%)")
            print(f"   Retention rate difference: Teacher vs Student average higher by {np.mean([r['retention_percentage_gap'] for r in retention_rates]):.2f}%")
        
        return retention_rates

    def _save_retention_rates(self):
        """Save retention rate data（Supporting Section 5.2.1 forgetting analysis）"""
        if 'retention_rates' in self.metrics and self.metrics['retention_rates']:
            # Save as JSON format
            retention_file = os.path.join(self.data_dir, 'retention_rates.json')
            with open(retention_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics['retention_rates'], f, indent=2, default=str)
            
            # Save as CSV format
            retention_csv_file = os.path.join(self.data_dir, 'retention_rates.csv')
            df_retention = pd.DataFrame(self.metrics['retention_rates'])
            df_retention.to_csv(retention_csv_file, index=False, encoding='utf-8')
            
            print(f" Retention rate data saved: {len(self.metrics['retention_rates'])} evaluation points")
            
            # Show retention rate analysis summary
            if self.metrics['retention_rates']:
                # Group analysis by old task
                old_task_analysis = {}
                for rate in self.metrics['retention_rates']:
                    old_task = rate['old_task_name']
                    if old_task not in old_task_analysis:
                        old_task_analysis[old_task] = []
                    old_task_analysis[old_task].append(rate)
                
                print(" Retention rate analysis for each old task:")
                for old_task, rates in old_task_analysis.items():
                    student_avg = np.mean([r['student_retention_percentage'] for r in rates])
                    teacher_avg = np.mean([r['teacher_retention_percentage'] for r in rates])
                    print(f"   {old_task}: Student={student_avg:.2f}%, Teacher={teacher_avg:.2f}%")

    def generate_retention_analysis_plots(self, save_path: str = None):
        """
        Generate retention rate analysis chart (Supporting Section 5.2.1 forgetting analysis)
        """
        if 'retention_rates' not in self.metrics or not self.metrics['retention_rates']:
            print(" No retention rate data, cannot generate chart")
            return
        
        retention_data = self.metrics['retention_rates']
        
        # Create chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Retention Rate Analysis (Based on FID_end(j))', fontsize=16)
        
        # 1. Retention rate trend over time
        ax1 = axes[0, 0]
        epochs = [r['evaluation_epoch'] for r in retention_data]
        student_rates = [r['student_retention_percentage'] for r in retention_data]
        teacher_rates = [r['teacher_retention_percentage'] for r in retention_data]
        
        ax1.plot(epochs, student_rates, 'o-', label='Student', alpha=0.7, markersize=4)
        ax1.plot(epochs, teacher_rates, 's-', label='Teacher', alpha=0.7, markersize=4)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Retention Rate (%)')
        ax1.set_title('Retention Rate Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Retention rate comparison for each old task
        ax2 = axes[0, 1]
        old_tasks = list(set([r['old_task_name'] for r in retention_data]))
        old_tasks.sort()
        
        student_avg_rates = []
        teacher_avg_rates = []
        for old_task in old_tasks:
            task_rates = [r for r in retention_data if r['old_task_name'] == old_task]
            student_avg = np.mean([r['student_retention_percentage'] for r in task_rates])
            teacher_avg = np.mean([r['teacher_retention_percentage'] for r in task_rates])
            student_avg_rates.append(student_avg)
            teacher_avg_rates.append(teacher_avg)
        
        x = np.arange(len(old_tasks))
        width = 0.35
        
        ax2.bar(x - width/2, student_avg_rates, width, label='Student', alpha=0.7)
        ax2.bar(x + width/2, teacher_avg_rates, width, label='Teacher', alpha=0.7)
        ax2.set_xlabel('Old Task')
        ax2.set_ylabel('Average Retention Rate (%)')
        ax2.set_title('Average Retention Rate by Old Task')
        ax2.set_xticks(x)
        ax2.set_xticklabels(old_tasks, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Retention rate distribution histogram
        ax3 = axes[1, 0]
        ax3.hist(student_rates, bins=20, alpha=0.7, label='Student', density=True)
        ax3.hist(teacher_rates, bins=20, alpha=0.7, label='Teacher', density=True)
        ax3.set_xlabel('Retention Rate (%)')
        ax3.set_ylabel('Density')
        ax3.set_title('Retention Rate Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Retention rate difference analysis
        ax4 = axes[1, 1]
        retention_gaps = [r['retention_percentage_gap'] for r in retention_data]
        ax4.hist(retention_gaps, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax4.axvline(np.mean(retention_gaps), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(retention_gaps):.2f}%')
        ax4.set_xlabel('Retention Rate Gap (Teacher - Student) (%)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Teacher vs Student Retention Rate Gap')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        if save_path is None:
            save_path = os.path.join(self.plots_dir, 'retention_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Retention rate analysis chart saved: {save_path}")
        
        return save_path

    def log_training_metrics(self, epoch: int, task_id: int, task_name: str,
                            reconstruction_loss: float = None, kl_loss: float = None,
                            total_loss: float = None, **kwargs):
        """
        Record reconstruction loss and KL divergence metrics during training
        Support Section 5.2-4/5 Recon/KL curve plotting
        """
        timestamp = time.time() - self.experiment_start_time
        
        training_metrics = {
            'epoch': epoch,
            'task_id': task_id,
            'task_name': task_name,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss,
            'timestamp': timestamp,
            **kwargs
        }
        
        # Store in training metrics
        if 'training_metrics' not in self.metrics:
            self.metrics['training_metrics'] = []
        self.metrics['training_metrics'].append(training_metrics)
        
        # Real-time display of key metrics
        if reconstruction_loss is not None or kl_loss is not None:
            print(f"  Epoch {epoch}: Recon={reconstruction_loss:.4f}, KL={kl_loss:.4f}" if reconstruction_loss and kl_loss else "")
    
    def log_representation_similarity(self, epoch: int, task_id: int, task_name: str,
                                    teacher_student_similarity: float,
                                    feature_alignment_score: float = None,
                                    **kwargs):
        """
        Record Teacher-Student representation alignment metrics
        
        """
        timestamp = time.time() - self.experiment_start_time
        
        similarity_data = {
            'epoch': epoch,
            'task_id': task_id,
            'task_name': task_name,
            'teacher_student_similarity': teacher_student_similarity,
            'feature_alignment_score': feature_alignment_score,
            'timestamp': timestamp,
            **kwargs
        }
        
        # Store in representation similarity metrics
        if 'representation_similarity' not in self.metrics:
            self.metrics['representation_similarity'] = []
        self.metrics['representation_similarity'].append(similarity_data)
        
        print(f"  Representation alignment: {teacher_student_similarity:.4f}")
    
    def log_sample_generation(self, epoch: int, task_id: int, task_name: str,
                             seed: int, sample_quality_score: float = None,
                             **kwargs):
        """
        Record fixed seed sample generation information
        
        """
        timestamp = time.time() - self.experiment_start_time
        
        sample_data = {
            'epoch': epoch,
            'task_id': task_id,
            'task_name': task_name,
            'seed': seed,
            'sample_quality_score': sample_quality_score,
            'timestamp': timestamp,
            **kwargs
        }
        
        # Store in sample generation metrics
        if 'sample_generation' not in self.metrics:
            self.metrics['sample_generation'] = []
        self.metrics['sample_generation'].append(sample_data)

    def _plot_task_completion_analysis(self):
        """Generate task completion analysis chart"""
        if 'task_data' not in self.metrics or not self.metrics['task_data']:
            print(" No task completion data, cannot generate task completion analysis chart")
            return
        
        # Prepare data
        task_data = self.metrics['task_data']
        task_names = [item.get('task_name', f'Task_{i}') for i, item in enumerate(task_data)]
        
        # Get Teacher and Student FID data
        teacher_fids = []
        student_fids = []
        for item in task_data:
            # Prioritize new separated FID data
            if 'teacher_final_fid' in item and 'student_final_fid' in item:
                teacher_fids.append(item['teacher_final_fid'])
                student_fids.append(item['student_final_fid'])
            else:
                # Compatible with old data format
                final_fid = item.get('final_fid', 0)
                teacher_fids.append(final_fid)
                student_fids.append(final_fid)
        
        # Create chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Task Completion Analysis (Task completion analysis)', fontsize=16)
        
        # 1. Teacher vs Student FID comparison bar chart
        ax1 = axes[0, 0]
        x = np.arange(len(task_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, teacher_fids, width, label='Teacher', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x + width/2, student_fids, width, label='Student', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Task')
        ax1.set_ylabel('FID Score')
        ax1.set_title('Teacher vs Student FID Comparison (Task end FID comparison)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(task_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add numeric labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 2. FID difference analysis
        ax2 = axes[0, 1]
        fid_gaps = [t - s for t, s in zip(teacher_fids, student_fids)]
        colors = ['red' if gap > 0 else 'green' for gap in fid_gaps]
        
        bars = ax2.bar(task_names, fid_gaps, color=colors, alpha=0.7)
        ax2.set_xlabel('Task')
        ax2.set_ylabel('FID Gap (Teacher - Student)')
        ax2.set_title('FID Performance Gap Analysis (FID performance gap analysis)')
        ax2.set_xticklabels(task_names, rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Add numeric labels
        for bar, gap in zip(bars, fid_gaps):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.1),
                    f'{gap:.1f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
        
        # 3. Training time analysis
        ax3 = axes[1, 0]
        training_times = [item.get('total_training_time', 0) for item in task_data]
        ax3.bar(task_names, training_times, alpha=0.7, color='lightgreen')
        ax3.set_xlabel('Task')
        ax3.set_ylabel('Training Time (seconds)')
        ax3.set_title('Training Time per Task (Training time per task)')
        ax3.set_xticklabels(task_names, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Number of experts change
        ax4 = axes[1, 1]
        expert_counts = [item.get('expert_count', 1) for item in task_data]
        ax4.plot(task_names, expert_counts, 'o-', linewidth=2, markersize=8, color='purple')
        ax4.set_xlabel('Task')
        ax4.set_ylabel('Expert Count')
        ax4.set_title('Expert Count Evolution (Number of experts evolution)')
        ax4.set_xticklabels(task_names, rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add numeric labels
        for i, count in enumerate(expert_counts):
            ax4.annotate(f'{count}', (i, count), textcoords="offset points", 
                         xytext=(0,10), ha='center', fontsize=10)
        
        plt.tight_layout()
        
        # Save chart
        plot_path = os.path.join(self.plots_dir, 'task_completion_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Task completion analysis chart saved: {plot_path}")
        return plot_path

    def generate_recon_kl_curves(self, save_path: str = None):
        """
        Generate reconstruction loss and KL divergence curves
        """
        if 'training_metrics' not in self.metrics or not self.metrics['training_metrics']:
            print(" No training metrics data, cannot generate Recon/KL curves")
            return
        
        training_data = self.metrics['training_metrics']
        
        # Group by task data
        task_groups = {}
        for metric in training_data:
            task_name = metric['task_name']
            if task_name not in task_groups:
                task_groups[task_name] = []
            task_groups[task_name].append(metric)
        
        # Create chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Loss Analysis (Training loss analysis)', fontsize=16)
        
        # 1. Reconstruction loss curve
        ax1 = axes[0, 0]
        for task_name, metrics in task_groups.items():
            epochs = [m['epoch'] for m in metrics if m['reconstruction_loss'] is not None]
            recon_losses = [m['reconstruction_loss'] for m in metrics if m['reconstruction_loss'] is not None]
            if epochs and recon_losses:
                ax1.plot(epochs, recon_losses, 'o-', label=task_name, markersize=4, alpha=0.8)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Reconstruction Loss')
        ax1.set_title('Reconstruction Loss Over Time (Reconstruction loss over time)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. KL divergence curve
        ax2 = axes[0, 1]
        for task_name, metrics in task_groups.items():
            epochs = [m['epoch'] for m in metrics if m['kl_loss'] is not None]
            kl_losses = [m['kl_loss'] for m in metrics if m['kl_loss'] is not None]
            if epochs and kl_losses:
                ax2.plot(epochs, kl_losses, 's-', label=task_name, markersize=4, alpha=0.8)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('KL Divergence Loss')
        ax2.set_title('KL Divergence Loss Over Time (KL divergence loss over time)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 3. Total loss curve
        ax3 = axes[1, 0]
        for task_name, metrics in task_groups.items():
            epochs = [m['epoch'] for m in metrics if m['total_loss'] is not None]
            total_losses = [m['total_loss'] for m in metrics if m['total_loss'] is not None]
            if epochs and total_losses:
                ax3.plot(epochs, total_losses, '^-', label=task_name, markersize=4, alpha=0.8)
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Total Loss')
        ax3.set_title('Total Loss Over Time (Total loss over time)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # 4. Loss components comparison (last task)
        ax4 = axes[1, 1]
        if task_groups:
            last_task = list(task_groups.keys())[-1]
            metrics = task_groups[last_task]
            
            epochs = [m['epoch'] for m in metrics if m['reconstruction_loss'] is not None and m['kl_loss'] is not None]
            recon_losses = [m['reconstruction_loss'] for m in metrics if m['reconstruction_loss'] is not None and m['kl_loss'] is not None]
            kl_losses = [m['kl_loss'] for m in metrics if m['reconstruction_loss'] is not None and m['kl_loss'] is not None]
            
            if epochs and recon_losses and kl_losses:
                ax4.plot(epochs, recon_losses, 'o-', label='Reconstruction Loss', markersize=4)
                ax4.plot(epochs, kl_losses, 's-', label='KL Loss', markersize=4)
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Loss Value')
                ax4.set_title(f'Loss Components Comparison - {last_task} (Loss components comparison)')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                ax4.set_yscale('log')
        
        plt.tight_layout()
        
        # Save chart
        if save_path is None:
            save_path = os.path.join(self.plots_dir, 'recon_kl_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Recon/KLcurve saved: {save_path}")
        return save_path

    def generate_fixed_seed_samples(self, save_path: str = None):
        """
        Generate fixed seed sample grid
        """
        if 'sample_generation' not in self.metrics or not self.metrics['sample_generation']:
            print(" No sample generation data, cannot generate fixed seed sample grid")
            return
        
        sample_data = self.metrics['sample_generation']
        
        # Group by task and seed
        task_seed_groups = {}
        for sample in sample_data:
            task_name = sample['task_name']
            seed = sample['seed']
            if task_name not in task_seed_groups:
                task_seed_groups[task_name] = {}
            if seed not in task_seed_groups[task_name]:
                task_seed_groups[task_name][seed] = []
            task_seed_groups[task_name][seed].append(sample)
        
        # Create chart
        n_tasks = len(task_seed_groups)
        n_seeds = max(len(seeds) for seeds in task_seed_groups.values()) if task_seed_groups else 1
        
        # Handle single task or single seed cases
        if n_tasks == 1 and n_seeds == 1:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            axes = [[ax]]  # Use list instead of numpy array
        elif n_tasks == 1:
            fig, axes = plt.subplots(1, n_seeds, figsize=(n_seeds * 3, 3))
            axes = axes.reshape(1, -1)
        elif n_seeds == 1:
            fig, axes = plt.subplots(n_tasks, 1, figsize=(6, n_tasks * 3))
            axes = axes.reshape(-1, 1)
        else:
            fig, axes = plt.subplots(n_tasks, n_seeds, figsize=(n_seeds * 3, n_tasks * 3))
        
        fig.suptitle('Fixed Seed Sample Grid (Fixed seed sample grid)', fontsize=16)
        
        # Fill sample grid
        for i, (task_name, seeds) in enumerate(task_seed_groups.items()):
            for j, seed in enumerate(sorted(seeds.keys())):
                # Correctly get axis object for different cases
                if n_tasks > 1 and n_seeds > 1:
                    ax = axes[i, j]
                elif n_seeds > 1:
                    ax = axes[i]
                elif n_tasks > 1:
                    ax = axes[i]
                else:
                    # Single task single seed case
                    ax = axes[0][0] if isinstance(axes, list) and len(axes) > 0 and len(axes[0]) > 0 else axes
                
                # Debug: verify axis object type and thoroughly fix
                if not hasattr(ax, 'imshow'):
                    print(f"    Warning: Axis object type error，type(ax)={type(ax)}, ax={ax}")
                    # Thorough fix: ensure ax is single item axis object
                    if isinstance(ax, np.ndarray):
                        if ax.size == 1:
                            ax = ax.item()  # Extract single item element
                        else:
                            # If array, take first item element
                            ax = ax.flat[0]
                    elif isinstance(axes, list):
                        if len(axes) > 0:
                            if isinstance(axes[0], list) and len(axes[0]) > 0:
                                ax = axes[0][0]
                            else:
                                ax = axes[0]
                        else:
                            ax = axes
                    else:
                        ax = axes
                    
                    print(f"    After fix: type(ax)={type(ax)}, hasattr(ax, 'imshow')={hasattr(ax, 'imshow')}")
                
                # Get sample data for this seed under this task
                samples = seeds[seed]
                if samples:
                    # Should display actual generated sample images here
                    # Since we have no actual image data, we create an item placeholder
                    sample_quality = samples[0].get('sample_quality_score', 0.5)
                    
                    # Create example image (should be replaced with real generated samples in actual use)
                    sample_img = np.random.rand(64, 64) * sample_quality
                    ax.imshow(sample_img, cmap='gray')
                    ax.set_title(f'{task_name}\nSeed: {seed}\nQuality: {sample_quality:.3f}')
                else:
                    ax.text(0.5, 0.5, 'No Sample', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{task_name}\nSeed: {seed}')
                
                ax.axis('off')
        
        plt.tight_layout()
        
        # Save chart
        if save_path is None:
            save_path = os.path.join(self.plots_dir, 'fixed_seed_samples.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Fixed seed sample grid saved: {save_path}")
        return save_path

    def generate_kd_timeline_with_fid(self, save_path: str = None):
        """
        Generate KD on/off timeline overlay FID curve
        """
        if 'key_events' not in self.metrics or not self.metrics['key_events']:
            print(" No key event data, cannot generate KD timeline")
            return
        
        # Get KD gating events
        kd_events = [event for event in self.metrics['key_events'] 
                     if event.get('event_type') == 'kd_gating']
        
        if not kd_events:
            print(" No KD gating event data, cannot generate KD timeline")
            return
        
        # Get FID curve data
        fid_curves = self.metrics.get('fid_curves', [])
        
        # Create chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('KD Gating Timeline with FID Overlay (KD gating timeline overlay FID)', fontsize=16)
        
        # 1. KD gating status timeline
        ax1.set_title('KD Gating Status Timeline (KD gating status timeline)')
        
        # Group by task KD events
        task_kd_events = {}
        for event in kd_events:
            task_name = event['task_name']
            if task_name not in task_kd_events:
                task_kd_events[task_name] = []
            task_kd_events[task_name].append(event)
        
        # Plot KD gating status
        colors = ['green', 'red', 'blue', 'orange', 'purple']
        for i, (task_name, events) in enumerate(task_kd_events.items()):
            color = colors[i % len(colors)]
            
            for event in events:
                epoch = event['epoch']
                gating_status = event['gating_details'].get('gating_status', 'unknown')
                kd_weight = event['gating_details'].get('kd_weight', 0)
                
                # Plot gating status points
                marker = 'o' if gating_status == 'on' else 'x'
                size = 100 if gating_status == 'on' else 80
                alpha = 0.8 if gating_status == 'on' else 0.6
                
                ax1.scatter(epoch, i, c=color, marker=marker, s=size, alpha=alpha, 
                           label=f'{task_name} ({gating_status})' if epoch == events[0]['epoch'] else "")
                
                # Add labels
                ax1.annotate(f'{gating_status.upper()}\n{kd_weight:.2f}', 
                            (epoch, i), xytext=(5, 5), textcoords='offset points',
                            fontsize=8, ha='left', va='bottom')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Task')
        ax1.set_yticks(range(len(task_kd_events)))
        ax1.set_yticklabels(list(task_kd_events.keys()))
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. FID curves overlay KD events
        ax2.set_title('FID Curves with KD Gating Events (FID curves with KD gating events)')
        
        if fid_curves:
            # Plot FID curves
            for curve in fid_curves:
                epochs = curve.get('epochs', [])
                fids = curve.get('fids', [])
                task_name = curve.get('task_name', 'Unknown')
                
                if epochs and fids:
                    ax2.plot(epochs, fids, 'o-', label=task_name, markersize=3, alpha=0.7)
            
            # Annotate KD events on FID curves
            for event in kd_events:
                epoch = event['epoch']
                gating_status = event['gating_details'].get('gating_status', 'unknown')
                
                # Find FID values for corresponding epoch
                event_fid = None
                for curve in fid_curves:
                    if epoch in curve.get('epochs', []):
                        idx = curve['epochs'].index(epoch)
                        event_fid = curve['fids'][idx]
                        break
                
                if event_fid is not None:
                    # Plot KD event markers
                    color = 'green' if gating_status == 'on' else 'red'
                    marker = '^' if gating_status == 'on' else 'v'
                    
                    ax2.scatter(epoch, event_fid, c=color, marker=marker, s=100, alpha=0.8,
                               edgecolors='black', linewidth=1)
                    
                    # Add labels
                    ax2.annotate(f'KD {gating_status.upper()}', 
                                (epoch, event_fid), xytext=(10, 10), textcoords='offset points',
                                fontsize=8, ha='left', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('FID Score')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save chart
        if save_path is None:
            save_path = os.path.join(self.plots_dir, 'kd_timeline_with_fid.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" KDTimeline overlay FID saved: {save_path}")
        return save_path

    def generate_representation_alignment_curves(self, save_path: str = None):
        """
        Generate representation alignment curves
        """
        if 'representation_similarity' not in self.metrics or not self.metrics['representation_similarity']:
            print(" No representation similarity data, cannot generate representation alignment curves")
            return
        
        similarity_data = self.metrics['representation_similarity']
        
        # Group by task data
        task_groups = {}
        for metric in similarity_data:
            task_name = metric['task_name']
            if task_name not in task_groups:
                task_groups[task_name] = []
            task_groups[task_name].append(metric)
        
        # Create chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Representation Alignment Analysis (Representation alignment analysis)', fontsize=16)
        
        # 1. Representation similarity over time
        ax1 = axes[0, 0]
        for task_name, metrics in task_groups.items():
            epochs = [m['epoch'] for m in metrics]
            similarities = [m['teacher_student_similarity'] for m in metrics]
            if epochs and similarities:
                ax1.plot(epochs, similarities, 'o-', label=task_name, markersize=4, alpha=0.8)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Teacher-Student Similarity')
        ax1.set_title('Representation Similarity Over Time (Representation similarity over time)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. Feature alignment analysis
        ax2 = axes[0, 1]
        for task_name, metrics in task_groups.items():
            epochs = [m['epoch'] for m in metrics if m.get('feature_alignment_score') is not None]
            alignments = [m['feature_alignment_score'] for m in metrics if m.get('feature_alignment_score') is not None]
            if epochs and alignments:
                ax2.plot(epochs, alignments, 's-', label=task_name, markersize=4, alpha=0.8)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Feature Alignment Score')
        ax2.set_title('Feature Alignment Score Over Time (Feature alignment score over time)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 3. Similarity distribution histogram
        ax3 = axes[1, 0]
        all_similarities = [m['teacher_student_similarity'] for m in similarity_data]
        ax3.hist(all_similarities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(np.mean(all_similarities), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(all_similarities):.3f}')
        ax3.set_xlabel('Teacher-Student Similarity')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Similarity Distribution (Similarity distribution)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Inter-task similarity comparison
        ax4 = axes[1, 1]
        task_avg_similarities = []
        task_names = []
        for task_name, metrics in task_groups.items():
            similarities = [m['teacher_student_similarity'] for m in metrics]
            if similarities:
                task_avg_similarities.append(np.mean(similarities))
                task_names.append(task_name)
        
        if task_avg_similarities:
            bars = ax4.bar(task_names, task_avg_similarities, alpha=0.7, color='lightcoral')
            ax4.set_xlabel('Task')
            ax4.set_ylabel('Average Similarity')
            ax4.set_title('Average Similarity by Task (Average similarity by task)')
            ax4.set_xticklabels(task_names, rotation=45)
            ax4.set_ylim(0, 1)
            ax4.grid(True, alpha=0.3)
            
            # Add numeric labels
            for bar, similarity in zip(bars, task_avg_similarities):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{similarity:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save chart
        if save_path is None:
            save_path = os.path.join(self.plots_dir, 'representation_alignment_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Representation alignmentcurve saved: {save_path}")
        return save_path

    def generate_retention_heatmap(self, save_path: str = None):
        """
        Generate retention rate heatmap
        """
        if 'retention_rates' not in self.metrics or not self.metrics['retention_rates']:
            print(" No retention rate data, cannot generate heatmap")
            return
        
        retention_data = self.metrics['retention_rates']
        
        # Organize data by old task and evaluation task
        old_tasks = list(set([r['old_task_name'] for r in retention_data]))
        old_tasks.sort()
        
        evaluation_tasks = list(set([r['evaluation_task'] for r in retention_data]))
        evaluation_tasks.sort()
        
        # Create data matrix
        student_matrix = np.zeros((len(old_tasks), len(evaluation_tasks)))
        teacher_matrix = np.zeros((len(old_tasks), len(evaluation_tasks)))
        
        # Fill data matrix
        for rate in retention_data:
            old_idx = old_tasks.index(rate['old_task_name'])
            eval_idx = evaluation_tasks.index(rate['evaluation_task'])
            
            student_matrix[old_idx, eval_idx] = rate['student_retention_percentage']
            teacher_matrix[old_idx, eval_idx] = rate['teacher_retention_percentage']
        
        # Create chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Retention Rate Heatmap Analysis (Retention rate heatmap analysis)', fontsize=16)
        
        # 1. Student retention rate heatmap
        ax1 = axes[0, 0]
        im1 = ax1.imshow(student_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)
        ax1.set_title('Student Retention Rate Heatmap (Student retention rate heatmap)')
        ax1.set_xlabel('Evaluation Task')
        ax1.set_ylabel('Old Task')
        ax1.set_xticks(range(len(evaluation_tasks)))
        ax1.set_xticklabels(evaluation_tasks, rotation=45)
        ax1.set_yticks(range(len(old_tasks)))
        ax1.set_yticklabels(old_tasks)
        
        # Add numeric labels
        for i in range(len(old_tasks)):
            for j in range(len(evaluation_tasks)):
                if student_matrix[i, j] > 0:
                    text = ax1.text(j, i, f'{student_matrix[i, j]:.1f}%',
                                   ha="center", va="center", color="black", fontsize=8)
        
        # Add color bar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Retention Rate (%)')
        
        # 2. Teacher retention rate heatmap
        ax2 = axes[0, 1]
        im2 = ax2.imshow(teacher_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)
        ax2.set_title('Teacher Retention Rate Heatmap (Teacher retention rate heatmap)')
        ax2.set_xlabel('Evaluation Task')
        ax2.set_ylabel('Old Task')
        ax2.set_xticks(range(len(evaluation_tasks)))
        ax2.set_xticklabels(evaluation_tasks, rotation=45)
        ax2.set_yticks(range(len(old_tasks)))
        ax2.set_yticklabels(old_tasks)
        
        # Add numeric labels
        for i in range(len(old_tasks)):
            for j in range(len(evaluation_tasks)):
                if teacher_matrix[i, j] > 0:
                    text = ax2.text(j, i, f'{teacher_matrix[i, j]:.1f}%',
                                   ha="center", va="center", color="black", fontsize=8)
        
        # Add color bar
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Retention Rate (%)')
        
        # 3. Retention rate difference heatmap
        ax3 = axes[1, 0]
        diff_matrix = teacher_matrix - student_matrix
        im3 = ax3.imshow(diff_matrix, cmap='RdBu_r', aspect='auto', 
                         vmin=-np.max(np.abs(diff_matrix)), vmax=np.max(np.abs(diff_matrix)))
        ax3.set_title('Retention Rate Difference (Teacher - Student) (Retention rate difference heatmap)')
        ax3.set_xlabel('Evaluation Task')
        ax3.set_ylabel('Old Task')
        ax3.set_xticks(range(len(evaluation_tasks)))
        ax3.set_xticklabels(evaluation_tasks, rotation=45)
        ax3.set_yticks(range(len(old_tasks)))
        ax3.set_yticklabels(old_tasks)
        
        # Add numeric labels
        for i in range(len(old_tasks)):
            for j in range(len(evaluation_tasks)):
                if diff_matrix[i, j] != 0:
                    text = ax3.text(j, i, f'{diff_matrix[i, j]:.1f}%',
                                   ha="center", va="center", color="black", fontsize=8)
        
        # Add color bar
        cbar3 = plt.colorbar(im3, ax=ax3)
        cbar3.set_label('Retention Rate Difference (%)')
        
        # 4. Average retention rate comparison bar chart
        ax4 = axes[1, 1]
        student_avg = np.mean(student_matrix, axis=1)
        teacher_avg = np.mean(teacher_matrix, axis=1)
        
        x = np.arange(len(old_tasks))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, student_avg, width, label='Student', alpha=0.8, color='lightcoral')
        bars2 = ax4.bar(x + width/2, teacher_avg, width, label='Teacher', alpha=0.8, color='skyblue')
        
        ax4.set_xlabel('Old Task')
        ax4.set_ylabel('Average Retention Rate (%)')
        ax4.set_title('Average Retention Rate by Old Task (Average retention rate by old task)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(old_tasks, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add numeric labels
        for bar, rate in zip(bars1, student_avg):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
        
        for bar, rate in zip(bars2, teacher_avg):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save chart
        if save_path is None:
            save_path = os.path.join(self.plots_dir, 'retention_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Retention rate heatmap saved: {save_path}")
        return save_path

    def _save_training_metrics(self):
        """Save training metrics data"""
        if 'training_metrics' in self.metrics and self.metrics['training_metrics']:
            # Save as JSON format
            training_file = os.path.join(self.data_dir, 'training_metrics.json')
            with open(training_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics['training_metrics'], f, indent=2, default=str)
            
            # Save as CSV format
            training_csv_file = os.path.join(self.data_dir, 'training_metrics.csv')
            df_training = pd.DataFrame(self.metrics['training_metrics'])
            df_training.to_csv(training_csv_file, index=False, encoding='utf-8')
            
            print(f" Training metrics data saved: {len(self.metrics['training_metrics'])} records")
    
    def _save_representation_similarity(self):
        """Save representation similarity data"""
        if 'representation_similarity' in self.metrics and self.metrics['representation_similarity']:
            # Save as JSON format
            similarity_file = os.path.join(self.data_dir, 'representation_similarity.json')
            with open(similarity_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics['representation_similarity'], f, indent=2, default=str)
            
            # Save as CSV format
            similarity_csv_file = os.path.join(self.data_dir, 'representation_similarity.csv')
            df_similarity = pd.DataFrame(self.metrics['representation_similarity'])
            df_similarity.to_csv(similarity_csv_file, index=False, encoding='utf-8')
            
            print(f" Representation similarity data saved: {len(self.metrics['representation_similarity'])} records")
    
    def _save_sample_generation(self):
        """Save sample generation data"""
        if 'sample_generation' in self.metrics and self.metrics['sample_generation']:
            # Save as JSON format
            sample_file = os.path.join(self.data_dir, 'sample_generation.json')
            with open(sample_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics['sample_generation'], f, indent=2, default=str)
            
            # Save as CSV format
            sample_csv_file = os.path.join(self.data_dir, 'sample_generation.csv')
            df_sample = pd.DataFrame(self.metrics['sample_generation'])
            df_sample.to_csv(sample_csv_file, index=False, encoding='utf-8')
            
            print(f" Sample generation data saved: {len(self.metrics['sample_generation'])} records")

    def log_kd_ablation_experiment(self, experiment_id: str, kd_enabled: bool,
                                  task_id: int, task_name: str,
                                  teacher_fid_end: float, student_fid_end: float,
                                  retention_rates: dict = None, **kwargs):
        """
        Log KD ablation experiment data
        
        Args:
            experiment_id: Experiment ID (e.g. 'kd_on', 'kd_off'）
            kd_enabled: Whether KD is enabled
            task_id: TaskID
            task_name: Task name
            teacher_fid_end: Teacher end FID
            student_fid_end: Student end FID
            retention_rates: Retention rate data
            **kwargs: Other parameters
        """
        timestamp = time.time() - self.experiment_start_time
        
        ablation_data = {
            'experiment_id': experiment_id,
            'kd_enabled': kd_enabled,
            'task_id': task_id,
            'task_name': task_name,
            'teacher_fid_end': teacher_fid_end,
            'student_fid_end': student_fid_end,
            'fid_gap': teacher_fid_end - student_fid_end,
            'timestamp': timestamp,
            **kwargs
        }
        
        # addRetention rate data
        if retention_rates:
            for old_task, rate in retention_rates.items():
                ablation_data[f'retention_{old_task}'] = rate
        
        # Store in KD ablation experiment metrics
        if 'kd_ablation_experiments' not in self.metrics:
            self.metrics['kd_ablation_experiments'] = []
        self.metrics['kd_ablation_experiments'].append(ablation_data)
        
        print(f"  KD ablation experiment logged: {experiment_id}, KD={'ON' if kd_enabled else 'OFF'}, "
              f"Teacher FID={teacher_fid_end:.2f}, Student FID={student_fid_end:.2f}")
    
    def generate_kd_ablation_comparison(self, save_path: str = None):
        """
        Generate KD ablation comparison analysis
        """
        if 'kd_ablation_experiments' not in self.metrics or not self.metrics['kd_ablation_experiments']:
            print(" No KD ablation experiment data, cannot generate comparison analysis")
            return
        
        ablation_data = self.metrics['kd_ablation_experiments']
        
        # Group by experiment ID
        experiment_groups = {}
        for data in ablation_data:
            exp_id = data['experiment_id']
            if exp_id not in experiment_groups:
                experiment_groups[exp_id] = []
            experiment_groups[exp_id].append(data)
        
        # Create chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('KD Ablation Study Analysis (KD ablation study analysis)', fontsize=16)
        
        # 1. End FID comparison
        ax1 = axes[0, 0]
        exp_ids = list(experiment_groups.keys())
        teacher_fids = []
        student_fids = []
        
        for exp_id in exp_ids:
            exp_data = experiment_groups[exp_id]
            avg_teacher_fid = np.mean([d['teacher_fid_end'] for d in exp_data])
            avg_student_fid = np.mean([d['student_fid_end'] for d in exp_data])
            teacher_fids.append(avg_teacher_fid)
            student_fids.append(avg_student_fid)
        
        x = np.arange(len(exp_ids))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, teacher_fids, width, label='Teacher', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x + width/2, student_fids, width, label='Student', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('FID Score')
        ax1.set_title('End-of-Task FID Comparison (Task end FID comparison)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(exp_ids, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add numeric labels
        for bar, fid in zip(bars1, teacher_fids):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{fid:.1f}', ha='center', va='bottom', fontsize=8)
        
        for bar, fid in zip(bars2, student_fids):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{fid:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 2. FID difference comparison
        ax2 = axes[0, 1]
        fid_gaps = [t - s for t, s in zip(teacher_fids, student_fids)]
        colors = ['red' if gap > 0 else 'green' for gap in fid_gaps]
        
        bars = ax2.bar(exp_ids, fid_gaps, color=colors, alpha=0.7)
        ax2.set_xlabel('Experiment')
        ax2.set_ylabel('FID Gap (Teacher - Student)')
        ax2.set_title('FID Performance Gap Analysis (FID performance gap analysis)')
        ax2.set_xticklabels(exp_ids, rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Add numeric labels
        for bar, gap in zip(bars, fid_gaps):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.1),
                    f'{gap:.1f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
        
        # 3. Retention rate comparison (if data available)
        ax3 = axes[1, 0]
        if any('retention_' in data for data in ablation_data):
            # extractRetention rate data
            retention_metrics = {}
            for exp_id in exp_ids:
                exp_data = experiment_groups[exp_id]
                retention_values = []
                for data in exp_data:
                    for key, value in data.items():
                        if key.startswith('retention_'):
                            retention_values.append(value)
                if retention_values:
                    retention_metrics[exp_id] = np.mean(retention_values)
            
            if retention_metrics:
                retention_values = list(retention_metrics.values())
                bars = ax3.bar(list(retention_metrics.keys()), retention_values, alpha=0.7, color='lightgreen')
                ax3.set_xlabel('Experiment')
                ax3.set_ylabel('Average Retention Rate (%)')
                ax3.set_title('Retention Rate Comparison ')
                ax3.set_xticklabels(list(retention_metrics.keys()), rotation=45)
                ax3.grid(True, alpha=0.3)
                
                # Add numeric labels
                for bar, rate in zip(bars, retention_values):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No Retention Data Available\n(No retention rate data)', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Retention Rate Comparison ')
        
        # 4. Convergence speed comparison (based on training metrics)
        ax4 = axes[1, 1]
        if 'training_metrics' in self.metrics and self.metrics['training_metrics']:
            # Group training metrics by experiment
            training_by_exp = {}
            for metric in self.metrics['training_metrics']:
                exp_id = metric.get('experiment_id', 'default')
                if exp_id not in training_by_exp:
                    training_by_exp[exp_id] = []
                training_by_exp[exp_id].append(metric)
            
            # Plot convergence curves
            for exp_id, metrics in training_by_exp.items():
                if exp_id in exp_ids:  # Only show ablation experiment data
                    epochs = [m['epoch'] for m in metrics if m.get('total_loss') is not None]
                    losses = [m['total_loss'] for m in metrics if m.get('total_loss') is not None]
                    if epochs and losses:
                        ax4.plot(epochs, losses, 'o-', label=f'{exp_id}', markersize=4, alpha=0.8)
            
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Total Loss')
            ax4.set_title('Convergence Speed Comparison (Convergence speed comparison)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')
        else:
            ax4.text(0.5, 0.5, 'No Training Metrics Available\n(No training metrics data)', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Convergence Speed Comparison (Convergence speed comparison)')
        
        plt.tight_layout()
        
        # Save chart
        if save_path is None:
            save_path = os.path.join(self.plots_dir, 'kd_ablation_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" KD ablation comparison analysis saved: {save_path}")
        return save_path

    def _save_kd_ablation_experiments(self):
        """Save KD ablation experiment data"""
        if 'kd_ablation_experiments' in self.metrics and self.metrics['kd_ablation_experiments']:
            # Save as JSON format
            ablation_file = os.path.join(self.data_dir, 'kd_ablation_experiments.json')
            with open(ablation_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics['kd_ablation_experiments'], f, indent=2, default=str)
            
            # Save as CSV format
            ablation_csv_file = os.path.join(self.data_dir, 'kd_ablation_experiments.csv')
            df_ablation = pd.DataFrame(self.metrics['kd_ablation_experiments'])
            df_ablation.to_csv(ablation_csv_file, index=False, encoding='utf-8')
            
            print(f" KD ablation experiment data saved: {len(self.metrics['kd_ablation_experiments'])} records")
            
            # Show ablation experiment statistics
            if self.metrics['kd_ablation_experiments']:
                # Group by experiment ID statistics
                exp_stats = {}
                for exp in self.metrics['kd_ablation_experiments']:
                    exp_id = exp['experiment_id']
                    if exp_id not in exp_stats:
                        exp_stats[exp_id] = {'teacher_fids': [], 'student_fids': []}
                    exp_stats[exp_id]['teacher_fids'].append(exp['teacher_fid_end'])
                    exp_stats[exp_id]['student_fids'].append(exp['student_fid_end'])
                
                print(" KD ablation experiment statistics:")
                for exp_id, stats in exp_stats.items():
                    avg_teacher = np.mean(stats['teacher_fids'])
                    avg_student = np.mean(stats['student_fids'])
                    avg_gap = avg_teacher - avg_student
                    print(f"   {exp_id}: Teacher={avg_teacher:.2f}, Student={avg_student:.2f}, Gap={avg_gap:.2f}")

    def log_fid_slope_analysis(self, task_id: int, task_name: str, epoch: int,
                              trigger_epoch: int, pre_trigger_slope: float,
                              post_trigger_slope: float, convergence_epoch: int = None,
                              **kwargs):
        """
        Log FID slope analysis data
        
        Args:
            task_id: TaskID
            task_name: Task name
            epoch: Current epoch
            trigger_epoch: Trigger epoch
            pre_trigger_slope: FID before trigger slope
            post_trigger_slope: FID after trigger slope
            convergence_epoch: convergenceepoch
            **kwargs: Other parameters
        """
        slope_data = {
            'run_id': getattr(self, 'run_id', 'unknown'),
            'seed': getattr(self, 'seed', 0),
            'task_id': task_id,
            'task_name': task_name,
            'epoch': epoch,
            'trigger_epoch': trigger_epoch,
            'pre_trigger_slope': pre_trigger_slope,
            'post_trigger_slope': post_trigger_slope,
            'slope_improvement': post_trigger_slope - pre_trigger_slope,
            'convergence_epoch': convergence_epoch,
            'timestamp': time.time() - self.experiment_start_time,
            'datetime': datetime.datetime.now().isoformat(),
            **kwargs
        }
        
        # Store in FID slope analysis data
        if 'fid_slope_analysis' not in self.metrics:
            self.metrics['fid_slope_analysis'] = []
        self.metrics['fid_slope_analysis'].append(slope_data)
        
        print(f"  FID slope analysis logged: Task{task_id}({task_name})")
        print(f"    Pre-trigger slope: {pre_trigger_slope:.6f}")
        print(f"    Post-trigger slope: {post_trigger_slope:.6f}")
        print(f"    Slope improvement: {slope_data['slope_improvement']:.6f}")
    
    def validate_data_diversity(self, min_slope_diff: float = 0.0001, min_fid_diff: float = 0.01):
        """
        Validate data diversity, ensure no overly similar data appears
        
        Args:
            min_slope_diff: Min slope difference threshold
            min_fid_diff: MinFIDdifferencethreshold
        """
        print("\n=== Data diversity validation ===")
        
        # Check FID slope analysis data
        if 'fid_slope_analysis' in self.metrics and self.metrics['fid_slope_analysis']:
            slope_data = self.metrics['fid_slope_analysis']
            
            # Check slope difference
            slope_diffs = []
            for i, data1 in enumerate(slope_data):
                for j, data2 in enumerate(slope_data[i+1:], i+1):
                    if (data1['task_id'] == data2['task_id'] and 
                        data1['task_name'] == data2['task_name']):
                        pre_diff = abs(data1['pre_trigger_slope'] - data2['pre_trigger_slope'])
                        post_diff = abs(data1['post_trigger_slope'] - data2['post_trigger_slope'])
                        slope_diffs.extend([pre_diff, post_diff])
            
            if slope_diffs:
                avg_slope_diff = np.mean(slope_diffs)
                min_slope_diff_actual = min(slope_diffs)
                print(f"  Slope difference statistics:")
                print(f"    Averagedifference: {avg_slope_diff:.6f}")
                print(f"    Mindifference: {min_slope_diff_actual:.6f}")
                
                if min_slope_diff_actual < min_slope_diff:
                    print(f"      Warning: Data with slope difference too small exists (< {min_slope_diff})")
                else:
                    print(f"     Slope difference normal (>= {min_slope_diff})")
        
        # Check ablation comparison data
        if 'ablation_comparison' in self.metrics and self.metrics['ablation_comparison']:
            ablation_data = self.metrics['ablation_comparison']
            
            # Check by ablation type group
            ablation_groups = {}
            for data in ablation_data:
                ab_type = data['ablation_type']
                if ab_type not in ablation_groups:
                    ablation_groups[ab_type] = {'triggered': [], 'not_triggered': []}
                
                if data['triggered']:
                    ablation_groups[ab_type]['triggered'].append(data)
                else:
                    ablation_groups[ab_type]['not_triggered'].append(data)
            
            print(f"  Ablation comparison data diversity:")
            for ab_type, groups in ablation_groups.items():
                triggered_fids = [d['final_fid'] for d in groups['triggered']]
                not_triggered_fids = [d['final_fid'] for d in groups['not_triggered']]
                
                if triggered_fids and not_triggered_fids:
                    avg_triggered = np.mean(triggered_fids)
                    avg_not_triggered = np.mean(not_triggered_fids)
                    fid_diff = abs(avg_triggered - avg_not_triggered)
                    
                    print(f"    {ab_type}:")
                    print(f"      Triggered group average FID: {avg_triggered:.4f}")
                    print(f"      Non-triggered group average FID: {avg_not_triggered:.4f}")
                    print(f"      Inter-group difference: {fid_diff:.4f}")
                    
                    if fid_diff < min_fid_diff:
                        print(f"        Warning: Data with inter-group difference too small exists (< {min_fid_diff})")
                    else:
                        print(f"       Inter-group difference normal (>= {min_fid_diff})")
        
        # Check expert expansion trigger log
        if 'expert_expansion_trigger_logs' in self.metrics and self.metrics['expert_expansion_trigger_logs']:
            trigger_logs = self.metrics['expert_expansion_trigger_logs']
            
            # Check FID change
            fid_changes = []
            for log in trigger_logs:
                if log['FID_before'] and log['FID_after']:
                    change = abs(log['FID_before'] - log['FID_after'])
                    fid_changes.append(change)
            
            if fid_changes:
                avg_fid_change = np.mean(fid_changes)
                min_fid_change = min(fid_changes)
                print(f"  FID change statistics:")
                print(f"    Average change: {avg_fid_change:.4f}")
                print(f"    Min change: {min_fid_change:.4f}")
                
                if min_fid_change < min_fid_diff:
                    print(f"      Warning: Data with FID change too small exists (< {min_fid_diff})")
                else:
                    print(f"       FID change normal (>= {min_fid_diff})")
        
        print("=== Data diversity validation completed ===\n")
    
    def log_ablation_comparison(self, task_id: int, task_name: str, 
                              ablation_type: str, triggered: bool,
                              final_fid: float, retention_rate: float,
                              convergence_time: float, **kwargs):
        """
        Log ablation comparison data
        
        Args:
            task_id: TaskID
            task_name: Task name
            ablation_type: Ablation type (e.g. "expert_expansion", "kd_gating", etc.)
            triggered: Whether triggered
            final_fid: endFID
            retention_rate: retention rate
            convergence_time: Convergence duration
            **kwargs: Other parameters
        """
        ablation_data = {
            'run_id': getattr(self, 'run_id', 'unknown'),
            'seed': getattr(self, 'seed', 0),
            'task_id': task_id,
            'task_name': task_name,
            'ablation_type': ablation_type,
            'triggered': triggered,
            'final_fid': final_fid,
            'retention_rate': retention_rate,
            'convergence_time': convergence_time,
            'timestamp': time.time() - self.experiment_start_time,
            'datetime': datetime.datetime.now().isoformat(),
            **kwargs
        }
        
        # Store in ablation comparison data
        if 'ablation_comparison' not in self.metrics:
            self.metrics['ablation_comparison'] = []
        self.metrics['ablation_comparison'].append(ablation_data)
        
        print(f"  Ablation comparison logged: {ablation_type} - Task{task_id}({task_name})")
        print(f"    Trigger status: {'Yes' if triggered else 'No'}")
        print(f"    endFID: {final_fid:.4f}")
        print(f"    retention rate: {retention_rate:.4f}")
        print(f"    Convergence duration: {convergence_time:.2f} epochs")
    
    def log_expert_switch_event(self, epoch: int, task_id: int, task_name: str,
                                old_expert_id: int, new_expert_id: int, 
                                switch_reason: str, **kwargs):
        """Log expert switch events"""
        event_details = {
            'old_expert_id': old_expert_id,
            'new_expert_id': new_expert_id,
            'switch_reason': switch_reason,
            'action': 'switch_expert'
        }
        self.log_key_event(epoch, task_id, task_name, 'expert_switch', event_details, **kwargs)
    
    def _plot_fid_slope_comparison(self):
        
        if 'fid_slope_analysis' not in self.metrics or not self.metrics['fid_slope_analysis']:
            print("  No FID slope analysis data to plot")
            return
        
        # Set Chinese font
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        slope_data = self.metrics['fid_slope_analysis']
        
        # Create chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('FID Slope Comparison: Before vs After Expert Expansion (Figure 5.3-2)', fontsize=16)
        
        # Left chart: Pre and post trigger slope comparison
        tasks = list(set([f"{d['task_id']}_{d['task_name']}" for d in slope_data]))
        tasks.sort()
        
        pre_slopes = []
        post_slopes = []
        improvements = []
        task_labels = []
        
        for task_key in tasks:
            task_slopes = [d for d in slope_data if f"{d['task_id']}_{d['task_name']}" == task_key]
            if task_slopes:
                avg_pre = np.mean([d['pre_trigger_slope'] for d in task_slopes])
                avg_post = np.mean([d['post_trigger_slope'] for d in task_slopes])
                avg_improvement = np.mean([d['slope_improvement'] for d in task_slopes])
                
                pre_slopes.append(avg_pre)
                post_slopes.append(avg_post)
                improvements.append(avg_improvement)
                task_labels.append(task_key)
        
        x = np.arange(len(task_labels))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, pre_slopes, width, label='Pre-Trigger Slope', 
                        color='lightcoral', alpha=0.8)
        bars2 = ax1.bar(x + width/2, post_slopes, width, label='Post-Trigger Slope', 
                        color='lightblue', alpha=0.8)
        
        ax1.set_xlabel('Tasks')
        ax1.set_ylabel('FID Slope')
        ax1.set_title('FID Slope Before vs After Expert Expansion')
        ax1.set_xticks(x)
        ax1.set_xticklabels(task_labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add numeric labels
        for i, (pre, post) in enumerate(zip(pre_slopes, post_slopes)):
            ax1.text(i - width/2, pre + 0.0001, f'{pre:.6f}', ha='center', va='bottom', fontsize=8)
            ax1.text(i + width/2, post + 0.0001, f'{post:.6f}', ha='center', va='bottom', fontsize=8)
        
        # Right chart: Slope improvement degree
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars3 = ax2.bar(task_labels, improvements, color=colors, alpha=0.7)
        ax2.set_xlabel('Tasks')
        ax2.set_ylabel('Slope Improvement (Post - Pre)')
        ax2.set_title('FID Slope Improvement After Expert Expansion')
        ax2.set_xticklabels(task_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add numeric labels
        for i, (bar, imp) in enumerate(zip(bars3, improvements)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                    f'{imp:.6f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save chart
        save_path = os.path.join(self.plots_dir, 'fid_slope_comparison_figure_5_3_2.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  FID slope comparison chart saved: {save_path}")
        
        # Save data
        slope_summary_file = os.path.join(self.data_dir, 'fid_slope_analysis_summary.csv')
        slope_summary_data = []
        for i, task_key in enumerate(task_labels):
            slope_summary_data.append({
                'Task': task_key,
                'Pre_Trigger_Slope': f"{pre_slopes[i]:.6f}",
                'Post_Trigger_Slope': f"{post_slopes[i]:.6f}",
                'Slope_Improvement': f"{improvements[i]:.6f}",
                'Improvement_Status': 'Positive' if improvements[i] > 0 else 'Negative'
            })
        
        df_slope_summary = pd.DataFrame(slope_summary_data)
        df_slope_summary.to_csv(slope_summary_file, index=False, encoding='utf-8')
        print(f"  FID slope analysis summary saved: {slope_summary_file}")
        
        plt.close()
    
    def _plot_ablation_comparison(self):
        """Plot ablation comparison chart (supporting Table 5.3-B: whether triggered/trigger timing ablation comparison)"""
        if 'ablation_comparison' not in self.metrics or not self.metrics['ablation_comparison']:
            print("  No ablation comparison data to plot")
            return
        
        # Set Chinese font
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        ablation_data = self.metrics['ablation_comparison']
        
        # Create chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Ablation Study: Expert Expansion Trigger Analysis (Table 5.3-B)', fontsize=16)
        
        # Group by ablation type
        ablation_types = list(set([d['ablation_type'] for d in ablation_data]))
        ablation_types.sort()
        
        # 1. End FID comparison
        final_fid_data = {ab_type: {'triggered': [], 'not_triggered': []} for ab_type in ablation_types}
        for d in ablation_data:
            if d['triggered']:
                final_fid_data[d['ablation_type']]['triggered'].append(d['final_fid'])
            else:
                final_fid_data[d['ablation_type']]['not_triggered'].append(d['final_fid'])
        
        x = np.arange(len(ablation_types))
        width = 0.35
        
        triggered_fids = [np.mean(final_fid_data[ab_type]['triggered']) if final_fid_data[ab_type]['triggered'] else 0 
                         for ab_type in ablation_types]
        not_triggered_fids = [np.mean(final_fid_data[ab_type]['not_triggered']) if final_fid_data[ab_type]['not_triggered'] else 0 
                             for ab_type in ablation_types]
        
        bars1 = ax1.bar(x - width/2, triggered_fids, width, label='Triggered', 
                        color='lightgreen', alpha=0.8)
        bars2 = ax1.bar(x + width/2, not_triggered_fids, width, label='Not Triggered', 
                        color='lightcoral', alpha=0.8)
        
        ax1.set_xlabel('Ablation Types')
        ax1.set_ylabel('Final FID')
        ax1.set_title('Final FID Comparison: Triggered vs Not Triggered')
        ax1.set_xticks(x)
        ax1.set_xticklabels(ablation_types, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. retention ratecomparison
        retention_data = {ab_type: {'triggered': [], 'not_triggered': []} for ab_type in ablation_types}
        for d in ablation_data:
            if d['triggered']:
                retention_data[d['ablation_type']]['triggered'].append(d['retention_rate'])
            else:
                retention_data[d['ablation_type']]['not_triggered'].append(d['retention_rate'])
        
        triggered_retentions = [np.mean(retention_data[ab_type]['triggered']) if retention_data[ab_type]['triggered'] else 0 
                               for ab_type in ablation_types]
        not_triggered_retentions = [np.mean(retention_data[ab_type]['not_triggered']) if retention_data[ab_type]['not_triggered'] else 0 
                                   for ab_type in ablation_types]
        
        bars3 = ax2.bar(x - width/2, triggered_retentions, width, label='Triggered', 
                        color='lightgreen', alpha=0.8)
        bars4 = ax2.bar(x + width/2, not_triggered_retentions, width, label='Not Triggered', 
                        color='lightcoral', alpha=0.8)
        
        ax2.set_xlabel('Ablation Types')
        ax2.set_ylabel('Retention Rate')
        ax2.set_title('Retention Rate Comparison: Triggered vs Not Triggered')
        ax2.set_xticks(x)
        ax2.set_xticklabels(ablation_types, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Convergence durationcomparison
        convergence_data = {ab_type: {'triggered': [], 'not_triggered': []} for ab_type in ablation_types}
        for d in ablation_data:
            if d['triggered']:
                convergence_data[d['ablation_type']]['triggered'].append(d['convergence_time'])
            else:
                convergence_data[d['ablation_type']]['not_triggered'].append(d['convergence_time'])
        
        triggered_convergences = [np.mean(convergence_data[ab_type]['triggered']) if convergence_data[ab_type]['triggered'] else 0 
                                 for ab_type in ablation_types]
        not_triggered_convergences = [np.mean(convergence_data[ab_type]['not_triggered']) if convergence_data[ab_type]['not_triggered'] else 0 
                                     for ab_type in ablation_types]
        
        bars5 = ax3.bar(x - width/2, triggered_convergences, width, label='Triggered', 
                        color='lightgreen', alpha=0.8)
        bars6 = ax3.bar(x + width/2, not_triggered_convergences, width, label='Not Triggered', 
                        color='lightcoral', alpha=0.8)
        
        ax3.set_xlabel('Ablation Types')
        ax3.set_ylabel('Convergence Time (epochs)')
        ax3.set_title('Convergence Time Comparison: Triggered vs Not Triggered')
        ax3.set_xticks(x)
        ax3.set_xticklabels(ablation_types, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Comprehensive performance radar chart
        ax4.remove()  
        ax4 = fig.add_subplot(2, 2, 4, projection='polar')
        
        # Calculate comprehensive performance metrics
        categories = ['Final FID', 'Retention Rate', 'Convergence Time']
        N = len(categories)
        
        triggered_performance = [
            1 / (np.mean(triggered_fids) + 1e-6),  # Lower FID is better
            np.mean(triggered_retentions),  # Higher retention rate is better
            1 / (np.mean(triggered_convergences) + 1e-6)  # Shorter convergence time is better
        ]
        
        not_triggered_performance = [
            1 / (np.mean(not_triggered_fids) + 1e-6),
            np.mean(not_triggered_retentions),
            1 / (np.mean(not_triggered_convergences) + 1e-6)
        ]
        
        # Normalize to 0-1 range
        max_val = max(max(triggered_performance), max(not_triggered_performance))
        triggered_performance = [v / max_val for v in triggered_performance]
        not_triggered_performance = [v / max_val for v in not_triggered_performance]
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close
        
        triggered_performance += triggered_performance[:1]
        not_triggered_performance += not_triggered_performance[:1]
        
        ax4.plot(angles, triggered_performance, 'o-', linewidth=2, label='Triggered', color='green')
        ax4.fill(angles, triggered_performance, alpha=0.25, color='green')
        ax4.plot(angles, not_triggered_performance, 'o-', linewidth=2, label='Not Triggered', color='red')
        ax4.fill(angles, not_triggered_performance, alpha=0.25, color='red')
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_ylim(0, 1)
        ax4.set_title('Overall Performance Comparison')
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        # Save chart
        save_path = os.path.join(self.plots_dir, 'ablation_comparison_table_5_3b.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" : {save_path}")
        
        ablation_summary_file = os.path.join(self.data_dir, 'ablation_comparison_table_5_3b.csv')
        ablation_summary_data = []
        
        for ab_type in ablation_types:
            # Triggered group statistics
            triggered_items = [d for d in ablation_data if d['ablation_type'] == ab_type and d['triggered']]
            not_triggered_items = [d for d in ablation_data if d['ablation_type'] == ab_type and not d['triggered']]
            
            if triggered_items:
                avg_final_fid_triggered = np.mean([d['final_fid'] for d in triggered_items])
                avg_retention_triggered = np.mean([d['retention_rate'] for d in triggered_items])
                avg_convergence_triggered = np.mean([d['convergence_time'] for d in triggered_items])
            else:
                avg_final_fid_triggered = avg_retention_triggered = avg_convergence_triggered = 0
            
            if not_triggered_items:
                avg_final_fid_not_triggered = np.mean([d['final_fid'] for d in not_triggered_items])
                avg_retention_not_triggered = np.mean([d['retention_rate'] for d in not_triggered_items])
                avg_convergence_not_triggered = np.mean([d['convergence_time'] for d in not_triggered_items])
            else:
                avg_final_fid_not_triggered = avg_retention_not_triggered = avg_convergence_not_triggered = 0
            
            ablation_summary_data.append({
                'Ablation_Type': ab_type,
                'Triggered_Count': len(triggered_items),
                'Not_Triggered_Count': len(not_triggered_items),
                'Final_FID_Triggered': f"{avg_final_fid_triggered:.4f}" if avg_final_fid_triggered > 0 else 'N/A',
                'Final_FID_Not_Triggered': f"{avg_final_fid_not_triggered:.4f}" if avg_final_fid_not_triggered > 0 else 'N/A',
                'Retention_Rate_Triggered': f"{avg_retention_triggered:.4f}" if avg_retention_triggered > 0 else 'N/A',
                'Retention_Rate_Not_Triggered': f"{avg_retention_not_triggered:.4f}" if avg_retention_not_triggered > 0 else 'N/A',
                'Convergence_Time_Triggered': f"{avg_convergence_triggered:.2f}" if avg_convergence_triggered > 0 else 'N/A',
                'Convergence_Time_Not_Triggered': f"{avg_convergence_not_triggered:.2f}" if avg_convergence_not_triggered > 0 else 'N/A'
            })
        
        df_ablation_summary = pd.DataFrame(ablation_summary_data)
        df_ablation_summary.to_csv(ablation_summary_file, index=False, encoding='utf-8')
        print(f"  ablationcomparisondatasaved: {ablation_summary_file}")
        
        plt.close()


# Convenience function
def create_experiment_logger(experiment_dir: str, config_dict: Dict = None) -> ExperimentLogger:
    """
    Convenience function: create and configure experiment logger
    
    Args:
        experiment_dir: Experiment directory
        config_dict: Configuration dictionary
        
    Returns:
        Configured experiment logger
    """
    logger = ExperimentLogger(experiment_dir)
    
    if config_dict:
        logger.log_config(config_dict)
    
    return logger