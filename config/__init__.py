"""
Configuration module 

"""

# Three core configuration modules
from .teacher_config import TeacherConfig, TeacherConfigHigh
from .student_config import StudentConfig, StudentConfigBalanced, StudentConfigHigh
from .experiment_config import ExperimentConfig, ExperimentConfigLong

__all__ = [
    # Teacher configurations
    'TeacherConfig', 'TeacherConfigHigh',
    
    # Student configurations
    'StudentConfig', 'StudentConfigBalanced', 'StudentConfigHigh',
    
    # Experiment configurations
    'ExperimentConfig', 'ExperimentConfigLong',
] 