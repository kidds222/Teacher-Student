"""
Utilities module - Advanced Dynamic Teaching System
Includes tools for training, image processing, FID computation, and experiment logging
"""

from .training_utils import (
    MetricsTracker,
    TrainingTimer,
    create_exp_directory,
    log_experiment_info,
    save_checkpoint,
    print_model_info
)

from .image_utils import (
    save_samples,
    create_image_grid_plot,
    tensor_to_pil,
    pil_to_tensor,
    test_image_utils
)

from .fid_utils import (
    InceptionV3FeatureExtractor,
    FIDCalculator,
    calculate_fid_from_dataloader_and_generator,
    test_fid_calculator
)

from .experiment_logger import (
    ExperimentLogger,
    create_experiment_logger
)

__all__ = [
    # Training utilities
    'MetricsTracker', 'TrainingTimer', 'create_exp_directory', 
    'log_experiment_info', 'save_checkpoint', 'print_model_info',
    
    # Image utilities
    'save_samples', 'create_image_grid_plot', 'tensor_to_pil', 'pil_to_tensor',
    
    # FID utilities (full)
    'InceptionV3FeatureExtractor', 'FIDCalculator', 'calculate_fid_from_dataloader_and_generator',
    
    # Experiment logger (full)
    'ExperimentLogger', 'create_experiment_logger',
    
    # Test functions
    'test_image_utils', 'test_fid_calculator'
] 