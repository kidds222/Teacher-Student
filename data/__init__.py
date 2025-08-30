"""
Data module initialization file
"""

from .datasets import (
    MNISTDataset,
    FashionMNISTDataset,
    LifelongDataset,
)

from .data_loaders import (
    get_data_loaders,
    get_lifelong_data_loaders,
    get_single_task_loader,
    calculate_dataset_stats,
    test_data_loaders,
)

__all__ = [
    # Dataset classes
    'MNISTDataset', 'FashionMNISTDataset', 'LifelongDataset',
    
    # Data loader functions
    'get_data_loaders',
    'get_lifelong_data_loaders', 
    'get_single_task_loader',
    'calculate_dataset_stats',
    'test_data_loaders',
] 