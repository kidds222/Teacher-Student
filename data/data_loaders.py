"""
Data loader utility functions
Provide unified data loading interfaces
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional

from .datasets import MNISTDataset, FashionMNISTDataset, LifelongDataset


def get_data_loaders(datasets_root: str = './datasets',
                    batch_size: int = 64,
                    num_workers: int = 4,
                    samples_per_task: Optional[int] = 10000) -> Dict[str, DataLoader]:
    """
    Get DataLoaders for all datasets
    
    Args:
        datasets_root: datasets root directory
        batch_size: batch size
        num_workers: number of data loading workers
        samples_per_task: per-task sample limit
    
    Returns:
        data_loaders: dict containing DataLoaders for all datasets
    """
    data_loaders = {}
    
    try:
        # MNIST loader
        print(f" Loading MNIST dataset...")
        mnist_dataset = MNISTDataset(root=datasets_root, train=True)
        if samples_per_task and len(mnist_dataset) > samples_per_task:
            indices = torch.randperm(len(mnist_dataset))[:samples_per_task]
            mnist_dataset = torch.utils.data.Subset(mnist_dataset, indices)
        
        data_loaders['mnist'] = DataLoader(
            mnist_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        print(f" MNIST DataLoader: {len(mnist_dataset)} samples")
        
    except Exception as e:
        print(f"  MNIST DataLoader creation failed: {e}")
        print(f" Please run: python scripts/download_datasets.py")
    
    try:
        # Fashion-MNIST loader
        print(f" Loading Fashion-MNIST dataset...")
        fashion_dataset = FashionMNISTDataset(root=datasets_root, train=True)
        if samples_per_task and len(fashion_dataset) > samples_per_task:
            indices = torch.randperm(len(fashion_dataset))[:samples_per_task]
            fashion_dataset = torch.utils.data.Subset(fashion_dataset, indices)
        
        data_loaders['fashion'] = DataLoader(
            fashion_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        print(f" Fashion-MNIST DataLoader: {len(fashion_dataset)} samples")
        
    except Exception as e:
        print(f"  Fashion-MNIST DataLoader creation failed: {e}")
        print(f" Please run: python scripts/download_datasets.py")
    
    return data_loaders


def get_lifelong_data_loaders(datasets_root: str = './datasets',
                             batch_size: int = 64,
                             num_workers: int = 4,
                             samples_per_task: Optional[int] = 10000) -> List[DataLoader]:
    """
    Get a sequence of DataLoaders for lifelong learning
    
    Args:
        datasets_root: datasets root directory
        batch_size: batch size
        num_workers: number of data loading workers
        samples_per_task: per-task sample limit
    
    Returns:
        loaders: list of DataLoaders ordered by task
    """
    loaders = []
    task_names = ['MNIST', 'Fashion-MNIST']  # two tasks only
    
    for task_id in range(len(task_names)):
        try:
            dataset = LifelongDataset(
                datasets_root=datasets_root,
                task_id=task_id,
                samples_per_task=samples_per_task
            )
            
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            
            loaders.append(loader)
            print(f" Task {task_id} ({dataset.get_task_name()}) DataLoader: {len(dataset)} samples")
            
        except Exception as e:
            print(f"  Task {task_id} DataLoader creation failed: {e}")
            # Add placeholder None loader
            loaders.append(None)
    
    return loaders


def get_single_task_loader(task_id: int,
                          datasets_root: str = './datasets',
                          batch_size: int = 64,
                          num_workers: int = 4,
                          samples_per_task: Optional[int] = 10000) -> Tuple[DataLoader, str]:
    """
    Get DataLoader for a single task
    
    Args:
        task_id: task id (0: MNIST, 1: Fashion-MNIST)
        datasets_root: datasets root directory
        batch_size: batch size
        num_workers: number of data loading workers
        samples_per_task: sample limit
    
    Returns:
        loader: DataLoader instance
        task_name: task name
    """
    dataset = LifelongDataset(
        datasets_root=datasets_root,
        task_id=task_id,
        samples_per_task=samples_per_task
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return loader, dataset.get_task_name()


def calculate_dataset_stats(data_loader: DataLoader) -> Dict[str, float]:
    """
    Compute dataset statistics
    
    Args:
        data_loader: DataLoader
    
    Returns:
        stats: dict of mean, std, and total sample count
    """
    print(" Computing dataset statistics...")
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for images, _ in data_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    stats = {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'total_samples': total_samples
    }
    
    print(f"   Mean: {mean.tolist()}")
    print(f"   Std: {std.tolist()}")
    print(f"   Total samples: {total_samples}")
    
    return stats


def test_data_loaders():
    """Test data loader utilities"""
    print(" Testing data loaders...")
    
    # Unified interface
    try:
        data_loaders = get_data_loaders(
            batch_size=32,
            samples_per_task=1000
        )
        
        for name, loader in data_loaders.items():
            if loader is not None:
                for images, labels in loader:
                    print(f" {name}: batch shape {images.shape}")
                    break
        
    except Exception as e:
        print(f" Unified interface test failed: {e}")
    
    # Lifelong interface
    try:
        lifelong_loaders = get_lifelong_data_loaders(
            batch_size=32,
            samples_per_task=1000
        )
        
        for i, loader in enumerate(lifelong_loaders):
            if loader is not None:
                for images, labels in loader:
                    print(f" Task {i}: batch shape {images.shape}")
                    break
        
    except Exception as e:
        print(f" Lifelong interface test failed: {e}")
    
    # Single task interface
    try:
        for task_id in range(2):  # test two tasks only
            loader, task_name = get_single_task_loader(
                task_id=task_id,
                batch_size=32,
                samples_per_task=500
            )
            
            for images, labels in loader:
                print(f" Single task {task_id} ({task_name}): batch shape {images.shape}")
                break
                
    except Exception as e:
        print(f" Single task interface test failed: {e}")
    
    print(" Data loader tests completed!")


if __name__ == "__main__":
    test_data_loaders() 