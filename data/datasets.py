"""
Dataset classes
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST
import numpy as np
import os
from PIL import Image
from typing import Tuple, Optional
import torchvision.datasets as datasets


def convert_to_rgb(x):
    """Convert grayscale images to RGB (serializable function)"""
    # If input is tensor, first convert to PIL image
    if isinstance(x, torch.Tensor):
        # If single channel tensor, convert to 3 channels
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        return x
    else:
        # If PIL image, convert to RGB
        return x.convert('RGB')


def check_dataset_exists(root: str, dataset_name: str) -> bool:
    """Check if dataset already exists"""
    dataset_path = os.path.join(root, dataset_name)
    return os.path.exists(dataset_path)


class MNISTDataset(Dataset):
    """MNIST dataset wrapper"""
    
    def __init__(self, root: str = './datasets', train: bool = True, download: bool = True):
        self.root = root
        self.train = train
        
        # Check if data already exists to avoid duplicate downloads
        data_exists = check_dataset_exists(root, 'MNIST')
        
        if data_exists:
            print(f" MNIST data already exists, skipping download")
            download = False  # If data exists, don't download
        else:
            print(f" MNIST data doesn't exist, will download to: {os.path.join(root, 'MNIST')}")
        
        # Use torchvision's MNIST dataset
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert to 3 channels
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class FashionMNISTDataset(Dataset):
    """Fashion-MNIST dataset wrapper"""
    
    def __init__(self, root: str = './datasets', train: bool = True, download: bool = True):
        self.root = root
        self.train = train
        
        # Check if data already exists to avoid duplicate downloads
        data_exists = check_dataset_exists(root, 'FashionMNIST')
        
        if data_exists:
            print(f" Fashion-MNIST data already exists, skipping download")
            download = False  # If data exists, don't download
        else:
            print(f" Fashion-MNIST data doesn't exist, will download to: {os.path.join(root, 'FashionMNIST')}")
        
        # Use torchvision's Fashion-MNIST dataset
        self.dataset = datasets.FashionMNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert to 3 channels
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class LifelongDataset(Dataset):
    """Lifelong learning dataset wrapper"""
    
    def __init__(self, 
                 datasets_root: str = './datasets',
                 task_id: int = 0,
                 samples_per_task: Optional[int] = 10000):
        self.datasets_root = datasets_root
        self.task_id = task_id
        self.samples_per_task = samples_per_task
        
        # Define task order - only includes two tasks
        self.task_names = ['MNIST', 'Fashion-MNIST']
        
        if task_id >= len(self.task_names):
            raise ValueError(f"task_id {task_id} out of range, maximum is {len(self.task_names)-1}")
        
        # Load dataset for corresponding task
        self.current_dataset = self._load_task_dataset(task_id)
        
        print(f" Loading task {task_id}: {self.task_names[task_id]}")
        print(f"   Dataset size: {len(self.current_dataset)}")
    
    def _load_task_dataset(self, task_id: int) -> Dataset:
        """Load corresponding dataset based on task ID"""
        if task_id == 0:  # MNIST
            dataset = MNISTDataset(root=self.datasets_root, train=True)
        elif task_id == 1:  # Fashion-MNIST
            dataset = FashionMNISTDataset(root=self.datasets_root, train=True)
        else:
            raise ValueError(f"Unsupported task_id: {task_id}")
        
        # Limit sample count
        if self.samples_per_task and len(dataset) > self.samples_per_task:
            # Create random subset
            indices = torch.randperm(len(dataset))[:self.samples_per_task]
            dataset = torch.utils.data.Subset(dataset, indices)
        
        return dataset
    
    def __len__(self):
        return len(self.current_dataset)
    
    def __getitem__(self, idx):
        return self.current_dataset[idx]
    
    def get_task_name(self) -> str:
        """Get current task name"""
        return self.task_names[self.task_id]


def test_datasets():
    """Test dataset loading"""
    print(" Testing dataset loading...")
    
    # Test MNIST
    try:
        mnist_dataset = MNISTDataset()  # Remove download=True, use smart check
        mnist_loader = DataLoader(mnist_dataset, batch_size=4, shuffle=True)
        
        for images, labels in mnist_loader:
            print(f" MNIST: images {images.shape}, labels {labels.shape}")
            print(f"   Image range: [{images.min():.3f}, {images.max():.3f}]")
            break
    except Exception as e:
        print(f" MNIST test failed: {e}")
    
    # Test Fashion-MNIST
    try:
        fashion_dataset = FashionMNISTDataset()  # Remove download=True, use smart check
        fashion_loader = DataLoader(fashion_dataset, batch_size=4, shuffle=True)
        
        for images, labels in fashion_loader:
            print(f" Fashion-MNIST: images {images.shape}, labels {labels.shape}")
            print(f"   Image range: [{images.min():.3f}, {images.max():.3f}]")
            break
    except Exception as e:
        print(f" Fashion-MNIST test failed: {e}")
    
    # Test lifelong learning dataset
    try:
        for task_id in range(2):  # Only test two tasks
            lifelong_dataset = LifelongDataset(task_id=task_id, samples_per_task=100)
            lifelong_loader = DataLoader(lifelong_dataset, batch_size=4, shuffle=True)
            
            for images, labels in lifelong_loader:
                print(f" Task {task_id} ({lifelong_dataset.get_task_name()}): "
                      f"images {images.shape}, labels {labels.shape}")
                break
    except Exception as e:
        print(f" Lifelong learning dataset test failed: {e}")
    
    print(" Dataset testing completed!")


if __name__ == "__main__":
    test_datasets() 