#!/usr/bin/env python3
"""
Dataset download script
Automatically download MNIST and Fashion-MNIST datasets
"""

import os
import sys
import argparse
import requests
from pathlib import Path
import torch
import torchvision
from torchvision.datasets import MNIST, FashionMNIST
import time
import urllib.request
import urllib.error

def setup_download_environment():
    """Setup download environment with timeout and retry"""
    # Set longer timeout
    import socket
    socket.setdefaulttimeout(60)  # 60 second timeout
    
    # Create custom opener
    opener = urllib.request.build_opener()
    opener.addheaders = [
        ('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'),
        ('Accept', '*/*'),
        ('Connection', 'keep-alive')
    ]
    urllib.request.install_opener(opener)

def download_mnist(root_dir):
    """Download MNIST dataset"""
    print(" Downloading MNIST dataset...")
    try:
        # Setup download environment
        setup_download_environment()
        
        # Training set
        print("    Downloading training set...")
        MNIST(root=root_dir, train=True, download=True)
        print("    MNIST training set download completed")
        
        # Test set
        print("    Downloading test set...")
        MNIST(root=root_dir, train=False, download=True)
        print("    MNIST test set download completed")
        
        return True
    except Exception as e:
        print(f" MNIST download failed: {e}")
        print(f" Please check network connection or try manual download")
        return False

def download_fashion_mnist(root_dir):
    """Download Fashion-MNIST dataset"""
    print(" Downloading Fashion-MNIST dataset...")
    try:
        # Setup download environment
        setup_download_environment()
        
        # Training set
        print("    Downloading training set...")
        FashionMNIST(root=root_dir, train=True, download=True)
        print("    Fashion-MNIST training set download completed")
        
        # Test set
        print("    Downloading test set...")
        FashionMNIST(root=root_dir, train=False, download=True)
        print("    Fashion-MNIST test set download completed")
        
        return True
    except Exception as e:
        print(f" Fashion-MNIST download failed: {e}")
        print(f" Please check network connection or try manual download")
        return False

def verify_datasets(root_dir):
    """Verify dataset integrity"""
    print("\n Verifying dataset integrity...")
    print(f" Dataset directory: {os.path.abspath(root_dir)}")
    
    success = True
    
    # Verify MNIST
    mnist_path = os.path.join(root_dir, 'MNIST')
    if os.path.exists(mnist_path):
        print(" MNIST directory exists")
        # Check specific files
        train_files = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte']
        test_files = ['t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']
        raw_folder = os.path.join(mnist_path, 'raw')
        
        if os.path.exists(raw_folder):
            for f in train_files + test_files:
                if os.path.exists(os.path.join(raw_folder, f)) or os.path.exists(os.path.join(raw_folder, f + '.gz')):
                    print(f"    {f}")
                else:
                    print(f"    {f} missing")
    else:
        print(" MNIST directory doesn't exist")
        success = False
    
    # Verify Fashion-MNIST
    fashion_path = os.path.join(root_dir, 'FashionMNIST')
    if os.path.exists(fashion_path):
        print(" Fashion-MNIST directory exists")
    else:
        print(" Fashion-MNIST directory doesn't exist")
        success = False
    
    # Provide path suggestions
    if not success:
        print(f"\n Solution suggestions:")
        print(f"1. Ensure running script in correct directory")
        print(f"2. Dataset will download to: {os.path.abspath(root_dir)}")
        print(f"3. Training program expects data in: ./datasets")
        print(f"4. If paths don't match, check datasets_root setting in config/experiment_config.py")
    
    return success

def main():
    parser = argparse.ArgumentParser(description='Download project datasets')
    parser.add_argument('--root', type=str, default=None, 
                       help='Dataset root directory (default: auto-detect project root)')
    parser.add_argument('--dataset', type=str, choices=['all', 'mnist', 'fashion'], 
                       default='all', help='Dataset to download')
    parser.add_argument('--verify-only', action='store_true', 
                       help='Only verify datasets, don\'t download')
    
    args = parser.parse_args()
    
    # Auto-detect project root directory
    if args.root is None:
        # Get script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Project root is parent of script directory
        project_root = os.path.dirname(script_dir)
        root_dir = os.path.join(project_root, 'datasets')
    else:
        root_dir = os.path.abspath(args.root)
    
    # Create dataset directory
    os.makedirs(root_dir, exist_ok=True)
    print(f" Dataset directory: {root_dir}")
    print(f" Current working directory: {os.getcwd()}")
    print(f" Project root directory: {os.path.dirname(root_dir)}")
    
    if args.verify_only:
        success = verify_datasets(root_dir)
        if success:
            print("\n All datasets verified successfully!")
        else:
            print("\n Dataset verification failed, please re-download")
        return
    
    # Download datasets
    results = []
    
    if args.dataset in ['all', 'mnist']:
        results.append(('MNIST', download_mnist(root_dir)))
    
    if args.dataset in ['all', 'fashion']:
        results.append(('Fashion-MNIST', download_fashion_mnist(root_dir)))
    
    # Summary of results
    print("\n Download results summary:")
    all_success = True
    for dataset_name, success in results:
        status = " Success" if success else " Failed"
        print(f"  {dataset_name}: {status}")
        if not success:
            all_success = False
    
    if all_success:
        print("\n All datasets downloaded successfully!")
        print(f" Dataset location: {root_dir}")
        print(f" When training, ensure running program from project root directory")
        verify_datasets(root_dir)
    else:
        print("\n Some datasets failed to download, please check network connection")
        print("\n Alternative solutions:")
        print("1. Check network connection")
        print("2. Try using VPN")
        print("3. Manually download datasets to the following directory:")
        print(f"   {root_dir}")
        print("4. Or try running: python -c \"import torchvision; torchvision.datasets.MNIST('./datasets', download=True)\"")

if __name__ == "__main__":
    main() 