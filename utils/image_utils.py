"""
Image processing utilities
Provide image saving and processing functions
"""

import torch
import torchvision.utils as vutils
import os
from PIL import Image
import numpy as np


def save_samples(images: torch.Tensor, 
                filepath: str, 
                nrow: int = 8, 
                normalize: bool = True,
                value_range: tuple = (-1, 1)):
    """
    Save image samples
    
    Args:
        images: image tensor (N, C, H, W)
        filepath: output path
        nrow: number of images per row
        normalize: whether to normalize
        value_range: pixel value range
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Ensure processing on CPU
    if images.is_cuda:
        images = images.cpu()
    
    # Use torchvision to save image grid
    vutils.save_image(
        images, 
        filepath, 
        nrow=nrow, 
        normalize=normalize, 
        value_range=value_range,
        padding=2
    )


def create_image_grid_plot(images: torch.Tensor, 
                          title: str = "Generated Images",
                          nrow: int = 8,
                          figsize: tuple = (12, 8)):
    """
    Create an image grid (for display or saving)
    
    Args:
        images: image tensor
        title: title
        nrow: number of images per row
        figsize: figure size
    
    Returns:
        PIL Image: generated grid image
    """
    try:
        import matplotlib.pyplot as plt
        
        # Create image grid
        grid = vutils.make_grid(
            images.cpu(),
            nrow=nrow,
            normalize=True,
            value_range=(-1, 1),
            padding=2
        )
        
        # Convert to numpy array
        grid_np = grid.numpy().transpose(1, 2, 0)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(grid_np)
        ax.set_title(title)
        ax.axis('off')
        
        # Convert to PIL image
        fig.canvas.draw()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(),
                             fig.canvas.tostring_rgb())
        plt.close(fig)
        
        return img
        
    except ImportError:
        print("  matplotlib is not installed, using simplified version")
        
        # Simplified version: directly use torchvision to create grid
        grid = vutils.make_grid(
            images.cpu(),
            nrow=nrow,
            normalize=True,
            value_range=(-1, 1),
            padding=2
        )
        
        # Convert to PIL Image
        grid_np = grid.numpy().transpose(1, 2, 0)
        grid_np = (grid_np * 255).astype(np.uint8)
        return Image.fromarray(grid_np)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert tensor to PIL Image
    
    Args:
        tensor: image tensor (C, H, W) or (1, C, H, W)
    
    Returns:
        PIL Image
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Normalize to [0, 1]
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    
    # To numpy
    np_img = tensor.cpu().numpy().transpose(1, 2, 0)
    np_img = (np_img * 255).astype(np.uint8)
    
    return Image.fromarray(np_img)


def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image to tensor
    
    Args:
        pil_img: PIL Image
    
    Returns:
        image tensor (C, H, W)
    """
    np_img = np.array(pil_img).astype(np.float32) / 255.0
    
    if len(np_img.shape) == 2:  # grayscale
        np_img = np.expand_dims(np_img, axis=2)
    
    tensor = torch.from_numpy(np_img.transpose(2, 0, 1))
    
    # Normalize to [-1, 1]
    tensor = tensor * 2 - 1
    
    return tensor


def test_image_utils():
    """Test image utility functions"""
    print(" Testing image utilities...")
    
    # Create test images
    test_images = torch.randn(16, 3, 32, 32)
    
    # Test saving images
    try:
        test_dir = "./test_images"
        os.makedirs(test_dir, exist_ok=True)
        
        save_samples(test_images, os.path.join(test_dir, "test_grid.png"))
        print(" Image saved successfully")
        
        # Clean up test files
        import shutil
        shutil.rmtree(test_dir)
        
    except Exception as e:
        print(f" Image save test failed: {e}")
    
    # Test conversion functions
    try:
        single_img = test_images[0]  # (C, H, W)
        pil_img = tensor_to_pil(single_img)
        tensor_back = pil_to_tensor(pil_img)
        
        print(f" Conversion test: {single_img.shape} -> PIL -> {tensor_back.shape}")
        
    except Exception as e:
        print(f" Conversion test failed: {e}")
    
    print(" Image utilities test completed!")


if __name__ == "__main__":
    test_image_utils() 