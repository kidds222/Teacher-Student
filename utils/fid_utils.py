

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg
from torch.utils.data import DataLoader
from torchvision import models, transforms
from typing import Optional, Tuple
import warnings


class InceptionV3FeatureExtractor(nn.Module):
    """
    Standard FID feature extractor using pretrained Inception v3
    """
    
    def __init__(self, normalize_input: bool = True, require_grad: bool = False):
        super(InceptionV3FeatureExtractor, self).__init__()
        
        # Load pretrained Inception v3
        inception = models.inception_v3(pretrained=True, transform_input=False)
        
        # Remove final FC/classifier, keep up to pooling
        self.blocks = nn.ModuleList([
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        ])
        
        # Image preprocessing
        self.normalize_input = normalize_input
        if normalize_input:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Freeze params
        if not require_grad:
            for param in self.parameters():
                param.requires_grad = False
        
        self.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning 2048-dim features
        
        Args:
            x: input images [N, 3, H, W], values in [-1, 1] or [0, 1]
            
        Returns:
            features: [N, 2048] feature vectors
        """
        # Ensure input in correct range
        if x.min() < 0:
            # Convert [-1, 1] to [0, 1]
            x = (x + 1) / 2
        
        # Resize to Inception input size (299, 299)
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalize
        if self.normalize_input:
            x = (x - self.mean) / self.std
        
        # Forward through blocks
        for block in self.blocks:
            x = block(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        return x


class FIDCalculator:
    """
    Standard FID calculator using Inception v3
    """
    
    def __init__(self, device: str = 'cuda', batch_size: int = 50):
        self.device = device
        self.batch_size = batch_size
        
        # Initialize Inception v3 feature extractor
        self.inception = InceptionV3FeatureExtractor().to(device)
        
        print(f" FID calculator initialized (device: {device})")
    
    def get_activations(self, images: torch.Tensor) -> np.ndarray:
        """
        Extract Inception features for images
        
        Args:
            images: image tensor [N, 3, H, W]
            
        Returns:
            activations: [N, 2048] numpy array
        """
        self.inception.eval()
        activations = []
        
        # Process in batches to save memory
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i + self.batch_size].to(self.device)
                batch_activations = self.inception(batch)
                activations.append(batch_activations.cpu().numpy())
        
        return np.concatenate(activations, axis=0)
    
    def get_activations_from_dataloader(self, dataloader: DataLoader, 
                                      max_samples: Optional[int] = None) -> np.ndarray:
        """
        Extract features from DataLoader
        
        Args:
            dataloader: dataloader
            max_samples: maximum number of samples
            
        Returns:
            activations: [N, 2048] numpy array
        """
        self.inception.eval()
        activations = []
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(dataloader):
                if max_samples and sample_count >= max_samples:
                    break
                
                images = images.to(self.device)
                batch_activations = self.inception(images)
                activations.append(batch_activations.cpu().numpy())
                
                sample_count += len(images)
                if batch_idx % 10 == 0:
                    print(f" Processed {sample_count} samples...")
        
        activations = np.concatenate(activations, axis=0)
        if max_samples:
            activations = activations[:max_samples]
        
        return activations
    
    def calculate_frechet_distance(self, mu1: np.ndarray, sigma1: np.ndarray,
                                 mu2: np.ndarray, sigma2: np.ndarray) -> float:
        """
        Compute Fréchet distance between two Gaussians
        
        Args:
            mu1, mu2: mean vectors
            sigma1, sigma2: covariance matrices
            
        Returns:
            FID score
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        assert mu1.shape == mu2.shape, "Mean vector shapes do not match"
        assert sigma1.shape == sigma2.shape, "Covariance matrix shapes do not match"
        
        diff = mu1 - mu2
        
        # Compute sqrt(sigma1 * sigma2)
        try:
            covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
            
            # Check None
            if covmean is None:
                print("WARNING: sqrtm returned None, applying numerical stabilization")
                offset = np.eye(sigma1.shape[0]) * 1e-6
                covmean, _ = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
                if covmean is None:
                    print(" Unable to compute matrix square root, returning inf")
                    return float('inf')
            
            # Numerical stability: handle non-finite
            if not np.isfinite(covmean).all():
                msg = ("fid calculation produces singular product; "
                       "adding %s to diagonal of cov estimates") % 1e-6
                print(msg)
                offset = np.eye(sigma1.shape[0]) * 1e-6
                covmean, _ = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
                if covmean is None:
                    return float('inf')
            
            # Remove imaginary component if present
            if np.iscomplexobj(covmean):
                if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                    print("  Imaginary part detected, removing imag component")
                covmean = covmean.real
            
            tr_covmean = np.trace(covmean)
            
            return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
            
        except Exception as e:
            print(f" FID calculation failed: {e}")
            return float('inf')
    
    def calculate_activation_statistics(self, activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute activation statistics (mean and covariance)
        
        Args:
            activations: [N, 2048] activations
            
        Returns:
            mu: mean vector
            sigma: covariance matrix
        """
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma
    
    def calculate_fid(self, real_activations: np.ndarray, 
                     fake_activations: np.ndarray) -> float:
        """
        Compute FID score
        
        Args:
            real_activations: real image activations
            fake_activations: generated image activations
            
        Returns:
            FID score
        """
        # Calculate activation statistics
        mu_real, sigma_real = self.calculate_activation_statistics(real_activations)
        mu_fake, sigma_fake = self.calculate_activation_statistics(fake_activations)
        
        # Calculate FID
        fid_score = self.calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        
        return float(fid_score)


def calculate_fid_from_dataloader_and_generator(
    real_dataloader: DataLoader,
    generator: nn.Module,
    device: str = 'cuda',
    num_fake_samples: int = 1000,
    max_real_samples: Optional[int] = None,
    z_dim: int = 256  # set to 256 to match TeacherGenerator
) -> float:
    """
    Compute FID score from dataloader and generator
    
    Args:
        real_dataloader: dataloader of real data
        generator: generator model
        device: compute device
        num_fake_samples: number of fake samples to generate
        max_real_samples: max number of real samples
        z_dim: noise vector dimension
        
    Returns:
        FID score
    """
    print(f" Starting FID computation...")
    print(f"   Max real samples: {max_real_samples or 'Unlimited'}")
    print(f"   Num fake samples: {num_fake_samples}")
    
    # Initialize FID calculator
    fid_calculator = FIDCalculator(device)
    
    # 1. Real activations
    print(f" Extracting activations for real images...")
    real_activations = fid_calculator.get_activations_from_dataloader(
        real_dataloader, max_real_samples
    )
    print(f" Real activations: {real_activations.shape}")
    
    # 2. Generate fakes and extract activations
    print(f" Generating fake images and extracting activations...")
    generator.eval()
    fake_images = []
    
    with torch.no_grad():
        for i in range(0, num_fake_samples, 64):  # Generate in batches
            batch_size = min(64, num_fake_samples - i)
            z = torch.randn(batch_size, z_dim, device=device)
            
            try:
                batch_fake = generator(z)
                fake_images.append(batch_fake.cpu())
            except Exception as e:
                print(f" Generator call failed: {e}")
                return float('inf')
    
    fake_images = torch.cat(fake_images, dim=0)
    print(f" Fake images: {fake_images.shape}")
    
    # Fake activations
    fake_activations = fid_calculator.get_activations(fake_images)
    print(f" Fake activations: {fake_activations.shape}")
    
    # 3. Compute FID
    print(f" Computing FID score...")
    fid_score = fid_calculator.calculate_fid(real_activations, fake_activations)
    
    print(f" FID computed: {fid_score:.2f}")
    print(f"   - Real samples: {len(real_activations)}")
    print(f"   - Fake samples: {len(fake_activations)}")
    
    return fid_score


def test_fid_calculator():
    
    print(" FID Calculator...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create test data
    real_images = torch.randn(50, 3, 32, 32, device=device)
    fake_images = torch.randn(50, 3, 32, 32, device=device)
    
    # Init calculator
    try:
        calculator = FIDCalculator(device)
        
        # Extract features
        print("Extracting features...")
        real_features = calculator.get_activations(real_images)
        fake_features = calculator.get_activations(fake_images)
        
        print(f" Features: real {real_features.shape}, fake {fake_features.shape}")
        
        # Compute FID
        fid_score = calculator.calculate_fid(real_features, fake_features)
        print(f" FID: {fid_score:.2f}")
        
        print(" Full FID calculator test passed!")
        
    except Exception as e:
        print(f" FID test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_fid_calculator() 