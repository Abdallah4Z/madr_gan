"""
Evaluation metrics for MADR-GAN

Implements:
- Fréchet Inception Distance (FID)
- Precision and Recall
- Density and Coverage
- Inception Score (IS)
- Mode Count (for Stacked MNIST)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3
from typing import Tuple, Optional
from tqdm import tqdm


class InceptionV3Features(nn.Module):
    """InceptionV3 feature extractor for FID calculation"""
    
    def __init__(self):
        super().__init__()
        inception = inception_v3(pretrained=True, transform_input=False)
        self.inception = inception
        self.inception.eval()
        
        # Freeze parameters
        for param in self.inception.parameters():
            param.requires_grad = False
        
        # Hook to extract features before final FC layer
        self.features = None
        def hook(module, input, output):
            self.features = output.detach()
        
        self.inception.fc.register_forward_hook(hook)
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Images in range [-1, 1] of shape (B, 3, H, W)
        Returns:
            Features of shape (B, 2048)
        """
        # Resize to 299x299 for Inception
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Inception expects [0, 1]
        x = (x + 1) / 2
        
        # Forward pass
        self.inception(x)
        
        return self.features.squeeze()


def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                               mu2: np.ndarray, sigma2: np.ndarray,
                               eps: float = 1e-6) -> float:
    """
    Calculate Fréchet distance between two Gaussians
    
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


@torch.no_grad()
def extract_inception_features(images: torch.Tensor, model: InceptionV3Features,
                               batch_size: int = 50) -> np.ndarray:
    """Extract Inception features from images"""
    device = next(model.parameters()).device
    
    features_list = []
    n_batches = (len(images) + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, len(images))
        batch = images[start:end].to(device)
        
        features = model(batch)
        features_list.append(features.cpu().numpy())
    
    return np.concatenate(features_list, axis=0)


def calculate_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """
    Calculate FID score
    
    Args:
        real_features: Real image features (N, 2048)
        fake_features: Generated image features (M, 2048)
    Returns:
        FID score (lower is better)
    """
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    
    return float(fid)


def calculate_precision_recall(real_features: np.ndarray, fake_features: np.ndarray,
                               k: int = 3) -> Tuple[float, float]:
    """
    Calculate Precision and Recall using k-NN
    
    Args:
        real_features: Real image features (N, D)
        fake_features: Fake image features (M, D)  
        k: Number of nearest neighbors
    Returns:
        (precision, recall) tuple
    """
    real_features = torch.from_numpy(real_features).float()
    fake_features = torch.from_numpy(fake_features).float()
    
    # Compute pairwise distances
    # Real k-NN ball radii
    real_dists = torch.cdist(real_features, real_features, p=2)
    real_knn_dists = torch.topk(real_dists, k + 1, largest=False, dim=1)[0][:, k]
    
    # Fake k-NN ball radii
    fake_dists = torch.cdist(fake_features, fake_features, p=2)
    fake_knn_dists = torch.topk(fake_dists, k + 1, largest=False, dim=1)[0][:, k]
    
    # Precision: fraction of fake samples inside real manifold
    # For each fake sample, check if it falls within any real sample's k-NN ball
    dists_fake_to_real = torch.cdist(fake_features, real_features, p=2)
    precision = 0.0
    for i in range(len(fake_features)):
        if (dists_fake_to_real[i] <= real_knn_dists).any():
            precision += 1.0
    precision /= len(fake_features)
    
    # Recall: fraction of real samples covered by fake manifold
    # For each real sample, check if it's within any fake sample's k-NN ball
    dists_real_to_fake = torch.cdist(real_features, fake_features, p=2)
    recall = 0.0
    for i in range(len(real_features)):
        if (dists_real_to_fake[i] <= fake_knn_dists).any():
            recall += 1.0
    recall /= len(real_features)
    
    return float(precision), float(recall)


def calculate_density_coverage(real_features: np.ndarray, fake_features: np.ndarray,
                               k: int = 3) -> Tuple[float, float]:
    """
    Calculate Density and Coverage (more robust than Precision/Recall)
    
    Args:
        real_features: Real image features (N, D)
        fake_features: Fake image features (M, D)
        k: Number of nearest neighbors
    Returns:
        (density, coverage) tuple
    """
    real_features = torch.from_numpy(real_features).float()
    fake_features = torch.from_numpy(fake_features).float()
    
    # Compute k-NN distances within real data
    real_dists = torch.cdist(real_features, real_features, p=2)
    real_knn_dists = torch.topk(real_dists, k + 1, largest=False, dim=1)[0][:, k]
    
    # For density: count how many fake samples fall into each real sample's k-NN ball
    dists_fake_to_real = torch.cdist(fake_features, real_features, p=2)
    
    density_list = []
    for i in range(len(real_features)):
        n_fake_in_ball = (dists_fake_to_real[:, i] <= real_knn_dists[i]).sum().item()
        density_list.append(n_fake_in_ball)
    
    # Density: average number of fake samples per real sample's k-NN ball
    density = np.mean(density_list) / k
    
    # Coverage: fraction of real samples that have at least one fake sample in k-NN ball
    coverage = np.mean([1.0 if d > 0 else 0.0 for d in density_list])
    
    return float(density), float(coverage)


def calculate_inception_score(fake_features: np.ndarray, splits: int = 10) -> Tuple[float, float]:
    """
    Calculate Inception Score
    
    Note: This is a simplified version that works with features.
    Proper IS requires the full Inception model output (class probabilities).
    
    Args:
        fake_features: Generated image features (N, 2048)
        splits: Number of splits for std calculation
    Returns:
        (mean_is, std_is) tuple
    """
    # This is a placeholder - proper IS needs softmax outputs
    # Would need to modify InceptionV3Features to return softmax outputs
    
    # For now, return placeholder
    return 0.0, 0.0


def count_modes_stacked_mnist(images: torch.Tensor, classifier: nn.Module) -> int:
    """
    Count modes for Stacked MNIST (maximum 777 modes)
    
    Args:
        images: Generated images (N, 3, H, W) where each channel is a digit
        classifier: Pretrained MNIST digit classifier
    Returns:
        Number of unique mode combinations
    """
    classifier.eval()
    device = next(classifier.parameters()).device
    
    mode_set = set()
    
    with torch.no_grad():
        for img in images:
            # Extract each channel (digit)
            digits = []
            for c in range(3):
                channel = img[c:c+1].unsqueeze(0).to(device)
                # Convert to grayscale if needed
                logits = classifier(channel)
                digit = logits.argmax(dim=1).item()
                digits.append(digit)
            
            mode_tuple = tuple(digits)
            mode_set.add(mode_tuple)
    
    return len(mode_set)


class GANEvaluator:
    """Comprehensive GAN evaluation"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.inception_model = InceptionV3Features().to(device)
        self.inception_model.eval()
    
    @torch.no_grad()
    def evaluate(self, real_images: torch.Tensor, fake_images: torch.Tensor,
                batch_size: int = 50, k: int = 3) -> dict:
        """
        Comprehensive evaluation
        
        Args:
            real_images: Real images (N, C, H, W) in range [-1, 1]
            fake_images: Generated images (M, C, H, W) in range [-1, 1]
            batch_size: Batch size for feature extraction
            k: Number of nearest neighbors for precision/recall
        Returns:
            Dictionary with all metrics
        """
        print("Extracting Inception features for real images...")
        real_features = extract_inception_features(
            real_images, self.inception_model, batch_size
        )
        
        print("Extracting Inception features for fake images...")
        fake_features = extract_inception_features(
            fake_images, self.inception_model, batch_size
        )
        
        print("Calculating FID...")
        fid = calculate_fid(real_features, fake_features)
        
        print("Calculating Precision and Recall...")
        precision, recall = calculate_precision_recall(real_features, fake_features, k=k)
        
        print("Calculating Density and Coverage...")
        density, coverage = calculate_density_coverage(real_features, fake_features, k=k)
        
        metrics = {
            'fid': fid,
            'precision': precision,
            'recall': recall,
            'density': density,
            'coverage': coverage
        }
        
        return metrics
    
    def print_metrics(self, metrics: dict):
        """Pretty print metrics"""
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        print(f"FID:       {metrics['fid']:.3f}  (lower is better)")
        print(f"Precision: {metrics['precision']:.3f}  (fidelity)")
        print(f"Recall:    {metrics['recall']:.3f}  (diversity)")
        print(f"Density:   {metrics['density']:.3f}")
        print(f"Coverage:  {metrics['coverage']:.3f}  (mode coverage)")
        print("="*50 + "\n")


def evaluate_checkpoint(checkpoint_path: str, real_images: torch.Tensor,
                       num_samples: int = 50000, batch_size: int = 128,
                       latent_dim: int = 128, device: torch.device = None):
    """
    Evaluate a saved checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        real_images: Real images for comparison (N, C, H, W)
        num_samples: Number of fake samples to generate
        batch_size: Batch size for generation
        latent_dim: Latent dimension
        device: Device to use
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    from models.madr_gan import MADRGAN
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model
    model = MADRGAN(latent_dim=latent_dim).to(device)
    model.generator.load_state_dict(checkpoint['generator_state_dict'])
    model.generator.eval()
    
    # Generate samples
    print(f"Generating {num_samples} samples...")
    fake_images_list = []
    
    with torch.no_grad():
        for _ in tqdm(range(0, num_samples, batch_size)):
            z = torch.randn(batch_size, latent_dim, device=device)
            fake = model.generator(z)
            fake_images_list.append(fake.cpu())
    
    fake_images = torch.cat(fake_images_list, dim=0)[:num_samples]
    
    # Evaluate
    evaluator = GANEvaluator(device)
    metrics = evaluator.evaluate(real_images, fake_images)
    evaluator.print_metrics(metrics)
    
    return metrics


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate MADR-GAN checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'celeba', 'stacked_mnist'])
    parser.add_argument('--num_samples', type=int, default=50000,
                       help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=128)
    
    args = parser.parse_args()
    
    # Load real images
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif args.dataset == 'celeba':
        dataset = datasets.CelebA(root='./data', split='train', download=True, transform=transform)
    else:
        dataset = datasets.MNIST(root='./data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.Resize(64),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                    transforms.Normalize([0.5]*3, [0.5]*3)
                                ]))
    
    # Sample real images
    real_images = torch.stack([dataset[i][0] for i in range(min(50000, len(dataset)))])
    
    # Evaluate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = evaluate_checkpoint(
        args.checkpoint,
        real_images,
        args.num_samples,
        args.batch_size,
        args.latent_dim,
        device
    )
