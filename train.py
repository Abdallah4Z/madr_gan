"""
Training script for Enhanced MADR-GAN

Implements Algorithm 1 from:
"Coverage-Guided Memory Replay for Mode-Diverse Generative Adversarial Networks"

Enhanced with --enhanced flag to enable all 7 improvements:
  FAISS indexing, progressive buffer, warm-start σ/EMA,
  auto τ_recall, multi-scale features, importance-weighted sampling.
"""

import os
import time
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm
import json
from pathlib import Path

from models.madr_gan import MADRGAN


def get_dataloader(dataset_name: str, batch_size: int, img_size: int = 64, num_workers: int = 4):
    """Load dataset and return dataloader"""

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    if dataset_name.lower() == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
    elif dataset_name.lower() == 'celeba':
        dataset = torchvision.datasets.CelebA(
            root='./data', split='train', download=True, transform=transform
        )
    elif dataset_name.lower() == 'stacked_mnist':
        dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True,
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize([0.5] * 3, [0.5] * 3)
            ])
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    return dataloader


def precompute_real_features(model: MADRGAN, dataloader: DataLoader,
                             device: torch.device, max_samples: int = 50000):
    """Precompute features for all real training images"""
    print("Precomputing features for real images...")

    all_features = []
    n_samples = 0

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Extracting features"):
            if n_samples >= max_samples:
                break
            images = images.to(device)
            features = model.feature_extractor(images)
            all_features.append(features.cpu())
            n_samples += images.size(0)

    all_features = torch.cat(all_features, dim=0)[:max_samples]
    model.real_features = all_features.to(device)

    # Enhancement 3: Warm-start bandwidth from real data
    model.bandwidth_estimator.warm_start(model.real_features)
    model.sigma = model.bandwidth_estimator.sigma

    print(f"Precomputed {len(model.real_features)} real image features (σ warm-start = {model.sigma:.4f})")


def save_checkpoint(model: MADRGAN, opt_g: optim.Optimizer, opt_d: optim.Optimizer,
                    iteration: int, save_dir: str):
    """Save model checkpoint"""
    checkpoint = {
        'iteration': iteration,
        'generator_state_dict': model.generator.state_dict(),
        'discriminator_state_dict': model.discriminator.state_dict(),
        'opt_g_state_dict': opt_g.state_dict(),
        'opt_d_state_dict': opt_d.state_dict(),
        'lambda_t': model.lambda_t,
        'sigma': model.sigma,
        'tau_recall': model.tau_recall,
        'buffer': model.buffer.buffer if len(model.buffer) > 0 else []
    }

    save_path = os.path.join(save_dir, f'checkpoint_iter_{iteration}.pt')
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def train(args):
    """Main training loop implementing Algorithm 1"""

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    # Save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Device
    if hasattr(args, 'gpu') and args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    dataloader = get_dataloader(args.dataset, args.batch_size, args.img_size, args.num_workers)
    print(f"Loaded {args.dataset} dataset with {len(dataloader.dataset)} images")

    # Resolve enhancement flags
    use_faiss = args.use_faiss or args.enhanced
    use_multiscale = args.use_multiscale or args.enhanced
    progressive_buffer = args.progressive_buffer or args.enhanced
    auto_tau = args.auto_tau or args.enhanced

    # Initialize model
    model = MADRGAN(
        latent_dim=args.latent_dim,
        img_channels=3,
        base_channels=args.base_channels,
        buffer_capacity=args.buffer_capacity,
        lambda_max=args.lambda_max,
        tau_recall=args.tau_recall,
        kappa=args.kappa,
        beta=args.beta,
        gamma_r1=args.gamma_r1,
        # Enhanced flags
        use_faiss=use_faiss,
        use_multiscale=use_multiscale,
        progressive_buffer=progressive_buffer,
        auto_tau=auto_tau,
    ).to(device)

    mode_label = "Enhanced" if args.enhanced else "Original"
    print(f"\nInitialized {mode_label} MADR-GAN")
    print(f"  Generator params:      {sum(p.numel() for p in model.generator.parameters()):,}")
    print(f"  Discriminator params:  {sum(p.numel() for p in model.discriminator.parameters()):,}")
    print(f"  FAISS:                 {'ON' if use_faiss else 'OFF'}")
    print(f"  Multi-scale features:  {'ON' if use_multiscale else 'OFF'}")
    print(f"  Progressive buffer:    {'ON' if progressive_buffer else 'OFF'}")
    print(f"  Auto τ_recall:         {'ON' if auto_tau else 'OFF'}")

    # Optimizers (TTUR)
    opt_g = optim.Adam(model.generator.parameters(), lr=args.lr_g, betas=(0.0, 0.99))
    opt_d = optim.Adam(model.discriminator.parameters(), lr=args.lr_d, betas=(0.0, 0.99))

    # Precompute real features
    precompute_real_features(model, dataloader, device, max_samples=args.max_real_samples)

    # Training loop
    print(f"\nStarting training for {args.num_iterations} iterations...")

    iteration = 0
    pbar = tqdm(total=args.num_iterations, desc="Training")
    fixed_z = torch.randn(64, args.latent_dim, device=device)
    metrics_log = []
    t_start = time.time()

    while iteration < args.num_iterations:
        for real_images, _ in dataloader:
            if iteration >= args.num_iterations:
                break

            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # ====================
            # Enhancement 2: Step progressive buffer capacity
            # ====================
            model.step_buffer(iteration)

            # ====================
            # Update Discriminator
            # ====================
            opt_d.zero_grad()
            z = torch.randn(batch_size, args.latent_dim, device=device)
            fake_images = model.generator(z)
            d_losses = model.discriminator_loss(real_images, fake_images)
            d_losses['loss_total'].backward()
            opt_d.step()

            # ====================
            # Update Memory Buffer (with importance weights for Enhancement 6)
            # ====================
            if iteration % args.buffer_update_interval == 0 and iteration > 0:
                with torch.no_grad():
                    z_buffer = torch.randn(batch_size, args.latent_dim, device=device)
                    gen_images_buffer = model.generator(z_buffer)
                    gen_features = model.feature_extractor(gen_images_buffer)

                    # Enhancement 6: compute importance weights for buffer insertion
                    importance_weights = None
                    if progressive_buffer and model.real_features is not None and len(model.buffer) > 0:
                        buffer_tensor = model.buffer.get_buffer()
                        if buffer_tensor is not None:
                            buffer_tensor = buffer_tensor.to(device)
                            # Low coverage in buffer → high importance
                            dists = torch.cdist(gen_features, buffer_tensor, p=2)
                            min_dists = dists.min(dim=1)[0]
                            importance_weights = min_dists / (min_dists.mean() + 1e-8)

                    model.buffer.update(gen_features, importance_weights)

            # ====================
            # Enhancement 3: Update Bandwidth (EMA)
            # ====================
            if iteration % 1000 == 0 and len(model.buffer) >= 10:
                model.update_bandwidth()
            elif iteration == args.buffer_update_interval and len(model.buffer) >= 10:
                model.update_bandwidth()

            # ====================
            # Update Adaptive Lambda (with Enhancement 4: auto τ)
            # ====================
            if iteration % 4 == 0:
                model.update_adaptive_lambda()

            # ====================
            # Update Generator
            # ====================
            opt_g.zero_grad()
            z = torch.randn(batch_size, args.latent_dim, device=device)
            fake_images = model.generator(z)
            fake_logits = model.discriminator(fake_images)
            g_losses = model.generator_loss(fake_logits, fake_images)
            g_losses['loss_total'].backward()
            opt_g.step()

            # ====================
            # Logging
            # ====================
            if iteration % args.log_interval == 0:
                recall = model.compute_online_recall()
                elapsed = time.time() - t_start

                metric_dict = {
                    'iteration': iteration,
                    'g_loss': g_losses['loss_total'].item(),
                    'g_adv': g_losses['loss_adv'].item(),
                    'g_cov': g_losses['loss_cov'].item(),
                    'd_loss': d_losses['loss_total'].item(),
                    'd_bce': d_losses['loss_bce'].item(),
                    'd_r1': d_losses['loss_r1'].item(),
                    'lambda_t': model.lambda_t,
                    'tau_recall': model.tau_recall,
                    'recall': recall,
                    'buffer_size': len(model.buffer),
                    'buffer_capacity': model.buffer.capacity if hasattr(model.buffer, 'capacity') else model.buffer.capacity,
                    'sigma': model.sigma if model.sigma is not None else 0.0,
                    'elapsed_sec': elapsed,
                }
                metrics_log.append(metric_dict)

                pbar.set_postfix({
                    'G': f"{g_losses['loss_total'].item():.3f}",
                    'D': f"{d_losses['loss_total'].item():.3f}",
                    'λ': f"{model.lambda_t:.3f}",
                    'τ': f"{model.tau_recall:.3f}",
                    'R': f"{recall:.3f}",
                    'buf': f"{len(model.buffer)}/{model.buffer.capacity if hasattr(model.buffer, 'capacity') else '?'}",
                })

            # ====================
            # Sample Generation
            # ====================
            if iteration % args.sample_interval == 0:
                model.generator.eval()
                with torch.no_grad():
                    samples = model.generator(fixed_z)
                    save_image(
                        samples,
                        os.path.join(args.output_dir, 'samples', f'iter_{iteration}.png'),
                        nrow=8, normalize=True, value_range=(-1, 1)
                    )
                model.generator.train()

            # ====================
            # Save Checkpoint
            # ====================
            if iteration % args.checkpoint_interval == 0 and iteration > 0:
                save_checkpoint(model, opt_g, opt_d, iteration,
                                os.path.join(args.output_dir, 'checkpoints'))
                with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
                    json.dump(metrics_log, f, indent=2)

            iteration += 1
            pbar.update(1)

    pbar.close()

    # Final save
    save_checkpoint(model, opt_g, opt_d, iteration,
                    os.path.join(args.output_dir, 'checkpoints'))
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_log, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nTraining complete! {iteration} iterations in {total_time:.1f}s")
    print(f"Results saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Enhanced MADR-GAN')

    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'celeba', 'stacked_mnist'])
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--max_real_samples', type=int, default=50000)

    # Model architecture
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--base_channels', type=int, default=128)

    # MADR-GAN hyperparameters
    parser.add_argument('--buffer_capacity', type=int, default=2048)
    parser.add_argument('--buffer_update_interval', type=int, default=100)
    parser.add_argument('--lambda_max', type=float, default=1.0)
    parser.add_argument('--tau_recall', type=float, default=0.80)
    parser.add_argument('--kappa', type=float, default=10.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma_r1', type=float, default=10.0)

    # Training
    parser.add_argument('--num_iterations', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr_g', type=float, default=1e-4)
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--num_workers', type=int, default=4)

    # Logging and checkpointing
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--sample_interval', type=int, default=1000)
    parser.add_argument('--checkpoint_interval', type=int, default=10000)
    parser.add_argument('--gpu', type=int, default=None)

    # --- Enhancement flags ---
    parser.add_argument('--enhanced', action='store_true',
                        help='Enable ALL 7 enhancements (shortcut)')
    parser.add_argument('--use_faiss', action='store_true',
                        help='Enhancement 1: FAISS-based coverage (10x faster)')
    parser.add_argument('--use_multiscale', action='store_true',
                        help='Enhancement 5: Multi-scale features (DINO+CLIP+ResNet)')
    parser.add_argument('--progressive_buffer', action='store_true',
                        help='Enhancement 2: Progressive buffer ramp (128→2048)')
    parser.add_argument('--auto_tau', action='store_true',
                        help='Enhancement 4: Auto-tuning τ_recall')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
