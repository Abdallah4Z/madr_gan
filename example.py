"""
Quick Start Examples for Enhanced MADR-GAN

Demonstrates:
  - Original MADR-GAN (backward compat)
  - Enhanced MADR-GAN with all 7 improvements
  - Individual component demonstrations
"""

import torch
import torch.optim as optim
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from models.madr_gan import (
    MADRGAN,
    ReservoirBuffer,
    ProgressiveReservoirBuffer,
    DINOFeatureExtractor,
    AdaptiveTauScheduler,
    EMABandwidthEstimator,
    FAISSCoverageIndex,
)


def minimal_example(gpu_id=None):
    """Minimal working example of original MADR-GAN (backward compat)"""

    print("=" * 60)
    print("MADR-GAN Minimal Example (Original)")
    print("=" * 60)

    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    latent_dim = 128
    batch_size = 64
    img_size = 64
    num_iterations = 1000

    print("\nLoading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    # Original model — no enhancements
    print("\nInitializing original MADR-GAN...")
    model = MADRGAN(
        latent_dim=latent_dim, img_channels=3, buffer_capacity=512,
        lambda_max=1.0, tau_recall=0.80,
        use_faiss=False, use_multiscale=False, progressive_buffer=False, auto_tau=False,
    ).to(device)

    opt_g = optim.Adam(model.generator.parameters(), lr=1e-4, betas=(0.0, 0.99))
    opt_d = optim.Adam(model.discriminator.parameters(), lr=2e-4, betas=(0.0, 0.99))

    # Precompute features
    print("\nPrecomputing DINO features for real images...")
    real_images_list = []
    for i, (imgs, _) in enumerate(dataloader):
        real_images_list.append(imgs)
        if i >= 5:
            break
    real_images = torch.cat(real_images_list, dim=0).to(device)
    with torch.no_grad():
        model.real_features = model.feature_extractor(real_images)
    model.bandwidth_estimator.warm_start(model.real_features)
    model.sigma = model.bandwidth_estimator.sigma
    print(f"Extracted {len(model.real_features)} features (σ = {model.sigma:.4f})")

    # Training loop
    print(f"\nTraining for {num_iterations} iterations...")
    data_iter = iter(dataloader)

    for iteration in range(num_iterations):
        try:
            real_images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            real_images, _ = next(data_iter)
        real_images = real_images.to(device)

        # D step
        opt_d.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_images = model.generator(z)
        d_losses = model.discriminator_loss(real_images, fake_images)
        d_losses['loss_total'].backward()
        opt_d.step()

        # Buffer update
        if iteration % 50 == 0:
            with torch.no_grad():
                z_buf = torch.randn(batch_size, latent_dim, device=device)
                gen_features = model.feature_extractor(model.generator(z_buf))
                model.buffer.update(gen_features)

        if iteration % 200 == 0 and len(model.buffer) > 10:
            model.update_bandwidth()
        if iteration % 4 == 0:
            model.update_adaptive_lambda()

        # G step
        opt_g.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_images = model.generator(z)
        fake_logits = model.discriminator(fake_images)
        g_losses = model.generator_loss(fake_logits, fake_images)
        g_losses['loss_total'].backward()
        opt_g.step()

        if iteration % 100 == 0:
            recall = model.compute_online_recall()
            print(f"Iter {iteration:4d} | G: {g_losses['loss_total'].item():.3f} | "
                  f"D: {d_losses['loss_total'].item():.3f} | λ: {model.lambda_t:.3f} | "
                  f"Recall: {recall:.3f} | Buffer: {len(model.buffer)}")

    print(f"\nFinal: Buffer={len(model.buffer)}, λ={model.lambda_t:.3f}, "
          f"Recall={model.compute_online_recall():.3f}")


def enhanced_example(gpu_id=None):
    """Enhanced MADR-GAN with all 7 improvements"""

    print("=" * 60)
    print("Enhanced MADR-GAN Example (All 7 Improvements)")
    print("=" * 60)

    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    latent_dim = 128
    batch_size = 64
    img_size = 64
    num_iterations = 1000

    print("\nLoading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    # Enhanced model — all improvements ON
    print("\nInitializing Enhanced MADR-GAN...")
    print("  ✅ FAISS indexing (10x faster coverage)")
    print("  ✅ Progressive buffer (128→2048 ramp)")
    print("  ✅ Warm-start σ + EMA smoothing")
    print("  ✅ Auto-tuning τ_recall")
    print("  ✅ Multi-scale features (DINO + ResNet)")  # CLIP optional
    print("  ✅ Importance-weighted reservoir sampling")

    model = MADRGAN(
        latent_dim=latent_dim, img_channels=3, buffer_capacity=512,
        lambda_max=1.0, tau_recall=0.80,
        # All enhancements ON
        use_faiss=True,
        use_multiscale=True,
        progressive_buffer=True,
        auto_tau=True,
        use_clip=False,      # Set True if open_clip is installed
        use_resnet=True,
        initial_buffer_capacity=64,
        buffer_ramp_interval=200,
    ).to(device)

    opt_g = optim.Adam(model.generator.parameters(), lr=1e-4, betas=(0.0, 0.99))
    opt_d = optim.Adam(model.discriminator.parameters(), lr=2e-4, betas=(0.0, 0.99))

    # Precompute features
    print("\nPrecomputing multi-scale features for real images...")
    real_images_list = []
    for i, (imgs, _) in enumerate(dataloader):
        real_images_list.append(imgs)
        if i >= 5:
            break
    real_images = torch.cat(real_images_list, dim=0).to(device)
    model.precompute_real_features(real_images)
    print(f"Extracted {len(model.real_features)} features (σ warm-start = {model.sigma:.4f})")

    # Training loop
    print(f"\nTraining for {num_iterations} iterations...")
    data_iter = iter(dataloader)

    for iteration in range(num_iterations):
        try:
            real_images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            real_images, _ = next(data_iter)
        real_images = real_images.to(device)

        # Enhancement 2: Step progressive buffer
        model.step_buffer(iteration)

        # D step
        opt_d.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_images = model.generator(z)
        d_losses = model.discriminator_loss(real_images, fake_images)
        d_losses['loss_total'].backward()
        opt_d.step()

        # Buffer update with importance weighting (Enhancement 6)
        if iteration % 50 == 0:
            with torch.no_grad():
                z_buf = torch.randn(batch_size, latent_dim, device=device)
                gen_features = model.feature_extractor(model.generator(z_buf))
                # Compute importance weights from buffer distance
                importance_weights = None
                if len(model.buffer) > 0:
                    buf = model.buffer.get_buffer()
                    if buf is not None:
                        buf = buf.to(device)
                        dists = torch.cdist(gen_features, buf, p=2)
                        min_dists = dists.min(dim=1)[0]
                        importance_weights = min_dists / (min_dists.mean() + 1e-8)
                model.buffer.update(gen_features, importance_weights)

        if iteration % 200 == 0 and len(model.buffer) > 10:
            model.update_bandwidth()
        if iteration % 4 == 0:
            model.update_adaptive_lambda()

        # G step
        opt_g.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_images = model.generator(z)
        fake_logits = model.discriminator(fake_images)
        g_losses = model.generator_loss(fake_logits, fake_images)
        g_losses['loss_total'].backward()
        opt_g.step()

        if iteration % 100 == 0:
            recall = model.compute_online_recall()
            buf_cap = model.buffer.capacity if hasattr(model.buffer, 'capacity') else '?'
            print(f"Iter {iteration:4d} | G: {g_losses['loss_total'].item():.3f} | "
                  f"D: {d_losses['loss_total'].item():.3f} | λ: {model.lambda_t:.3f} | "
                  f"τ: {model.tau_recall:.3f} | Recall: {recall:.3f} | "
                  f"Buffer: {len(model.buffer)}/{buf_cap} | σ: {model.sigma:.4f}")

    print(f"\nFinal: Buffer={len(model.buffer)}, λ={model.lambda_t:.3f}, "
          f"τ={model.tau_recall:.3f}, Recall={model.compute_online_recall():.3f}, σ={model.sigma:.4f}")


def demonstrate_components():
    """Demonstrate key enhanced components individually"""

    print("\n" + "=" * 60)
    print("Demonstrating Enhanced MADR-GAN Components")
    print("=" * 60)

    # 1. Progressive Reservoir Buffer
    print("\n1. Progressive Reservoir Buffer (Enhancement 2 + 6)")
    print("-" * 60)

    buffer = ProgressiveReservoirBuffer(max_capacity=256, initial_capacity=16, ramp_interval=3, feature_dim=4)
    print(f"Initial capacity: {buffer.capacity}")

    for step in range(15):
        buffer.step_capacity(step)
        feat = torch.randn(2, 4)
        weights = torch.tensor([1.5, 0.5])  # First sample has higher importance
        buffer.update(feat, importance_weights=weights)
        if step % 3 == 0:
            print(f"  Step {step:2d}: capacity={buffer.capacity}, size={len(buffer)}")

    # 2. EMA Bandwidth Estimator
    print("\n2. EMA Bandwidth Estimator (Enhancement 3)")
    print("-" * 60)

    bw = EMABandwidthEstimator(ema_alpha=0.3, sigma_min=0.1)
    print(f"Initial σ: {bw.sigma:.4f}")

    # Warm-start from real data
    real_feats = torch.randn(100, 64)
    bw.warm_start(real_feats)
    print(f"After warm-start: σ = {bw.sigma:.4f}")

    # EMA updates
    for i in range(5):
        buffer_feats = torch.randn(50, 64) * (0.5 + i * 0.3)
        new_sigma = bw.update(buffer_feats)
        print(f"  EMA update {i+1}: σ = {new_sigma:.4f}")

    # 3. Adaptive τ_recall Scheduler
    print("\n3. Adaptive τ_recall Scheduler (Enhancement 4)")
    print("-" * 60)

    scheduler = AdaptiveTauScheduler(initial_tau=0.80, window_size=20, warmup_steps=10)
    print(f"Initial τ: {scheduler.tau:.3f}")

    import numpy as np
    for step in range(60):
        # Simulate recall climbing from 0.3 to 0.8
        recall = 0.3 + 0.5 * min(1.0, step / 50) + np.random.normal(0, 0.02)
        tau = scheduler.step(recall)
        if step % 10 == 0:
            print(f"  Step {step:2d}: recall={recall:.3f} → τ={tau:.3f}")

    # 4. FAISS Coverage Index
    print("\n4. FAISS Coverage Index (Enhancement 1)")
    print("-" * 60)

    from models.madr_gan import FAISS_AVAILABLE
    if FAISS_AVAILABLE:
        idx = FAISSCoverageIndex(feature_dim=64)
        buffer_feats = torch.randn(200, 64)
        query_feats = torch.randn(50, 64)
        scores = idx.compute_coverage_faiss(query_feats, buffer_feats, sigma=1.0, k=10)
        print(f"FAISS coverage scores shape: {scores.shape}")
        print(f"Mean coverage: {scores.mean():.4f}")
    else:
        print("FAISS not installed — using brute-force fallback")
        idx = FAISSCoverageIndex(feature_dim=64)
        buffer_feats = torch.randn(200, 64)
        query_feats = torch.randn(50, 64)
        scores = idx.compute_coverage_faiss(query_feats, buffer_feats, sigma=1.0, k=10)
        print(f"Brute-force coverage scores shape: {scores.shape}")

    print("\n" + "=" * 60)
    print("Component demonstration complete!")
    print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MADR-GAN Examples')
    parser.add_argument('--mode', type=str, default='enhanced',
                        choices=['minimal', 'enhanced', 'components', 'all'],
                        help='Which example to run')
    parser.add_argument('--gpu', type=int, default=None)

    args = parser.parse_args()

    if args.mode in ['components', 'all']:
        demonstrate_components()

    if args.mode in ['minimal', 'all']:
        minimal_example(gpu_id=args.gpu)

    if args.mode in ['enhanced', 'all']:
        enhanced_example(gpu_id=args.gpu)
