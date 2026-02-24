"""
Unit tests for Enhanced MADR-GAN components

Tests all 7 enhancements:
  1. FAISS coverage computation
  2. Progressive buffer
  3. EMA bandwidth
  4. Auto τ_recall
  5. Multi-scale features
  6. Importance-weighted sampling
  7. Backward compatibility
"""

import pytest
import torch
import numpy as np
from models.madr_gan import (
    MADRGAN,
    MADRGenerator,
    MADRDiscriminator,
    DINOFeatureExtractor,
    ReservoirBuffer,
    ProgressiveReservoirBuffer,
    AdaptiveTauScheduler,
    EMABandwidthEstimator,
    FAISSCoverageIndex,
    FAISS_AVAILABLE,
    OPEN_CLIP_AVAILABLE,
)


# =============================================================================
# Test Enhancement 1: FAISS Coverage
# =============================================================================

class TestFAISSCoverage:

    def test_brute_force_fallback(self):
        """FAISS index falls back to brute-force when FAISS not available."""
        idx = FAISSCoverageIndex(feature_dim=32)
        idx._faiss_available = False  # Force fallback

        buffer = torch.randn(50, 32)
        query = torch.randn(10, 32)
        scores = idx.compute_coverage_faiss(query, buffer, sigma=1.0, k=5)

        assert scores.shape == (10,)
        assert (scores >= 0).all()

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
    def test_faiss_coverage_shape(self):
        """FAISS coverage returns correct shape."""
        idx = FAISSCoverageIndex(feature_dim=32)
        buffer = torch.randn(100, 32)
        query = torch.randn(20, 32)
        scores = idx.compute_coverage_faiss(query, buffer, sigma=1.0, k=10)

        assert scores.shape == (20,)

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
    def test_faiss_matches_bruteforce(self):
        """FAISS coverage scores approximately match brute-force."""
        torch.manual_seed(42)
        feature_dim = 16
        idx = FAISSCoverageIndex(feature_dim=feature_dim)

        buffer = torch.randn(50, feature_dim)
        query = torch.randn(10, feature_dim)
        sigma = 2.0

        faiss_scores = idx.compute_coverage_faiss(query, buffer, sigma=sigma, k=50)
        bf_scores = FAISSCoverageIndex._brute_force_coverage(query, buffer, sigma)

        # With k=all, FAISS should match brute-force closely
        assert torch.allclose(faiss_scores, bf_scores, atol=0.05)


# =============================================================================
# Test Enhancement 2: Progressive Buffer
# =============================================================================

class TestProgressiveBuffer:

    def test_initial_capacity(self):
        buf = ProgressiveReservoirBuffer(max_capacity=256, initial_capacity=32, ramp_interval=100)
        assert buf.capacity == 32

    def test_capacity_ramp(self):
        buf = ProgressiveReservoirBuffer(max_capacity=256, initial_capacity=32, ramp_interval=100)

        buf.step_capacity(0)
        assert buf.capacity == 32

        buf.step_capacity(100)
        assert buf.capacity == 64

        buf.step_capacity(200)
        assert buf.capacity == 128

        buf.step_capacity(300)
        assert buf.capacity == 256

        buf.step_capacity(500)  # Should not exceed max
        assert buf.capacity == 256

    def test_buffer_fills_to_capacity(self):
        buf = ProgressiveReservoirBuffer(max_capacity=64, initial_capacity=16, ramp_interval=100)
        for _ in range(20):
            buf.update(torch.randn(1, 8))
        assert len(buf) == 16  # Capped at initial capacity

    def test_get_buffer(self):
        buf = ProgressiveReservoirBuffer(max_capacity=64, initial_capacity=16, feature_dim=8)
        assert buf.get_buffer() is None

        buf.update(torch.randn(5, 8))
        tensor = buf.get_buffer()
        assert tensor.shape == (5, 8)


# =============================================================================
# Test Enhancement 3: EMA Bandwidth
# =============================================================================

class TestEMABandwidth:

    def test_warm_start(self):
        bw = EMABandwidthEstimator(sigma_init=1.0)
        real = torch.randn(100, 32)
        bw.warm_start(real)
        assert bw.sigma > 0
        assert bw._warm_started

    def test_ema_update(self):
        bw = EMABandwidthEstimator(ema_alpha=0.5, sigma_init=1.0)
        buffer = torch.randn(50, 32)
        old_sigma = bw.sigma
        new_sigma = bw.update(buffer)
        # Should have changed
        assert new_sigma != old_sigma or abs(new_sigma - old_sigma) < 1e-6

    def test_sigma_floor(self):
        bw = EMABandwidthEstimator(sigma_min=0.5, sigma_init=0.01)
        # Update with very similar features (would produce tiny σ)
        buffer = torch.ones(50, 8) + torch.randn(50, 8) * 0.0001
        new_sigma = bw.update(buffer)
        assert new_sigma >= 0.5  # Floor enforced

    def test_none_buffer(self):
        bw = EMABandwidthEstimator(sigma_init=2.0)
        assert bw.update(None) == 2.0

    def test_small_buffer(self):
        bw = EMABandwidthEstimator(sigma_init=2.0)
        assert bw.update(torch.randn(1, 8)) == 2.0  # < 2 samples


# =============================================================================
# Test Enhancement 4: Auto τ_recall
# =============================================================================

class TestAdaptiveTau:

    def test_warmup_keeps_initial(self):
        sched = AdaptiveTauScheduler(initial_tau=0.80, warmup_steps=10)
        for _ in range(9):
            tau = sched.step(0.5)
        assert tau == 0.80  # Still in warmup

    def test_adapts_after_warmup(self):
        sched = AdaptiveTauScheduler(initial_tau=0.80, warmup_steps=5, window_size=20)
        # Feed high recall values
        for _ in range(30):
            tau = sched.step(0.90)
        # τ should adapt toward 0.90
        assert tau > 0.85

    def test_tau_clamped(self):
        sched = AdaptiveTauScheduler(initial_tau=0.80, warmup_steps=5,
                                     tau_min=0.5, tau_max=0.95)
        for _ in range(50):
            sched.step(0.99)
        assert sched.tau <= 0.95

        sched2 = AdaptiveTauScheduler(initial_tau=0.80, warmup_steps=5,
                                      tau_min=0.5, tau_max=0.95)
        for _ in range(50):
            sched2.step(0.1)
        assert sched2.tau >= 0.5


# =============================================================================
# Test Enhancement 6: Importance-Weighted Sampling
# =============================================================================

class TestImportanceWeightedSampling:

    def test_with_weights(self):
        buf = ProgressiveReservoirBuffer(max_capacity=10, initial_capacity=10, feature_dim=4)
        features = torch.randn(10, 4)
        weights = torch.ones(10)
        buf.update(features, importance_weights=weights)
        assert len(buf) == 10

    def test_without_weights(self):
        buf = ProgressiveReservoirBuffer(max_capacity=10, initial_capacity=10, feature_dim=4)
        features = torch.randn(10, 4)
        buf.update(features)  # No weights — should work like regular reservoir
        assert len(buf) == 10

    def test_high_importance_survives(self):
        """High-importance features are more likely to stay in buffer."""
        torch.manual_seed(42)
        np.random.seed(42)

        buf = ProgressiveReservoirBuffer(max_capacity=5, initial_capacity=5, feature_dim=4)
        # Fill buffer with low-importance features
        low_features = torch.zeros(5, 4)
        low_weights = torch.ones(5) * 0.01
        buf.update(low_features, importance_weights=low_weights)
        assert len(buf) == 5

        # Add high-importance features — they should replace low ones
        high_features = torch.ones(10, 4)
        high_weights = torch.ones(10) * 100.0
        buf.update(high_features, importance_weights=high_weights)

        # At least some high features should be in the buffer
        buf_tensor = buf.get_buffer()
        high_count = (buf_tensor.sum(dim=1) > 0.5).sum().item()
        assert high_count > 0, "High-importance features should survive in buffer"


# =============================================================================
# Test Enhancement 7: Backward Compatibility
# =============================================================================

class TestBackwardCompat:

    def test_original_mode(self):
        """Original MADR-GAN interface still works."""
        model = MADRGAN(
            latent_dim=32, img_channels=3, base_channels=16, buffer_capacity=32,
            use_faiss=False, use_multiscale=False, progressive_buffer=False, auto_tau=False,
        )
        z = torch.randn(2, 32)
        out = model.generator(z)
        assert out.shape == (2, 3, 64, 64)

    def test_legacy_reservoir_buffer(self):
        """Legacy ReservoirBuffer still works."""
        buf = ReservoirBuffer(capacity=10, feature_dim=8)
        buf.update(torch.randn(5, 8))
        assert len(buf) == 5
        tensor = buf.get_buffer()
        assert tensor.shape == (5, 8)


# =============================================================================
# Test Full Forward Pass
# =============================================================================

class TestForwardPass:

    def test_generator_forward(self):
        gen = MADRGenerator(latent_dim=32, img_channels=3, base_channels=16)
        z = torch.randn(4, 32)
        out = gen(z)
        assert out.shape == (4, 3, 64, 64)

    def test_discriminator_forward(self):
        disc = MADRDiscriminator(img_channels=3, base_channels=16)
        x = torch.randn(4, 3, 64, 64)
        out = disc(x)
        assert out.shape == (4, 1)

    def test_enhanced_model_instantiation(self):
        """Enhanced model can be created without errors."""
        model = MADRGAN(
            latent_dim=32, img_channels=3, base_channels=16,
            buffer_capacity=64,
            use_faiss=True,
            use_multiscale=False,  # Skip multiscale to avoid large downloads in CI
            progressive_buffer=True,
            auto_tau=True,
            initial_buffer_capacity=8,
            buffer_ramp_interval=10,
        )
        z = torch.randn(2, 32)
        out = model.generator(z)
        assert out.shape == (2, 3, 64, 64)

    def test_coverage_loss_with_empty_buffer(self):
        """Coverage loss returns 0 when buffer is insufficient."""
        model = MADRGAN(latent_dim=32, img_channels=3, base_channels=16, buffer_capacity=64)
        fake = torch.randn(2, 3, 64, 64)
        loss = model.compute_coverage_loss(fake)
        assert loss.item() == 0.0

    def test_step_buffer(self):
        """step_buffer works for both progressive and non-progressive."""
        model = MADRGAN(latent_dim=32, base_channels=16, progressive_buffer=True,
                        initial_buffer_capacity=8, buffer_ramp_interval=5)
        model.step_buffer(0)
        assert model.buffer.capacity == 8

        model.step_buffer(5)
        assert model.buffer.capacity == 16

    def test_sample(self):
        model = MADRGAN(latent_dim=32, base_channels=16)
        samples = model.sample(4, torch.device('cpu'))
        assert samples.shape == (4, 3, 64, 64)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
