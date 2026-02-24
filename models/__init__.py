"""
MADR-GAN Models Package

Exports all model components including enhanced versions.
"""

from .madr_gan import (
    MADRGAN,
    MADRGenerator,
    MADRDiscriminator,
    DINOFeatureExtractor,
    ReservoirBuffer,
    # Enhanced components
    MultiScaleFeatureExtractor,
    ProgressiveReservoirBuffer,
    AdaptiveTauScheduler,
    EMABandwidthEstimator,
    FAISSCoverageIndex,
    CLIPFeatureExtractor,
    ResNetFeatureExtractor,
)

__all__ = [
    'MADRGAN',
    'MADRGenerator',
    'MADRDiscriminator',
    'DINOFeatureExtractor',
    'ReservoirBuffer',
    'MultiScaleFeatureExtractor',
    'ProgressiveReservoirBuffer',
    'AdaptiveTauScheduler',
    'EMABandwidthEstimator',
    'FAISSCoverageIndex',
    'CLIPFeatureExtractor',
    'ResNetFeatureExtractor',
]
