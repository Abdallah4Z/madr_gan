"""
MADR-GAN: Memory-Augmented Diversity Replay GAN

Coverage-Guided Memory Replay for Mode-Diverse Generative Adversarial Networks

This package implements the Enhanced MADR-GAN method described in:
"Coverage-Guided Memory Replay for Mode-Diverse Generative Adversarial Networks: MADR-GAN"
IEEE Transactions on Neural Networks and Learning Systems (PREPRINT), 2026

"""

__version__ = "2.0.0"

from .models import (
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

from .evaluation import (
    GANEvaluator,
    calculate_fid,
    calculate_precision_recall,
    calculate_density_coverage
)

from .utils import (
    load_config,
    save_config,
    count_parameters,
    get_device
)

__all__ = [
    # Core Models
    'MADRGAN',
    'MADRGenerator',
    'MADRDiscriminator',
    'DINOFeatureExtractor',
    'ReservoirBuffer',

    # Enhanced Components
    'MultiScaleFeatureExtractor',
    'ProgressiveReservoirBuffer',
    'AdaptiveTauScheduler',
    'EMABandwidthEstimator',
    'FAISSCoverageIndex',
    'CLIPFeatureExtractor',
    'ResNetFeatureExtractor',

    # Evaluation
    'GANEvaluator',
    'calculate_fid',
    'calculate_precision_recall',
    'calculate_density_coverage',

    # Utils
    'load_config',
    'save_config',
    'count_parameters',
    'get_device',
]
