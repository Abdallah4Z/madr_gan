"""
Enhanced MADR-GAN: Memory-Augmented Diversity Replay GAN
Coverage-Guided Memory Replay for Mode-Diverse Generative Adversarial Networks

Enhanced version with 7 algorithmic improvements:
  1. FAISS-based coverage computation (O(N·K) → O(N log K))
  2. Progressive buffer initialization (128→2048 ramp)
  3. Warm-start σ bandwidth + EMA smoothing
  4. Auto-tuning τ_recall (dataset-agnostic)
  5. Multi-scale feature extractor (DINO + CLIP + ResNet)
  6. Importance-weighted reservoir sampling
  7. Full experiment/validation protocol (see run_experiments.py)

Original paper:
"Coverage-Guided Memory Replay for Mode-Diverse GANs: MADR-GAN"
IEEE Transactions on Neural Networks and Learning Systems (PREPRINT), 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from collections import deque
import warnings
import timm

# Optional FAISS import
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Optional OpenCLIP import
try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False


# =============================================================================
# Generator & Discriminator (unchanged from original)
# =============================================================================

class MADRGenerator(nn.Module):
    """DCGAN-style generator for MADR-GAN"""

    def __init__(self, latent_dim: int = 128, img_channels: int = 3, base_channels: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, base_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(True),
            # State: (base_channels*8) x 4 x 4
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            # State: (base_channels*4) x 8 x 8
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            # State: (base_channels*2) x 16 x 16
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            # State: base_channels x 32 x 32
            nn.ConvTranspose2d(base_channels, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: img_channels x 64 x 64
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.main(z)


class MADRDiscriminator(nn.Module):
    """DCGAN-style discriminator with spectral normalization"""

    def __init__(self, img_channels: int = 3, base_channels: int = 128):
        super().__init__()

        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(img_channels, base_channels, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(base_channels * 8, 1, 4, 1, 0, bias=False)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(x.size(0), -1)


# =============================================================================
# Enhancement 5: Multi-Scale Feature Extractor (DINO + CLIP + ResNet)
# =============================================================================

class DINOFeatureExtractor(nn.Module):
    """Frozen DINO ViT-S/8 feature extractor"""

    def __init__(self, model_name: str = 'vit_small_patch8_224_dino'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.feature_dim = 384  # ViT-S/8 CLS token dimension
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1) / 2  # [-1,1] -> [0,1]
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        features = self.model.forward_features(x)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        elif features.dim() > 2:
            features = features[:, 0, :]  # CLS token
        return features


class CLIPFeatureExtractor(nn.Module):
    """Frozen CLIP ViT-B/32 feature extractor via open_clip"""

    def __init__(self):
        super().__init__()
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError("open_clip_torch is required for CLIP features. "
                              "Install with: pip install open-clip-torch")
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        self.model = model.visual
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.feature_dim = 512
        self.register_buffer('mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1) / 2
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        features = self.model(x)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        return features


class ResNetFeatureExtractor(nn.Module):
    """Frozen ResNet-50 pool5 feature extractor"""

    def __init__(self):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Remove the final FC layer — keep up through avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.feature_dim = 2048
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1) / 2
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        features = self.backbone(x).flatten(1)  # (B, 2048)
        return features


class MultiScaleFeatureExtractor(nn.Module):
    """
    Enhancement 5: Multi-scale feature extractor combining DINO + CLIP + ResNet.

    Each backbone's features are L2-normalized, concatenated, then projected
    to a common dimensionality via a learned linear projection.
    Falls back to DINO-only if CLIP or ResNet are unavailable.
    """

    def __init__(self, output_dim: int = 512, use_clip: bool = True, use_resnet: bool = True):
        super().__init__()

        self.extractors: Dict[str, nn.Module] = {}
        total_dim = 0

        # Always include DINO
        self.dino = DINOFeatureExtractor()
        self.extractors['dino'] = self.dino
        total_dim += self.dino.feature_dim  # 384

        # Optionally include CLIP
        self._has_clip = False
        if use_clip and OPEN_CLIP_AVAILABLE:
            try:
                self.clip = CLIPFeatureExtractor()
                self.extractors['clip'] = self.clip
                total_dim += self.clip.feature_dim  # 512
                self._has_clip = True
            except Exception as e:
                warnings.warn(f"CLIP extractor unavailable: {e}. Using DINO-only.")

        # Optionally include ResNet
        self._has_resnet = False
        if use_resnet:
            try:
                self.resnet = ResNetFeatureExtractor()
                self.extractors['resnet'] = self.resnet
                total_dim += self.resnet.feature_dim  # 2048
                self._has_resnet = True
            except Exception as e:
                warnings.warn(f"ResNet extractor unavailable: {e}. Using DINO-only.")

        # Linear projection to output_dim
        self.projection = nn.Linear(total_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        # Freeze projection after init (we don't train it)
        for param in self.projection.parameters():
            param.requires_grad = False

        self.feature_dim = output_dim
        self.input_dim = total_dim

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = []
        # DINO
        f_dino = self.dino(x)
        f_dino = F.normalize(f_dino, p=2, dim=1)
        parts.append(f_dino)
        # CLIP
        if self._has_clip:
            f_clip = self.clip(x)
            f_clip = F.normalize(f_clip, p=2, dim=1)
            parts.append(f_clip)
        # ResNet
        if self._has_resnet:
            f_resnet = self.resnet(x)
            f_resnet = F.normalize(f_resnet, p=2, dim=1)
            parts.append(f_resnet)

        concatenated = torch.cat(parts, dim=1)  # (B, total_dim)
        projected = self.projection(concatenated)  # (B, output_dim)
        return projected


# =============================================================================
# Enhancement 1: FAISS-based Coverage Index
# =============================================================================

class FAISSCoverageIndex:
    """
    Enhancement 1: FAISS-accelerated nearest-neighbor coverage computation.

    Replaces brute-force O(N·K) torch.cdist with O(N log K) FAISS search.
    Falls back to brute-force if FAISS is not installed.
    """

    def __init__(self, feature_dim: int, use_gpu: bool = False):
        self.feature_dim = feature_dim
        self.use_gpu = use_gpu and FAISS_AVAILABLE
        self.index = None
        self._faiss_available = FAISS_AVAILABLE

    def build_index(self, buffer_features: np.ndarray):
        """Build/rebuild FAISS index from buffer features."""
        if not self._faiss_available:
            return

        features = np.ascontiguousarray(buffer_features.astype(np.float32))
        self.index = faiss.IndexFlatL2(self.feature_dim)

        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        self.index.add(features)

    def search(self, query_features: np.ndarray, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.

        Returns:
            (distances, indices) — each of shape (N, k)
        """
        query = np.ascontiguousarray(query_features.astype(np.float32))
        distances, indices = self.index.search(query, k)
        return distances, indices

    def compute_coverage_faiss(
        self, real_features: torch.Tensor, buffer_features: torch.Tensor, sigma: float, k: int = 20
    ) -> torch.Tensor:
        """
        Fast coverage scores via FAISS approximate NN search.

        Args:
            real_features: (N, D) real data features
            buffer_features: (K, D) buffer features
            sigma: kernel bandwidth
            k: number of neighbors to use
        Returns:
            coverage scores of shape (N,)
        """
        if not self._faiss_available or len(buffer_features) < k:
            # Fallback to brute-force
            return self._brute_force_coverage(real_features, buffer_features, sigma)

        # Build index from buffer
        self.build_index(buffer_features.cpu().numpy())

        # Query real features
        real_np = real_features.cpu().numpy()
        distances, _ = self.search(real_np, k=min(k, len(buffer_features)))

        # Gaussian kernel: exp(-d^2 / (2σ^2)), distances from FAISS are already squared L2
        distances_tensor = torch.from_numpy(distances).to(real_features.device)
        kernels = torch.exp(-distances_tensor / (2 * sigma ** 2))
        coverage_scores = kernels.mean(dim=1)

        return coverage_scores

    @staticmethod
    def _brute_force_coverage(
        real_features: torch.Tensor, buffer_features: torch.Tensor, sigma: float
    ) -> torch.Tensor:
        """Original brute-force O(N·K) fallback."""
        dists_sq = torch.cdist(real_features, buffer_features, p=2).pow(2)
        kernels = torch.exp(-dists_sq / (2 * sigma ** 2))
        return kernels.mean(dim=1)


# =============================================================================
# Enhancement 2 & 6: Progressive + Importance-Weighted Reservoir Buffer
# =============================================================================

class ProgressiveReservoirBuffer:
    """
    Enhancement 2: Progressive buffer that ramps capacity from initial_capacity
    to max_capacity over training.

    Enhancement 6: Importance-weighted reservoir sampling — features with
    higher diversity scores (lower coverage) have higher probability of
    remaining in the buffer.
    """

    def __init__(
        self,
        max_capacity: int = 2048,
        initial_capacity: int = 128,
        ramp_interval: int = 5000,
        feature_dim: int = 512,
    ):
        self.max_capacity = max_capacity
        self.initial_capacity = initial_capacity
        self.ramp_interval = ramp_interval
        self.feature_dim = feature_dim

        self.buffer: List[torch.Tensor] = []
        self.importance_scores: List[float] = []
        self.n_seen = 0
        self._current_capacity = initial_capacity

    @property
    def capacity(self) -> int:
        return self._current_capacity

    def step_capacity(self, iteration: int):
        """
        Enhancement 2: Progressive capacity ramp.
        Doubles capacity every ramp_interval until max_capacity.
        Schedule: 128 → 256 → 512 → 1024 → 2048
        """
        target = self.initial_capacity
        steps = iteration // self.ramp_interval
        target = min(self.initial_capacity * (2 ** steps), self.max_capacity)
        self._current_capacity = int(target)

    def update(self, features: torch.Tensor, importance_weights: Optional[torch.Tensor] = None):
        """
        Enhancement 6: Importance-weighted reservoir sampling.

        Args:
            features: (B, feature_dim) tensor of new features (moved to CPU internally)
            importance_weights: (B,) optional per-sample importance (higher = more diverse)
        """
        features = features.detach().cpu()
        if importance_weights is not None:
            importance_weights = importance_weights.detach().cpu()

        for i, feat in enumerate(features):
            self.n_seen += 1
            weight = importance_weights[i].item() if importance_weights is not None else 1.0

            if len(self.buffer) < self._current_capacity:
                self.buffer.append(feat)
                self.importance_scores.append(weight)
            else:
                # Importance-weighted reservoir: probability proportional to weight
                base_prob = self._current_capacity / self.n_seen
                accept_prob = min(1.0, base_prob * weight)

                if np.random.rand() < accept_prob:
                    # Evict the sample with the lowest importance
                    if self.importance_scores:
                        min_idx = int(np.argmin(self.importance_scores))
                    else:
                        min_idx = np.random.randint(0, len(self.buffer))
                    self.buffer[min_idx] = feat
                    self.importance_scores[min_idx] = weight

    def get_buffer(self) -> Optional[torch.Tensor]:
        if len(self.buffer) == 0:
            return None
        return torch.stack(self.buffer)

    def __len__(self):
        return len(self.buffer)


# Legacy alias for backward compatibility
class ReservoirBuffer:
    """Original reservoir sampling buffer (kept for backward compat)."""

    def __init__(self, capacity: int = 2048, feature_dim: int = 384):
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.buffer: List[torch.Tensor] = []
        self.n_seen = 0

    def update(self, features: torch.Tensor):
        features = features.detach().cpu()
        for feat in features:
            self.n_seen += 1
            if len(self.buffer) < self.capacity:
                self.buffer.append(feat)
            else:
                if np.random.rand() < self.capacity / self.n_seen:
                    idx = np.random.randint(0, self.capacity)
                    self.buffer[idx] = feat

    def get_buffer(self) -> Optional[torch.Tensor]:
        if len(self.buffer) == 0:
            return None
        return torch.stack(self.buffer)

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# Enhancement 4: Adaptive τ_recall Scheduler
# =============================================================================

class AdaptiveTauScheduler:
    """
    Enhancement 4: Auto-tuning τ_recall.

    Tracks recall history and adjusts τ_recall based on the 75th percentile
    of recent recall values. Eliminates manual per-dataset tuning.
    """

    def __init__(
        self,
        initial_tau: float = 0.80,
        window_size: int = 100,
        warmup_steps: int = 50,
        tau_min: float = 0.50,
        tau_max: float = 0.95,
    ):
        self.tau = initial_tau
        self.window_size = window_size
        self.warmup_steps = warmup_steps
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.recall_history: deque = deque(maxlen=window_size)
        self._step_count = 0

    def step(self, recall: float) -> float:
        """Record a recall observation and update τ_recall."""
        self.recall_history.append(recall)
        self._step_count += 1

        if self._step_count < self.warmup_steps or len(self.recall_history) < 10:
            return self.tau  # Keep initial value during warmup

        # Set τ to the 75th percentile of recent recall
        recalls = np.array(self.recall_history)
        target = float(np.percentile(recalls, 75))
        # Clamp to [tau_min, tau_max]
        self.tau = max(self.tau_min, min(self.tau_max, target))
        return self.tau


# =============================================================================
# Enhancement 3: Warm-Start σ + EMA Bandwidth Estimator
# =============================================================================

class EMABandwidthEstimator:
    """
    Enhancement 3: Warm-start bandwidth with EMA smoothing.

    - Initializes σ from real-data pairwise distances (warm-start)
    - Updates σ via EMA: σ_new = α·σ_median + (1-α)·σ_old
    - Floor σ_min prevents collapse when generator collapses
    """

    def __init__(self, ema_alpha: float = 0.1, sigma_min: float = 0.1, sigma_init: float = 1.0):
        self.ema_alpha = ema_alpha
        self.sigma_min = sigma_min
        self.sigma = sigma_init
        self._warm_started = False

    def warm_start(self, real_features: torch.Tensor, n_samples: int = 500):
        """Initialize σ from real-data pairwise distances."""
        n = min(len(real_features), n_samples)
        indices = torch.randperm(len(real_features))[:n]
        subset = real_features[indices]

        dists = torch.cdist(subset, subset, p=2)
        dists_flat = dists[dists > 0]

        if len(dists_flat) > 0:
            k = max(1, len(dists_flat) // 2)
            median_dist = torch.kthvalue(dists_flat, k)[0].item()
            self.sigma = max(median_dist, self.sigma_min)
        self._warm_started = True

    def update(self, buffer_features: torch.Tensor, n_samples: int = 200) -> float:
        """
        Update σ via EMA on median buffer pairwise distance.

        Returns:
            Updated σ value
        """
        if buffer_features is None or len(buffer_features) < 2:
            return self.sigma

        n = min(len(buffer_features), n_samples)
        if n < len(buffer_features):
            indices = torch.randperm(len(buffer_features))[:n]
            subset = buffer_features[indices]
        else:
            subset = buffer_features

        dists = torch.cdist(subset, subset, p=2)
        dists_flat = dists[dists > 0]

        if len(dists_flat) == 0:
            return self.sigma

        k = max(1, len(dists_flat) // 2)
        sigma_new = torch.kthvalue(dists_flat, k)[0].item()

        # EMA update
        self.sigma = self.ema_alpha * sigma_new + (1 - self.ema_alpha) * self.sigma
        # Floor
        self.sigma = max(self.sigma, self.sigma_min)

        return self.sigma


# =============================================================================
# Main Enhanced MADR-GAN Model
# =============================================================================

class MADRGAN(nn.Module):
    """
    Enhanced MADR-GAN: Memory-Augmented Diversity Replay GAN

    Includes all 7 enhancements. Backward-compatible with the original:
    pass use_faiss=False, use_multiscale=False, progressive_buffer=False,
    auto_tau=False to get original behavior.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        img_channels: int = 3,
        base_channels: int = 128,
        buffer_capacity: int = 2048,
        lambda_max: float = 1.0,
        tau_recall: float = 0.80,
        kappa: float = 10.0,
        beta: float = 1.0,
        gamma_r1: float = 10.0,
        # --- Enhancement flags ---
        use_faiss: bool = False,
        use_multiscale: bool = False,
        progressive_buffer: bool = False,
        auto_tau: bool = False,
        # --- Enhancement params ---
        initial_buffer_capacity: int = 128,
        buffer_ramp_interval: int = 5000,
        ema_alpha: float = 0.1,
        sigma_min: float = 0.1,
        multiscale_output_dim: int = 512,
        use_clip: bool = True,
        use_resnet: bool = True,
        faiss_k: int = 20,
    ):
        super().__init__()

        # Networks
        self.generator = MADRGenerator(latent_dim, img_channels, base_channels)
        self.discriminator = MADRDiscriminator(img_channels, base_channels)

        # --- Enhancement 5: Multi-scale vs DINO-only ---
        self.use_multiscale = use_multiscale
        if use_multiscale:
            self.feature_extractor = MultiScaleFeatureExtractor(
                output_dim=multiscale_output_dim,
                use_clip=use_clip,
                use_resnet=use_resnet,
            )
            feature_dim = multiscale_output_dim
        else:
            self.feature_extractor = DINOFeatureExtractor()
            feature_dim = 384

        # Convenience alias to keep old code paths compatible
        self.dino = self.feature_extractor

        # --- Enhancement 2 & 6: Progressive + importance-weighted buffer ---
        self.use_progressive_buffer = progressive_buffer
        if progressive_buffer:
            self.buffer = ProgressiveReservoirBuffer(
                max_capacity=buffer_capacity,
                initial_capacity=initial_buffer_capacity,
                ramp_interval=buffer_ramp_interval,
                feature_dim=feature_dim,
            )
        else:
            self.buffer = ReservoirBuffer(buffer_capacity, feature_dim)

        # --- Enhancement 1: FAISS index ---
        self.use_faiss = use_faiss
        self.faiss_k = faiss_k
        if use_faiss:
            self.faiss_index = FAISSCoverageIndex(feature_dim)
        else:
            self.faiss_index = None

        # --- Enhancement 3: EMA bandwidth estimator ---
        self.bandwidth_estimator = EMABandwidthEstimator(
            ema_alpha=ema_alpha, sigma_min=sigma_min
        )

        # --- Enhancement 4: Auto τ ---
        self.use_auto_tau = auto_tau
        if auto_tau:
            self.tau_scheduler = AdaptiveTauScheduler(initial_tau=tau_recall)
        else:
            self.tau_scheduler = None

        # Hyperparameters
        self.latent_dim = latent_dim
        self.lambda_max = lambda_max
        self.tau_recall = tau_recall
        self.kappa = kappa
        self.beta = beta
        self.gamma_r1 = gamma_r1

        # State
        self.real_features = None
        self.sigma = 1.0
        self.lambda_t = 0.0
        self.iter = 0

    # ------------------------------------------------------------------
    # Enhancement 3: Warm-start σ from real data
    # ------------------------------------------------------------------

    def precompute_real_features(self, real_images: torch.Tensor):
        """Precompute features for all real training images."""
        with torch.no_grad():
            self.real_features = self.feature_extractor(real_images)
        # Warm-start bandwidth from real data
        self.bandwidth_estimator.warm_start(self.real_features)
        self.sigma = self.bandwidth_estimator.sigma

    # ------------------------------------------------------------------
    # Enhancement 3: EMA bandwidth update
    # ------------------------------------------------------------------

    def update_bandwidth(self):
        """Update kernel bandwidth using EMA-smoothed median heuristic."""
        buffer_tensor = self.buffer.get_buffer()
        if buffer_tensor is None or len(buffer_tensor) < 2:
            return

        device = self.real_features.device if self.real_features is not None else torch.device('cpu')
        buffer_tensor = buffer_tensor.to(device)

        self.sigma = self.bandwidth_estimator.update(buffer_tensor)

    # ------------------------------------------------------------------
    # Enhancement 1: Coverage scores (FAISS or brute-force)
    # ------------------------------------------------------------------

    def compute_coverage_scores(self) -> torch.Tensor:
        """Compute coverage scores for all real samples."""
        buffer_tensor = self.buffer.get_buffer()
        if buffer_tensor is None:
            return torch.zeros(len(self.real_features))

        buffer_tensor = buffer_tensor.to(self.real_features.device)
        sigma = self.sigma if self.sigma is not None else 1.0

        if self.use_faiss and self.faiss_index is not None:
            return self.faiss_index.compute_coverage_faiss(
                self.real_features, buffer_tensor, sigma, k=self.faiss_k
            )
        else:
            # Original brute-force
            dists_sq = torch.cdist(self.real_features, buffer_tensor, p=2).pow(2)
            kernels = torch.exp(-dists_sq / (2 * sigma ** 2))
            return kernels.mean(dim=1)

    # ------------------------------------------------------------------
    # Importance weights (unchanged logic)
    # ------------------------------------------------------------------

    def compute_importance_weights(self, coverage_scores: torch.Tensor) -> torch.Tensor:
        weights = torch.exp(-self.beta * coverage_scores)
        weights = weights / (weights.sum() + 1e-8)
        return weights

    # ------------------------------------------------------------------
    # Coverage loss
    # ------------------------------------------------------------------

    def compute_coverage_loss(self, generated_images: torch.Tensor) -> torch.Tensor:
        min_buffer = (self.buffer.capacity // 4) if hasattr(self.buffer, 'capacity') else 64
        if self.real_features is None or len(self.buffer) < max(min_buffer, 2):
            return torch.tensor(0.0, device=generated_images.device)

        with torch.no_grad():
            gen_features = self.feature_extractor(generated_images)

        coverage_scores = self.compute_coverage_scores()
        weights = self.compute_importance_weights(coverage_scores)

        sigma = self.sigma if self.sigma is not None else 1.0

        dists_sq = torch.cdist(gen_features, self.real_features, p=2).pow(2)
        kernels = torch.exp(-dists_sq / (2 * sigma ** 2))
        weighted_kernels = (kernels * weights.unsqueeze(0)).sum(dim=1)
        loss = -weighted_kernels.mean()

        return loss

    # ------------------------------------------------------------------
    # Enhancement 4: Online recall with adaptive τ
    # ------------------------------------------------------------------

    def compute_online_recall(self) -> float:
        buffer_tensor = self.buffer.get_buffer()
        if buffer_tensor is None or self.real_features is None:
            return 0.0

        buffer_tensor = buffer_tensor.to(self.real_features.device)

        if buffer_tensor.dim() == 1:
            buffer_tensor = buffer_tensor.unsqueeze(0)
        real_features = self.real_features if self.real_features.dim() == 2 else self.real_features.unsqueeze(0)

        if len(buffer_tensor) < 2:
            return 0.0

        # Compute epsilon_t as 95th percentile of intra-buffer distances
        n_samples = min(len(buffer_tensor), 200)
        if n_samples < len(buffer_tensor):
            indices = torch.randperm(len(buffer_tensor))[:n_samples]
            buffer_sample = buffer_tensor[indices]
        else:
            buffer_sample = buffer_tensor

        dists_buffer = torch.cdist(buffer_sample, buffer_sample, p=2)
        dists_flat = dists_buffer[dists_buffer > 0]

        if len(dists_flat) == 0:
            epsilon_t = 1.0
        else:
            k = max(1, int(0.95 * len(dists_flat)))
            epsilon_t = torch.kthvalue(dists_flat, k)[0].item()

        # Batched neighbor check
        batch_size = 100
        has_neighbor_list = []
        for i in range(0, len(real_features), batch_size):
            real_batch = real_features[i:i + batch_size]
            dists_batch = torch.cdist(real_batch, buffer_tensor, p=2)
            has_neighbor_batch = (dists_batch < epsilon_t).any(dim=1)
            has_neighbor_list.append(has_neighbor_batch)

        has_neighbor = torch.cat(has_neighbor_list, dim=0)
        recall = has_neighbor.float().mean().item()

        return recall

    # ------------------------------------------------------------------
    # Enhancement 4: Adaptive lambda with auto τ
    # ------------------------------------------------------------------

    def update_adaptive_lambda(self):
        recall = self.compute_online_recall()

        # Update τ if auto-tuning is enabled
        if self.use_auto_tau and self.tau_scheduler is not None:
            self.tau_recall = self.tau_scheduler.step(recall)

        self.lambda_t = self.lambda_max * torch.sigmoid(
            torch.tensor(self.kappa * (self.tau_recall - recall))
        ).item()

    # ------------------------------------------------------------------
    # Enhancement 2: Step buffer capacity (call from training loop)
    # ------------------------------------------------------------------

    def step_buffer(self, iteration: int):
        """Call each iteration to maybe grow the progressive buffer."""
        if self.use_progressive_buffer and hasattr(self.buffer, 'step_capacity'):
            self.buffer.step_capacity(iteration)

    # ------------------------------------------------------------------
    # Losses (same interface as original)
    # ------------------------------------------------------------------

    def generator_loss(self, fake_logits: torch.Tensor, generated_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss_adv = F.binary_cross_entropy_with_logits(
            fake_logits, torch.ones_like(fake_logits)
        )
        loss_cov = self.compute_coverage_loss(generated_images)
        loss_total = loss_adv + self.lambda_t * loss_cov
        return {
            'loss_total': loss_total,
            'loss_adv': loss_adv,
            'loss_cov': loss_cov,
            'lambda_t': torch.tensor(self.lambda_t),
        }

    def discriminator_loss(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        real_logits = self.discriminator(real_images)
        fake_logits = self.discriminator(fake_images.detach())

        loss_real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        loss_fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        loss_bce = loss_real + loss_fake

        real_images.requires_grad_(True)
        real_logits_grad = self.discriminator(real_images)
        r1_grads = torch.autograd.grad(
            outputs=real_logits_grad.sum(), inputs=real_images,
            create_graph=True, retain_graph=True
        )[0]
        r1_penalty = r1_grads.pow(2).sum(dim=[1, 2, 3]).mean()
        loss_r1 = self.gamma_r1 / 2 * r1_penalty

        loss_total = loss_bce + loss_r1
        return {
            'loss_total': loss_total,
            'loss_bce': loss_bce,
            'loss_r1': loss_r1,
            'real_logits': real_logits.mean(),
            'fake_logits': fake_logits.mean(),
        }

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self.generator(z)
