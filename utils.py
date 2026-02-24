"""
Utility functions for MADR-GAN
"""

import yaml
import json
import torch
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file"""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def dict_to_namespace(d: Dict[str, Any]):
    """Convert dictionary to namespace for argparse compatibility"""
    from argparse import Namespace
    return Namespace(**d)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(gpu_id: int = 0) -> torch.device:
    """Get torch device"""
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


def save_metrics(metrics: Dict[str, Any], save_path: str):
    """Save metrics to JSON file"""
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics(load_path: str) -> Dict[str, Any]:
    """Load metrics from JSON file"""
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    return metrics


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_exp_dir(base_dir: str, exp_name: str) -> Path:
    """Create experiment directory with subdirectories"""
    exp_dir = Path(base_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / 'samples').mkdir(exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    
    return exp_dir


def log_metrics_to_tensorboard(writer, metrics: Dict[str, float], step: int):
    """Log metrics to TensorBoard"""
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(key, value, step)
