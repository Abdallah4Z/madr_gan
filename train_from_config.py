"""
Train MADR-GAN from configuration file

Usage:
    python train_from_config.py --config configs/cifar10.yaml
    python train_from_config.py --config configs/celeba.yaml
    python train_from_config.py --config configs/stacked_mnist.yaml
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_config, dict_to_namespace
from train import train


def main():
    parser = argparse.ArgumentParser(description='Train MADR-GAN from config file')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    
    # Allow overriding config parameters
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Override output directory')
    parser.add_argument('--num_iterations', type=int, default=None,
                       help='Override number of iterations')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--lambda_max', type=float, default=None,
                       help='Override lambda_max')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device ID (e.g., 0, 1, 6)')
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    if args.num_iterations is not None:
        config['num_iterations'] = args.num_iterations
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lambda_max is not None:
        config['lambda_max'] = args.lambda_max
    if args.gpu is not None:
        config['gpu'] = args.gpu
    
    # Convert to namespace
    config_ns = dict_to_namespace(config)
    
    print("\nConfiguration:")
    print("-" * 50)
    for key, value in config.items():
        print(f"{key:30s}: {value}")
    print("-" * 50)
    print()
    
    # Train
    train(config_ns)


if __name__ == '__main__':
    main()
