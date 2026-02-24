"""
Visualization utilities for MADR-GAN

Generate plots and visualizations of training progress
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict


def plot_training_curves(metrics_file: str, save_dir: str = None):
    """
    Plot training curves from metrics.json
    
    Args:
        metrics_file: Path to metrics.json
        save_dir: Directory to save plots (if None, will display)
    """
    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Extract data
    iterations = [m['iteration'] for m in metrics]
    g_loss = [m['g_loss'] for m in metrics]
    d_loss = [m['d_loss'] for m in metrics]
    lambda_t = [m['lambda_t'] for m in metrics]
    recall = [m['recall'] for m in metrics]
    buffer_size = [m['buffer_size'] for m in metrics]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('MADR-GAN Training Progress', fontsize=16, fontweight='bold')
    
    # Generator and Discriminator Loss
    ax = axes[0, 0]
    ax.plot(iterations, g_loss, label='Generator', color='blue', alpha=0.7)
    ax.plot(iterations, d_loss, label='Discriminator', color='red', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Generator and Discriminator Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Adaptive Lambda
    ax = axes[0, 1]
    ax.plot(iterations, lambda_t, color='green', linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='λ = 0.5')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('λ(t)')
    ax.set_title('Adaptive Coverage Weight λ(t)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Online Recall
    ax = axes[0, 2]
    ax.plot(iterations, recall, color='purple', linewidth=2)
    ax.axhline(y=0.80, color='red', linestyle='--', alpha=0.5, label='Target τ=0.80')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Recall')
    ax.set_title('Online Recall Estimate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Buffer Size
    ax = axes[1, 0]
    ax.plot(iterations, buffer_size, color='orange', linewidth=2)
    ax.axhline(y=2048, color='red', linestyle='--', alpha=0.5, label='Capacity K=2048')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Buffer Size')
    ax.set_title('Episodic Buffer Growth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Lambda vs Recall (phase diagram)
    ax = axes[1, 1]
    scatter = ax.scatter(recall, lambda_t, c=iterations, cmap='viridis', 
                        alpha=0.6, s=10)
    ax.set_xlabel('Recall')
    ax.set_ylabel('λ(t)')
    ax.set_title('Adaptive Controller Phase Diagram')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Iteration')
    
    # Coverage Loss Component
    if 'g_cov' in metrics[0]:
        g_cov = [m['g_cov'] for m in metrics]
        ax = axes[1, 2]
        ax.plot(iterations, g_cov, color='brown', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('L_cov')
        ax.set_title('Coverage Loss Component')
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_dir is not None:
        save_path = Path(save_dir) / 'training_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()


def plot_evaluation_comparison(results: Dict[str, Dict[str, float]], 
                              save_path: str = None):
    """
    Plot comparison of evaluation metrics across methods
    
    Args:
        results: Dictionary mapping method names to their metrics
                 e.g., {'MADR-GAN': {'fid': 15.8, 'recall': 0.82, ...}, ...}
        save_path: Path to save plot (if None, will display)
    """
    methods = list(results.keys())
    metrics_names = ['FID', 'Precision', 'Recall', 'Coverage']
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('MADR-GAN Evaluation Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    for i, metric_name in enumerate(metrics_names):
        ax = axes[i]
        metric_key = metric_name.lower()
        
        values = [results[m].get(metric_key, 0) for m in methods]
        bars = ax.bar(range(len(methods)), values, color=colors)
        
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()


def visualize_mode_coverage_evolution(metrics_file: str, save_path: str = None):
    """
    Visualize how mode coverage evolves over training
    
    For Stacked MNIST: plot number of modes captured over time
    """
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    iterations = [m['iteration'] for m in metrics]
    recall = [m['recall'] for m in metrics]
    coverage = [m.get('coverage', m['recall']) for m in metrics]  # Use recall if coverage not available
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(iterations, recall, label='Recall', color='blue', linewidth=2)
    ax.plot(iterations, coverage, label='Coverage', color='red', linewidth=2)
    ax.axhline(y=0.80, color='gray', linestyle='--', alpha=0.5, label='Target τ=0.80')
    
    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Mode Coverage Evolution During Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved coverage evolution plot to {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize MADR-GAN training')
    parser.add_argument('--metrics', type=str, required=True,
                       help='Path to metrics.json file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Plot training curves
    plot_training_curves(args.metrics, args.output_dir)
    
    # Plot mode coverage evolution
    visualize_mode_coverage_evolution(args.metrics, 
                                     Path(args.output_dir) / 'coverage_evolution.png' if args.output_dir else None)
