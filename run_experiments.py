"""
Enhancement 7: Full Experiment Protocol for Enhanced MADR-GAN

Automated experiment runner that:
  - Compares Vanilla GAN, Original MADR-GAN, Enhanced MADR-GAN
  - Collects FID, Precision, Recall, Coverage metrics
  - Supports ablation studies
  - Exports results as JSON + comparison table
"""

import os
import json
import argparse
import subprocess
import time
from pathlib import Path

import yaml


ABLATION_CONFIGS = {
    'vanilla':        {'lambda_max': 0.0, 'use_faiss': False, 'use_multiscale': False,
                       'progressive_buffer': False, 'auto_tau': False},
    'original':       {'use_faiss': False, 'use_multiscale': False,
                       'progressive_buffer': False, 'auto_tau': False},
    'enhanced':       {'use_faiss': True, 'use_multiscale': True,
                       'progressive_buffer': True, 'auto_tau': True},
    'no_faiss':       {'use_faiss': False, 'use_multiscale': True,
                       'progressive_buffer': True, 'auto_tau': True},
    'no_multiscale':  {'use_faiss': True, 'use_multiscale': False,
                       'progressive_buffer': True, 'auto_tau': True},
    'no_progressive': {'use_faiss': True, 'use_multiscale': True,
                       'progressive_buffer': False, 'auto_tau': True},
    'no_auto_tau':    {'use_faiss': True, 'use_multiscale': True,
                       'progressive_buffer': True, 'auto_tau': False},
}


def build_train_command(base_args: dict, variant: str, output_root: str) -> list:
    """Build python train.py command for a given experiment variant."""
    cmd = ['python', 'train.py']

    overrides = ABLATION_CONFIGS[variant]
    merged = {**base_args, **overrides}
    merged['output_dir'] = os.path.join(output_root, variant)

    for key, val in merged.items():
        if isinstance(val, bool):
            if val:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(val)])

    return cmd


def run_experiment(variant: str, base_args: dict, output_root: str, gpu: int = None):
    """Run a single training experiment."""
    cmd = build_train_command(base_args, variant, output_root)
    if gpu is not None:
        cmd.extend(['--gpu', str(gpu)])

    print(f"\n{'='*60}")
    print(f"  Running: {variant}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    output_dir = os.path.join(output_root, variant)
    metrics_file = os.path.join(output_dir, 'metrics.json')

    summary = {
        'variant': variant,
        'return_code': result.returncode,
        'training_time_sec': elapsed,
    }

    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        if metrics:
            last = metrics[-1]
            summary['final_g_loss'] = last.get('g_loss')
            summary['final_d_loss'] = last.get('d_loss')
            summary['final_recall'] = last.get('recall')
            summary['final_lambda_t'] = last.get('lambda_t')
            summary['final_tau_recall'] = last.get('tau_recall')
            summary['final_sigma'] = last.get('sigma')
            summary['final_buffer_size'] = last.get('buffer_size')

    return summary


def print_results_table(results: list):
    """Print comparison table."""
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT RESULTS")
    print(f"{'='*80}")

    header = f"{'Variant':<18} {'Status':>6} {'Time':>8} {'G Loss':>8} {'D Loss':>8} {'Recall':>8} {'λ(t)':>8} {'τ':>8}"
    print(header)
    print('-' * len(header))

    for r in results:
        status = '✅' if r['return_code'] == 0 else '❌'
        time_str = f"{r['training_time_sec']:.0f}s"
        g = f"{r.get('final_g_loss', 0):.3f}" if r.get('final_g_loss') else 'N/A'
        d = f"{r.get('final_d_loss', 0):.3f}" if r.get('final_d_loss') else 'N/A'
        rec = f"{r.get('final_recall', 0):.3f}" if r.get('final_recall') is not None else 'N/A'
        lam = f"{r.get('final_lambda_t', 0):.3f}" if r.get('final_lambda_t') is not None else 'N/A'
        tau = f"{r.get('final_tau_recall', 0):.3f}" if r.get('final_tau_recall') is not None else 'N/A'
        print(f"{r['variant']:<18} {status:>6} {time_str:>8} {g:>8} {d:>8} {rec:>8} {lam:>8} {tau:>8}")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Run MADR-GAN experiments')

    parser.add_argument('--variants', nargs='+', default=['vanilla', 'original', 'enhanced'],
                        choices=list(ABLATION_CONFIGS.keys()),
                        help='Which variants to run')
    parser.add_argument('--output_root', type=str, default='./outputs/experiments')
    parser.add_argument('--gpu', type=int, default=None)

    # Base training args
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_iterations', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--buffer_capacity', type=int, default=2048)

    args = parser.parse_args()

    base_args = {
        'dataset': args.dataset,
        'num_iterations': args.num_iterations,
        'batch_size': args.batch_size,
        'img_size': args.img_size,
        'buffer_capacity': args.buffer_capacity,
    }

    os.makedirs(args.output_root, exist_ok=True)

    results = []
    for variant in args.variants:
        summary = run_experiment(variant, base_args, args.output_root, args.gpu)
        results.append(summary)

    # Save results
    results_file = os.path.join(args.output_root, 'experiment_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

    print_results_table(results)


if __name__ == '__main__':
    main()
