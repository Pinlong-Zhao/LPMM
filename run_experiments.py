"""
Script to run complete experiments with different configurations.
"""

import os
import subprocess
import argparse


def run_experiment(dataset, ipc, gpu):
    """
    Run a single experiment.
    
    Args:
        dataset: Dataset name ('cifar10', 'cifar100', 'imagenet')
        ipc: Images per class
        gpu: GPU device ID
    """
    config_file = f"config/{dataset}.yaml"
    
    if not os.path.exists(config_file):
        print(f"Config file {config_file} not found!")
        return
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {dataset.upper()} with IPC={ipc}")
    print(f"{'='*60}\n")
    
    cmd = [
        "python", "main.py",
        "--config", config_file,
        "--ipc", str(ipc),
        "--gpu", str(gpu),
        "--mode", "train"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ Experiment completed: {dataset} IPC={ipc}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Experiment failed: {dataset} IPC={ipc}")
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run LPMM experiments')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'cifar10', 'cifar100', 'imagenet'],
                        help='Dataset to run experiments on')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID')
    args = parser.parse_args()
    
    # Define experiment configurations
    experiments = {
        'cifar10': [1, 10, 50],
        'cifar100': [10, 50],
        'imagenet': [10]
    }
    
    if args.dataset == 'all':
        datasets_to_run = ['cifar10', 'cifar100']
    else:
        datasets_to_run = [args.dataset]
    
    # Run experiments
    for dataset in datasets_to_run:
        for ipc in experiments[dataset]:
            run_experiment(dataset, ipc, args.gpu)
    
    print("" + "="*60)
    print("All experiments completed!")
    print("="*60)


if __name__ == '__main__':
    main()
