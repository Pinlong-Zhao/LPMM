# Information-Theoretic Dataset Distillation via Disentangled Purity and Multi-Descriptor Consensus

PyTorch implementation of **Information-Theoretic Dataset Distillation via Disentangled Purity and Multi-Descriptor Consensus**.

This repository provides the code for **LPMM**, a dataset distillation method that aims to synthesize compact training sets by preserving label-relevant information and reducing bias from single-descriptor matching. The current release includes an end-to-end CIFAR-10 long-tailed distillation example together with modular components for information purification and multi-descriptor consensus.

## Overview

Dataset distillation compresses a large training set into a small synthetic set that can still be used to train effective models. LPMM focuses on two issues that may affect matching-based distillation methods:

- **Information contamination**: label-irrelevant factors may be preserved together with task-relevant information.
- **Matching bias**: relying on a single descriptor may provide a biased view of the data distribution.

LPMM addresses these issues through two main components:

- **Information Purification (IP)**: disentangles label-relevant and label-irrelevant latent representations with an information-bottleneck-inspired objective.
- **Multi-Descriptor Consensus (MDC)**: matches real and synthetic data using multiple training descriptors, such as loss, uncertainty, and gradient statistics.

## Repository Structure

```text
LPMM-main/
├── weighted_distill.py              # Main runnable CIFAR-10-LT distillation example
├── modules/
│   ├── information_purification.py  # Information Purification module
│   └── mdc.py                       # Multi-Descriptor Consensus module
├── utils/                           # Data loading, evaluation, and visualization utilities
├── config/                          # Configuration templates for CIFAR/ImageNet-style experiments
├── scripts/                         # Shell script templates for extended experiments
├── Requirements.txt                 # Python dependencies
├── Installation.sh                  # Minimal installation helper
├── setup.py                         # Package metadata
└── LICENSE
```

The main executable file in this release is `weighted_distill.py`. The configuration files and shell scripts are kept as templates for extending the codebase to larger experiment pipelines.

## Environment

The code is intended for Python 3.8+ and PyTorch 1.12+. A CUDA-enabled GPU is recommended.

Create an environment and install the dependencies:

```bash
conda create -n lpmm python=3.8
conda activate lpmm
pip install -r Requirements.txt
```

For CUDA-specific PyTorch installation, please install the PyTorch and torchvision versions that match your CUDA driver before installing the remaining dependencies.

## Data

The default example uses CIFAR-10. When running `weighted_distill.py`, CIFAR-10 is downloaded automatically to `./data` if it is not already available.

The script constructs a long-tailed CIFAR-10 training set with imbalance factor `IMB_FACTOR = 10` by default, and then evaluates the distilled data on the original CIFAR-10 test set.

## Quick Start

Run the default CIFAR-10 long-tailed distillation example:

```bash
python weighted_distill.py
```

The script performs the following steps:

1. Build a long-tailed CIFAR-10 training set.
2. Train a ResNet-18 model on the long-tailed training set.
3. Generate class-wise distilled samples using the encoder-decoder distillation pipeline.
4. Match real and generated samples with descriptor statistics including loss, gradient norm, and uncertainty.
5. Train a ResNet-18 model on the distilled set and report test accuracy.
6. Train another ResNet-18 model on the original long-tailed set for comparison.

TensorBoard logs are written to:

```text
runs/my_experiment/
```

## Main Settings

The default hyperparameters are defined near the beginning of `weighted_distill.py`:

```python
IMB_FACTOR = 10
NUM_CLASSES = 10
BATCH_SIZE = 64
LATENT_DIM = 1028
M = 5
EPOCHS_ORIG = 20
EPOCHS_GENERATED = 20
DISTILL_EPOCHS = 20
TRAINING_EPOCHS = 50
```

Common settings to adjust are:

- `IMB_FACTOR`: imbalance factor for the long-tailed CIFAR-10 split.
- `M`: number of generated representations per input sample.
- `LATENT_DIM`: latent dimension used by the encoder and decoder.
- `EPOCHS_ORIG`, `EPOCHS_GENERATED`, `DISTILL_EPOCHS`, and `TRAINING_EPOCHS`: training schedules for the baseline, generation, distillation, and final evaluation stages.

---
