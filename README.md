# Sparse Frequency-Aware Transformer

This repository contains the implementation of **Sparse Frequency-Aware Transformer (SFAT)** for spiking neural networks (SNNs). The method introduces structured sparsity via Haar wavelet decomposition and a dual-path MLP routing mechanism for frequency compensation.

## Baseline

The baseline `SWformer` model and the original Haar wavelet layers are adapted from:

- **Repository:** https://github.com/bic-L/Spiking-Wavelet-Transformer
- **Role in this codebase:** The non-sparse `swformer` baseline and `wavelet_layers.py` are retained for comparison against our sparse (`swformer_sparse`) and frequency-aware (`swformer_freq_aware`) variants.

## Repository Structure

```
.
├── cifar10-100/          # CIFAR-10 and CIFAR-100 experiments
├── TinyImageNet/         # Tiny ImageNet experiments
├── event/                # Neuromorphic event-based experiments (CIFAR10-DVS, DVS-Gesture)
└── figures/              # Visualization assets
```

Each experiment directory contains:
- `train_freq_aware.py` — main training script
- `model_frequency_aware.py` — SFAT model definition
- `model_sparse.py` — sparse SWformer backbone
- `frequency_modules.py` — dual-path frequency compensation MLP
- `wavelet_layers_sparse.py` — Haar wavelet layers with structured sparsity
- `*.yaml` — configuration files for hyperparameters

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA-capable GPU (recommended)

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Optional Dependencies

- **wandb** — for experiment logging (optional, disabled by default)
- **apex** — for mixed-precision training (optional)

## Running Experiments

All experiments are launched via `train_freq_aware.py` inside each directory using YAML config files.

### CIFAR-10

```bash
cd cifar10-100
python train_freq_aware.py -c cifar10_sparse_m_parallel.yaml \
    --model swformer_freq_aware \
    --data-dir ./data
```

### CIFAR-100

```bash
cd cifar10-100
python train_freq_aware.py -c cifar100_sparse_m_parallel.yaml \
    --model swformer_freq_aware \
    --data-dir ./data
```

### Tiny ImageNet

```bash
cd TinyImageNet
python train_freq_aware.py -c tiny_imagenet_sparse_m_parallel.yaml \
    --model swformer_freq_aware \
    --data-dir ./data
```

### CIFAR10-DVS (Event Data)

```bash
cd event
python train_freq_aware.py -c cifar10dvs_paper_lightweight.yaml \
    --model swformer_freq_aware \
    --data-dir ./data
```

### DVS-Gesture (Event Data)

```bash
cd event
python train_freq_aware.py -c dvsgesture_sparse_m_parallel.yaml \
    --model swformer_freq_aware \
    --data-dir ./data
```

### Available Models

| Model | Description |
|-------|-------------|
| `swformer` | Baseline SWformer (non-sparse) |
| `swformer_sparse` | Sparse variant with two-level Haar sparsity |
| `swformer_freq_aware` | Full SFAT with dual-path frequency compensation |

### Key Arguments

- `-c, --config` — YAML config file path
- `--model` — Model architecture (`swformer`, `swformer_sparse`, `swformer_freq_aware`)
- `--data-dir` — Dataset root directory
- `--time-step` — Simulation time steps (default: 4 for static, 16 for event)
- `--layer` — Number of transformer layers
- `--dim` — Embedding dimension
- `--batch-size` — Training batch size
- `--epochs` — Total training epochs
- `--log-wandb` — Enable Weights & Biases logging

## Reproduction

To reproduce the main results:

1. Download and prepare datasets into `./data` under each experiment directory.
2. Use the provided `*_parallel.yaml` or `*_lightweight.yaml` configs for the corresponding model variant.
3. Run `train_freq_aware.py` with the config file.

Each YAML config contains all hyperparameters (thresholds, sparsity settings, learning rate schedules) required for reproduction. No additional hyperparameter tuning is needed beyond the provided configs.

## Data Preparation

### CIFAR-10 / CIFAR-100
Datasets are auto-downloaded by `torchvision` when `--data-dir ./data` is specified.

### Tiny ImageNet
Download from http://cs231n.stanford.edu/tiny-imagenet-200.zip and extract to `./data/tiny-imagenet-200`.

### CIFAR10-DVS / DVS-Gesture
These neuromorphic datasets are handled automatically by `spikingjelly` or `tonic` when running the event experiments.

## License

Third-party code (e.g., `wavelet_layers.py`, `model.py`, `loader.py`, `aa_snn.py`, `transforms_factory.py`, `neuron.py`) retains the original licenses of their respective authors. See file headers for attribution details.
