# Teacher-Student Knowledge Distillation System

## Project Overview

This system implements a complete teacher-student learning architecture, combining the following core technologies:
- **WGAN-GP Teacher Network**: High-quality sample generation with gradient penalty
- **β-VAE Student Network**: Disentangled representation learning with task-conditional priors
- **Dynamic Knowledge Distillation**: Adaptive teacher-student knowledge transfer mechanism
- **Continual Learning**: Multi-task learning without catastrophic forgetting

## Key Features

- **Multi-Expert Teacher System**: Dynamic expert addition based on FID thresholds
- **Contrastive Learning**: Enhanced representation quality through contrastive objectives
- **Task-Conditional Priors**: Domain-specific latent representations
- **Forgetting Evaluation**: Comprehensive analysis of knowledge retention
- **Mixed Sample Generation**: Advanced data augmentation techniques

## Project Structure

```
project/
├── config/                    # Configuration files
│   ├── teacher_config.py     # Teacher network configuration
│   ├── student_config.py     # Student network configuration
│   └── experiment_config.py  # Experiment settings
├── models/                   # Neural network architectures
│   ├── teacher.py           # Teacher model implementation
│   ├── student.py           # Student model implementation
│   └── dynamic_teacher_student.py # Main training system
├── data/                    # Data loading and preprocessing
│   ├── data_loaders.py     # Data loaders
│   └── datasets.py         # Dataset definitions
├── utils/                   # Utility functions
│   ├── fid_utils.py        # FID calculation utilities
│   ├── forgetting_evaluator.py # Forgetting analysis
│   ├── training_utils.py   # Training helper functions
│   └── experiment_logger.py # Experiment logging
├── experiments/             # Experiment runners
│   └── run.py              # Experiment execution script
├── scripts/                 # Utility scripts
│   ├── download_datasets.py # Dataset download script
│   └── evaluate_forgetting.py # Forgetting evaluation script
├── results/                 # Output directory
└── datasets/               # Dataset storage directory
```


## Usage Instructions

### Step 1: Download Datasets
Before running the main program, you need to download the necessary datasets:

```bash
cd scripts
python download_datasets.py
cd ..
```

### Step 2: Run Main Program
```bash
python main.py
```

## Configuration Details

### Teacher Configuration (teacher_config.py)
- **Architecture Parameters**: Latent dimensions, hidden layers, attention mechanisms
- **Training Parameters**: Learning rates, WGAN-GP parameters, gradient penalty
- **Expert Management**: FID thresholds, adaptive parameters

### Student Configuration (student_config.py)  
- **β-VAE Parameters**: Latent dimensions (z_dim, u_dim), β coefficient
- **Contrastive Learning**: Temperature parameters, negative sample count, loss weights
- **Knowledge Distillation**: KD weight, temperature, adaptive mechanisms

### Experiment Configuration (experiment_config.py)
- **Training Settings**: Epochs, batch size, samples per task
- **Dataset Settings**: Task sequence, image size, data augmentation
- **Evaluation Settings**: FID thresholds, saving intervals, sample generation

## Supported Datasets

- **MNIST**: Handwritten digit recognition
- **Fashion-MNIST**: Fashion item classification

Datasets will be automatically downloaded to the `./datasets/` directory.

## Output Results

Training results are saved in the `./results/` directory, including:
- Model checkpoints
- Generated samples
- Training curves
- FID scores
- Forgetting analysis reports

## Core Components

### Teacher Network (WGAN-GP)
- Generator with spectral normalization
- Discriminator with gradient penalty
- Multi-expert architecture for continual learning

### Student Network (β-VAE)
- Encoder-decoder architecture with β regularization
- Task-conditional latent variables
- Contrastive learning for representation quality

### Knowledge Distillation
- Adaptive distillation based on teacher performance
- Feature-level and output-level knowledge transfer
- Dynamic temperature adjustment