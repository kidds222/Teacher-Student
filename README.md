# Advanced Dynamic Teaching System

A sophisticated lifelong learning framework implementing Teacher-Student architecture with dynamic knowledge distillation and β-VAE-based continual learning.

## Overview

This system implements an advanced dynamic teaching mechanism that combines:
- **WGAN-GP Teacher Network**: High-quality sample generation with gradient penalty
- **β-VAE Student Network**: Disentangled representation learning with task-conditional priors
- **Dynamic Knowledge Distillation**: Adaptive knowledge transfer between teacher and student
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
├── config/              # Configuration files
│   ├── teacher_config.py    # Teacher network parameters
│   ├── student_config.py    # Student network parameters
│   └── experiment_config.py # Experiment settings
├── models/              # Neural network architectures
│   ├── teacher.py           # WGAN-GP teacher implementation
│   ├── student.py           # β-VAE student implementation
│   └── dynamic_teacher_student.py # Main training system
├── data/                # Data loading and preprocessing
├── utils/               # Utility functions
│   ├── fid_utils.py         # FID calculation utilities
│   ├── forgetting_evaluator.py # Forgetting analysis
│   └── training_utils.py    # Training helpers
├── experiments/         # Experiment runners
├── scripts/             # Utility scripts
└── results/             # Output directory
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd test8
   ```

2. **Create conda environment**
   ```bash
   conda create -n dynamic_teaching python=3.8
   conda activate dynamic_teaching
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchvision
   pip install numpy scipy matplotlib
   pip install tensorboard
   pip install scikit-learn
   ```

## Quick Start

### Basic Usage

Run the complete experiment with default settings:
```bash
python main.py
```

### Configuration

Modify configuration files to customize the experiment:

- `config/teacher_config.py`: Teacher network architecture and training parameters
- `config/student_config.py`: Student network and β-VAE parameters  
- `config/experiment_config.py`: Dataset, training, and evaluation settings

### Advanced Options

Enable mixed sample generation:
```bash
python main.py --enable_mixed_samples
```

Generate mixed samples only (requires checkpoint):
```bash
python main.py --mixed_samples_only --checkpoint path/to/checkpoint.pth
```

## Configuration Options

### Teacher Configuration
- **Architecture**: Latent dimension, hidden layers, attention mechanisms
- **Training**: Learning rates, WGAN-GP parameters, gradient penalty
- **Expert Management**: FID thresholds, adaptive parameters

### Student Configuration  
- **β-VAE Parameters**: Latent dimensions (z_dim, u_dim), β coefficient
- **Contrastive Learning**: Temperature, negative samples, loss weights
- **Knowledge Distillation**: KD weight, temperature, adaptive mechanisms

### Experiment Configuration
- **Training**: Epochs, batch size, samples per task
- **Datasets**: Task sequence, image size, data augmentation
- **Evaluation**: FID thresholds, saving intervals, sample generation

## Datasets

Supported datasets:
- MNIST
- FashionMNIST
- CIFAR-10 (experimental)

Datasets are automatically downloaded to `./datasets/` directory.

## Results

Training results are saved to `./results/` including:
- Model checkpoints
- Generated samples
- Training curves
- FID scores
- Forgetting analysis

## Key Components

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

## Evaluation Metrics

- **FID Score**: Image quality assessment
- **Reconstruction Loss**: VAE reconstruction quality
- **Contrastive Loss**: Representation learning effectiveness
- **Forgetting Score**: Knowledge retention analysis

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{advanced_dynamic_teaching,
  title={Advanced Dynamic Teaching System},
  author={Advanced Dynamic Teaching Team},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/username/advanced-dynamic-teaching}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Contact

For questions and support, please open an issue on GitHub. 