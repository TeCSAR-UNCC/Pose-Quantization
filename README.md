# 3D Pose Quantization and Compression using VQ-GAN

This repository contains the code and resources for the paper titled *"Optimizing Human Motion Representation: Spatial-temporal Quantization and Compression with VQ-GANs"*. The approach introduces a novel method of discretizing continuous human motion using a Vector Quantized Generative Adversarial Network (VQ-GAN) for accurate and efficient motion representation.

## Table of Contents
- [Overview](#overview)
- [Model Performance Summary](#model-performance-summary)
- [Training the Model](#training-the-model)
- [Configuration Files](#configuration-files)
- [Testing the Model](#testing-the-model)
- [Setting Up the CMU Panoptic Dataset](#setting-up-the-cmu-panoptic-dataset)
- [Dataset Structure](#dataset-structure)
- [Getting Started](#getting-started)
- [Citation](#citation)

## Overview

The project focuses on:
- Discretizing motion information using spatiotemporal heatmaps.
- Extreme compression of human pose data while retaining essential information.
- Decoupling embedding creation from Vision Transformers, allowing for more flexible motion analysis.

The experiments demonstrate the effectiveness of this approach across different camera perspectives (egocentric and exocentric) and various compression rates.

## Model Performance Summary

| Model Type         | Compression | Vocab Size | SSIM↑ | PSNR↑ | L1↓ | T-Std↓ | Q-Loss↓ |
|-------------------|-------------|------------|-------|-------|------|--------|---------|
| Single Egocentric | F8          | 512        | 0.975 | 31.23 | 0.005| 0.212  | 0.0013  |
| Single Egocentric | F16         | 512        | 0.950 | 28.06 | 0.008| 0.217  | 0.0033  |
| Single Egocentric | F16         | 256        | 0.953 | 28.30 | 0.008| 0.217  | 0.0034  |
| Multi Exocentric  | F8          | 1024       | 0.921 | 25.37 | 0.011| 0.230  | 0.0037  |
| 3D Projection     | F8          | 1024       | 0.934 | 31.65 | 0.005| 0.151  | 0.0014  |

## Training the Model

To train the model, you can use different configurations depending on the task and architecture. Below are some common examples:

```bash
python train.py --cfg configs/c2d_2d_vqgan.yml
python train.py --cfg configs/m2d_m2d_vqgan.yml
python train.py --cfg configs/3d_3d_vqgan.yml
```

## Configuration Files
The configuration files are located in the configs/ directory. These files allow you to adjust key features like compression rate, number of codebook vectors, and other hyperparameters.

You can change:
- Compression Rate: Adjust how much data is compressed (e.g., F8, F16, F32).
- Number of Codebook Vectors: Control the size of the latent space.

To resume training, specify the path to your checkpoint file in the configuration under resume. For example:

```yaml
resume: 'experiments/sandy-jazz-107/3dvqgan_e9_sandy-jazz-107.pt'
```

## Testing the Model
To test multiple models, use the test.py script with a configuration file like configs/test_multiple_models.yml:

```bash
python test.py --cfg configs/test_multiple_models.yml
```

In the test_multiple_models.yml file, you can list multiple model configurations:

```yaml
models:
  clear-dawn-120:
    config_file: 'c2d_2d_vqgan.yml'
    <<: *F8
    num_codebook_vectors: 512
  
  Additional-model-here:
    ...
```

## Setting Up the CMU Panoptic Dataset
1. Download the CMU Panoptic dataset following the instructions in the panoptic-toolbox.
2. Extract the data to your desired path and link it in the configuration file:

root: '/home/gmaldon2/panoptic-toolbox/data/'

3. Replace the bash files in the panoptic toolbox with the files located in \panoptic_sh\ to extract only keypoint data.
4. The correct files should be generated automatically when you run the code for the first time.

### Dataset Structure
The directory structure should look like this:

```yaml
|-- panoptic-toolbox
    |-- data
        |-- 16060224_haggling1
        |   |-- hdPose3d_stage1_coco19
        |   |-- calibration_160224_haggling1.json
        |-- 160226_haggling1  
        |-- ...
```

## Getting Started
Clone the repository:

git clone https://github.com/your-username/your-repository.git

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the experiments:

```bash
python train.py --cfg configs/c2d_2d_vqgan.yml
```

## Citation

Will be updated

```bibtex
@article{your_citation_here,
  title={Optimizing Human Motion Representation: Spatial-temporal Quantization and Compression with VQ-GANs},
  author={Anonymous},
  journal={AAAI 2024},
  year={2024}
}
```