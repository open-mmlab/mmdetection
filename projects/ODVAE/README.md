# ODVAE: Generative Priors for 3D Object Detection

## Overview

ODVAE is a project focused on leveraging generative priors for 3D object detection tasks. It builds on the OpenMMLab Detection Toolbox and Benchmark, extending its functionality to incorporate deep generative models for improved accuracy and robustness in 3D object detection scenarios.

This project includes configurations, training scripts, and utilities to facilitate the training and evaluation of ODVAE models for 3D object detection, with a focus on datasets like NuScenes.

## Features

- **Generative Priors:** Utilizes variational autoencoders (VAEs) as generative priors to enhance 3D object detection.
- **Flexible Configuration:** Easily configurable training and testing pipelines with support for multiple datasets.
- **Integration with OpenMMLab:** Seamlessly integrates with the robust OpenMMLab framework for object detection tasks.

## Requirements

Before running the project, ensure the following dependencies are installed:

- Python >= 3.6
- PyTorch >= 1.8.0
- MMDetection
- Additional dependencies as specified in the OpenMMLab documentation

Refer to the [MMDetection Installation Guide](https://github.com/open-mmlab/mmdetection/blob/main/docs/get_started.md) for detailed setup instructions.

## Usage

### Training

To train the ODVAE model, use the following command:

```bash
python projects/ODVAE/tools/train.py \
   projects/ODVAE/configs/detr_vae_encoder_8xb2_nuscenes_2d.py \
   --work-dir .out/work_dirs/detr_vae_encoder_nuscenes_2d \
   --launcher slurm
```

- **`train.py`**: The training script.
- **`detr_vae_encoder_8xb2_nuscenes_2d.py`**: Configuration file for the ODVAE model.
- **`--work-dir`**: Directory where training outputs (e.g., checkpoints, logs) will be saved.
- **`--launcher slurm`**: Specifies the distributed launcher (e.g., Slurm).

### Evaluation

To be added soon...

## Project Structure

```
projects/ODVAE/
├── configs/                   # Configuration files for ODVAE models
├── tools/                     # Training and utility scripts
└── other_directories...       # Additional project files
```

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request.

## License

This project is distributed under the [Creative Commons License](LICENSE).
