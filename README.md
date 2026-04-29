# Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention

Official PyTorch implementation for **Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial
and Temporal Attention**.

This repository provides training and evaluation code for event-camera object detection on the Prophesee Gen1 and Gen4 datasets.

## Overview

The codebase supports:

- **Gen1** event-camera object detection, with 240x304 input resolution and 2 classes.
- **Gen4 / 1 Mpx** event-camera object detection, with 720x1280 input resolution and 3 classes.
- Dataset-specific experiment configurations through Hydra.
- Training, validation, and test entry points through shell scripts and Python modules.

## Installation

We recommend using `uv` for reproducible dependency installation:

```bash
uv sync
source .venv/bin/activate
```

The main dependencies are specified in `pyproject.toml`.

## Datasets

This project uses the preprocessed Gen1 and Gen4 datasets released with **Recurrent Vision Transformers for Object Detection with Event Cameras**.

| Dataset | Download | CRC32 |
| --- | --- | --- |
| Gen1 | [gen1.tar](https://download.ifi.uzh.ch/rpg/RVT/datasets/preprocessed/gen1.tar) | `5acab6f3` |
| Gen4 | [gen4.tar](https://download.ifi.uzh.ch/rpg/RVT/datasets/preprocessed/gen4.tar) | `c5ec7c38` |

After downloading, update the dataset path in the relevant training or testing script, or pass it through the Hydra configuration.

## Checkpoints

Pretrained checkpoints are available from [Google Drive](https://drive.google.com/drive/folders/1M16UZ_3p2CvV3Q1io5PpwSGWIvnpkPMF?usp=sharing).

Reported checkpoint results:

| Dataset | Checkpoint | Test mAP |
| --- | --- | --- |
| Gen1 | `ckpt_files/gen1/gen1_mAP36.ckpt` | 0.36 |
| Gen4 | `ckpt_files/gen4/gen4_mAP29.ckpt` | 0.29 |

## Training and Evaluation

The repository includes shell scripts for common training and testing runs. These scripts are intended as editable entry points for local dataset paths, GPU IDs, batch sizes, and checkpoint paths.

Before running the scripts, update the placeholder paths such as `/path/to/gen1`, `/path/to/gen4`, and the checkpoint paths to match your local dataset and checkpoint locations.

| Purpose | Script |
| --- | --- |
| Train Gen1 | `train_gen1.sh` |
| Train Gen4 | `train_gen4.sh` |
| Test Gen1 | `test_gen1.sh` |
| Test Gen4 | `test_gen4.sh` |

The main Python entry points are:

- `train.py` for training.
- `validation.py` for validation and test evaluation.

## Project Links

- [Project page](https://soikathasanahmed.github.io/hybrid/)
- [Video](https://www.youtube.com/watch?v=9OA2cnTd8WA)
- [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Ahmed_Efficient_Event-Based_Object_Detection_A_Hybrid_Neural_Network_with_Spatial_CVPR_2025_paper.pdf)

## Code Acknowledgements

This repository builds on ideas and code from the following projects:

- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), for the detection head and PAFPN components.
- [RVT](https://github.com/uzh-rpg/RVT), for event-camera object detection infrastructure and preprocessed dataset support.

We thank the authors of these projects for releasing their code and datasets.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{ahmed2025efficient,
  title={Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention},
  author={Ahmed, Soikat Hasan and Finkbeiner, Jan and Neftci, Emre},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13970--13979},
  year={2025}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
