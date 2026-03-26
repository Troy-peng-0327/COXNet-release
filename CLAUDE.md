# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

COXNet is an object detection framework for **RGB-Thermal (RGBT) drone-based tiny person detection**, built on MMDetection 2.x. It implements several multi-spectral fusion detectors (QFDet, COXNet, ATSS-HF) published in ISPRS 2023.

## Environment

```
mmdet 2.25.1
mmcv-full 1.6.1
pytorch 1.10.0
cuda 10.2
```

Install following [MMDetection 2.x installation guide](https://github.com/open-mmlab/mmdetection/tree/2.x).

## Commands

**Train:**
```bash
python tools/train.py qfdet_configs/qfdet_r50_fpn_1x_rgbtdroneperson.py
python tools/train.py qfdet_configs/qfdet_star_r50_fpn_1x_rgbtdroneperson.py
```

**Multi-GPU train:**
```bash
bash tools/dist_train.sh <config> <num_gpus>
```

**Test/Evaluate:**
```bash
python tools/test.py qfdet_configs/qfdet_r50_fpn_1x_rgbtdroneperson.py \
  work_dir/qfdet_r50_fpn/rgbtdroneperson/epoch_11_qfdet_rgbtdroneperson.pth --eval bbox
```

**Run tests:**
```bash
pytest tests/
```

## Architecture

### Model Pipeline

All detectors follow MMDetection's two-stream design for RGBT inputs:

1. **Dual Backbone** — Two ResNet-50 networks processing RGB and thermal images independently
2. **Fusion** — Spectral features merged via learned fusion strategies
3. **Neck (FPN)** — Multi-scale feature pyramid
4. **Head** — Quality-aware ATSS detection head

### Key Models

| Model | Detector class | Config dir |
|-------|---------------|------------|
| QFDet | `mmdet/models/detectors/qfdet.py` | `qfdet_configs/` |
| QFDet* | same | `qfdet_configs/` |
| COXNet | `mmdet/models/detectors/coxnet.py` | `configs/coxnet/` |
| ATSS-HF | `mmdet/models/detectors/atss.py` | `configs/atss/` |

### COXNet Fusion Layer (`mmdet/models/utils/fusion_layer_coxnet.py`)

The core innovation. Implements several fusion strategies selectable via config:
- **CLFM** — Cross-Level Fusion Module using wavelet decomposition (DWT/IDWT)
- **AAM** — Adaptive Attention Module with multi-scale spatial attention
- **WF** — Wavelet Feature fusion
- **MSF** — Multi-Scale Feature fusion

### Data Pipeline

Custom transforms in `mmdet/datasets/pipelines/`:
- `LoadImagePairFromFile` — loads paired RGB+thermal images
- `MultiNormalize` — normalizes each modality with its own mean/std

Dataset class `DronePerson` (`mmdet/datasets/droneperson.py`) wraps COCO-format annotations with 3 classes: `person`, `rider`, `crowd`.

Normalization stats used in configs:
```python
mean_list=([115.37, 121.82, 122.63], [93.10, 93.10, 93.10])
std_list=([85.13, 89.01, 88.27], [50.24, 50.24, 50.24])
```

### Config Inheritance

Configs use MMDetection's `_base_` inheritance. Dataset, schedule, and runtime are defined in `configs/_base_/` and composed via `_base_ = [...]` in model configs. The `qfdet_configs/` directory contains self-contained configs for the primary QFDet model.

### Outputs

Training outputs (checkpoints, logs) go to `work_dir/` (gitignored).

## Datasets

- **RGBTDronePerson** — primary benchmark: [https://nnnnerd.github.io/RGBTDronePerson/](https://nnnnerd.github.io/RGBTDronePerson/)
- **VTUAV-det** — aerial detection benchmark

Pretrained checkpoints available on Google Drive (see README.md for links).
