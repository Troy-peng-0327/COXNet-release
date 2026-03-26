# COXNet: Cross-Layer Fusion With Adaptive Alignment and Scale Integration for RGBT Tiny Object Detection

**IEEE Transactions on Circuits and Systems for Video Technology**, Vol. 36, No. 1, January 2026

[![paper](https://img.shields.io/badge/IEEE%20TCSVT-2026-blue)](https://doi.org/10.1109/TCSVT.2025.3595147)

**Authors:** Peiran Peng, Tingfa Xu, Liqiang Song, Mengqi Zhu, Yuqiang Fang, Jianan Li

---

## Introduction

COXNet is an RGBT tiny object detection framework that jointly addresses cross-modal fusion, misalignment, and scale variation in drone-based multi-spectral imagery. The core innovations are: **(1) CLFM** (Cross-Layer Fusion Module), which leverages wavelet decomposition to align and fuse complementary RGB and thermal features across pyramid levels; **(2) DASR** (Dynamic Adaptive Scale Refinement), which recalibrates spatial correspondences and integrates multi-scale contextual cues for robust tiny object localization; and **(3) a GeoShape-based label assignment strategy** that better fits the irregular geometry of tiny aerial targets, improving recall under severe scale imbalance.

---

## Main Results

### RGBTDronePerson

| Model | mAP50 | mAP50 (tiny) | mAP25 |
|-------|------:|-------------:|------:|
| COXNet | **50.04** | — | — |

### VTUAV-det

| Model | mAP50 |
|-------|------:|
| COXNet | **76.1** |

### NII-CU

| Model | mAP50 |
|-------|------:|
| COXNet | **98.2** |

---

## Installation

**Requirements:** CUDA 11.3 · Python 3.8

**Step 1 — Clone the repository**

```bash
git clone https://github.com/your-username/COXNet-release.git
cd COXNet-release
```

**Step 2 — Install PyTorch**

```bash
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

**Step 3 — Install mmcv-full**

```bash
pip install mmcv-full==1.7.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

**Step 4 — Install remaining dependencies**

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Dataset Preparation

COXNet is evaluated on three RGBT benchmarks:

| Dataset | Description | Link |
|---------|-------------|------|
| **RGBTDronePerson** | Drone-based RGB-thermal person detection | [Project page](https://nnnnerd.github.io/RGBTDronePerson/) |
| **VTUAV-det** | Aerial vehicle and UAV detection | [Project page](https://nnnnerd.github.io/RGBTDronePerson/) |
| **NII-CU** | From "Deep learning with RGB and thermal images onboard a drone for monitoring operations" | See paper |

Organize datasets under `data/` as follows:

```
data/
├── RGBTDronePerson/
│   ├── train/
│   │   ├── visible/
│   │   └── infrared/
│   └── val/
│       ├── visible/
│       └── infrared/
└── VTUAV/
    ├── train/
    └── val/
```

Update the `data_root` paths in the corresponding config files under `configs/_base_/datasets/` before training.

---

## Training

**Single GPU**

```bash
python tools/train.py configs/coxnet/coxnet_r50_fpn_1x_rgbtdroneperson.py
```

**Multi-GPU (e.g., 4 GPUs)**

```bash
bash tools/dist_train.sh configs/coxnet/coxnet_r50_fpn_1x_rgbtdroneperson.py 4
```

Available configs:

```
configs/coxnet/
├── coxnet_r50_fpn_1x_rgbtdroneperson.py
├── coxnet_star_r50_fpn_1x_rgbtdroneperson.py
├── coxnet_r50_fpn_1x_vtuav.py
└── coxnet_star_r50_fpn_1x_vtuav.py
```

---

## Evaluation

```bash
python tools/test.py \
    configs/coxnet/coxnet_r50_fpn_1x_rgbtdroneperson.py \
    /path/to/checkpoint.pth \
    --eval bbox
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{peng2025coxnet,
  title={COXNet: Cross-layer fusion with adaptive alignment and scale integration for RGBT tiny object detection},
  author={Peng, Peiran and Xu, Tingfa and Zhu, Mengqi Zhu and Fang, Yuqiang and Li, Jianan},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025},
  publisher={IEEE}
}
```

---

## Acknowledgement

This work was supported by the Natural Science Foundation of Chongqing, China, under Grant cstc2021jcyj-msxmX1130.

This codebase is built upon [MMDetection](https://github.com/open-mmlab/mmdetection). We thank the OpenMMLab team for their excellent open-source framework.
