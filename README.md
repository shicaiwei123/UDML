# UDML
😺 [CVPR'26 Findings] Unbiased Dynamic Multimodal Fusion

## Overview

The training entry is [`main_auxi_weight_udml.py`](./main_auxi_weight_udml.py). At a high level, the project:

- extracts audio and visual features with two ResNet-18 backbones,
- performs multimodal fusion with `concat`, `sum`, `gated`, or `film`,
- adds auxiliary unimodal classification heads for audio and visual branches,
- estimates modality uncertainty and uses it to reweight audio/visual features before fusion,
- introduces noise-aware training after a warm-up stage.

## Main Dependencies

This repository does not provide a `requirements.txt` yet. Based on the source code, the main dependencies are:

- Python 3.8+
- PyTorch 1.12.1
- torchvision

## Repository Structure

```text
UDML-Copy1/
|-- dataset/
|   |-- CramedDataset.py
|   |-- KSDataset.py
|   `-- data/
|       |-- CREMAD/
|       |   |-- train.csv
|       |   `-- test.csv
|       `-- KineticSound/
|           `-- class.txt
|-- models/
|   |-- backbone.py
|   |-- basic_model.py
|   `-- fusion_modules.py
|-- utils/
|   `-- utils.py
|-- main_auxi_weight_udml.py
|-- cramed_auxi.sh
`-- ks_auxi.sh
```

## Method Highlights

The main model class is `AVClassifier_AUXI_UDML` in [`models/basic_model.py`](./models/basic_model.py).

The final loss in the current implementation is:

```python
loss = loss_cls + regurize_loss * args.beta + variance_fc_loss * 0.1
```

where:

- `loss_cls = loss_f + (loss_a + loss_v) * args.gamma`
- `loss_f` is the multimodal classification loss
- `loss_a` / `loss_v` are auxiliary unimodal losses

The uncertainty-aware weighting is implemented by predicting audio and visual variances and using them to compute feature weights before fusion.

## Data Preparation

Download Dataset：
[CREMA-D](https://pan.baidu.com/s/11ISqU53QK7MY3E8P2qXEyw?pwd=4isj), [Kinetics-Sounds](https://pan.baidu.com/s/1E9E7h1s5NfPYFXLa1INUJQ?pwd=rcts).
Here we provide the processed dataset directly. 

The original dataset can be seen in the following links,
[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D),
[Kinetics-Sounds](https://github.com/cvdfoundation/kinetics-dataset).
[VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/),

 And you need to process the dataset following the instruction below.

### Pre-processing

For CREMA-D and VGGSound dataset, we provide code to pre-process videos into RGB frames and audio wav files in the directory ```data/```.

#### CREMA-D 

As the original CREMA-D dataset has provided the original audio and video files, we simply extract the video frames by running the code:

```python data/CREMAD/video_preprecessing.py```

Note that, the relevant path/dir should be changed according your own env.  

### Data path

you should move the download dataset into the folder *train_test_data*, or make a soft link in this folder.

## Training

### Quick Start

The repository provides two example scripts:

```bash
bash cramed_auxi.sh
bash ks_auxi.sh
```

These scripts are written for Linux-style shells. On Windows, it is usually easier to run the equivalent Python command directly.

### Example: Train on CREMAD

```bash
python main_auxi_weight_udml.py \
  --ckpt_path ./results/cramed/udml \
  --modality full \
  --dataset CREMAD \
  --gpu_ids 0 \
  --modulation Normal \
  --train \
  --num_frame 1 \
  --pe 1 \
  --beta 1e-5 \
  --gamma 4.0
```

### Example: Train on KineticSound

```bash
python main_auxi_weight_udml.py \
  --ckpt_path ./results/ks/udml \
  --modality full \
  --dataset KineticSound \
  --gpu_ids 0 \
  --modulation Normal \
  --train \
  --num_frame 3 \
  --pe 1 \
  --beta 0 \
  --gamma 2.5
```

### Important Arguments

- `--pe`: enables the uncertainty estimation branch used by this repository
- `--beta`: weight of the regularization term
- `--gamma`: weight of the auxiliary unimodal losses
- `--cylcle_epoch`: epoch after which noisy training data is used
- `--ckpt_path`: directory for logs and saved checkpoints

## Evaluation

The current evaluation path is implemented inside [`main_auxi_weight_udml.py`](./main_auxi_weight_udml.py), but the checkpoint to load is hard-coded in the `else` branch of `main()`.

To evaluate your own model:

1. modify the `torch.load(...)` path in `main_auxi_weight_udml.py`,
2. run the script without `--train`.

For example:

```bash
python main_auxi_weight_udml.py \
  --dataset CREMAD \
  --modality full \
  --fusion_method concat \
  --num_frame 1 \
  --pe 1 \
  --gamma 4.0 \
  --beta 1e-5 \
  --gpu_ids 0
```

The script reports:

- multimodal accuracy,
- audio-only accuracy,
- visual-only accuracy.

## Acknowledgement

This README is organized with reference to the public structure of [ICCV2025-GDL](https://github.com/shicaiwei123/ICCV2025-GDL), especially the way it presents dependencies, data preparation, and quick-start commands. The supported datasets here are also partially overlapping with that repository.


