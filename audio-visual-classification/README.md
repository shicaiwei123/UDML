# audio-visual-classification

UDML for audio-visual classification.

Run all commands below from inside `audio-visual-classification/`.

## Main Entry

- Training / evaluation entry: `main_auxi_weight_udml.py`

## Main Dependencies

- Python 3.8+
- PyTorch 1.12.1
- torchvision

## Repository Layout

```text
audio-visual-classification/
|- dataset/
|  |- CramedDataset.py
|  |- KSDataset.py
|  `- data/
|- models/
|- utils/
|- main_auxi_weight_udml.py
|- cramed_auxi.sh
`- ks_auxi.sh
```

## Data

Supported datasets in the current code:

- CREMAD
- KineticSound

The repository includes lightweight metadata files under `dataset/data/`, but dataset assets should be prepared separately according to your environment.

## Train

Quick start:

```bash
bash cramed_auxi.sh
bash ks_auxi.sh
```

Example: CREMAD

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

Example: KineticSound

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

## Evaluate

Evaluation is implemented in `main_auxi_weight_udml.py`.

To evaluate a checkpoint, update the checkpoint loading path in the evaluation branch of `main()` and run the script without `--train`.

Example:

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
