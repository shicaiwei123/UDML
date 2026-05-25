# text-image-classification

UDML-based multimodal sentiment classification on **MVSA-Single**.

This branch follows the `text-image-classification` layout used in QMF. Run all commands below from inside `text-image-classification/`.

## Tested Environment

- Python `3.12.13`
- CUDA `12.4`
- cuDNN `9.1.0`
- PyTorch `2.5.1+cu124`
- torchvision `0.20.1+cu124`
- numpy `2.4.3`
- scikit-learn `1.8.0`
- Pillow `12.1.1`
- tqdm `4.67.3`
- pytorch-pretrained-bert `0.6.2`

## Install

```bash
conda create -n mvsa_task python=3.12 -y
conda activate mvsa_task
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com
```

## Expected Layout

```text
text-image-classification/
|- datasets/
|  `- MVSA_Single/
|     |- train.jsonl
|     |- dev.jsonl
|     |- test.jsonl
|     `- ...
|- bert-base-uncased/
|- src/
|- train_udml_noise_base.py
`- eval_udml_noise.py
```

If you prefer, `datasets/MVSA_Single` and `bert-base-uncased` can also be created as soft links.

## Main Files

- `train_udml_noise_base.py`: training entry
- `eval_udml_noise.py`: noise evaluation entry
- `run_udml_noise_base.sh`: training wrapper
- `eval_udml_noise.sh`: evaluation wrapper

## Train

```bash
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0 python -u train_udml_noise_base.py \
  --task MVSA_Single \
  --data_path ./datasets \
  --name noise_gamma4_15_3 \
  --savedir ./checkpoint \
  --batch_sz 32 \
  --lr 5e-5 \
  --max_epochs 100 \
  --patience 10 \
  --n_workers 4 \
  --fusion_dim 2048 \
  --gamma 4.0 \
  --beta 1e-3 \
  --cylcle_epoch 10 \
  --audio_depend 1.0 \
  --visual_depend 1.0
```

Or:

```bash
GPU=0 NAME=noise_gamma4_15_3 FORCE_FRESH=1 bash run_udml_noise_base.sh
```

### Training Output

Outputs are saved to `./checkpoint/<name>/`.

Important files:

- `model_best.pt`
- `model_best_depend.pt`
- `logfile.log`

### Important Note

If `./checkpoint/<name>/model_best.pt` already exists, `train_udml_noise_base.py` will load it and enter evaluation instead of starting a fresh run.

## Evaluate

```bash
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0 python eval_udml_noise.py \
  --checkpoint ./checkpoint/noise_gamma4_15_3/model_best.pt \
  --depend ./checkpoint/noise_gamma4_15_3/model_best_depend.pt \
  --strengths 0,5,10 \
  --batch_sz 32
```

Or:

```bash
GPU=0 \
CKPT=./checkpoint/noise_gamma4_15_3/model_best.pt \
DEPEND=./checkpoint/noise_gamma4_15_3/model_best_depend.pt \
STRENGTHS=0,5,10 \
bash eval_udml_noise.sh
```

## Noise Protocol

Evaluation loads test data strictly through:

```python
def get_udml_test(args, add_noise, txt_noise_level=None, img_noise_level=None)
```

Strength mapping:

- `strength = 0`: clean test set
- `strength > 0`:
  - text uses `txt_noise_level = strength + 1`
  - image uses `img_noise_level = strength`

Current helper behavior:

- text noise: 50% probability, word masking ratio `(tv - 1) / 10`
- image noise: 50% probability, `AddGaussianNoiseUDML(variance=vv)`
- Gaussian image noise uses `scale = variance**2`

## Reference

- QMF: <https://github.com/QingyangZhang/QMF>
