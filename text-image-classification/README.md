# UDML text-image classification

This directory contains the MVSA-Single text-image experiments. The implementation uses BERT for text, ResNet-152 for images, a variational latent representation for each modality, and dynamic late fusion.

Run every command below from `text-image-classification/`.

## Environment

The current release was tested on:

- Python 3.12.13
- PyTorch 2.7.0+cu128
- torchvision 0.22.0+cu128
- CUDA 12.8 and cuDNN 9.7.1
- RTX 5090
- pytorch-pretrained-bert 0.6.2

Use the existing server environment:

```bash
source /root/miniconda3/bin/activate torch2.5.1
pip install pytorch-pretrained-bert==0.6.2
export HF_ENDPOINT=https://hf-mirror.com
```

Do not downgrade the installed CUDA-enabled PyTorch build on RTX 5090.

## Data and pretrained model

The expected layout is:

```text
text-image-classification/
|- bert-base-uncased/
|  |- bert_config.json
|  |- pytorch_model.bin
|  `- vocab.txt
|- datasets/
|  `- MVSA_Single/
|     |- data/
|     |  `- <image>.jpg
|     |- train.jsonl
|     |- dev.jsonl
|     `- test.jsonl
|- train_udml_noise_base.py
|- eval_udml_noise.py
|- run_udml_noise_base.sh
`- eval_udml_noise.sh
```

The official split sizes used here are 1,555 train, 518 dev, and 519 test examples. Each JSONL row must contain `id`, `label`, `text`, and an image path relative to the split directory, for example `"img": "data/477.jpg"`.

The dataset and BERT directories may be real directories or symbolic links. On the server they can be configured as:

```bash
mkdir -p datasets
ln -s /root/autodl-tmp/data/MVSA_Single datasets/MVSA_Single
ln -s /root/autodl-tmp/data/bert-base-uncased bert-base-uncased
```

To download BERT through the Hugging Face mirror when the local copy is absent:

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download bert-base-uncased \
  --local-dir /root/autodl-tmp/data/bert-base-uncased
```

## Training

The recommended entry point is the wrapper:

```bash
GPU=0 \
NAME=udml_mvsa_noise \
FORCE_FRESH=1 \
bash run_udml_noise_base.sh
```

`FORCE_FRESH=1` deletes `checkpoint/$NAME` before training. Omit it to preserve an existing run. If `model_best.pt` already exists under that directory, the Python entry loads it and runs evaluation instead of training again.

The wrapper exposes these environment variables:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `GPU` | `0` | CUDA device visible to the process |
| `NAME` | `udml_noise_base_2` | Run name and checkpoint subdirectory |
| `DATA_PATH` | `./datasets` | Parent directory containing `MVSA_Single` |
| `BATCH_SZ` | `32` | Batch size |
| `LR` | `5e-5` | Adam learning rate |
| `MAX_EPOCHS` | `100` | Maximum number of epochs |
| `PATIENCE` | `10` | Early-stopping patience after checkpoint selection begins |
| `N_WORKERS` | `4` | DataLoader workers |
| `SAVEDIR` | `./checkpoint` | Checkpoint root |
| `FUSION_DIM` | `2048` | Latent dimension for both modalities |
| `GAMMA` | `4.0` | Weight of the two unimodal classification losses |
| `BETA` | `1e-3` | Weight of KL regularization against `N(0, I)` |
| `CYL_CLE` | `10` | First epoch that uses the noisy training loader |
| `AUDIO_DEPEND` | `1.0` | Initial text dependence; the legacy argument name is retained |
| `VISUAL_DEPEND` | `1.0` | Initial image dependence |

Additional Python arguments can be appended after the script name. For example:

```bash
GPU=1 NAME=udml_beta_1e4 BETA=1e-4 \
bash run_udml_noise_base.sh --seed 3
```

The equivalent explicit command is:

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_udml_noise_base.py \
  --task MVSA_Single \
  --data_path ./datasets \
  --bert_model ./bert-base-uncased \
  --name udml_mvsa_noise \
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

### Training schedule

With `cylcle_epoch=10`:

1. Epochs 0-9 use the clean training loader.
2. Epoch 10 onward uses the noisy loader. Text and image noise are sampled independently for each example, each with a 0.5 application probability. Active text masking samples a level in `[1, 11)`; active image Gaussian noise samples an integer level from 1 through 11.
3. Training uses equal modality weights through epoch 19. Dynamic weights begin at epoch 20.
4. Best-checkpoint selection begins at epoch 15, which is `cylcle_epoch + 5`.
5. Clean dev and clean test accuracy, including fused, text-only, and image-only accuracy, are reported after every epoch.

The training loss is:

```text
fused CE
+ gamma * (text CE + image CE)
+ beta * (KL_text + KL_image)
+ 0.1 * (text variance MSE + image variance MSE)
```

Both KL terms constrain their latent distributions toward mean 0 and variance 1.

Text and image dependence are estimated from the epoch-level mean absolute unimodal logits. The estimates are used from the next epoch. When validation improves, the code saves the model and the dependence values used for that validation epoch together. After training it reloads both before the final test.

## Checkpoints

A run writes:

```text
checkpoint/<name>/
|- model_best.pt
|- model_best_depend.pt
`- logfile.log
```

`model_best_depend.pt` stores `text_depend`, `visual_depend`, the best `epoch`, and its `val_acc`. Older dependence files containing only the two dependence values remain supported.

## Evaluation

Evaluate clean, Gaussian level 5, and Gaussian level 10:

```bash
GPU=0 \
CKPT=./checkpoint/udml_mvsa_noise/model_best.pt \
DEPEND=./checkpoint/udml_mvsa_noise/model_best_depend.pt \
STRENGTHS=0,5,10 \
bash eval_udml_noise.sh
```

A clean-only evaluation uses:

```bash
GPU=0 \
CKPT=./checkpoint/udml_mvsa_noise/model_best.pt \
DEPEND=./checkpoint/udml_mvsa_noise/model_best_depend.pt \
STRENGTHS=0 \
bash eval_udml_noise.sh
```

The direct Python form is:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_udml_noise.py \
  --checkpoint ./checkpoint/udml_mvsa_noise/model_best.pt \
  --depend ./checkpoint/udml_mvsa_noise/model_best_depend.pt \
  --strengths 0,5,10 \
  --batch_sz 32
```

For evaluation, strength 0 selects the clean dataset. A positive public strength `s` maps to text helper level `s + 1` and image Gaussian level `s`. Each modality still has its independent 0.5 outer application probability. Use `--max_samples N` only for a quick sanity check; omit it for full test-set evaluation.

The evaluator infers `fusion_dim` from the checkpoint and automatically loads the adjacent `model_best_depend.pt` when `--depend` is omitted.

## Reproducibility

The current server configuration differs from older paper or repository environments because RTX 5090 requires the CUDA 12.8-compatible PyTorch build. Exact accuracy can also vary with the random seed, stochastic latent sampling, independently sampled training noise, torchvision behavior, and checkpoint selection on the dev split.

## Reference

- [QMF](https://github.com/QingyangZhang/QMF)
