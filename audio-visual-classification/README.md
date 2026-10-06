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

mkdir train_test_data (same level with the main_auxi_weight_udml.py) and put the dataset in the floder

## Train

Each launcher contains two fixed commands and runs them in order: clean UDML,
then Gaussian noise training after the default cycle epoch 50.

```bash
bash cramed_auxi.sh
bash ks_auxi.sh
```

## Evaluate

Use the standalone `test.py` entry. It accepts the checkpoint and corruption settings on the command line, evaluates every test sample by default, and reports fused, audio-only, and visual-only accuracy.

With no noise flags, `test.py` follows the first noisy condition in UDML Table 3: Gaussian level 5 with independent audio/visual application probability 0.5. The other official robustness settings are Gaussian 10 and Salt 5/10. Pass `--noise_type None` for clean evaluation.

Clean CREMAD evaluation:

```bash
python test.py \
  --dataset CREMAD \
  --pretrained_model <checkpoint.pth> \
  --noise_type None \
  --fusion_method concat \
  --num_frame 1 \
  --pe 1 \
  --gpu_ids 0
```

Gaussian noise with one shared level:

```bash
python test.py \
  --dataset CREMAD \
  --pretrained_model <checkpoint.pth> \
  --noise_type Gaussian \
  --noise_level 5 \
  --gpu_ids 0 \
  --output ./results/test_gaussian_level5.json
```

`--visual_variance` and `--audio_variance` set modality-specific fixed levels and override `--noise_level`:

```bash
python test.py \
  --dataset CREMAD \
  --pretrained_model <checkpoint.pth> \
  --noise_type Gaussian \
  --visual_variance 5 \
  --audio_variance 2 \
  --visual_noise_prob 0.5 \
  --audio_noise_prob 0.5 \
  --gpu_ids 0
```

salt-and-pepper noise uses levels in `[0, 100]`:

```bash
python test.py \
  --dataset CREMAD \
  --pretrained_model <checkpoint.pth> \
  --noise_type Salt \
  --noise_level 5 \
  --gpu_ids 0
```

### Noise test launch scripts

Run one checkpoint with explicit modality strengths:

```bash
bash test_noise.sh \
  CREMAD \
  results/cramed/udml/best_model.pth \
  Gaussian \
  5 \
  5 \
  0
```

The positional arguments are `DATASET CHECKPOINT NOISE_TYPE VISUAL_VARIANCE AUDIO_VARIANCE [GPU_ID] [OUTPUT_JSON]`. For example, replace `Gaussian 5 5` with `Salt 10 10`, or use different modality strengths such as `Gaussian 5 2`.

Run every discovered checkpoint under the four official robustness conditions Gaussian 5/10 and Salt 5/10:

```bash
bash run_noise_matrix.sh results/noise_matrix_seed0_p05
```

Both scripts default to audio/visual probability 0.5, seed 0, batch size 64, and 8 workers. These can be changed through `AUDIO_NOISE_PROB`, `VISUAL_NOISE_PROB`, `SEED`, `BATCH_SIZE`, and `NUM_WORKERS` environment variables.
