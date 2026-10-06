#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

OUTPUT_DIR="${1:-results/noise_matrix_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/torch2.5.1/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
VISUAL_NOISE_PROB="${VISUAL_NOISE_PROB:-0.5}"
AUDIO_NOISE_PROB="${AUDIO_NOISE_PROB:-0.5}"

"$PYTHON_BIN" run_noise_matrix.py \
    --output-dir "$OUTPUT_DIR" \
    --conditions Gaussian:5 Gaussian:10 Salt:5 Salt:10 \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --visual-noise-prob "$VISUAL_NOISE_PROB" \
    --audio-noise-prob "$AUDIO_NOISE_PROB"

echo "Saved matrix to $OUTPUT_DIR"
