#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

NOISE_TYPE="${1:-Gaussian}"
CYCLE_EPOCH="${2:-50}"
GPU_ID="${3:-0}"

case "$NOISE_TYPE" in
    Gaussian|Salt)
        ;;
    *)
        echo "NOISE_TYPE must be Gaussian or Salt"
        echo "Usage: bash train_udml_noise_cycle.sh [NOISE_TYPE] [CYCLE_EPOCH] [GPU_ID] [CKPT_DIR]"
        exit 1
        ;;
esac

NOISE_NAME="$(printf '%s' "$NOISE_TYPE" | tr '[:upper:]' '[:lower:]')"
CKPT_DIR="${4:-./results/cramed/udml_noise_cycle${CYCLE_EPOCH}_${NOISE_NAME}_0_11}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/torch2.5.1/bin/python}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
AUDIO_VARIANCE_MIN="${AUDIO_VARIANCE_MIN:-0}"
AUDIO_VARIANCE_MAX="${AUDIO_VARIANCE_MAX:-11}"
VISUAL_VARIANCE_MIN="${VISUAL_VARIANCE_MIN:-0}"
VISUAL_VARIANCE_MAX="${VISUAL_VARIANCE_MAX:-11}"
AUDIO_NOISE_PROB="${AUDIO_NOISE_PROB:-0.5}"
VISUAL_NOISE_PROB="${VISUAL_NOISE_PROB:-0.5}"

mkdir -p "$CKPT_DIR"

"$PYTHON_BIN" main_auxi_weight_udml.py \
    --dataset CREMAD \
    --ckpt_path "$CKPT_DIR" \
    --modality full \
    --fusion_method concat \
    --modulation Normal \
    --train \
    --num_frame 1 \
    --fps 1 \
    --pe 1 \
    --noise_type "$NOISE_TYPE" \
    --train_audio_variance_min "$AUDIO_VARIANCE_MIN" \
    --train_audio_variance_max "$AUDIO_VARIANCE_MAX" \
    --train_visual_variance_min "$VISUAL_VARIANCE_MIN" \
    --train_visual_variance_max "$VISUAL_VARIANCE_MAX" \
    --train_audio_noise_prob "$AUDIO_NOISE_PROB" \
    --train_visual_noise_prob "$VISUAL_NOISE_PROB" \
    --cylcle_epoch "$CYCLE_EPOCH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --optimizer sgd \
    --learning_rate 0.001 \
    --beta 1e-5 \
    --gamma 4.0 \
    --gpu_ids "$GPU_ID"
