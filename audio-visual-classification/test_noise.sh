#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ "$#" -lt 5 ]; then
    echo "Usage: bash test_noise.sh DATASET CHECKPOINT NOISE_TYPE VISUAL_VARIANCE AUDIO_VARIANCE [GPU_ID] [OUTPUT_JSON]"
    echo "Example: bash test_noise.sh CREMAD results/cramed/udml/best_model.pth Gaussian 5 5 0"
    exit 1
fi

DATASET="$1"
CHECKPOINT="$2"
NOISE_TYPE="$3"
VISUAL_VARIANCE="$4"
AUDIO_VARIANCE="$5"
GPU_ID="${6:-0}"

case "$DATASET" in
    CREMAD)
        NUM_FRAME=1
        ;;
    KineticSound)
        NUM_FRAME=3
        ;;
    *)
        echo "DATASET must be CREMAD or KineticSound"
        exit 1
        ;;
esac

case "$NOISE_TYPE" in
    Gaussian|Salt|None)
        ;;
    *)
        echo "NOISE_TYPE must be Gaussian, Salt, or None"
        exit 1
        ;;
esac

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/torch2.5.1/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
VISUAL_NOISE_PROB="${VISUAL_NOISE_PROB:-0.5}"
AUDIO_NOISE_PROB="${AUDIO_NOISE_PROB:-0.5}"

CHECKPOINT_NAME="$(basename "$CHECKPOINT" .pth)"
NOISE_NAME="$(printf '%s' "$NOISE_TYPE" | tr '[:upper:]' '[:lower:]')"
DEFAULT_OUTPUT="results/noise_tests/${CHECKPOINT_NAME}__${DATASET}__${NOISE_NAME}_v${VISUAL_VARIANCE}_a${AUDIO_VARIANCE}.json"
OUTPUT_JSON="${7:-$DEFAULT_OUTPUT}"
mkdir -p "$(dirname "$OUTPUT_JSON")"

SHARED_LEVEL_ARGS=()
if [ "$VISUAL_VARIANCE" = "$AUDIO_VARIANCE" ]; then
    SHARED_LEVEL_ARGS=(--noise_level "$VISUAL_VARIANCE")
fi

"$PYTHON_BIN" test.py \
    --dataset "$DATASET" \
    --pretrained_model "$CHECKPOINT" \
    --noise_type "$NOISE_TYPE" \
    "${SHARED_LEVEL_ARGS[@]}" \
    --visual_variance "$VISUAL_VARIANCE" \
    --audio_variance "$AUDIO_VARIANCE" \
    --visual_noise_prob "$VISUAL_NOISE_PROB" \
    --audio_noise_prob "$AUDIO_NOISE_PROB" \
    --num_frame "$NUM_FRAME" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --gpu_ids "$GPU_ID" \
    --output "$OUTPUT_JSON"

echo "Saved result to $OUTPUT_JSON"
