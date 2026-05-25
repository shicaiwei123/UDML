#!/bin/bash
#
# eval_udml_noise.sh — 评估训好的 UDML 模型在不同噪声强度下的表现
#
# 使用方法:
#   bash eval_udml_noise.sh                                         # 默认强度 0~5
#   bash eval_udml_noise.sh --checkpoint <path> --strengths "0,1,3,5"
#
# 默认 checkpoint: ./checkpoint/udml_noise_base/model_best.pt
#

set -e

source /root/miniconda3/bin/activate torch2.5
export HF_ENDPOINT=https://hf-mirror.com

GPU="${GPU:-0}"
CKPT="${CKPT:-./checkpoint/noise_gamma4_15/model_best.pt}"
DEPEND="${DEPEND:-./checkpoint/noise_gamma4_15/model_best_depend.pt}"
TASK="${TASK:-MVSA_Single}"
STRENGTHS="${STRENGTHS:-0,5,10}"
BATCH_SZ="${BATCH_SZ:-32}"

EXTRA_ARGS="$@"

CUDA_VISIBLE_DEVICES="$GPU" python -u eval_udml_noise.py \
    --checkpoint "$CKPT" \
    --depend "$DEPEND" \
    --task "$TASK" \
    --strengths "$STRENGTHS" \
    --batch_sz "$BATCH_SZ" \
    $EXTRA_ARGS
