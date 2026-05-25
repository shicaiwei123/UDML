#!/bin/bash
#
# run_baseline_clean.sh — Baseline 无噪声训练+测试
#
# 和 run_baseline.sh 的区别:
#   --noise_level 0  → 测试集不加任何噪声（Gaussian/Salt）
#   训练和测试都在 clean 数据上进行
#
# 参考: Clean Test ~79%
#

set -e

source /root/miniconda3/bin/activate torch2.5
export HF_ENDPOINT=https://hf-mirror.com

GPU="${GPU:-0}"
NAME="${NAME:-baseline_clean}"
DATA_PATH="${DATA_PATH:-./datasets}"
TASK="${TASK:-MVSA_Single}"
MODEL="${MODEL:-latefusion}"
BATCH_SZ="${BATCH_SZ:-32}"
LR="${LR:-5e-5}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
PATIENCE="${PATIENCE:-50}"
GRAD_ACC="${GRAD_ACC:-1}"
N_WORKERS="${N_WORKERS:-4}"
SAVEDIR="${SAVEDIR:-./checkpoint}"

EXTRA_ARGS="$@"

CKPT_DIR="$SAVEDIR/$NAME"
if [ -d "$CKPT_DIR" ] && [ "${FORCE_FRESH:-0}" = "1" ]; then
    echo "删除旧 checkpoint: $CKPT_DIR"
    rm -rf "$CKPT_DIR"
elif [ -d "$CKPT_DIR" ]; then
    echo "旧 checkpoint 存在: $CKPT_DIR，从中断恢复。FORCE_FRESH=1 可重新开始。"
fi

CUDA_VISIBLE_DEVICES="$GPU" python -u train_qmf.py \
    --model "$MODEL" \
    --task "$TASK" \
    --data_path "$DATA_PATH" \
    --batch_sz "$BATCH_SZ" \
    --lr "$LR" \
    --max_epochs "$MAX_EPOCHS" \
    --patience "$PATIENCE" \
    --gradient_accumulation_steps "$GRAD_ACC" \
    --n_workers "$N_WORKERS" \
    --name "$NAME" \
    --savedir "$SAVEDIR" \
    --noise_level 0 \
    $EXTRA_ARGS

echo ""
echo "训练完成。Best model: $SAVEDIR/$NAME/model_best.pt"
