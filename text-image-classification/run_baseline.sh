#!/bin/bash
#
# run_baseline.sh — 训练 Baseline (latefusion) on MVSA_Single
#
# 性能参考: Clean Test 79.19%, Noisy Test 79.38%
# 训练环境: torch 2.5.1, CUDA 12.4, RTX 4090
#
# 使用方法:
#   bash run_baseline.sh                  # 默认训练
#   bash run_baseline.sh --max_epochs 50  # 自定义 epoch
#

set -e

# ── 1. 环境激活 ──────────────────────────────────────────────
source /root/miniconda3/bin/activate torch2.5
export HF_ENDPOINT=https://hf-mirror.com

# ── 2. 参数 ───────────────────────────────────────────────────
GPU="${GPU:-0}"
NAME="${NAME:-baseline_mvsa}"
DATA_PATH="${DATA_PATH:-./datasets}"
TASK="${TASK:-MVSA_Single}"
MODEL="${MODEL:-latefusion}"
BATCH_SZ="${BATCH_SZ:-32}"
LR="${LR:-5e-5}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
PATIENCE="${PATIENCE:-5}"
GRAD_ACC="${GRAD_ACC:-1}"
N_WORKERS="${N_WORKERS:-4}"
SAVEDIR="${SAVEDIR:-./checkpoint}"

# 额外传入的参数会追加到末尾
EXTRA_ARGS="$@"

# ── 3. 清理旧 checkpoint（默认不清理，防止误删）───────────────
CKPT_DIR="$SAVEDIR/$NAME"
if [ -d "$CKPT_DIR" ] && [ "${FORCE_FRESH:-0}" = "1" ]; then
    echo "⚠️  删除旧 checkpoint: $CKPT_DIR"
    rm -rf "$CKPT_DIR"
elif [ -d "$CKPT_DIR" ]; then
    echo "ℹ️  旧 checkpoint 存在: $CKPT_DIR"
    echo "   从中断处恢复训练。若要重新开始，设置: FORCE_FRESH=1"
fi

# ── 4. 训练 ───────────────────────────────────────────────────
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
    $EXTRA_ARGS

echo ""
echo "训练完成。Best model saved at: $SAVEDIR/$NAME/model_best.pt"
