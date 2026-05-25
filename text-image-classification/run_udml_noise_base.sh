#!/bin/bash
#
# run_udml_noise_base.sh — 启动 train_udml_noise_base.py (新dataset: word级mask + 50% guard)
#
# 使用方法:
#   bash run_udml_noise_base.sh                    # 两阶段 noisy 训练
#   bash run_udml_noise_base.sh --noise_free 1     # clean-only
#
# 两阶段 (cylcle_epoch=10):
#   epoch < 10  : clean_loader（原始数据）
#   epoch >= 10 : noise_loader（word masking + 高斯噪声，各50%概率不施加）
#   epoch >= 20 : warmup 结束，动态权重生效
#

set -e

# ── 1. 环境 ──────────────────────────────────────────────
source /root/miniconda3/bin/activate torch2.5
export HF_ENDPOINT=https://hf-mirror.com

# ── 2. 参数 ───────────────────────────────────────────────
GPU="${GPU:-0}"                  # 指定 GPU，如 GPU=0 或 GPU=0,1
NAME="${NAME:-udml_noise_base_2}"
DATA_PATH="${DATA_PATH:-./datasets}"
TASK="${TASK:-MVSA_Single}"
BATCH_SZ="${BATCH_SZ:-32}"
LR="${LR:-5e-5}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"
N_WORKERS="${N_WORKERS:-4}"
SAVEDIR="${SAVEDIR:-./checkpoint}"

# UDML 参数
FUSION_DIM="${FUSION_DIM:-2048}"
GAMMA="${GAMMA:-4.0}"
BETA="${BETA:-1e-3}"
CYL_CLE="${CYL_CLE:-10}"
AUDIO_DEPEND="${AUDIO_DEPEND:-1.0}"
VISUAL_DEPEND="${VISUAL_DEPEND:-1.0}"

EXTRA_ARGS="$@"

# ── 3. 清理旧 checkpoint ──────────────────────────────────
CKPT_DIR="$SAVEDIR/$NAME"
if [ -d "$CKPT_DIR" ] && [ "${FORCE_FRESH:-0}" = "1" ]; then
    echo "删除旧 checkpoint: $CKPT_DIR"
    rm -rf "$CKPT_DIR"
elif [ -d "$CKPT_DIR" ]; then
    echo "旧 checkpoint 存在: $CKPT_DIR，将恢复训练。FORCE_FRESH=1 可重新开始。"
fi

# ── 4. 训练 ───────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="$GPU" python -u train_udml_noise_base.py \
    --task "$TASK" \
    --data_path "$DATA_PATH" \
    --batch_sz "$BATCH_SZ" \
    --lr "$LR" \
    --max_epochs "$MAX_EPOCHS" \
    --patience "$PATIENCE" \
    --n_workers "$N_WORKERS" \
    --name "$NAME" \
    --savedir "$SAVEDIR" \
    --fusion_dim "$FUSION_DIM" \
    --gamma "$GAMMA" \
    --beta "$BETA" \
    --cylcle_epoch "$CYL_CLE" \
    --audio_depend "${AUDIO_DEPEND}" \
    --visual_depend "${VISUAL_DEPEND}" \
    $EXTRA_ARGS

echo ""
echo "训练完成。Best model: $SAVEDIR/$NAME/model_best.pt"
