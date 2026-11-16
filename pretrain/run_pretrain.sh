#!/bin/bash
# Continued multimodal pre-training (paper setting: 128 x 4096 tokens per step, lr 2e-5, one epoch).
# Usage: bash run_pretrain.sh [extra train.py args]   (re-running resumes from the latest checkpoint)
# Multi-node: set NNODES / NODE_RANK / MASTER_ADDR on every node.
set -e
cd "$(dirname "$0")"

MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}                 # or a local path, e.g. Qwen3.8-27B
OUTPUT=${OUTPUT:-./outputs/pretrain_qwen3vl_8b}
# 'path[:weight]'  weight = passes per epoch (fractions subsample, >1 repeats)
DATA=${DATA:-"/path/to/PubMedVision-Plus:1"}

SEQ_LEN=${SEQ_LEN:-32768}                         # packed row per GPU, samples stay isolated inside it
MAX_LEN=${MAX_LEN:-4096}                          # max tokens of one sample
GBS=${GBS:-$((128 * 4096 / SEQ_LEN))}             # rows per step: 128 x 4096 tokens

GPUS=${GPUS:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUTPUT"
torchrun --nproc_per_node $GPUS --nnodes $NNODES --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    train.py \
    --model_path "$MODEL" \
    --data $DATA \
    --output_dir "$OUTPUT" \
    --seq_len $SEQ_LEN \
    --max_len $MAX_LEN \
    --global_batch_size $GBS \
    --lr 2e-5 \
    --warmup_ratio 0.01 \
    --num_epochs 1 \
    --max_pixels 2097152 \
    --save_steps 1000 \
    --resume \
    "$@" 2>&1 | tee -a "$OUTPUT/train.log"
