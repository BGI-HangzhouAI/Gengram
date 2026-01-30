#!/bin/bash

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=6005
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=$1        # <Specify path>
TENSORBOARD_LOGS_PATH=$2  # <Specify path>
DATA_PATH=$3              # <Specify path and file prefix>_text_document

# --- Tokenizer (SentencePiece) ---
TOKENIZER_TYPE=SentencePieceTokenizer
TOKENIZER_MODEL=/path/one_hot.bpe.model

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 16
    --hidden-size 1024
    --num-attention-heads 16
    --seq-length 2048
    --max-position-embeddings 2048
    --attention-backend auto
    --attention-output-gate

    # ---- Gengram ----
    --gengram-enabled \
    --gengram-layer-ids 6 \
    --gengram-ngram-sizes 2 3 4 5 6 \
    --gengram-embed-dim-per-ngram 512 \
    --gengram-token-ids 8 5 6 7 9 \
    --gengram-use-short-conv \
    --gengram-short-conv-kernel-size 4 \
    --gengram-window-size 12


    --group-query-attention      
    --num-query-groups 8        

    --normalization RMSNorm
    --apply-layernorm-1p

    --rotary-percent 0.25
    --rotary-base 10000000
    --apply-layernorm-1p

    --position-embedding-type rope
    --moe-shared-expert-gate
    --moe-shared-expert-intermediate-size 512

    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
)

TRAINING_ARGS=(
    --micro-batch-size 4
    --global-batch-size 32
    --train-iters 500
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --bf16
    --lr 2.0e-5
    --lr-decay-style cosine
    --min-lr 2.0e-6
    --lr-warmup-fraction .1
    --lr-decay-iters 430
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --tokenizer-type $TOKENIZER_TYPE
    --tokenizer-model $TOKENIZER_MODEL
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 20
    --save-interval 260
    --eval-interval 20
    --save $CHECKPOINT_PATH
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
