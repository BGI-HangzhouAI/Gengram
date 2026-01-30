#!/bin/bash

# Mixtral Forward对比检查脚本
# 基于用户提供的模型参数和路径配置

# 设置环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=/mnt/zzbnew/peixunban/xxhn/Megatron-LM:$PYTHONPATH

# 模型路径配置
MG_CKPT_PATH="/mnt/zzbnew/peixunban/xxhn/checkpoints"
TOKENIZER_PATH="/mnt/workspace/users/xz/tokenizer/one_hot.bpe.model"

# 并行配置
TENSOR_MODEL_PARALLEL_SIZE=1
PIPELINE_MODEL_PARALLEL_SIZE=1
EXPERT_MODEL_PARALLEL_SIZE=1


# 运行检查脚本
# python /mnt/workspace/users/chenjh356/embedding/mg_embedding.py \
NCCL_DEBUG=WARN torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=6000 mg_embedding_new.py \
    --mg-ckpt-path $MG_CKPT_PATH \
    --load $MG_CKPT_PATH \
    --tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE \
    --pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE \
    --expert-model-parallel-size $EXPERT_MODEL_PARALLEL_SIZE \
    --use-flash-attn \
    --bf16 \
    --use-hf-tokenizer \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model $TOKENIZER_PATH \
    --disable-bias-linear \
    --sequence-parallel \
    --context-parallel-size 1 \
    --num-layers 16 --hidden-size 1024 --num-attention-heads 16 --seq-length 2048 --max-position-embeddings 2048 --attention-backend auto --attention-output-gate --no-weight-decay-cond-type qwen3_next --linear-attention-type gated_delta_net --linear-attention-freq 4 --linear-conv-kernel-dim 4 --linear-key-head-dim 128 --linear-value-head-dim 128 --linear-num-key-heads 16 --linear-num-value-heads 32 --normalization RMSNorm --apply-layernorm-1p --rotary-percent 0.25 --rotary-base 10000000 --apply-layernorm-1p --position-embedding-type rope --moe-shared-expert-gate --moe-shared-expert-intermediate-size 512 --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1 --micro-batch-size 4 --global-batch-size 32 --train-iters 5 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.006 --clip-grad 1.0 --bf16 --lr 2.0e-5 --lr-decay-style cosine --min-lr 2.0e-6 --lr-warmup-fraction .1 --lr-decay-iters 430 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --data-path --tokenizer-type SentencePieceTokenizer --tokenizer-model /mnt/zzbnew/peixunban/xxhn/one_hot.bpe.model --split 949,50,1 --log-interval 20 --save-interval 2 --eval-interval 20
    # --num-layers $NUM_LAYERS \
    # --hidden-size $HIDDEN_SIZE \
    # --num-attention-heads $NUM_ATTENTION_HEADS \
    # --ffn-hidden-size $INTERMEDIATE_SIZE \
    # --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    # --vocab-size $VOCAB_SIZE \
    # --num-experts $NUM_EXPERTS \
    # --moe-router-topk $NUM_EXPERTS_PER_TOK \
echo "检查完成!"