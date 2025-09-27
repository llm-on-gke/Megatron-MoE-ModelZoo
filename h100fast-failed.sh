#!/bin/bash
set -euxo pipefail

source /usr/local/gib/scripts/set_nccl_env.sh
export NCCL_SOCKET_IFNAME="eth0,eth1"
export NCCL_TUNER_CONFIG_PATH=/usr/local/gib/configs/tuner_config_a4.txtpb

export TRITON_CACHE_DIR="/tmp/triton-cache/"
#export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NVTE_FWD_LAYERNORM_SM_MARGIN=20
export NVTE_BWD_LAYERNORM_SM_MARGIN=20
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=1
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false # to avoid HF warnings
export DEEPEP_COMM_TIMEOUT_MS=30000

DISTRIBUTED_ARGS=(
    --nproc_per_node 8
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --rdzv_id="${JOB_IDENTIFIER}"
    --rdzv_backend static
)

TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model deepseek-ai/DeepSeek-V3
    --make-vocab-size-divisible-by 3232
)

DATA_ARGS=(
#     --data-path /mnt/nvme/deepseek-v3-data2/processed_data_text_document
    --split 99,1,0
    --no-create-attention-mask-in-dataloader
    --no-mmap-bin-files
    --num-workers 6
#     --data-cache-path /mnt/nvme/tmp/dataset_cache
    # Alternatively, if you don't want to use a dataset, you can use mock data:
    --mock-data \
    # --lr-warmup-samples 1536000

    # Iteration-based training
    --train-iters 10000
    --lr-decay-iters 10000
    --lr-warmup-iters 1000

    # Sample-based training
    # --train-samples 268554688 \
    # --lr-decay-samples 584765624
    # --lr-warmup-samples 1536000
)

PERF_ARGS=(
    # Parallel args
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 8
    --expert-model-parallel-size 32
    --context-parallel-size 1
    --expert-tensor-parallel-size 1

    # layout
    #B200 best practice
    #PR=mxfp8 A2A_OVERLAP=1 TP=1 PP=8 EP=32 NNODES=32 GBS=2048 bash sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules mla_up_proj mlp --pipeline-model-parallel-layout "Et*3|(tt|)*29|mL"

    # `"Et*2|(tt|)*22t|(tt|)*7mL"` would include the `mtp` loss; the checkpoint we converted did not include the `mtp` loss
    # --pipeline-model-parallel-layout "Et*2|(tt|)*22t|(tt|)*7L"
    --pipeline-model-parallel-layout "Et*3|(tt|)*29|mL"

    # # Recompute args (activation checkpointing)
    # --recompute-granularity full
    # --recompute-method uniform
    # --recompute-num-layers 1
    # Instead of the above, you can use selective recomputation
    # but this doesn't really work well with our EFA setup
    # --recompute-granularity selective
    #rick
    --recompute-granularity selective
    --recompute-modules mla_up_proj moe mlp layernorm moe_act core_attn

    # # Offload args
    # --optimizer-cpu-offload
    # --overlap-cpu-optimizer-d2h-h2d

    # Overlap args
    --overlap-grad-reduce
    --overlap-param-gather
)

TRAINING_ARGS=(
    # Key args
    --seq-length 4096
    --micro-batch-size 1
    --global-batch-size 2048

    # Optimizer args
    --lr-warmup-init 3.9e-7
    --lr 3.9e-6
    --min-lr 3.9e-7
    --lr-decay-style cosine
    --adam-beta1 0.9
    --adam-beta2 0.95

    # Distributed optimizer args
    --use-distributed-optimizer
    --use-precision-aware-optimizer
    --main-grads-dtype fp32
    --main-params-dtype fp32
    --exp-avg-dtype bf16
    --exp-avg-sq-dtype bf16

    # Training args
    #--sequence-parallel
    --use-flash-attn
    --no-save-optim
    --no-check-for-nan-in-loss-and-grad
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --manual-gc
    --manual-gc-interval 10
    --transformer-impl transformer_engine

    # Regularization args
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --clip-grad 1.0
    --weight-decay 0.1
    --qk-layernorm

    # Initialization args
    --init-method-std 0.02

    # Add mixed precision args
    --bf16

    # Enable experimental args
    --enable-experimental

    # Misc args
    --distributed-timeout-minutes 60
)

CHECKPOINTING_ARGS=(
    --load /mnt/nvme/model
    --save /mnt/nvme/model
    --save-interval 500
    --no-load-optim
    --no-load-rng
    --auto-detect-ckpt-format
    --dist-ckpt-strictness log_all
)

LOGGING_ARGS=(
    #--log-memory-to-tensorboard
    #--log-validation-ppl-to-tensorboard
    --log-throughput
    --log-interval 1
    --logging-level 40
    --tensorboard-dir /mnt/nvme/model/tensorboard
)

# EVAL_ARGS=(
#     --eval-iters 10000
#     --eval-interval 10000000
# )

NETWORK_ARGS=(
    --disable-bias-linear
    --num-layers 61
    --hidden-size 7168
    --ffn-hidden-size 18432
    --num-attention-heads 128
    --kv-channels 128
    --max-position-embeddings 4096
    --position-embedding-type rope
    --rotary-base 10000
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --swiglu
    --untie-embeddings-and-output-weights
    --multi-latent-attention
)

MOE_ARGS=(
    --num-experts 256
    --moe-layer-freq "([0]*3+[1]*58)"
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk 8
    --moe-router-pre-softmax
    --moe-grouped-gemm
    --moe-aux-loss-coeff 1e-4
    --moe-router-group-topk 4
    --moe-router-num-groups 8
    --moe-router-topk-scaling-factor 2.5
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-router-dtype fp32
    --moe-permute-fusion
    # --moe-router-fusion
    # --moe-router-padding-for-fp8

    # --moe-token-dispatcher-type alltoall
    # The following are not compatible with our EFA setup
    # They are infiniband only: they can make the training go much faster
    --moe-enable-deepep
    --moe-token-dispatcher-type flex
)

MLA_ARGS=(
    --q-lora-rank 1536
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --rotary-scaling-factor 40
    --mscale 1.0
    --mscale-all-dim 1.0
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
)

FP8_ARGS=(
     #-fp8-recipe mxfp8
     #--fp8-format e4m3
)

NEW_1F1A_ARGS=(
    # --delay-wgrad-compute
    # --overlap-moe-expert-parallel-comm
)

torchrun \
    ${DISTRIBUTED_ARGS[@]} /home/Megatron-LM/pretrain_gpt.py  \
    ${DATA_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${CHECKPOINTING_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${EVAL_ARGS[@]} \
    ${NETWORK_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${MLA_ARGS[@]} \
    ${FP8_ARGS[@]} \
    ${NEW_1F1A_ARGS[@]} \
    "$@" # pass in extra or override arguments