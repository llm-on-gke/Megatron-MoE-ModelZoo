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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=1
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false # to avoid HF warnings
export DEEPEP_COMM_TIMEOUT_MS=30000

TP=1
PP=8
EP=32
CP=1
MBS=1
GBS=1024
SEQ_LEN=4096
OUTPUT_PATH="/home/Megatron-MoE-ModelZoo/output/"
WORKSPACE="/home/Megatron-MoE-ModelZoo"

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
    #--tokenizer-type HuggingFaceTokenizer
    #--tokenizer-model deepseek-ai/DeepSeek-V3
    --make-vocab-size-divisible-by 3232
)

MODEL_ARGS=(
  # Distributed args
  --distributed-timeout-minutes: 60
  --tensor-model-parallel-size: ${TP}
  --pipeline-model-parallel-size: ${PP}
  --expert-model-parallel-size: ${EP}
  --context-parallel-size: ${CP}
  --expert-tensor-parallel-size: 1
  --use-distributed-optimizer: true

  # Training args
  --use-mcore-models: true
  --sequence-parallel: true
  --use-flash-attn: true
  --disable-bias-linear: true
  --micro-batch-size: ${MBS}
  --global-batch-size: ${GBS}
  --train-samples: 585937500
  --exit-duration-in-mins: 220
  --no-save-optim: false # set to False to save optim state. TODO(lit): check the ckpt size.
  --no-check-for-nan-in-loss-and-grad: true
  --cross-entropy-loss-fusion: true
  --cross-entropy-fusion-impl: native #te
  --manual-gc: true
  --manual-gc-interval: 10

  # Transformer Engine args
  --transformer-impl: transformer_engine

  # Data args
  --seq-length: ${SEQ_LEN}
  --data-cache-path: ${WORKSPACE}/data_cache
  --tokenizer-type: HuggingFaceTokenizer
  --tokenizer-model: deepseek-ai/DeepSeek-V3
  #--data-path: ${DATA_PATH}
  --mock-data
  --split: 99,1,0
  --no-mmap-bin-files: true
  --no-create-attention-mask-in-dataloader: true
  --num-workers: 6

  # Add network size args
  --num-layers: 14 # original 61 layers
  --hidden-size: 7168
  --ffn-hidden-size: 18432
  --num-attention-heads: 128
  --kv-channels: 128
  --max-position-embeddings: 4096
  --position-embedding-type: rope
  --rotary-base: 10000
  --make-vocab-size-divisible-by: 3232
  --normalization: RMSNorm
  --norm-epsilon: 1e-6
  --swiglu: true
  --untie-embeddings-and-output-weights: true
  --multi-latent-attention: true

  # Add regularization args
  --attention-dropout: 0.0
  --hidden-dropout: 0.0
  --clip-grad: 1.0
  --weight-decay: 0.1
  --qk-layernorm: true

  # Add learning rate args
  --lr-decay-samples: 584765624
  --lr-warmup-samples: 1536000
  # Learning rate scaled down from 7.3e-6 (DeepSeek-V3 technical report, GBS=15360) to 3.9e-6 (GBS=8192)
  --lr-warmup-init: 3.9e-7
  --lr: 3.9e-6
  --min-lr: 3.9e-7
  --lr-decay-style: cosine
  --adam-beta1: 0.9
  --adam-beta2: 0.95

  # Add MoE args
  --num-experts: 64 # local 4 + 1 shared, EP16
  --moe-layer-freq: "([0]*3+[1]*11)"
  --moe-ffn-hidden-size: 2048
  --moe-shared-expert-intermediate-size: 2048
  --moe-router-load-balancing-type: seq_aux_loss
  --moe-router-topk: 8
  # --moe-token-dispatcher-type: alltoall
  --moe-token-dispatcher-type: flex
  --moe-enable-deepep: true
  --moe-router-pre-softmax: true
  --moe-grouped-gemm: true
  --moe-aux-loss-coeff: 1e-4
  --moe-router-group-topk: 4
  --moe-router-num-groups: 8
  --moe-router-topk-scaling-factor: 2.5
  --moe-router-score-function: sigmoid
  --moe-router-enable-expert-bias: true
  --moe-router-bias-update-rate: 1e-3
  --moe-router-dtype: fp32
  --moe-permute-fusion: true

  # Add MLA args
  --q-lora-rank: 1536
  --kv-lora-rank: 512
  --qk-head-dim: 128
  --qk-pos-emb-head-dim: 64
  --v-head-dim: 128
  --rotary-scaling-factor: 40
  --mscale: 1.0
  --mscale-all-dim: 1.0

  --mtp-num-layers: 1
  --mtp-loss-scaling-factor: 0.1

  # Add validation args
  --eval-iters: 32
  --eval-interval: 200

  # Add checkpointing args
  --finetune: false
  --no-load-optim: true
  --no-load-rng: true
  --auto-detect-ckpt-format: true
  --load: ${OUTPUT_PATH}
  --save: ${OUTPUT_PATH}/checkpoints
  --save-interval: 500
  --dist-ckpt-strictness: log_all

  # Add initialization args
  --init-method-std: 0.02

  # Add logging args
  #--log-timers-to-tensorboard: true
  #--log-memory-to-tensorboard: true
  --log-num-zeros-in-grad: false
  --log-params-norm: false
  --log-validation-ppl-to-tensorboard: true
  --log-throughput: true
  --log-interval: 1
  --logging-level: 40
  --tensorboard-dir: ${OUTPUT_PATH}/tensorboard
  #--wandb-project: ${WANDB_PROJECT}
  #--wandb-exp-name: DeepSeek-V3-Proxy-TP${TP}PP${PP}EP${EP}CP${CP}VPP${VPP}-MBS${MBS}GBS${GBS}-${COMMENT}

  # Add mixed precision args
  --bf16: true

  # enable experimental
  --enable-experimental: true
)

torchrun \
    ${DISTRIBUTED_ARGS[@]} /home/Megatron-LM/pretrain_gpt.py  \
    ${TOKENIZER_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    "$@" # pass in extra or override arguments