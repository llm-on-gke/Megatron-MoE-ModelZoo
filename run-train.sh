set -euxo pipefail

source /usr/local/gib/scripts/set_nccl_env.sh
export NCCL_SOCKET_IFNAME="eth0,eth1"
export NCCL_TUNER_CONFIG_PATH=/usr/local/gib/configs/tuner_config_a4.txtpb

export TRITON_CACHE_DIR="/tmp/triton-cache/"
#export 
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=1
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export PYTHONWARNINGS="ignore"

#add
export DEEPEP_COMM_TIMEOUT_MS=30000 

chmod +x /home/Megatron-LM/pretrain_gpt.py

#--moe-shared-expert-overlap \
#--moe-token-dispatcher-type alltoall \
#--recompute-granularity selective \
#--recompute-modules mla_up_proj moe mlp layernorm \
#--fp8-recipe blockwise, GEMM does not support B200 
#--moe-router-padding-for-fp8 \
#--fp8-recipe $FP8_RECIPE \
#--fp8-format e4m3 \
TP=1
PP=16
EP=8
CP=1
MBS=1
GBS=512
SEQ_LEN=2048
OUTPUT_PATH="/home/Megatron-MoE-ModelZoo/output/"
WORKSPACE="/home/Megatron-MoE-ModelZoo"

NVSHMEM_DEBUG=INFO PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" OMP_NUM_THREADS=8 PYTHON_PATH=/home/Megatron-LM TORCH_NCCL_ENABLE_MONITORING=0 DEEPEP_COMM_TIMEOUT_MS=30000 torchrun \
        --nproc_per_node 8 \
        --nnodes $NNODES \
        --node_rank $NODE_RANK \
        --master_addr $MASTER_ADDR \
        --rdzv_id="${JOB_IDENTIFIER}" \
        --rdzv_backend static \
        --master_port $MASTER_PORT /home/Megatron-LM/pretrain_gpt.py \
        --distributed-timeout-minutes 60 \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --expert-model-parallel-size ${EP} \
        --context-parallel-size ${CP} \
        --expert-tensor-parallel-size 1 \
        --use-distributed-optimizer  \
        --use-mcore-models  \
        --sequence-parallel  \
        --use-flash-attn  \
        --disable-bias-linear  \
        --micro-batch-size ${MBS} \
        --global-batch-size ${GBS} \
        --train-samples 6552800 \
        --no-save-optim  \
        --no-check-for-nan-in-loss-and-grad  \
        --cross-entropy-loss-fusion  \
        --cross-entropy-fusion-impl te \
        --manual-gc  \
        --manual-gc-interval 10 \
        --transformer-impl transformer_engine \
        --seq-length ${SEQ_LEN} \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model deepseek-ai/DeepSeek-V3 \
        --mock-data  \
        --split 99,1,0 \
        --no-mmap-bin-files  \
        --no-create-attention-mask-in-dataloader  \
        --num-workers 6 \
        --num-layers 61 \
        --hidden-size 7168 \
        --ffn-hidden-size 18432 \
        --num-attention-heads 128 \
        --kv-channels 128 \
        --max-position-embeddings 4096 \
        --position-embedding-type rope \
        --rotary-base 10000 \
        --make-vocab-size-divisible-by 3232 \
        --normalization RMSNorm \
        --norm-epsilon 1e-6 \
        --swiglu  \
        --untie-embeddings-and-output-weights  \
        --multi-latent-attention  \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --qk-layernorm  \
        --lr-decay-samples 584765624 \
        --lr-warmup-samples 1536000 \
        --lr-warmup-init 3.9e-7 \
        --lr 3.9e-7 \
        --min-lr 3.9e-7 \
        --lr-decay-style cosine \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --num-experts 8 \
        --moe-layer-freq "([0]*3+[1]*58)" \
        --moe-ffn-hidden-size 2048 \
        --moe-shared-expert-intermediate-size 2048 \
        --moe-router-load-balancing-type seq_aux_loss \
        --moe-router-topk 2 \
        --moe-token-dispatcher-type flex \
        --moe-enable-deepep  \
        --moe-router-pre-softmax  \
        --moe-aux-loss-coeff 1e-4 \
        --moe-router-group-topk 4 \
        --moe-router-num-groups 8 \
        --moe-router-topk-scaling-factor 2.5 \
        --moe-router-score-function sigmoid \
        --moe-router-enable-expert-bias  \
        --moe-router-bias-update-rate 1e-3 \
        --moe-router-dtype fp32 \
        --moe-permute-fusion  \
        --q-lora-rank 1536 \
        --kv-lora-rank 512 \
        --qk-head-dim 128 \
        --qk-pos-emb-head-dim 64 \
        --v-head-dim 128 \
        --rotary-scaling-factor 40 \
        --mscale 1.0 \
        --mscale-all-dim 1.0 \
        --mtp-num-layers 1 \
        --mtp-loss-scaling-factor 0.1 \
        --eval-iters 32 \
        --eval-interval 10000000 \
        --no-load-optim  \
        --no-load-rng  \
        --auto-detect-ckpt-format  \
        --save ${OUTPUT_PATH}/Megatron-MoE-ModelZoo-workspace/Megatron-MoE-ModelZoo/output/mcore-benchmarking-vyour_own_megatron_version/DeepSeek-V3-TP1PP8EP32VPP4CP1-MBS1GBS8192/checkpoints \
        --save-interval 10000000 \
        --dist-ckpt-strictness log_all \
        --init-method-std 0.02 \
        --log-throughput  \
        --log-interval 1 \
        --logging-level 40 \
        --tensorboard-dir ${OUTPUT_PATH}/Megatron-MoE-ModelZoo-workspace/Megatron-MoE-ModelZoo/output/mcore-benchmarking-vyour_own_megatron_version/DeepSeek-V3-TP1PP8EP32VPP4CP1-MBS1GBS8192/tensorboard \
        --bf16  \
        --moe-router-padding-for-fp8 \
        --fp8-recipe $FP8_RECIPE \
        --fp8-format e4m3 \
        --enable-experimental   \
        --recompute-granularity selective \
        --recompute-modules mla_up_proj moe mlp layernorm \
        --pipeline-model-parallel-layout "Et*2|(tt|)*22t|(tt|)*7mL" \
        --use-precision-aware-optimizer \
        --main-grads-dtype fp32 \
        --main-params-dtype fp32 \
        --exp-avg-dtype bf16 \
        --exp-avg-sq-dtype bf16 \
        --overlap-grad-reduce \
        --overlap-param-gather