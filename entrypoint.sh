#!/bin/bash
# vLLM startup for Qwen3.6 on RTX PRO 6000 Blackwell (96GB)
# Applies both 27B-FP8 and 35B-A3B-FP8 via MODEL env var.
set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_FLOAT32_MATMUL_PRECISION=high
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export NCCL_CUMEM_ENABLE=0
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=8

# ── Configurable defaults ────────────────────────────────────────────────────
MODEL="${MODEL:-/models/Qwen3.6-27B-FP8}"
MODEL_NAME="${MODEL_NAME:-qwen3.6-27b}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"
PORT="${PORT:-8000}"

echo "Starting vLLM: model=$MODEL, name=$MODEL_NAME, context=$MAX_MODEL_LEN, mem=$GPU_MEM_UTIL, MTP n=$NUM_SPEC_TOKENS"

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --trust-remote-code \
  --dtype auto \
  --attention-backend flashinfer \
  --kv-cache-dtype fp8_e5m2 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens 2048 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --language-model-only \
  --speculative-config "{\"model\": \"$MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"draft_model_uses_mrope\": true, \"draft_model_uses_xdrope_dim\": 0}"
