# Qwen3.6 on RTX PRO 6000 Blackwell — 120 t/s per GPU, step by step

Qwen3.6 27B runs well on the RTX PRO 6000 Blackwell (96GB), but getting vLLM + MTP speculative decoding to actually work on sm_120 in WSL2 requires workarounds that aren't documented anywhere in one place. This is that guide — from a fresh WSL2 install to a working OpenAI-compatible API server. Tested on WSL2; bare-metal Linux should be identical or faster.

*Last tested: May 2026 — vLLM v0.19.2rc1.dev214, CUDA 13.0 (PyPI), NVIDIA driver 581.42, Ubuntu 24.04 on WSL2.*

## The Numbers (empirically verified, May 2026)

All numbers measured on RTX PRO 6000 Blackwell (96GB, 1792 GB/s), 3+ runs of 1024 output tokens each:

| Config | Hardware | t/s per GPU | Notes |
|---|---|---|---|
| Eager mode, no MTP, no CUDA graphs | 1x RTX PRO 6000 | **24** | Bandwidth floor |
| CUDA graphs + flash_attn, no MTP | 1x RTX PRO 6000 | **75** | 3.1x — matches 1792 GB/s theory |
| **CUDA graphs + flash_attn + MTP n=3** | **1x RTX PRO 6000** | **120** | **27B Dense, INT4** |
| CUDA graphs + flash_attn + MTP n=3 | 1x RTX PRO 6000 | **200** | **35B-A3B MoE, FP8** |
| CUDA graphs + flash_attn + MTP n=5 | 1x RTX PRO 6000 | **125** | +4% over n=3, marginal |

MTP speculative decoding metrics: mean acceptance length 3.19, per-position acceptance 0.87/0.72/0.60, avg draft acceptance 73%.

**Multi-GPU:** Each GPU runs its own independent vLLM instance with `--max-num-seqs 8`. With 2x RTX PRO 6000, that's two independent servers handling 8 concurrent users each — no NCCL, no tensor parallel, no shared state. vLLM's continuous batching handles concurrency within each GPU. A lightweight bridge in front does health-aware least-connections routing.

### Bandwidth parity note

The RTX PRO 6000 and RTX 5090 share **identical memory bandwidth** (1792 GB/s, 512-bit GDDR7 @ 28 Gbps). The PRO 6000's advantage is purely capacity (96 vs 32 GB), not speed per token. If you have a 5090, expect the same tok/s at equivalent configs.

---

## Prerequisites

- RTX PRO 6000 Blackwell (96GB) or RTX 5090 (32GB — use INT4 quant instead of FP8)
- Windows 11 + WSL2 with Ubuntu 24.04 (tested; bare-metal Linux should be identical)
- ~25GB free disk space for the INT4 model, ~40GB for FP8
- Basic comfort with the terminal

### CRITICAL: Prevent WSL2 auto-shutdown

By default, WSL2 shuts down after ~15 seconds of idle (no active terminal sessions). This kills long-running vLLM services when you disconnect SSH. Add `vmIdleTimeout=-1` to your `.wslconfig` **before** setting up vLLM:

```ini
# C:\Users\<you>\.wslconfig  (on Windows, NOT inside WSL)
[wsl2]
memory=200GB
swap=32GB
processors=24
vmIdleTimeout=-1
```

Then `wsl --shutdown` from PowerShell and reconnect.

---

## Step 1 — WSL2 + Ubuntu 24.04

**Why:** vLLM is Linux-native. WSL2 provides a full Linux kernel and near-zero-overhead CUDA pass-through.

```bash
# In PowerShell (as Admin)
wsl --install -d Ubuntu-24.04
wsl --set-default-version 2
# Reboot, set up your Ubuntu user, then:
wsl -d Ubuntu-24.04
```

---

## Step 2 — System packages

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv ninja-build curl
```

`ninja-build` is **mandatory** — flashinfer's JIT compiler invokes it directly.

You do **not** need a system-wide CUDA toolkit (`apt install cuda-toolkit-12-8`). We use the PyPI-shipped CUDA 13 components instead. If you have `/usr/local/cuda` pointing at CUDA 12.8 from an earlier install, that's fine — the setup explicitly works around it.

---

## Step 3 — Create the vLLM virtualenv

```bash
sudo mkdir -p /opt
sudo python3.12 -m venv /opt/vllm-env
sudo /opt/vllm-env/bin/pip install --upgrade pip wheel
```

---

## Step 4 — Install vLLM nightly + CUDA 13

The stable vLLM 0.19.1 has a WSL2 MTP bug (CUDA driver error during subprocess fork). Use the nightly:

```bash
WHEEL=$(curl -s "https://wheels.vllm.ai/nightly/cu130/vllm/" \
  | grep -oE 'href="[^"]+x86_64\.whl"' | head -1 \
  | sed 's/href="//;s/"$//')
URL="https://wheels.vllm.ai/nightly/cu130/vllm/${WHEEL}"
echo "Installing $URL"
sudo /opt/vllm-env/bin/pip install --upgrade "$URL"

# Add CUDA 13 dev pieces for flashinfer
sudo /opt/vllm-env/bin/pip install nvidia-cuda-nvcc nvidia-cuda-cccl
```

### Sanity check

```bash
/opt/vllm-env/bin/python -c "
import torch, vllm
print(f'vLLM    {vllm.__version__}')
print(f'torch   {torch.__version__}')
print(f'CUDA    {torch.version.cuda}')
print(f'GPU     {torch.cuda.get_device_name(0)}')
print(f'sm      {torch.cuda.get_device_capability(0)}')
"
```

---

## Step 5 — Download the model

```bash
sudo mkdir -p /opt/models
export HF_TOKEN=hf_xxxxxxxxxx
sudo HF_TOKEN=$HF_TOKEN /opt/vllm-env/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Lorbus/Qwen3.6-27B-int4-AutoRound',
    local_dir='/opt/models/Qwen3.6-27B-int4-AutoRound',
    max_workers=4,
)
"
```

### GPU 1 — 35B-A3B MoE FP8 (optional)

The MoE variant delivers ~200 t/s on the second GPU. Requires the official
FP8 model (the GPTQ-Int4 quant lacks MTP weights):

```bash
sudo mkdir -p /opt/models
export HF_TOKEN=hf_xxxxxxxxxx
sudo HF_TOKEN=$HF_TOKEN /opt/vllm-env/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3.6-35B-A3B-FP8',
    local_dir='/opt/models/Qwen3.6-35B-A3B-FP8',
    max_workers=4,
)
"
```

### Model choice guide

| Variant | Size | Speed (with MTP) | When to use |
|---|---|---|---|
| `Lorbus/Qwen3.6-27B-int4-AutoRound` | 19 GB | **120 t/s** | **Default — best speed/quality on 96GB** |
| `Qwen/Qwen3.6-35B-A3B-FP8` | 35 GB | **200 t/s** | MoE — highest throughput |
| `Qwen/Qwen3.6-27B-FP8` | 38 GB | ~80 t/s (no MTP)* | Near-lossless quality, lower speed |

*FP8 27B with MTP on WSL2 is blocked by the flashinfer JIT bug (see below).
The MoE FP8 works because its attention pattern avoids the problematic kernel.

**Do NOT use the GPTQ-Int4 MoE** — it lacks MTP weights (mtp_num_layers
missing from config.json) and crashes with speculative decoding.

---

## Step 6 — Create the startup script

```bash
sudo tee /usr/local/bin/start-vllm.sh > /dev/null <<'EOF'
#!/bin/bash
set -euo pipefail

# CRITICAL: Don't put /usr/local/cuda/bin in PATH if it's CUDA 12.8.
export PATH=/usr/bin:/usr/sbin
export CUDA_HOME=/opt/vllm-env/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64

export CUDA_VISIBLE_DEVICES=0
export FLASHINFER_CUDA_ARCH_LIST="12.0"
export TORCH_CUDA_ARCH_LIST="12.0"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_FLOAT32_MATMUL_PRECISION=high
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export NCCL_CUMEM_ENABLE=0
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=8

# CRITICAL: Do NOT set these on WSL2:
# - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (breaks MTP subprocess)
# - VLLM_USE_FLASHINFER_SAMPLER=1 (JIT fails on sm_120)
# - --kv-cache-dtype fp8_e5m2 (triggers broken flashinfer JIT during CUDA graph capture)
# - --attention-backend flashinfer (CCCL header incompatibility on PyPI CUDA 13)

exec /opt/vllm-env/bin/python -m vllm.entrypoints.openai.api_server \
  --model /opt/models/Qwen3.6-27B-int4-AutoRound \
  --served-model-name qwen3.6-27b \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --quantization auto_round \
  --attention-backend flash_attn \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 8 \
  --skip-mm-profiling \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --language-model-only \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
EOF

sudo chmod +x /usr/local/bin/start-vllm.sh
```

### Why `flash_attn` not `flashinfer`

The v1 README recommended `flashinfer`. On WSL2 with PyPI-only CUDA 13 (no system toolkit), this **crashes** during CUDA graph capture:

```
RuntimeError: Ninja build failed...
error: "CUDA compiler and CUDA toolkit headers are incompatible"
```

The CCCL headers in the PyPI `nvidia-cuda-nvcc` package conflict with flashinfer's JIT compilation for the `batch_prefill_with_kv_cache` kernel on sm_120f. This is also why `--kv-cache-dtype fp8_e5m2` crashes — it triggers the same flashinfer JIT path.

`flash_attn` ships precompiled wheels that work on Blackwell out of the box. The tradeoff: no FP8 KV cache. At 96GB with INT4 weights (17GB), you still get ~228K tokens of BF16 KV cache — plenty for 256K context.

If you have the **full system CUDA 13 toolkit** (`apt install cuda-toolkit-13-0`), you can switch to `flashinfer` + `--kv-cache-dtype fp8_e4m3` for ~10-15% more speed on long contexts.

### Flag reference

| Flag | What it does |
|------|--------------|
| `--attention-backend flash_attn` | Precompiled attention — avoids broken flashinfer JIT on WSL2 |
| `--quantization auto_round` | INT4 AutoRound quantization (Lorbus model) |
| `--gpu-memory-utilization 0.92` | 92% of 96GB for model + cache |
| `--max-num-seqs 8` | Concurrent request slots. vLLM's continuous batching interleaves decode across all 8. Increase to 16 for heavier multi-user loads; decrease to 2 for single-user lowest-latency |
| `--enable-chunked-prefill` | Splits long prefills into chunks interleaved with generation — prevents one long prompt from blocking short ones |
| `--enable-prefix-caching` | KV for identical prompt prefixes computed once and reused across requests. Huge win when many users share the same system prompt |
| `--language-model-only` | Skips vision encoder. Saves ~2GB VRAM |
| `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` | MTP speculative decoding — 3 draft tokens per forward pass |

---

## Step 7 — Systemd service

```bash
sudo tee /etc/systemd/system/vllm.service > /dev/null <<'EOF'
[Unit]
Description=vLLM Qwen3.6-27B INT4-AutoRound + MTP n=3 (Blackwell, ~120 tps)
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/start-vllm.sh
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal
Environment=HOME=/root

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable vllm
sudo systemctl start vllm
```

First boot takes ~90-170s (torch.compile + CUDA graph capture). Subsequent boots: ~40-65s (cached).

```bash
# Watch startup
sudo journalctl -u vllm -f

# Look for:
# "Application startup complete."
# "Avg generation throughput: 120.0 tokens/s"
```

---

## Step 8 — Verify

```bash
# Fast (no thinking — default)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Say hello and nothing else."}],
    "max_tokens": 50
  }'

# With reasoning chain (append :think)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b:think",
    "messages": [{"role": "user", "content": "Explain Fermat\'s Last Theorem"}],
    "max_tokens": 200
  }'
```

---

## Dual-GPU: One Instance Per GPU

If you have 2x Blackwell GPUs, run one independent vLLM instance per GPU behind a bridge/load-balancer. **Do not use tensor-parallel (TP=2)** — NCCL all-reduce over PCIe adds more latency than it saves for a 27B model.

Each GPU runs its own full vLLM server with `--max-num-seqs 8`. vLLM's continuous batching handles concurrency within each instance. A lightweight bridge in front routes requests to the least-loaded GPU and handles health checking.

### GPU 1 instance (35B FP8 MoE)

```bash
# GPU 1 runs the FP8 MoE with MTP n=3. Create a separate start script:
sudo tee /usr/local/bin/start-vllm-gpu1.sh > /dev/null <<'VLLM'
#!/bin/bash
set -euo pipefail
# ... same env vars as GPU 0 but CUDA_VISIBLE_DEVICES=1 ...
exec /opt/vllm-env/bin/python -m vllm.entrypoints.openai.api_server \
  --model /opt/models/Qwen3.6-35B-A3B-FP8 \
  --served-model-name qwen3.6-35b-a3b \
  --host 127.0.0.1 --port 8001 \
  --trust-remote-code \
  --dtype auto \
  --attention-backend flash_attn \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --language-model-only \
  --skip-mm-profiling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
VLLM
sudo chmod +x /usr/local/bin/start-vllm-gpu1.sh
```

### Systemd unit for GPU 1 (sequential start)

GPU 1's service waits for GPU 0 to be healthy before starting. This prevents concurrent `torch.compile` from deadlocking on shared CPU (load avg > 7 when both compile simultaneously).

```bash
sudo tee /etc/systemd/system/vllm-gpu1.service > /dev/null <<'EOF'
[Unit]
Description=vLLM GPU 1 — Qwen3.6-35B-A3B FP8 + MTP n=3 (~200 tps)
After=vllm.service

[Service]
Type=simple
# Wait for GPU 0 to finish booting before starting GPU 1
ExecStartPre=/bin/bash -c 'until curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; do sleep 5; done'
ExecStart=/usr/local/bin/start-vllm-gpu1.sh
Restart=on-failure
RestartSec=30
Environment=HOME=/root

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable vllm-gpu1
```

### Bridge (replaces nginx)

This repo includes a Go bridge binary (`vllm-bridge`) that runs inside WSL2 alongside vLLM. It replaces nginx with built-in:
- **Model-aware routing** — probes `/v1/models` on each backend, routes requests to the GPU that serves the requested model
- **Thinking off by default** — injects `enable_thinking: false` for fast time-to-first-token. Append `:think` to the model name to opt into reasoning (e.g. `qwen3.6-35b-a3b:think`)
- **Least-connections load balancing** across backends sharing the same model
- **Per-backend health tracking** with 10s probe interval and circuit breaker
- **SSE streaming** — flush every chunk, no write timeout
- **Retry on transient errors** for GET requests

```bash
# Install the bridge binary
sudo cp vllm-bridge /usr/local/bin/vllm-bridge
sudo chmod +x /usr/local/bin/vllm-bridge

# Systemd unit
sudo tee /etc/systemd/system/vllm-bridge.service > /dev/null <<'EOF'
[Unit]
Description=vLLM Bridge — health-aware LB + SSE streaming
After=vllm.service
Wants=vllm.service vllm-gpu1.service

[Service]
Type=simple
ExecStart=/usr/local/bin/vllm-bridge
Environment=PORT=8098
Environment=VLLM_BACKENDS=http://127.0.0.1:8000,http://127.0.0.1:8001
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable vllm-bridge
```

### Start everything

```bash
sudo systemctl start vllm          # GPU 0 — ~90s first boot, ~40s cached
sudo systemctl start vllm-gpu1     # GPU 1 — waits for GPU 0, then ~40s
sudo systemctl start vllm-bridge   # Bridge — waits for at least one backend
```

### Hit the bridge

```bash
curl http://localhost:8098/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'

# Check backend health
curl http://localhost:8098/health
```

### Single-user low-latency variant

If you're a single user firing parallel API calls (agent tool calls, batch processing), you can also run two replicas of the same model behind the bridge. This trades memory efficiency for per-request parallelism — your C=4 requests are spread across GPUs instead of queued. Measured: **234 tok/s aggregate at C=4** with this pattern. The downside: model weights are loaded twice (17GB x2). For most use cases, the default one-instance-per-GPU with `--max-num-seqs 8` is better.

---

## Troubleshooting

### WSL2 crashes during vLLM startup

**Cause:** `vmIdleTimeout` not set — WSL shuts down 15s after your SSH session ends, killing the vLLM process mid-startup. The next systemd restart triggers a crash loop that can destabilize WSL entirely.

**Fix:** Add `vmIdleTimeout=-1` to `.wslconfig` (see Prerequisites above).

### `RuntimeError: Ninja build failed ... headers are incompatible`

**Cause:** flashinfer JIT compilation is broken on sm_120 with PyPI CUDA 13 fragments. Triggered by `--attention-backend flashinfer` or `--kv-cache-dtype fp8_e5m2`.

**Fix:** Use `--attention-backend flash_attn` and remove `--kv-cache-dtype` (this guide's default).

### `RuntimeError: CUDA driver error: unknown error` at `torch.zeros()`

**Cause:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set. WSL's CUDA driver can't handle the expandable-segments allocator across fork/spawn.

**Fix:** `unset PYTORCH_CUDA_ALLOC_CONF` — do NOT set this on WSL2.

### `FlashInfer requires GPUs with sm75 or higher`

**Cause:** flashinfer can't detect Blackwell — it falls back to the system CUDA 12.8 nvcc.

**Fix:** Ensure `CUDA_HOME` points at the venv's CUDA 13: `export CUDA_HOME=/opt/vllm-env/lib/python3.12/site-packages/nvidia/cu13`

### Speeds < 50 tok/s — MTP not active

Check: `journalctl -u vllm | grep -E "MTP|SpecDecoding"`

Must see:
- `Resolved architecture: Qwen3_5MTP`
- `Detected MTP model. Sharing target model embedding/lm_head weights`
- `SpecDecoding metrics: Mean acceptance length: N.NN`

### Both replicas fail to start (dual-GPU)

**Cause:** Starting both simultaneously causes torch.compile CPU contention (load avg > 7). Neither finishes compiling.

**Fix:** Always start sequentially — wait for R1 to serve before starting R2.

### Slow first request (~3-5 minutes)

Normal. vLLM AOT-compiles the model graph and caches it at `~/.cache/vllm/torch_compile_cache/`. Second boot skips the compile (~40s startup).

---

## Docker (Alternative to Manual Setup)

The repo includes Dockerfiles and docker-compose for both single-GPU and dual-GPU setups:

```bash
git clone https://github.com/lastloop-ai/vllm-blackwell-guide.git
cd vllm-blackwell-guide

# Download model
pip install huggingface-hub
huggingface-cli download Lorbus/Qwen3.6-27B-int4-AutoRound \
  --local-dir ./models/Qwen3.6-27B-int4-AutoRound \
  --local-dir-use-symlinks False

# Single GPU
docker compose up -d

# Dual GPU (both GPUs + bridge LB)
docker compose --profile dual-gpu up -d
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `/models/Qwen3.6-27B-int4-AutoRound` | Model path inside container |
| `MODEL_NAME` | `qwen3.6-27b` | Name exposed via API |
| `MAX_MODEL_LEN` | `262144` | Context length (256K) |
| `GPU_MEM_UTIL` | `0.92` | VRAM fraction |
| `MAX_NUM_SEQS` | `8` | Concurrent request slots per GPU |
| `NUM_SPEC_TOKENS` | `3` | MTP draft tokens per step |

---

## Community Comparison

| Source | Hardware | Config | tok/s (per GPU) | Notes |
|---|---|---|
| **This guide** | RTX PRO 6000 96GB | 27B INT4 + MTP n=3 | **120** | WSL2, flash_attn |
| **This guide** | RTX PRO 6000 96GB | 35B FP8 MoE + MTP n=3 | **200** | WSL2, flash_attn |
| [CobraPhil](https://github.com/CobraPhil/qwen36-27b-single-5090) | RTX 5090 32GB | INT4 + Genesis + MTP n=3 | ~160 | Blackwell 32GB |
| [Ollman blog](https://alexander-ollman.github.io/qwen3.6-on-rtx3090/) | RTX 3090 24GB | INT4 + Genesis + MTP n=5 | 100 | Native Linux |
| [noonghunna](https://github.com/noonghunna/qwen36-27b-single-3090) | RTX 3090 24GB | INT4 + Genesis + MTP n=3 | 50-70 | Single 3090 |
| [Sandermage/Genesis](https://github.com/Sandermage/genesis-vllm-patches) | 2x A5000 40GB | INT4 + TQ k8v4 + MTP n=3 | 103 | Patch framework |

---

## Credits

This recipe builds on work from several people:

- [u/Kindly-Cantaloupe978](https://www.reddit.com/r/LocalLLaMA/comments/1sv8eua/) — original 80 t/s recipe on RTX 3090
- [Wasif Basharat's Medium write-up](https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914) — documented the `.tolist()` CUDA graph bug
- [noonghunna/qwen36-27b-single-3090](https://github.com/noonghunna/qwen36-27b-single-3090) — the `patch_tolist_cudagraph.py` patch
- [Alexander Ollman](https://alexander-ollman.github.io/qwen3.6-on-rtx3090/) — dual-3090 benchmark methodology, 225 t/s aggregate
- [Sandermage/genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches) — runtime patching framework for vLLM
- The vLLM team for [PR #36325](https://github.com/vllm-project/vllm/pull/36325) (Blackwell TMA fix)

What I added: identified the flashinfer JIT root cause on WSL2 (`flash_attn` workaround), WSL2 `vmIdleTimeout` fix, multi-GPU architecture with health-aware bridge (replaces nginx), sequential startup to avoid torch.compile deadlock, and empirical benchmark data across 5 configs.

---

*Tested May 2026. Questions or issues, [open an issue](https://github.com/lastloop-ai/vllm-blackwell-guide/issues).*
