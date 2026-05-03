# Qwen3.6 on RTX PRO 6000 Blackwell — 120 t/s (27B) / 234 t/s (dual-GPU LB), step by step

Qwen3.6 27B runs well on the RTX PRO 6000 Blackwell (96GB), but getting vLLM + MTP speculative decoding to actually work on sm_120 in WSL2 requires workarounds that aren't documented anywhere in one place. This is that guide — from a fresh WSL2 install to a working OpenAI-compatible API server. Tested on WSL2; bare-metal Linux should be identical or faster.

*Last tested: May 2026 — vLLM v0.19.2rc1.dev214, CUDA 13.0 (PyPI), NVIDIA driver 581.42, Ubuntu 24.04 on WSL2.*

## The Numbers (empirically verified, May 2026)

All numbers measured on RTX PRO 6000 Blackwell (96GB, 1792 GB/s), 3 runs of 1024 output tokens each:

| Config | Hardware | t/s | Notes |
|---|---|---|---|
| Eager mode, no MTP, no CUDA graphs | 1x RTX PRO 6000 | **24** | Bandwidth floor (dispatch overhead dominates) |
| CUDA graphs + flash_attn, no MTP | 1x RTX PRO 6000 | **75** | 3.1x over eager — matches 1792 GB/s theory |
| **CUDA graphs + flash_attn + MTP n=3** | **1x RTX PRO 6000** | **120** | **This guide's target config** |
| CUDA graphs + flash_attn + MTP n=5 | 1x RTX PRO 6000 | **125** | +4% over n=3, marginal gain |
| **Dual-replica nginx LB, C=4** | **2x RTX PRO 6000** | **234 aggregate** | **Two independent servers, no NCCL** |

MTP speculative decoding metrics: mean acceptance length 3.19, per-position acceptance 0.87/0.72/0.60, avg draft acceptance 73%.

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

### Model choice guide

| Variant | Size | Speed (with MTP) | When to use |
|---|---|---|---|
| `Lorbus/Qwen3.6-27B-int4-AutoRound` | 19 GB | **120 t/s** | **Default — best speed/quality on 96GB** |
| `Qwen/Qwen3.6-27B-FP8` | 31 GB | ~80 t/s (no MTP)* | Near-lossless quality, lower speed |
| `Qwen/Qwen3.6-35B-A3B-FP8` | 35 GB | ~200 t/s | MoE — highest throughput |

*FP8 with MTP on WSL2 is blocked by the flashinfer JIT bug (see below).

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
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --quantization auto_round \
  --attention-backend flash_attn \
  --performance-mode interactivity \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 2 \
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

`flash_attn` ships precompiled wheels that work on Blackwell out of the box. The tradeoff: no FP8 KV cache. At 96GB with INT4 weights (17GB), you still get ~228K tokens of BF16 KV cache — plenty for 128K context.

If you have the **full system CUDA 13 toolkit** (`apt install cuda-toolkit-13-0`), you can switch to `flashinfer` + `--kv-cache-dtype fp8_e4m3` for ~10-15% more speed on long contexts.

### Flag reference

| Flag | What it does |
|------|--------------|
| `--attention-backend flash_attn` | Precompiled attention — avoids broken flashinfer JIT on WSL2 |
| `--quantization auto_round` | INT4 AutoRound quantization (Lorbus model) |
| `--gpu-memory-utilization 0.90` | 90% of 96GB for model + cache |
| `--max-num-seqs 2` | Concurrent request slots (2 for single-user, 4 for multi-user) |
| `--enable-chunked-prefill` | Prevents long prompts from blocking short ones |
| `--enable-prefix-caching` | System prompt KV computed once and reused |
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
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Say hello and nothing else."}],
    "max_tokens": 50
  }'
```

---

## Dual-GPU: 234 tok/s aggregate with nginx LB

If you have 2x RTX PRO 6000, run two independent replicas behind nginx instead of tensor-parallel. TP=2 adds NCCL all-reduce overhead over PCIe that exceeds the benefit for a 27B model.

### When to use dual-replica vs single-replica

| Scenario | Recommendation | Why |
|---|---|---|
| **Single user, parallel API calls** (agents, batch) | **Dual-replica LB** | Spreads your C=4 calls across both GPUs — 234 tok/s aggregate |
| **Multi-user server** (many independent users) | **Single replica per GPU, high `--max-num-seqs`** | vLLM's continuous batching already parallelizes users; avoids duplicating 17GB model weights + CUDA graph memory on each GPU |
| **Maximum context length** | **Single replica, `--gpu-memory-utilization 0.95`** | All 96GB for one KV cache pool |

The dual-replica pattern **doubles model memory usage** (17GB x2) and CUDA graph memory. It optimizes for **per-user latency across parallel requests**, not for total multi-user throughput. For a shared server, one replica with `--max-num-seqs 8` on each GPU independently (different ports, no LB) is more efficient.

### Measured results (May 2026)

| Concurrency | Aggregate tok/s | Notes |
|---|---|---|
| C=1 | 103-135 | Single stream, routed to one GPU |
| C=2 | 110-125 | Both GPUs active |
| C=4 | **198-234** | Sweet spot for agentic tool-call workloads |

### Setup

**1. GPU 1 start script** (copy of main, change GPU and port):

```bash
sudo cp /usr/local/bin/start-vllm.sh /usr/local/bin/start-vllm-gpu1.sh
sudo sed -i 's/CUDA_VISIBLE_DEVICES=0/CUDA_VISIBLE_DEVICES=1/' /usr/local/bin/start-vllm-gpu1.sh
sudo sed -i 's/--port 8000/--port 8001/' /usr/local/bin/start-vllm-gpu1.sh
```

**2. Systemd unit for replica 2:**

```bash
sudo tee /etc/systemd/system/vllm-gpu1.service > /dev/null <<'EOF'
[Unit]
Description=vLLM replica 2 (GPU 1, port 8001)
After=network.target
[Service]
Type=simple
ExecStart=/usr/local/bin/start-vllm-gpu1.sh
Restart=on-failure
RestartSec=30
Environment=HOME=/root
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
```

**3. nginx LB:**

```bash
sudo apt install -y nginx
sudo tee /etc/nginx/sites-available/vllm-lb > /dev/null <<'NGINX'
upstream vllm_pool {
    least_conn;
    server 127.0.0.1:8000 max_fails=3 fail_timeout=10s;
    server 127.0.0.1:8001 max_fails=3 fail_timeout=10s;
}
server {
    listen 8400;
    proxy_read_timeout    900s;
    proxy_send_timeout    900s;
    proxy_buffering       off;
    proxy_request_buffering off;
    location / {
        proxy_pass http://vllm_pool;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header X-Accel-Buffering no;
    }
}
NGINX
sudo ln -sf /etc/nginx/sites-available/vllm-lb /etc/nginx/sites-enabled/vllm-lb
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

**4. Start sequentially** (concurrent torch.compile deadlocks on shared CPU):

```bash
sudo systemctl start vllm        # GPU 0 — wait ~90s
# ... wait for "Application startup complete" in journal ...
sudo systemctl start vllm-gpu1   # GPU 1 — ~40s (cached compile)
```

**5. Hit the LB endpoint:**

```bash
curl http://localhost:8400/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'
```

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

The repo includes a Dockerfile and docker-compose that bake in the working config:

```bash
git clone https://github.com/lastloop-ai/vllm-blackwell-guide.git
cd vllm-blackwell-guide

pip install huggingface-hub
huggingface-cli download Lorbus/Qwen3.6-27B-int4-AutoRound \
  --local-dir ./models/Qwen3.6-27B-int4-AutoRound \
  --local-dir-use-symlinks False

docker compose --profile 27b up -d
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `/models/Qwen3.6-27B-int4-AutoRound` | Model path inside container |
| `MODEL_NAME` | `qwen3.6-27b` | Name exposed via API |
| `MAX_MODEL_LEN` | `131072` | Context length |
| `GPU_MEM_UTIL` | `0.90` | VRAM fraction |
| `NUM_SPEC_TOKENS` | `3` | MTP draft tokens per step |
| `CUDA_VISIBLE_DEVICES` | `0` | Which GPU |

---

## Community Comparison

| Source | Hardware | Config | tok/s | Notes |
|---|---|---|---|---|
| **This guide** | 1x RTX PRO 6000 | INT4 + MTP n=3 | **120** | WSL2, flash_attn |
| **This guide** | 2x RTX PRO 6000 | 2-replica LB, C=4 | **234 agg** | WSL2, nginx |
| [Ollman blog](https://alexander-ollman.github.io/qwen3.6-on-rtx3090/) | 2x RTX 3090 | INT4 + Genesis + MTP n=5 | 100 (single) | Native Linux |
| [Ollman blog](https://alexander-ollman.github.io/qwen3.6-on-rtx3090/) | 2x RTX 3090 | 2-replica LB, C=4 | **225 agg** | Native Linux |
| [noonghunna](https://github.com/noonghunna/qwen36-27b-single-3090) | RTX 3090 | INT4 + Genesis + MTP n=3 | 50-70 | Single 3090 |
| [CobraPhil](https://github.com/CobraPhil/qwen36-27b-single-5090) | RTX 5090 | INT4 + Genesis + MTP n=3 | ~160 | Blackwell 32GB |
| [Sandermage/Genesis](https://github.com/Sandermage/genesis-vllm-patches) | 2x A5000 | INT4 + TQ k8v4 + MTP n=3 | 103 | Patch framework |

---

## Credits

This recipe builds on work from several people:

- [u/Kindly-Cantaloupe978](https://www.reddit.com/r/LocalLLaMA/comments/1sv8eua/) — original 80 t/s recipe on RTX 3090
- [Wasif Basharat's Medium write-up](https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914) — documented the `.tolist()` CUDA graph bug
- [noonghunna/qwen36-27b-single-3090](https://github.com/noonghunna/qwen36-27b-single-3090) — the `patch_tolist_cudagraph.py` patch
- [Alexander Ollman](https://alexander-ollman.github.io/qwen3.6-on-rtx3090/) — dual-3090 benchmark methodology, 225 t/s aggregate
- [Sandermage/genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches) — runtime patching framework for vLLM
- The vLLM team for [PR #36325](https://github.com/vllm-project/vllm/pull/36325) (Blackwell TMA fix)

What I added: identified the flashinfer JIT root cause on WSL2 (`flash_attn` workaround), WSL2 vmIdleTimeout fix, dual-replica nginx LB recipe, and empirical benchmark data across 5 configs.

---

*Tested May 2026. Questions or issues, [open an issue](https://github.com/lastloop-ai/vllm-blackwell-guide/issues).*
