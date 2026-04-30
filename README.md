# Qwen3.6 on RTX PRO 6000 Blackwell — 120 t/s (27B) / 200 t/s (35B MoE), step by step

## The Numbers

| Setup | Hardware | Context | t/s |
|---|---|---|---|
| llama.cpp (best GGUF) | RTX PRO 6000 96GB | 128K | ~35 |
| vLLM FP8 (no MTP) | RTX PRO 6000 96GB | 128K | ~39 |
| **vLLM FP8 + MTP n=3** | **RTX PRO 6000 96GB** | **256K** | **~115–120** |
| **vLLM FP8 + MTP n=3 (35B-A3B MoE)** | **RTX PRO 6000 96GB** | **256K** | **~200** |

MTP = Multi-Token Prediction speculative decoding. It generates 3 draft tokens per forward pass, so you get roughly 3× the throughput for "free" once the draft model warms up.

---

## Prerequisites

- RTX PRO 6000 Blackwell (96GB) — the recipe is specifically for this card
- Windows 11 + WSL2 with Ubuntu 24.04 (tested; bare-metal Linux should be identical)
- ~40GB free disk space for the 27B model, ~55GB for the 35B MoE
- Basic comfort with the terminal

---

## Step 1 — WSL2 + Ubuntu 24.04

**Why:** vLLM is Linux-native. WSL2 provides a full Linux kernel and near-zero-overhead CUDA pass-through. The setup is identical to bare-metal Linux once you're inside the distro.

```bash
# In PowerShell (as Admin)
wsl --install -d Ubuntu-24.04
wsl --set-default-version 2
# Reboot, set up your Ubuntu user, then:
wsl -d Ubuntu-24.04
```

---

## Step 2 — NVIDIA Drivers + CUDA 12.8

**Why:** Blackwell (sm_120) is only fully supported from CUDA 12.8 onwards. The Windows host driver must be 570.xx or newer. Inside WSL2, the CUDA toolkit is installed separately from the host driver.

```bash
# Verify host driver passes through
nvidia-smi
# Should show your RTX PRO 6000 and CUDA 12.8

# Install CUDA 12.8 toolkit inside WSL2
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## Step 3 — Python 3.12 + Virtual Environment

**Why:** vLLM v0.19.1 is built against Python 3.12. Using a venv prevents system package conflicts and makes the entire setup trivial to nuke and rebuild.

```bash
sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv python3-pip git

mkdir -p ~/vllm-setup && cd ~/vllm-setup
python3.12 -m venv venv
source venv/bin/activate

# Verify
python --version  # 3.12.x
```

---

## Step 4 — Install vLLM v0.19.1 (`cu130` wheel)

**Why:** v0.19.1 is the first release that contains the Blackwell TMA fix and the MTP loader. The `cu130` wheel bundles the PyPI CUDA 13 toolchain, so you don't need a full system CUDA development install.

```bash
# Inside the venv
pip install --upgrade pip

# Install vLLM with CUDA 13 support
pip install vllm==0.19.1 --extra-index-url https://download.pytorch.org/whl/cu130

# Verify
vllm --version  # Should show 0.19.1
```

---

## Step 5 — Apply the Blackwell TMA Patch

**Why:** vLLM's Triton autotuner checks `is_tma_supported` and returns `True` for any compute capability >= 9. Blackwell consumer (sm_120) doesn't actually implement TMA — the descriptor buffer allocations blow up VRAM during kernel warmup and you get a silent OOM.

```bash
# Locate the file inside your venv
VENV_PATH=$(python -c "import vllm; print(vllm.__path__[0])")
FILE="$VENV_PATH/model_executor/layers/fla/ops/utils.py"

# Backup
cp "$FILE" "$FILE.bak"

# Patch: cap TMA support at < sm_120 (sed is fragile here, use Python)
python -c "
import pathlib
p = pathlib.Path('$FILE')
src = p.read_text()
patched = src.replace(
    'torch.cuda.get_device_capability(0)[0] >= 9) and (',
    'torch.cuda.get_device_capability(0)[0] >= 9 and torch.cuda.get_device_capability(0)[0] < 12) and ('
)
assert patched != src, 'Pattern not found — check vLLM version'
p.write_text(patched)
print('Patched. TMA now disabled for Blackwell (sm_120).')
"
```

---

## Step 6 — Install `flashinfer` Attention Backend

**Why:** FlashInfer ships with precompiled Blackwell kernels. If you use `flash_attn`, it tries to JIT-compile custom kernels at runtime, which requires a full CUDA toolkit that WSL2 doesn't cleanly provide. FlashInfer avoids this entirely.

```bash
pip install flashinfer-python

# Verify it can see your GPU
python -c "import flashinfer; print('FlashInfer OK')"
```

---

## Step 7 — Download Qwen3.6-27B-FP8

**Why:** FP8 is the sweet spot for Blackwell — it runs on the native tensor cores without the quality loss of heavier quantization, and at 96GB you have room for the full model (~38GB on disk) plus a massive KV cache. No GGUF, no AutoRound, no compromises needed at this VRAM tier.

```bash
# Create model directory
mkdir -p ~/models
cd ~/models

# Download with huggingface-cli
pip install huggingface-hub

huggingface-cli download Qwen/Qwen3.6-27B-FP8 \
  --local-dir ./Qwen3.6-27B-FP8 \
  --local-dir-use-symlinks False

# Verify size (~38GB)
du -sh ./Qwen3.6-27B-FP8
```

---

## Step 8 — Apply `patch_tolist_cudagraph.py`

**Why:** Qwen's MTP loader calls `.tolist()` during CUDA graph capture warmup. That `.tolist()` forces a CPU/GPU synchronization that silently breaks graph compilation. Without this patch, CUDA graphs never get captured and you lose 40–77% of your performance.

```bash
# Download the patch
cd ~/vllm-setup
wget https://raw.githubusercontent.com/noonghunna/qwen36-27b-single-3090/main/patches/patch_tolist_cudagraph.py

# Run it against your vLLM install
python patch_tolist_cudagraph.py --vllm-path $(python -c "import vllm; print(vllm.__path__[0])")

echo "CUDA graph patch applied."
```

---

## Step 9 — Build the `vllm serve` Command with MTP n=3

**Why:** Multi-Token Prediction speculative decoding runs a small draft head alongside the main model. It predicts 3 tokens ahead per forward pass, and the main model verifies them in parallel. Acceptance rate peaks at ~95% once warm, giving you effectively 3× the tokens per unit time.

```bash
# Create a startup script (note: uses $HOME, not ~, so paths resolve correctly)
cat > ~/vllm-setup/start-vllm.sh << EOF
#!/bin/bash
set -e

source $HOME/vllm-setup/venv/bin/activate

python -m vllm.entrypoints.openai.api_server \\
  --model $HOME/models/Qwen3.6-27B-FP8 \\
  --served-model-name qwen3.6-27b \\
  --host 0.0.0.0 \\
  --port 8000 \\
  --trust-remote-code \\
  --dtype auto \\
  --attention-backend flashinfer \\
  --kv-cache-dtype fp8_e5m2 \\
  --max-model-len 131072 \\
  --gpu-memory-utilization 0.92 \\
  --max-num-seqs 4 \\
  --max-num-batched-tokens 2048 \\
  --enable-prefix-caching \\
  --enable-chunked-prefill \\
  --enable-auto-tool-choice \\
  --tool-call-parser qwen3_coder \\
  --reasoning-parser qwen3 \\
  --language-model-only \\
  --speculative-config '{"model": "$HOME/models/Qwen3.6-27B-FP8", "num_speculative_tokens": 3, "draft_model_uses_mrope": true, "draft_model_uses_xdrope_dim": 0}'
EOF

chmod +x ~/vllm-setup/start-vllm.sh
```

**Flag reference:**

| Flag | What it does |
|------|--------------|
| `--kv-cache-dtype fp8_e5m2` | Stores KV cache in FP8 instead of FP16 — doubles the number of tokens that fit in cache for the same VRAM |
| `--gpu-memory-utilization 0.92` | Reserves 92% of 96GB (~88GB) for model + cache. Push to `0.97` for 256K context |
| `--max-num-seqs 4` | Continuous batching — 4 concurrent requests share the GPU instead of queueing serially |
| `--max-num-batched-tokens 2048` | Chunk size for batched prefill |
| `--enable-chunked-prefill` | Splits long prefills into chunks, interleaved with generation — prevents one long prompt from blocking short ones |
| `--enable-prefix-caching` | Caches KV for identical prompt prefixes — your system prompt is computed once and reused across requests |
| `--language-model-only` | Skips loading the vision encoder (Qwen3.6 is multimodal). Saves ~2GB VRAM |

**Note:** `--max-model-len 131072` gives 128K context. If you want the full 256K, change it to `256000` and bump `--gpu-memory-utilization` to `0.97`.

---

## Step 10 — Verify with a Warmup Request

**Why:** Blackwell CUDA graph compilation failures are completely silent until the first real inference call. The server starts fine, then dies on the first token. A warmup request forces graph capture and exposes any remaining issues immediately.

```bash
# Start the server in one terminal
~/vllm-setup/start-vllm.sh

# Wait for "Application startup complete." (~2–3 minutes on first boot)

# In another terminal, run a warmup:
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Say hello and nothing else."}],
    "max_tokens": 50
  }'

# You should get a JSON response with generated text.
# Check server logs for the throughput number (t/s) in the last line.
```

---

## Bonus Variant — 35B-A3B MoE at ~200 t/s

Same exact recipe, swap the model. The 35B-A3B is a Mixture-of-Experts with 256 total experts and 8 active per token. It uses less memory than you'd think (~35GB in FP8) and the Marlin MoE backend is well-optimized for Blackwell.

```bash
# Download the MoE variant
huggingface-cli download Qwen/Qwen3.6-35B-A3B-FP8 \
  --local-dir ./Qwen3.6-35B-A3B-FP8 \
  --local-dir-use-symlinks False

# Swap the model path in your startup script
# Change: --model ~/models/Qwen3.6-27B-FP8
# To:     --model ~/models/Qwen3.6-35B-A3B-FP8
# And update: --served-model-name qwen3.6-35b-a3b
#
# 256K context fits on 96GB with the MoE too — same --max-model-len 256000 works.
```

Expected performance: **~200 tok/s** sustained, 256K context. Use this when you want raw throughput. Use the 27B when MoE expert noise is undesirable for your use case (e.g., long coherent prose).

---

## Systemd Service (Auto-Start on WSL Boot)

```bash
sudo tee /etc/systemd/system/vllm.service > /dev/null << 'EOF'
[Unit]
Description=vLLM OpenAI API Server (Qwen3.6-27B-FP8 + MTP n=3)
After=network.target

[Service]
Type=simple
ExecStart=/home/YOUR_USER/vllm-setup/start-vllm.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable vllm
sudo systemctl start vllm

# Check status
sudo systemctl status vllm
```

---

## Performance Validation

Once warm (after ~5–10 requests), check your server logs. You should see lines like:

```
Avg prompt throughput: X tok/s
Avg generation throughput: 115.4 tok/s
```

If you're seeing ~40 tok/s or less, MTP is not active. Common causes:
- Missing the `patch_tolist_cudagraph.py` patch (CUDA graphs failed to capture)
- `--speculative-config` malformed or missing
- Model not FP8 (e.g., accidentally downloaded BF16)

If the server crashes on first request with a Triton OOM, the Blackwell TMA patch (Step 5) was not applied correctly.

---

## What This Setup Is

This is a **local OpenAI-compatible API server** running on your own GPU. You can point anything that speaks the OpenAI client protocol at `http://localhost:8000/v1`:

- OpenWebUI
- AnythingLLM
- Continue.dev
- Custom scripts via `openai` Python package

No cloud dependency, no API keys, no rate limits. Just a workstation with a very large GPU.

---

## Credits

This recipe builds on work from several people:

- [u/Kindly-Cantaloupe978](https://www.reddit.com/r/LocalLLaMA/comments/1sv8eua/qwen3627b_at_80_tps_with_218k_context_window_on/) — original 80 t/s recipe on RTX 3090
- [Wasif Basharat's Medium write-up](https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914) — 85 t/s overnight stack, documented the `.tolist()` CUDA graph bug
- [noonghunna/qwen36-27b-single-3090](https://github.com/noonghunna/qwen36-27b-single-3090) — the `patch_tolist_cudagraph.py` patch
- The vLLM team for [PR #36325](https://github.com/vllm-project/vllm/pull/36325) (Blackwell TMA fix)

What I added: adapted the 24GB/3090 recipes to the 96GB RTX PRO 6000 where FP8 + MTP n=3 just fits without the OOM workarounds the smaller cards need.

---

*Questions or issues, drop them below.*
