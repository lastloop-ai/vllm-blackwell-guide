FROM vllm/vllm-openai:v0.19.1

# ── Blackwell TMA patch ──────────────────────────────────────────────────────
# sm_120 falsely reports TMA support → Triton autotuner OOMs during warmup.
# Cap is_tma_supported at compute capability < 12.
# Reference: https://github.com/vllm-project/vllm/pull/36325
RUN VLLM_PATH=$(python3 -c "import vllm; print(vllm.__path__[0])") && \
    FILE="$VLLM_PATH/model_executor/layers/fla/ops/utils.py" && \
    cp "$FILE" "$FILE.bak" && \
    python3 -c "\
import pathlib; \
p = pathlib.Path('$FILE'); \
src = p.read_text(); \
patched = src.replace( \
    'torch.cuda.get_device_capability(0)[0] >= 9) and (', \
    'torch.cuda.get_device_capability(0)[0] >= 9 and torch.cuda.get_device_capability(0)[0] < 12) and (' \
); \
assert patched != src, 'TMA pattern not found — check vLLM version'; \
p.write_text(patched); \
print('Patched: TMA disabled for Blackwell (sm_120).')"

# ── CUDA graph .tolist() patch (best-effort) ─────────────────────────────────
# Qwen MTP loader calls .tolist() during CUDA graph capture warmup, causing a
# CPU/GPU sync that breaks graph compilation. This patch is critical for
# TurboQuant KV (24GB cards) and beneficial for FP8 KV (96GB cards).
# On v0.19.1 stable the target file may not exist yet — apply if found, skip
# gracefully if not (FP8 KV path is less affected than TurboQuant).
# Source: https://github.com/noonghunna/qwen36-27b-single-3090
RUN VLLM_PATH=$(python3 -c "import vllm; print(vllm.__path__[0])") && \
    python3 -c "\
import pathlib, zipfile, io, urllib.request; \
url = 'https://github.com/noonghunna/qwen36-27b-single-3090/archive/refs/heads/master.zip'; \
data = urllib.request.urlopen(url).read(); \
z = zipfile.ZipFile(io.BytesIO(data)); \
patch_name = [n for n in z.namelist() if 'patch_tolist_cudagraph' in n][0]; \
pathlib.Path('/tmp/patch_tolist_cudagraph.py').write_bytes(z.read(patch_name)); \
print(f'Extracted {patch_name}')" && \
    python3 /tmp/patch_tolist_cudagraph.py --vllm-path "$VLLM_PATH" || \
    echo "WARN: tolist cudagraph patch skipped (target not found in this vLLM version — OK for FP8 KV)" && \
    rm -f /tmp/patch_tolist_cudagraph.py

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
