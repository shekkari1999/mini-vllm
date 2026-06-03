#!/usr/bin/env bash
# Run this as soon as you SSH into a Vast.ai GPU instance.
set -euo pipefail

cd "$(dirname "$0")"

echo "=== mini-vllm (Llama-2-7B) ==="

if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. Rent a CUDA template on Vast.ai."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [[ "$VRAM_MB" -lt 20000 ]]; then
  echo "WARNING: Llama-2-7B needs ~20+ GB VRAM. You have ${VRAM_MB} MB."
  echo "         Use a 24 GB GPU (RTX 3090 / 4090) or reduce num_blocks in config.py."
fi

if ! command -v uv &>/dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "Using HF_TOKEN for Hugging Face auth..."
  uv run huggingface-cli login --token "$HF_TOKEN"
elif [[ ! -f "$HOME/.cache/huggingface/token" ]]; then
  echo "WARNING: No HF_TOKEN set. Export it before running:"
  echo "  export HF_TOKEN=hf_..."
  echo "Also accept the Llama 2 license: https://huggingface.co/meta-llama/Llama-2-7b-hf"
fi

echo "Installing dependencies..."
uv sync

echo "Checking CUDA..."
uv run python -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available())"

echo ""
echo "=== Demo (meta-llama/Llama-2-7b-hf) ==="
uv run python - <<'PY'
from minivllm import LLM, SamplingParams

llm = LLM()
prompts = [
    "The capital of France is",
    "In machine learning, a tensor is",
]
results = llm.generate(prompts, SamplingParams(max_tokens=32))
for rid, text in sorted(results.items()):
    print(f"[{rid}] {text}")
PY

echo ""
echo "=== Benchmark ==="
uv run python bench.py --warmup --max-tokens 64 --batch-size 4

echo ""
echo "Done."
