#!/usr/bin/env bash
# One-shot demo + benchmarks on a rented GPU.
set -euo pipefail

cd "$(dirname "$0")"

echo "=== mini-vllm (Qwen2.5-7B-Instruct) ==="

if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. Rent a CUDA instance."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  uv run huggingface-cli login --token "$HF_TOKEN"
elif [[ ! -f "$HOME/.cache/huggingface/token" ]]; then
  echo "Set HF_TOKEN or run: huggingface-cli login"
fi

uv sync
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo ""
echo "=== Demo ==="
uv run python - <<'PY'
from minivllm import LLM, SamplingParams

llm = LLM()
for rid, text in sorted(llm.generate(
    ["The capital of France is", "Machine learning is"],
    SamplingParams(max_tokens=32),
).items()):
    print(f"[{rid}] {text}")
PY

echo ""
echo "=== Benchmarks (8 × 128 tokens, 16-request TTFT) ==="
uv run python benchmarks/run_all.py --warmup
uv run python benchmarks/plot_results.py

echo "Done."
