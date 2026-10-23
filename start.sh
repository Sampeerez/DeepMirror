#!/usr/bin/env sh
# OpenVoiceChanger launcher for Linux and macOS: ./start.sh [--port 8080] [--device cpu]
# Installs uv (per user, no root) if needed; uv then fetches Python and every
# dependency into .venv and starts the studio.
set -eu
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv, the Python package manager (one time only)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  PATH="$HOME/.local/bin:$PATH"
  export PATH
fi

# PyTorch build: CUDA when an NVIDIA driver is present, CPU otherwise (macOS
# gets the standard build, which uses the Apple GPU through MPS).
# Force one with OVC_TORCH=cpu or OVC_TORCH=cuda.
if [ -z "${OVC_TORCH:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then OVC_TORCH=cuda; else OVC_TORCH=cpu; fi
fi

exec uv run --extra "$OVC_TORCH" --extra neural-f0 openvoicechanger "$@"
