#!/usr/bin/env sh
# Voice Studio launcher for Linux and macOS: ./start.sh [--port 8080] [--device cpu]
# Installs uv (per user, no root) if needed and prepares Python and every
# dependency in .venv (visible here; the first time takes a few minutes). The
# studio then runs in the background and opens in the browser: this terminal
# can be closed, the log is in the studio's Logs tab, and closing the studio
# page stops it. On Linux an app-menu entry named after the app is added.
# Use --foreground to keep it attached to the terminal.
set -eu
cd "$(dirname "$0")"
HERE="$(pwd)"
# The app's name (scripts/rename_project.py changes it everywhere).
APP_NAME="Voice Studio"
APP_SLUG="voicestudio"

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

echo "Preparing $APP_NAME ($OVC_TORCH)..."
uv sync --extra "$OVC_TORCH" --extra neural-f0

FOREGROUND=0
for arg in "$@"; do [ "$arg" = "--foreground" ] && FOREGROUND=1; done
ARGS=""
for arg in "$@"; do [ "$arg" != "--foreground" ] && ARGS="$ARGS $arg"; done

if [ "$(uname)" = "Linux" ] && [ -d "$HOME/.local/share" ]; then
  mkdir -p "$HOME/.local/share/applications"
  UV_BIN="$(command -v uv)"
  rm -f "$HOME/.local/share/applications/openvoicechanger.desktop"
  cat > "$HOME/.local/share/applications/$APP_SLUG.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=$APP_NAME
Comment=Realtime voice changer studio
Exec=sh -c 'cd "$HERE" && exec "$UV_BIN" run --no-sync openvoicechanger'
Path=$HERE
Icon=audio-input-microphone
Terminal=false
Categories=AudioVideo;Audio;
EOF
fi

if [ "$FOREGROUND" = "1" ]; then
  # shellcheck disable=SC2086
  exec uv run --no-sync openvoicechanger $ARGS
fi
# shellcheck disable=SC2086
if command -v setsid >/dev/null 2>&1; then
  setsid uv run --no-sync openvoicechanger $ARGS >/dev/null 2>&1 < /dev/null &
else
  nohup uv run --no-sync openvoicechanger $ARGS >/dev/null 2>&1 < /dev/null &
fi
echo "$APP_NAME is starting in your browser. You can close this terminal."
